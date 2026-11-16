"""Download the upstream public Orbax checkpoint, pinned by GCS generation.

List first, check capacity, validate MD5 when provided, and save SHA256 for every
object. A partial download is resumable; it is never marked ready on failure.
"""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.parse
import urllib.request

from exp_code.baselines.data import digest

BUCKET = 'gemma-data'
PREFIX = 'checkpoints/diffusiongemma-26B-A4B-it/'


def list_objects(prefix):
  result, token = [], None
  while True:
    query = dict(prefix=prefix, maxResults=1000)
    if token:
      query['pageToken'] = token
    url = f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?' + urllib.parse.urlencode(query)
    with urllib.request.urlopen(url, timeout=60) as stream:
      page = json.load(stream)
    result.extend(page.get('items', []))
    token = page.get('nextPageToken')
    if not token:
      return result


def download(item, destination):
  if destination.is_file() and destination.stat().st_size == int(item['size']):
    # Re-hash existing files too: length alone is not an integrity check.
    pass
  else:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = (f'https://storage.googleapis.com/{BUCKET}/'
           + urllib.parse.quote(item['name'], safe='/')
           + '?generation=' + item['generation'])
    partial = destination.with_name(destination.name + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > int(item['size']):
      raise ValueError(f'Oversized partial file: {partial}')
    if offset < int(item['size']):
      request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
      with urllib.request.urlopen(request, timeout=120) as source:
        append = offset > 0 and source.status == 206
        if append and not source.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
          raise ValueError('Unexpected range response')
        with partial.open('ab' if append else 'wb') as out:
          shutil.copyfileobj(source, out, length=8*1024*1024)
    if partial.stat().st_size != int(item['size']):
      raise ValueError(f'Incomplete object: {item["name"]}')
    partial.replace(destination)
  sha, md5 = hashlib.sha256(), hashlib.md5()
  with destination.open('rb') as stream:
    for block in iter(lambda: stream.read(8*1024*1024), b''):
      sha.update(block)
      md5.update(block)
  if item.get('md5Hash') and base64.b64encode(md5.digest()).decode() != item['md5Hash']:
    raise ValueError(f'MD5 mismatch; remove damaged file and retry: {destination}')
  return dict(name=item['name'], generation=item['generation'],
              size=int(item['size']), sha256=sha.hexdigest())


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--list-only', action='store_true')
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  listing_file = args.output / 'gcs-objects.json'
  if listing_file.exists():
    objects = json.loads(listing_file.read_text())
  else:
    objects = [o for o in list_objects(PREFIX) if not o['name'].endswith('/')]
    tokenizer = list_objects('tokenizers/tokenizer_gemma4.model')
    objects.extend(o for o in tokenizer if o['name'] == 'tokenizers/tokenizer_gemma4.model')
    if not objects or not any(o['name'].endswith('_METADATA') for o in objects):
      raise ValueError('No valid checkpoint listing')
    listing_file.write_text(json.dumps(objects, indent=2))
  total = sum(int(o['size']) for o in objects)
  free = shutil.disk_usage(args.output).free
  print(json.dumps(dict(objects=len(objects), bytes=total, free_bytes=free)), flush=True)
  if args.list_only:
    return
  if free < total * 1.2:
    raise RuntimeError('Insufficient headroom for checkpoint download')

  def get_object(item):
    name = item['name']
    relative = name[len(PREFIX):] if name.startswith(PREFIX) else 'tokenizer.model'
    destination = args.output / relative
    if not destination.resolve().is_relative_to(args.output.resolve()):
      raise ValueError('Object name escapes destination')
    result = download(item, destination)
    print('VERIFIED', name, flush=True)
    return result

  with ThreadPoolExecutor(max_workers=4) as pool:
    verified = list(pool.map(get_object, objects))
  manifest = dict(source='gs://' + BUCKET + '/' + PREFIX,
                  files=verified, sha256=digest(verified))
  (args.output / 'verified-manifest.json').write_text(json.dumps(manifest, indent=2))
  print('CHECKPOINT_READY', manifest['sha256'], flush=True)


if __name__ == '__main__':
  main()
