"""GSM8K exact match and one-sample MBPP pass@1.

MBPP execution uses a disposable child process with a timeout, NOT a security
sandbox. Run generated code in an isolated compute job with no credentials.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile


NUMBER = r'[-+]?\d[\d,]*(?:\.\d+)?'


def extract_number(text):
  # Prefer the requested final marker; never pick an arbitrary earlier CoT value.
  markers = re.findall(r'####\s*(' + NUMBER + r')', text)
  boxed = re.findall(r'\\boxed\{\s*(' + NUMBER + r')\s*\}', text)
  if markers or boxed:
    value = (markers or boxed)[-1]
  else:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
      return None
    match = re.fullmatch(r'(?:The answer is\s*)?\$?(' + NUMBER + r')\.?', lines[-1], re.I)
    if not match:
      return None
    value = match.group(1)
  try:
    return str(Decimal(value.replace(',', '')).normalize())
  except InvalidOperation:
    return None


def score_gsm8k(text, reference):
  actual, expected = extract_number(text), extract_number(reference)
  if expected is None:
    raise ValueError('Reference has no parseable numeric answer')
  return dict(passed=actual is not None and actual == expected,
              prediction=actual, expected=expected,
              status='ok' if actual is not None else 'unparseable')


def extract_code(text):
  # Gemma thinking text must never be executed as Python.
  for marker in ('</think>', '<|channel>final', '<|channel|>final'):
    if marker in text:
      text = text.rsplit(marker, 1)[1]
  blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S | re.I)
  if blocks:
    return blocks[-1].strip()
  return text.strip()


def score_mbpp(text, tests, test_imports=(), timeout=5):
  code = extract_code(text)
  program = '\n'.join([*test_imports, code, *tests])
  with tempfile.TemporaryDirectory(prefix='allm-mbpp-') as temporary:
    script = Path(temporary) / 'solution.py'
    script.write_text(program, encoding='utf-8')
    # Discard output to avoid an unbounded stream from a faulty candidate.
    process = subprocess.Popen(
        [sys.executable, '-I', str(script)], cwd=temporary,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=(os.name != 'nt'),
        env={'PATH': os.environ.get('PATH', ''),
             'SYSTEMROOT': os.environ.get('SYSTEMROOT', ''),
             'PYTHONIOENCODING': 'utf-8'},
    )
    try:
      result = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
      if os.name == 'nt':
        process.kill()
      else:
        os.killpg(process.pid, signal.SIGKILL)
      process.wait()
      return dict(passed=False, status='timeout')
  return dict(passed=result == 0, status='passed' if result == 0 else 'failed',
              exit_code=result)
