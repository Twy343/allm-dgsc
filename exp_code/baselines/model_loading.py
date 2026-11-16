"""Restore unchanged weights using Kauldron's existing FSDP sharding rule."""
def load_model(checkpoint,model_shards=1):
  import jax
  from gemma import gm,diffusion
  from kauldron import kd
  if jax.device_count()!=model_shards:
    raise ValueError(f'Expected {model_shards} devices, got {jax.devices()}')
  model=diffusion.DiffusionGemma_26B_A4B()
  rule=kd.sharding.FSDPSharding() if model_shards>1 else None
  params=gm.ckpts.load_params(str(checkpoint),text_only=True,
                            sharding=rule,restore_concurrent_gb=4)
  tokenizer=gm.text.Gemma4Tokenizer(path=str(checkpoint/'tokenizer.model'))
  nbytes=sum(x.size*x.dtype.itemsize for x in jax.tree.leaves(params))
  print('PARAMETER_LOGICAL_BYTES',nbytes,'MODEL_SHARDS',model_shards,flush=True)
  print('DEVICE_MEMORY_AFTER_RESTORE',
        [dict(device=str(d),stats=d.memory_stats()) for d in jax.devices()],flush=True)
  return model,params,tokenizer,(kd.sharding.REPLICATED if model_shards>1 else None)
