from functools import partial
import importlib,json
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from common.jax_compile_cache import ensure_jax_compile_cache,compile_cache_stats
owner=importlib.import_module('distrib_la.matmul')
mesh=Mesh(np.asarray(jax.devices()).reshape(2,2),('x','y'))
sharding=NamedSharding(mesh,P(None,'x','y'))
ensure_jax_compile_cache()
@partial(jax.jit,static_argnums=(0,1,2))
def shared(shape,dtype,sharding):
 return jax.lax.with_sharding_constraint(jnp.zeros(shape,dtype=dtype),sharding)
rows=[]
for name,fn in [('incumbent',owner._zeros),('hoisted',shared)]:
 before=compile_cache_stats()
 for i in range(10):
  v=fn((4,8,12),jnp.dtype('complex128'),sharding)
  v.block_until_ready()
  assert v.sharding==sharding,v.sharding
  assert np.array_equal(np.asarray(v),np.zeros((4,8,12),complex))
 after=compile_cache_stats()
 rows.append(dict(name=name,calls=10,compiles=after['compiles']-before['compiles'],compiler_seconds=after['compile_secs']-before['compile_secs']))
print(json.dumps(rows,indent=2))
open('zero_probe.json','w').write(json.dumps(rows,indent=2)+'\n')
assert rows[0]['compiles']==10 and rows[1]['compiles']==1,rows
