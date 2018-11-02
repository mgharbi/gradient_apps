import cntk as C
import numpy as np

c = np.asarray([0,1]).astype('f')
x = C.input_variable((2), needs_gradient=True, dynamic_axes=[])
y = C.input_variable((6), needs_gradient=False, dynamic_axes=[])
output = C.gather(x, y)
loss = C.reduce_sum(output)
print(loss.grad({y: np.arange(6).reshape(6).astype('f'), x: c}))
