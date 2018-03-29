import numpy as np
from refocus import refocusAlpha
from scipy.misc import imsave
import time

LF = np.load('LF_0001.npy')/255
imsave('real.png', LF[4,4,:,:,:])
start = time.perf_counter()
imout = refocusAlpha(LF, 1.5)
end = time.perf_counter()
print('Refocus time: {} s'.format(end-start))
imsave('out.png', imout)
