import numpy as np
from scipy.interpolate import RegularGridInterpolator

def refocusAlpha(LF, alpha):
    Iout = np.zeros(LF.shape[2:])
    xs, ys = np.arange(LF.shape[3]), np.arange(LF.shape[2])
    normalized = 1 / (LF.shape[0] * LF.shape[1])
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    for ky in range(LF.shape[0]):
        for kx in range(LF.shape[1]):
            for c in range(LF.shape[-1]):
                interpFunc = RegularGridInterpolator((ys,xs), LF[ky,kx,:,:,c], bounds_error=False, fill_value=None)
                newyy = yy - (alpha-1) * (ky - LF.shape[0]//2)
                newxx = xx + (alpha-1) * (kx - LF.shape[1]//2)
                Iout[:,:,c] += interpFunc(np.dstack([newyy,newxx])) * normalized
    return Iout