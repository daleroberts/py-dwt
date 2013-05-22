#A Python/GSL Wavelet library
# Dale Roberts <dale.o.roberts@gmail.com>

import numpy as np
from ctypes import CDLL, POINTER as PTR, RTLD_GLOBAL, Structure, c_size_t, c_double, c_char_p, c_int

# load BLAS and GSL
from ctypes.util import find_library
libblas = CDLL(find_library("libblas"), RTLD_GLOBAL)
libgslcblas = CDLL(find_library("gslcblas"), RTLD_GLOBAL)
libgsl = CDLL(find_library("gsl"), RTLD_GLOBAL)

class gsl_wavelet_type(Structure):
    pass

class gsl_wavelet(Structure):
    pass

class gsl_wavelet_workspace(Structure):
    pass

p_gsl_wavelet_type = PTR(gsl_wavelet_type)

def _p_const(fname):
    return p_gsl_wavelet_type.in_dll(libgsl,"gsl_wavelet_" + fname)

def _types(fname, restype, argtypes):
    f = libgsl.__getattr__("gsl_wavelet_" + fname)
    if restype != None:
        f.restype = restype
    f.argtypes = argtypes
    return f

_types("alloc", PTR(gsl_wavelet), [p_gsl_wavelet_type, c_size_t])
_types("free", None, [PTR(gsl_wavelet)])
_types("name", c_char_p, [PTR(gsl_wavelet)])
_types("workspace_alloc", PTR(gsl_wavelet_workspace), [c_size_t])
_types("workspace_free", None, [PTR(gsl_wavelet_workspace)])
_types("transform_forward", c_int, [PTR(gsl_wavelet), PTR(c_double), c_size_t, c_size_t, PTR(gsl_wavelet_workspace)])
_types("transform_inverse", c_int, [PTR(gsl_wavelet), PTR(c_double), c_size_t, c_size_t, PTR(gsl_wavelet_workspace)])

def gsl_dwt(x, family="haar", k=2, stride=1):
    # create a new array as GSL destroys input data, coerce to double too
    z = np.array(x, dtype=np.double)
    # allocate workspace and wavelet
    t = libgsl.gsl_wavelet_workspace_alloc(len(z))
    w = libgsl.gsl_wavelet_alloc(_p_const(family), k)
    # do the transform
    libgsl.gsl_wavelet_transform_forward(w,
        z.ctypes.data_as(PTR(c_double)),
        stride, len(z), t)
    # free objects
    libgsl.gsl_wavelet_free(w)
    libgsl.gsl_wavelet_workspace_free(t)
    # return result
    return z

def gsl_idwt(x, family="haar", k=2, stride=1):
    # create a new array as GSL destroys input data, coerce to double too
    z = np.array(x, dtype=np.double)
    # allocate workspace and wavelet
    t = libgsl.gsl_wavelet_workspace_alloc(len(z))
    w = libgsl.gsl_wavelet_alloc(_p_const(family), k)
    # do the transform
    libgsl.gsl_wavelet_transform_inverse(w,
        z.ctypes.data_as(PTR(c_double)),
        stride, len(z), t)
    # free objects
    libgsl.gsl_wavelet_free(w)
    libgsl.gsl_wavelet_workspace_free(t)
    # return result
    return z

if __name__ == '__main__':
    x = np.array([0.]*16)
    x[0]=1.
    print x
    print '-'*8
    print gsl_dwt(x)
