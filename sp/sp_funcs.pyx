# Import necessary Cython and NumPy functions
import numpy as np
cimport numpy as cnp

cdef extern from "sp.cpp":
    void get_data(double * data, size_t size)

def get_data_np(cnp.ndarray[cnp.float64_t, ndim=1] input_array):
    get_data(&input_array[0], input_array.shape[0])
