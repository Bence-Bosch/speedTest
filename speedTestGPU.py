# Ensure that your system has a CUDA-enabled GPU and the necessary drivers installed.
# You may need to install the CUDA toolkit and verify GPU availability using `numba.cuda.is_available()`.

import numpy as np
from timeit import default_timer as timer

def pow(a, b):
    return np.power(a, b)

def main():
    vec_size = 100000000

    a = b = np.array(np.random.sample(vec_size), dtype=np.float32)
    c = np.zeros(vec_size, dtype=np.float32)

    start = timer()

    c = pow(a, b)

    duration = timer() - start

    print("duration taken on GPU:", duration)

if __name__ == '__main__':
    main()
