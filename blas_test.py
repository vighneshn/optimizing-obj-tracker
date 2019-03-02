import numpy as np
import scipy.linalg.blas as slb
import time

M = 8400
N = 10
#k_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

np.show_config()

start = time.time()
for K in range(100):
        a = np.array(np.random.random((M, N)), dtype=np.double, order='C', copy=False)
        b = np.array(np.random.random((N, M)), dtype=np.double, order='C', copy=False)
        c = np.array(np.random.random((M, 1)), dtype=np.double, order='C', copy=False)
        A = np.matrix(a, dtype=np.double, copy=False)
        B = np.matrix(b, dtype=np.double, copy=False)
        C = np.matrix(c, dtype=np.double, copy=False)

        C = slb.dgemm(1.0, a=A,b=slb.dgemm(1.0, a=B, b=C))
        #print (C.shape)
        #end = time.time()

        #tm = end - start
        #print ('{0:4}, {1:9.7}'.format(K, tm))
print (time.time() - start)
