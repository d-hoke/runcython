runcython lets you compile and run cython in one line. OSX and Linux are supported.

<h2>Installation</h2>

    pip install runcython

For OSX, make sure that the file python.pc is on your PKG_CONFIG_PATH. I achieved this by adding to my ~/.bashrc:

    export PKG_CONFIG_PATH=/System/Library/Frameworks/Python.framework/Versions/2.7/lib/pkgconfig
    
<h2>Usage</h2>

    # hello.pyx
    print 'hello, world'
  
    $ runcython hello.pyx
    hello, world
    
  You can use `runcython file.pyx` just like you would use `python file.py`. The difference is that runcython will run a file with arbitrary cython code.
  
    # accum.pyx
    cdef int i, n, accum
    accum = 0
    n = 10**4
    for i in range(n):
        accum += i
    print i
  
    $ runcython accum.pyx
    49995000
    
  There's no need to muck around with distutils or intermediate files. Using cython the typical way would require creating 5 distinct files, `accum.pyx`, `accum.c`, `accum.so`, `setup.py`, and `use.py`. That's a lot of moving parts to keep track of. `runcython` keeps things simple so that you can just focus on writing fast code. If you want to output a module for use in your other python files, you can always use `makecython` instead:
  
    # primes.pyx
    def primes(int kmax):
        cdef int n, k, i
        cdef int p[1000000]
        result = []
        if kmax > 1000000:
            kmax = 1000000
        k = 0
        n = 2
        while k < kmax:
            i = 0
            while i < k and n % p[i] != 0:
                i = i + 1
            if i == k:
                p[k] = n
                k = k + 1
                result.append(n)
            n = n + 1
        return result
    
    $ makecython primes.pyx
    $ ls
    primes.pyx primes.so
    $ python -c 'import primes; print primes.primes(10)'
    [2, 3, 5, 7, 11, 13, 17, 23, 29]
    
<h2> Advanced Usage </h2>

  Of course, none of this would be much better than the pyximport tool if it didn't work for complex cython builds with lots of dependencies. But unlike pyximport, runcython doesn't force you to adopt an entirely new strategy for complex builds. You get 2 extra arguments to runcython, one for passing additional flags to `cython file.pyx ...`, and one for passing additional flags to `gcc a.c ...`. Lets see how this works for calling a c file:
  
    # square.c
    int square(int x) {
        return x * x;
    }
  
    # use_square.pyx
    cdef extern int square(int)
    print square(5)
  
  Now if we don't add any extra parameters, `runcython use_square.pyx` will first run `cython use_square.pyx` to produce use_square.c, and call `gcc -shared -fPIC use_square.c -o use_square.so` to produce use_square.so. But we need to tell gcc that it should also compile the square.c file. Doing this just requres tagging on square.c to the gcc command, giving `gcc -shared -fPIC use_square.c -o use_square.so square.c`. To tell runcython to do this, we just tell it to add the string "square.c" to the end of the gcc command:
  
    $ runcython square.pyx "" "square.c"
    25
  
  We can also tell runcython to pass special flags to the cython command. For example the `-a` flag tells cython to produce a nicely formatted html file with a summary of which lines in the input file were successfully optimized by cython. We can do that for the above `primes.pyx`:
  
    $ runcython primes.pyx "-a"
    $ ls
    primes.pyx primes.html
    $ firefox primes.html
    # firefox will show us the areas that have been sped up

<h2> Binding cuda kernels with runcython </h2>

  To convince you that `runcython` really does scale to rather complex build processes, here's a pipeline I built recently to call cuda kernels directly using `runcython++`. Note that runcython++ is just like runcython, but using g++ for compilation rather than gcc:
  
    // kernel.h
    #define N 16
    void cscan(float* hostArray);
    
  kernel.h declares a simple interface to a cuda kernel that will take an array of floats and return an array of the partial sums. This is a great parallel primitive that would be nice to do on a GPU if it were an important part of your computation. The kernel itself is defined with `kernel.cu`.
  
    // kernel.cu
    #include "kernel.h"
    __global__ void scan(float *g_odata, float *g_idata, int n)  {
        extern __shared__ float temp[]; // allocated on invocation  
        int thid = threadIdx.x;
        int pout = 0, pin = 1;
        // Load input into shared memory.  
        // This is exclusive scan, so shift right by one  
        // and set first element to 0  
        temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;
        __syncthreads();
        for (int offset = 1; offset < n; offset *= 2)
        {
            pout = 1 - pout; // swap double buffer indices
            pin = 1 - pout;
            if (thid >= offset) {
              temp[pout*n+thid] = temp[pin*n+thid - offset] + temp[pin*n+thid];
            } else {
              temp[pout*n+thid] = temp[pin*n+thid];
            }
            __syncthreads();
        }
        g_odata[thid] = temp[pout*n+thid]; // write output
    }
    
    void cscan(float* hostArray)
    {
        float* deviceArray;
        float* deviceArrayOut;
        const float zero = 0.0f;
        const int arrayLength = N;
        const unsigned int memSize = sizeof(float) * arrayLength;
    
        cudaMalloc((void**) &deviceArray, memSize);
        cudaMalloc((void**) &deviceArrayOut, memSize);
        cudaMemset( deviceArray, zero, memSize);
        cudaMemset( deviceArrayOut, zero, memSize);
    
        cudaMemcpy(deviceArray, hostArray, memSize, cudaMemcpyHostToDevice);
        scan <<< 1, N, 32 >>> (deviceArrayOut, deviceArray, N);
        cudaMemcpy(hostArray, deviceArrayOut, memSize, cudaMemcpyDeviceToHost);
    
    
        cudaFree(deviceArrayOut);
        cudaFree(deviceArray);
    
        return;
    }

  Finally, we're going to wrap this `cscan` function using cython with the following code:
  
    # use_kernel.pyx
    from libc.stdlib cimport malloc, free

    cdef extern from "kernel.h":
        void cscan(float*)
    
    def main():
        cdef int number = 16
        cdef float *my_array = <float *>malloc(number * sizeof(float))
        print "before"
        for x in range(number):
            my_array[x] = x**2
            print my_array[x]
        cscan(my_array)
        print "after"
        for x in range(number):
            print my_array[x]
    
    main()

  To compile, first we need to turn the cuda code into c++ object code:
  
    $ nvcc -c kernel.cu  --shared --compiler-options '-fPIC' -o kernel.o
    
  The results are stored in `kernel.o`, which we would normall call from c++ with
  
    $ g++ use_kernel.cpp kernel.o -L/usr/local/cuda-5.5/lib64 -lcudart
    
  Hence, we just need to pass the equivalent flags to runcython++, and we'll get what we want
  
    $ runcython++ user_kernel.pyx "" "kernel.o -L/usr/local/cuda-5.5/lib64 -lcudart"
    before
    0.0
    1.0
    4.0
    9.0
    16.0
    25.0
    36.0
    49.0
    64.0
    81.0
    100.0
    121.0
    144.0
    169.0
    196.0
    225.0
    after
    0.0
    0.0
    1.0
    5.0
    14.0
    30.0
    55.0
    91.0
    140.0
    204.0
    285.0
    385.0
    506.0
    650.0
    819.0
    1015.0
