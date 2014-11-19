runcython lets you compile and run cython in one line

<h2>Installation</h2>

    pip install runcython
    
<h2>Usage</h2>

    # hello.pyx
    print 'hello, world'
  
    $ runcython hello.pyx
    hello, world
    
  You can use `runcython file.pyx` just like you would use `python file.py`. The difference is of course that `runcython` will run a file with arbitrary cython code.
  
    # accum.pyx
    cdef int i, n, accum
    accum = 0
    n = 10**4
    for i in range(n):
        accum += i
    print i
  
    $ runcython accum.pyx
    49995000
    
  There's no need to muck around with distutils or intermediate files. Using cython the typical way would require creating 5 distinct files, `accum.pyx`, `accum.c`, `accum.so`, `setup.py`, and `use.py`. That's a lot of moving parts to keep track of. `runcython` keeps things simple so that you can just focus on writing fast code. If you want to output a module for use in your other python files, you can always use `makecython instead`
  
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
    
  Of course, none of this would be any better than `pyximport` if it didn't work for complex cython builds with lots of dependencies. But runcython let's you handle complex builds by giving 2 extra arguments, one for passing additional flags to `cython file.pyx ...`, and one for passing additional flags to `gcc a.c ...`. Lets see how this works for calling a c file:
  
    # square.c
    int square(int x) {
        return x * x;
    }
  
    # use_square.pyx
    cdef extern int square(int)
    print square(5)
  
  Now we'll need gcc to compile `square.c` along with `use_square.c`, so we can do 
  
    $ runcython square.pyx "" " square.c"
    25
  
  On the other hand, if we want to pass special flags to cython, such as perhaps `-a`, which tells cython to tell us which lines of the files are cython optimized, and which are regular python code, we can do that for the above `primes.pyx`:
  
    $ runcython primes.pyx "-a"
    $ ls
    primes.pyx primes.html
    $ firefox primes.html
    # firefox will show us the areas that have been sped up
    
  
    
    
    
    
