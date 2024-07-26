# Numba-accelerated n-body from programming languages benchmarks game

See: https://benchmarksgame-team.pages.debian.net/benchmarksgame/performance/nbody.html

I just took the Python 3 code and jitted it, using a few optimisation tips from the numba team. On my machine, it is about 1.1x slower than the best C code, which translates to about 1.5x of the best-performing code. Not bad for a snake!

```
$ time ./nbody.gcc-6.gcc_run 50000000
-0.169075164
-0.169059907
./nbody.gcc-6.gcc_run 50000000  3.19s user 0.00s system 99% cpu 3.204 total

$ time python nbody-numba.py 50000000
-0.169075164
-0.169059907
python nbody-numba.py 50000000  3.55s user 0.07s system 99% cpu 3.636 total
```
