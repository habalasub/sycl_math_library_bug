When using Intel icpx to run both a sycl kernel and a sequential loop, there are differences in the pow and the exp functions. The following example is a reproducer of these errors. 

Steps to reproduce the error:
1. Initialise the oneAPI environment on your machine.
2. Run the code (example below): The code was run using the following flags since the purpose is precision, not performance.

   icpx -fsycl -fp-model=precise -O0 power.cpp -o power
   ./power $INPUT

where $INPUT is the number of elements to be tested on.

Note:
The code can be run with both floats and doubles by manually changing the values of 'real_t' in the code. 
