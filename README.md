When using Intel icpx to run both a sycl kernel and a sequential loop, there are differences in the pow and the exp functions. The following example is a reproducer of these errors. 

Steps to reproduce the error:
1. Initialise the oneAPI environment on your machine.
2. Run the code (example below): The code was run on a CPU using the following flags since the purpose is precision, not performance.

   icpx -fsycl -fp-model=precise -O0 power.cpp -o power
   
   ./power $INPUT

4. This can also be tested on (nvidia) GPUs and notice differences. An example command is given below

   icpx -fsycl -fp-model=precise -fsycl-targets=nvptx64-nvidia-cuda,spir64  -O0 power.cpp -o power
   
   ONEAPI_DEVICE_SELECTOR="ext_oneapi_cuda:*" ./power $INPUT

where $INPUT is the number of elements to be tested on.

Note:
The code can be run with both floats and doubles by manually changing the value of 'real_t' in the code. 

