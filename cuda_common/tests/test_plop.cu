#include <stdio.h>

#include "plop.cuh"

template <unsigned int SIZE>
__global__ void kernel(float * in_data, float * out_data)
{
   plop::DeviceArray<float, SIZE> kern_in_arr(in_data);

   plop::DeviceArray<float, SIZE> kern_out_arr(out_data);

   int tid = threadIdx.x;
   if (tid < SIZE)
   {
      out_data[tid] = in_data[tid];
   }
}

int main(void)
{
   plop::HostArray<float, 10> host_arr;

   for (int i = 0; i < 10; ++i)
      host_arr[i] = (float )i / 3.f;

   plop::HostArray<float, 10> out_host_arr;

   plop::DeviceArray<float, 10> device_arr;
   device_arr = host_arr;

   plop::DeviceArray<float, 10> out_device_arr;

   kernel<10><<<1, 10>>>(device_arr.begin(), out_device_arr.begin());

   plop::copy(out_device_arr.begin(), out_device_arr.end(), out_host_arr.begin());

   for (int i = 0; i < 10; ++i)
      printf("%f\n", out_host_arr[i]);

   device_arr.clear();

   out_device_arr.clear();


}
