#include <stdio.h>
#include <cmath>

#include "buffer_wrapper.cuh"
#include "transit_buffer.cuh"
#include "vector3.cuh"

#define NUM_VECS 1024

void local_kernel(
   BufferWrapper<cumath::Vector3> & vec_in, BufferWrapper<cumath::Vector3> & vec_out
)
{
   printf("local kernel\n");
   for (int i = 0; i < vec_in.size(); ++i)
   {
      vec_out[i] = vec_in[i] * 2.f;
   }
}

__global__ void kernel(
   BufferWrapper<cumath::Vector3> vec_in, BufferWrapper<cumath::Vector3> vec_out
)
{
   int tid = threadIdx.x;

   vec_out[tid] = vec_in[tid] * 2.f;
}

int main(void)
{
   TransitBuffer<cumath::Vector3, NUM_VECS> input_buf;
   TransitBuffer<cumath::Vector3, NUM_VECS> output_buf;

   BufferWrapper<cumath::Vector3> verify_buf(NUM_VECS);

   for (int i = 0; i < NUM_VECS; ++i)
   {
      for (int j = 0; j < 3; ++j)
      {
         input_buf[i][j] = sinf((3 * i + j) * M_PI / 256);
      }
   }

   input_buf.copy_host_to_device();

   kernel<<<1, NUM_VECS>>>(input_buf.device_buffer_wrapper(), output_buf.device_buffer_wrapper());

   output_buf.copy_device_to_host();

   BufferWrapper<cumath::Vector3> input_buf_wrap = input_buf.host_buffer_wrapper();

   local_kernel(input_buf_wrap, verify_buf);

   bool badness = false;
   BufferWrapper<cumath::Vector3> output_buf_host = output_buf.host_buffer_wrapper();
   for (int i = 0; i < NUM_VECS; ++i)
   {
      if (!output_buf_host[i].almostEqual(verify_buf[i]))
      {
         printf("error at element %d\n", i);
         badness = true;
         break;
      }
      printf(
         "(%f, %f, %f), (%f, %f, %f)\n",
         output_buf_host[i][0], output_buf_host[i][1], output_buf_host[i][2],
         verify_buf[i][0], verify_buf[i][1], verify_buf[i][2]
      );
   }

   if (!badness)
   {
      printf("things are lookin good\n");
   }

   verify_buf.clear();

   return 0;
}
