#include "segment.cuh"

#include "buffer_wrapper.cuh"
#include "transit_buffer.cuh"

// #include "segment.hpp"

#define NUM_VECS 1024

void local_kernel(
   BufferWrapper<cumath::Vector3> & as,
   BufferWrapper<cumath::Vector3> & bs,
   BufferWrapper<cumath::Vector3> & qs,
   BufferWrapper<cumath::Vector3> & cs
)
{
   // for (int i = 0; i < NUM_VECS; ++i)
   // {
   //    cs[i] = geometry::segment::closestPointToPoint(

   //    )
   // }
}

__global__ void kernel(
   BufferWrapper<cumath::Vector3> as,
   BufferWrapper<cumath::Vector3> bs,
   BufferWrapper<cumath::Vector3> qs,
   BufferWrapper<cumath::Vector3> cs
)
{
   int tid = threadIdx.x;

   cs[tid] = cugeom::segment::closestPointToPoint(
      as[tid], bs[tid], qs[tid]
   ).point;
}

int main(void)
{
   TransitBuffer<cumath::Vector3, NUM_VECS> seg_a_buf;
   TransitBuffer<cumath::Vector3, NUM_VECS> seg_b_buf;
   TransitBuffer<cumath::Vector3, NUM_VECS> q_buf;
   TransitBuffer<cumath::Vector3, NUM_VECS> closest_points_buf;

   for (unsigned int i = 0; i < NUM_VECS; ++i)
   {
      seg_a_buf[i][0] = (float )NUM_VECS/2.f - i;
      seg_a_buf[i][1] = (float )NUM_VECS/3.f - i + i / 10.f + (float )(i % 5);
      seg_a_buf[i][2] = (float )NUM_VECS/4.f - i + i * 2.f;

      seg_b_buf[i][0] = (float )NUM_VECS/4.f + i + i * 2.f;
      seg_b_buf[i][1] = -1.f * (float )NUM_VECS/2.f + i;
      seg_b_buf[i][2] = (float )NUM_VECS/3.f + i + i / 10.f - (float )(i % 5);

      q_buf[i][0] = ((float )NUM_VECS/2.f - i) * sinf(M_PI * i / 256);
      q_buf[i][1] = ((float )NUM_VECS/2.f - i) * 4.f * cosf(M_PI * i / 256);
      q_buf[i][1] = ((float )NUM_VECS/2.f - i) * 4.f * cosf(M_PI * i / 256)* sinf(M_PI * i / 256);
   }

   seg_a_buf.copy_host_to_device();
   seg_b_buf.copy_host_to_device();
   q_buf.copy_host_to_device();

   kernel<<<1, NUM_VECS>>>(
      seg_a_buf.device_buffer_wrapper(),
      seg_b_buf.device_buffer_wrapper(),
      q_buf.device_buffer_wrapper(),
      closest_points_buf.device_buffer_wrapper()
   );

   closest_points_buf.copy_host_to_device();

   return 0;
}
