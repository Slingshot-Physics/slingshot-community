#ifndef VECTOR3_CUDA_HEADER
#define VECTOR3_CUDA_HEADER

#include "defines.cuh"

#include <algorithm>
#include <math.h>

namespace cumath
{
   class Vector3
   {
      public:

         CUDA_HOST_CALLABLE Vector3 (float x, float y, float z)
         {
            Initialize(x, y, z);
         }

         CUDA_HOST_CALLABLE Vector3 (const Vector3 & in)
         {
            Initialize(in[0], in[1], in[2]);
         }

         CUDA_HOST_CALLABLE Vector3 (const float in[3])
         {
            Initialize(in[0], in[1], in[2]);
         }

         CUDA_HOST_CALLABLE Vector3 (void)
         {
            Initialize(0.0, 0.0, 0.0);
         }

         CUDA_HOST_CALLABLE void Initialize (float x, float y, float z)
         {
            base_[0] = x;
            base_[1] = y;
            base_[2] = z;
         }

         CUDA_HOST_CALLABLE Vector3 crossProduct (const Vector3 & a) const
         {
            Vector3 c(
               base_[1] * a.base_[2] - base_[2] * a.base_[1],
               base_[2] * a.base_[0] - base_[0] * a.base_[2],
               base_[0] * a.base_[1] - base_[1] * a.base_[0]
            );
            return c;
         }

         CUDA_HOST_CALLABLE Vector3 unitVector (void) const
         {
            Vector3 dir(*this);
            dir.Normalize();
            return dir;
         }

         CUDA_HOST_CALLABLE float magnitude (void) const
         {
            float mag = sqrt(magnitudeSquared());
            return mag;
         }

         CUDA_HOST_CALLABLE float magnitudeSquared (void) const
         {
            float magSquared = \
               base_[0]*base_[0] + base_[1]*base_[1] + base_[2]*base_[2];
            return magSquared;
         }

         CUDA_HOST_CALLABLE float dot (const Vector3 & a) const
         {
            return base_[0]*a[0] + base_[1]*a[1] + base_[2]*a[2];
         }

         CUDA_HOST_CALLABLE bool hasNan(void) const
         {
            return isnan(base_[0]) || isnan(base_[1]) || isnan(base_[2]);
         }

         CUDA_HOST_CALLABLE bool hasInf(void) const
         {
            return isinf(base_[0]) || isinf(base_[1]) || isinf(base_[2]);
         }

         CUDA_HOST_CALLABLE Vector3 & Normalize(void)
         {
            float mag = magnitude();
#ifdef __CUDACC__
            base_[0] /= fmaxf(mag, 1e-7f);
            base_[1] /= fmaxf(mag, 1e-7f);
            base_[2] /= fmaxf(mag, 1e-7f);
#else
            base_[0] /= std::max(mag, 1e-7f);
            base_[1] /= std::max(mag, 1e-7f);
            base_[2] /= std::max(mag, 1e-7f);
#endif
            return *this;
         }

         CUDA_HOST_CALLABLE Vector3 & operator= (const Vector3 & a)
         {
            if (&a == this)
            {
               return *this;
            }

            Initialize(a[0], a[1], a[2]);
            return *this;
         }

         CUDA_HOST_CALLABLE const Vector3 operator= (const Vector3 & a) const
         {
            if (&a == this)
            {
               return *this;
            }
            Vector3 result(a[0], a[1], a[2]);
            return result;
         }

         CUDA_HOST_CALLABLE float & operator[] (unsigned int i)
         {
            return base_[i];
         }

         CUDA_HOST_CALLABLE const float & operator[] (unsigned int i) const
         {
            return base_[i];
         }

         CUDA_HOST_CALLABLE Vector3 operator+ (const Vector3 & a) const
         {
            return Vector3(base_[0] + a[0], base_[1] + a[1], base_[2] + a[2]);
         }

         CUDA_HOST_CALLABLE Vector3 & operator+= (const Vector3 & a)
         {
            for (unsigned int i = 0; i < 3; ++i)
            {
               base_[i] += a[i];
            }
            return *this;
         }

         CUDA_HOST_CALLABLE Vector3 operator- (const Vector3 & a) const
         {
            Vector3 result(base_[0] - a[0], base_[1] - a[1], base_[2] - a[2]);
            return result;
         }

         CUDA_HOST_CALLABLE Vector3 & operator-= (const Vector3 & a)
         {
            for (unsigned int i = 0; i < 3; ++i)
            {
               base_[i] -= a[i];
            }
            return *this;
         }

         CUDA_HOST_CALLABLE Vector3 operator* (const Vector3 & a) const
         {
            Vector3 result(base_[0]*a[0], base_[1]*a[1], base_[2]*a[2]);
            return result;
         }

         CUDA_HOST_CALLABLE Vector3 operator* (const float & v) const
         {
            return Vector3(base_[0]*v, base_[1]*v, base_[2]*v);
         }

         CUDA_HOST_CALLABLE Vector3 & operator*= (const float & v)
         {
            base_[0] *= v;
            base_[1] *= v;
            base_[2] *= v;
            return *this;
         }

         CUDA_HOST_CALLABLE Vector3 & operator/= (const float & v)
         {
            base_[0] /= v;
            base_[1] /= v;
            base_[2] /= v;
            return *this;
         }

         CUDA_HOST_CALLABLE Vector3 operator/ (const float & v) const
         {
            Vector3 result(*this);
            return result*(1.0/v);
         }

         CUDA_HOST_CALLABLE Vector3 operator& (const Vector3 & v)
         {
            return Vector3(base_[0]*v[0], base_[1]*v[1], base_[2]*v[2]);
         }

         CUDA_HOST_CALLABLE bool operator== (const Vector3 & v) const
         {
            return base_[0] == v[0] && base_[1] == v[1] && base_[2] == v[2];
         }

         CUDA_HOST_CALLABLE bool operator!= (const Vector3 & v) const
         {
            return !(*this == v);
         }

         CUDA_HOST_CALLABLE bool almostEqual (const Vector3 & v) const
         {
#ifdef __CUDACC__
            return (
               (
                  (fabs(base_[0] - v.base_[0]) / fmaxf(fabs(base_[0]), fabs(v.base_[0])) < 1e-6)
                  || (base_[0] == v.base_[0])
               ) &&
               (
                  (fabs(base_[1] - v.base_[1]) / fmaxf(fabs(base_[1]), fabs(v.base_[1])) < 1e-6)
                  || (base_[1] == v.base_[1])
               ) &&
               (
                  (fabs(base_[2] - v.base_[2]) / fmaxf(fabs(base_[2]), fabs(v.base_[2])) < 1e-6)
                  || (base_[2] == v.base_[2])
               )
            );
#else
            return (
               (
                  (fabs(base_[0] - v.base_[0]) / std::max(fabs(base_[0]), fabs(v.base_[0])) < 1e-6)
                  || (base_[0] == v.base_[0])
               ) &&
               (
                  (fabs(base_[1] - v.base_[1]) / std::max(fabs(base_[1]), fabs(v.base_[1])) < 1e-6)
                  || (base_[1] == v.base_[1])
               ) &&
               (
                  (fabs(base_[2] - v.base_[2]) / std::max(fabs(base_[2]), fabs(v.base_[2])) < 1e-6)
                  || (base_[2] == v.base_[2])
               )
            );
#endif
         }

         CUDA_HOST_CALLABLE bool almostEqual (float a, float b, float c) const
         {
#ifdef __CUDACC__
            return (
               (
                  (fabs(base_[0] - a) / fmaxf(fabs(base_[0]), fabs(a)) < 1e-6)
                  || (base_[0] == a)
               ) &&
               (
                  (fabs(base_[1] - b) / fmaxf(fabs(base_[1]), fabs(b)) < 1e-6)
                  || (base_[1] == b)
               ) &&
               (
                  (fabs(base_[2] - c) / fmaxf(fabs(base_[2]), fabs(c)) < 1e-6)
                  || (base_[2] == c)
               )
            );
#else
            return (
               (
                  (fabs(base_[0] - a) / std::max(fabs(base_[0]), fabs(a)) < 1e-6)
                  || (base_[0] == a)
               ) &&
               (
                  (fabs(base_[1] - b) / std::max(fabs(base_[1]), fabs(b)) < 1e-6)
                  || (base_[1] == b)
               ) &&
               (
                  (fabs(base_[2] - c) / std::max(fabs(base_[2]), fabs(c)) < 1e-6)
                  || (base_[2] == c)
               )
            );
#endif
         }

      private:

         float base_[3];

   };

   CUDA_HOST_CALLABLE Vector3 operator* (const float & v, const cumath::Vector3 & a);

}

#endif
