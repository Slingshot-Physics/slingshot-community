#ifndef VECTOR4_HEADER
#define VECTOR4_HEADER

#include <algorithm>
#include <iomanip>
#include <cmath>

class Vector4
{
   public:

      Vector4 (float w, float x, float y, float z)
      {
         Initialize(w, x, y, z);
      }

      Vector4 (const Vector4 & in)
      {
         Initialize(in[0], in[1], in[2], in[3]);
      }

      Vector4 (const float in[4])
      {
         Initialize(in[0], in[1], in[2], in[3]);
      }

      Vector4 (void)
      {
         Initialize(0.f, 0.f, 0.f, 0.f);
      }

      void Initialize (float w, float x, float y, float z)
      {
         base_[0] = w;
         base_[1] = x;
         base_[2] = y;
         base_[3] = z;
      }

      Vector4 unitVector (void) const
      {
         Vector4 dir(*this);
         dir.Normalize();
         return dir;
      }

      float magnitude (void) const
      {
         float mag = sqrt(magnitudeSquared());
         return mag;
      }

      float magnitudeSquared (void) const
      {
         float magSquared = \
            base_[0]*base_[0] + base_[1]*base_[1] + base_[2]*base_[2] + base_[3]*base_[3];
         return magSquared;
      }

      Vector4 & Normalize(void)
      {
         float mag = magnitude();
         base_[0] /= std::max(mag, 1e-7f);
         base_[1] /= std::max(mag, 1e-7f);
         base_[2] /= std::max(mag, 1e-7f);
         return *this;
      }

      bool hasNan(void) const
      {
         return (
            std::isnan(base_[0]) ||
            std::isnan(base_[1]) ||
            std::isnan(base_[2]) ||
            std::isnan(base_[3])
         );
      }

      bool hasInf(void) const
      {
         return (
            std::isinf(base_[0]) ||
            std::isinf(base_[1]) ||
            std::isinf(base_[2]) ||
            std::isinf(base_[3])
         );
      }

      Vector4 & operator= (const Vector4 & a)
      {
         if (&a == this)
         {
            return *this;
         }

         Initialize(a[0], a[1], a[2], a[3]);
         return *this;
      }

      const Vector4 operator= (const Vector4 & a) const
      {
         if (&a == this)
         {
            return *this;
         }
         Vector4 result(a[0], a[1], a[2], a[3]);
         return result;
      }

      float & operator[] (unsigned int i)
      {
         return base_[i];
      }

      const float & operator[] (unsigned int i) const
      {
         return base_[i];
      }

      bool operator== (const Vector4 & v) const
      {
         return (
            (base_[0] == v[0]) &&
            (base_[1] == v[1]) &&
            (base_[2] == v[2]) &&
            (base_[3] == v[3])
         );
      }

      bool operator!= (const Vector4 & v) const
      {
         return !(*this == v);
      }

   private:

      float base_[4];

};

std::ostream & operator << (std::ostream & out, const Vector4 & a);

#endif
