#ifndef VECTOR3_HEADER
#define VECTOR3_HEADER

#include <algorithm>
#include <iomanip>
#include <cmath>

class Vector3
{
   public:

      Vector3 (float x, float y, float z)
      {
         Initialize(x, y, z);
      }

      Vector3 (const Vector3 & in)
      {
         Initialize(in[0], in[1], in[2]);
      }

      Vector3 (const float in[3])
      {
         Initialize(in[0], in[1], in[2]);
      }

      Vector3 (void)
      {
         Initialize(0.f, 0.f, 0.f);
      }

      void Initialize (float x, float y, float z)
      {
         base_[0] = x;
         base_[1] = y;
         base_[2] = z;
      }

      Vector3 crossProduct (const Vector3 & a) const
      {
         Vector3 c(
            base_[1]*a[2] - base_[2]*a[1],
            base_[2]*a[0] - base_[0]*a[2],
            base_[0]*a[1] - base_[1]*a[0]
         );
         return c;
      }

      Vector3 unitVector (void) const
      {
         Vector3 dir(*this);
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
            base_[0]*base_[0] + base_[1]*base_[1] + base_[2]*base_[2];
         return magSquared;
      }

      float dot (const Vector3 & a) const
      {
         return base_[0]*a[0] + base_[1]*a[1] + base_[2]*a[2];
      }

      bool hasNan(void) const
      {
         return (
            std::isnan(base_[0]) ||
            std::isnan(base_[1]) ||
            std::isnan(base_[2])
         );
      }

      bool hasInf(void) const
      {
         return (
            std::isinf(base_[0]) ||
            std::isinf(base_[1]) ||
            std::isinf(base_[2])
         );
      }

      Vector3 & Normalize(void)
      {
         float mag = magnitude();
         base_[0] /= std::max(mag, 1e-7f);
         base_[1] /= std::max(mag, 1e-7f);
         base_[2] /= std::max(mag, 1e-7f);
         return *this;
      }

      Vector3 & operator= (const Vector3 & a)
      {
         if (&a == this)
         {
            return *this;
         }

         Initialize(a[0], a[1], a[2]);
         return *this;
      }

      const Vector3 operator= (const Vector3 & a) const
      {
         if (&a == this)
         {
            return *this;
         }
         Vector3 result(a[0], a[1], a[2]);
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

      Vector3 operator+ (const Vector3 & a) const
      {
         return Vector3(base_[0] + a[0], base_[1] + a[1], base_[2] + a[2]);
      }

      Vector3 & operator+= (const Vector3 & a)
      {
         for (unsigned int i = 0; i < 3; ++i)
         {
            base_[i] += a[i];
         }
         return *this;
      }

      Vector3 operator- (const Vector3 & a) const
      {
         Vector3 result(base_[0] - a[0], base_[1] - a[1], base_[2] - a[2]);
         return result;
      }

      Vector3 & operator-= (const Vector3 & a)
      {
         for (unsigned int i = 0; i < 3; ++i)
         {
            base_[i] -= a[i];
         }
         return *this;
      }

      Vector3 operator* (const Vector3 & a) const
      {
         Vector3 result(base_[0]*a[0], base_[1]*a[1], base_[2]*a[2]);
         return result;
      }

      Vector3 operator* (const float & v) const
      {
         return Vector3(base_[0]*v, base_[1]*v, base_[2]*v);
      }

      Vector3 & operator*= (const float & v)
      {
         base_[0] *= v;
         base_[1] *= v;
         base_[2] *= v;
         return *this;
      }

      Vector3 & operator/= (const float & v)
      {
         base_[0] /= v;
         base_[1] /= v;
         base_[2] /= v;
         return *this;
      }

      Vector3 operator/ (const float & v) const
      {
         Vector3 result(*this);
         return result*(1.0/v);
      }

      Vector3 operator& (const Vector3 & v)
      {
         return Vector3(base_[0]*v[0], base_[1]*v[1], base_[2]*v[2]);
      }

      bool operator== (const Vector3 & v) const
      {
         return base_[0] == v[0] && base_[1] == v[1] && base_[2] == v[2];
      }

      bool operator!= (const Vector3 & v) const
      {
         return !(*this == v);
      }

      bool almostEqual (const Vector3 & v, float epsilon=1e-6f) const
      {
         return (
            (
               (fabs(base_[0] - v.base_[0]) / std::max(fabs(base_[0]), fabs(v.base_[0])) < epsilon)
               || (base_[0] == v.base_[0])
            ) &&
            (
               (fabs(base_[1] - v.base_[1]) / std::max(fabs(base_[1]), fabs(v.base_[1])) < epsilon)
               || (base_[1] == v.base_[1])
            ) &&
            (
               (fabs(base_[2] - v.base_[2]) / std::max(fabs(base_[2]), fabs(v.base_[2])) < epsilon)
               || (base_[2] == v.base_[2])
            )
         );
      }

      bool almostEqual (float a, float b, float c, float epsilon=1e-6f) const
      {
         return this->almostEqual(Vector3(a, b, c), epsilon);
      }

   private:

      float base_[3];

};

std::ostream & operator << (std::ostream & out, const Vector3 & a);

Vector3 operator* (const float & v, const Vector3 & a);

#endif
