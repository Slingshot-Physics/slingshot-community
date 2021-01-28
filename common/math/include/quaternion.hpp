#ifndef QUATERNION
#define QUATERNION

#include "vector3.hpp"
#include "matrix33.hpp"

// Quaternion implementation where the scalar part is the first element, and
// the vector part is the second element.

// This math library uses the convention that quaternion rotations are achieved
// via:
//    v_b = ~q_b/a * v_a * q_b/a

class Quaternion
{
   public:

      Quaternion (void)
#ifdef MATHDEBUGGING
         : logger_("quaterniondebug.log", "Quaternion")
#endif
      {
         Initialize(0.0, 0.0, 0.0, 0.0);
      }

      Quaternion (const float & a0, const Vector3 & b)
#ifdef MATHDEBUGGING
         : logger_("quaterniondebug.log", "Quaternion")
#endif
      {
         Initialize(a0, b[0], b[1], b[2]);
         //std::cout << "the vector part of the quaternion: " << vector_ << std::endl;
      }

      Quaternion (const Quaternion & a)
#ifdef MATHDEBUGGING
         : logger_("quaterniondebug.log", "Quaternion")
#endif
      {
         Initialize(a[0], a[1], a[2], a[3]);
      }

      Quaternion (float a0, float a1, float a2, float a3)
#ifdef MATHDEBUGGING
         : logger_("quaterniondebug.log", "Quaternion")
#endif
      {
         Initialize(a0, a1, a2, a3);
      }

      void Initialize (float a0, float a1, float a2, float a3)
      {
         scalar_ = a0;
         vector_[0] = a1;
         vector_[1] = a2;
         vector_[2] = a3;
      }

      float magnitude (void) const
      {
         float m = sqrt(scalar_*scalar_ + vector_.magnitude()*vector_.magnitude());
         return m;
      }

      float magnitudeSquared (void) const
      {
         float m = scalar_*scalar_ + vector_.magnitude()*vector_.magnitude();
         return m;
      }

      Vector3 & vector (void)
      {
         return vector_;
      }

      const Vector3 & vector (void) const
      {
         return vector_;
      }

      float & scalar (void)
      {
         return scalar_;
      }

      const float & scalar (void) const
      {
         return scalar_;
      }

      Quaternion conjugate (void) const
      {
         Quaternion result(scalar(), -1.0*vector());
         return result;
      }

      // For a unit quaternion, performs the equivalent of rotating the vector
      // with the quaternion expressed as a rotation matrix.
      //
      //    x_prime = quat.rotationMatrix() * x
      //
      //    x_prime = quat.sandwich(x)
      //
      // This saves a conversion and several multiply/adds.
      Vector3 sandwich (const Vector3 & input) const
      {
         Quaternion unit = (*this)/this->magnitude();
         float q0 = unit[0];
         float q1 = unit[1];
         float q2 = unit[2];
         float q3 = unit[3];

         Vector3 output;

         // 00, 11, 22, 33, 12, 03, 13, 02, 23, 01

         output[0] = (
            (q0*q0 + q1*q1 - q2*q2 - q3*q3) * input[0] + \
            2.f * (q1*q2 + q0*q3) * input[1] + \
            2.f * (q1*q3 - q0*q2) * input[2]
         );
         output[1] = (
            2.f * (q1*q2 - q0*q3) * input[0] + \
            (q0*q0 - q1*q1 + q2*q2 - q3*q3) * input[1] + \
            2.f * (q2*q3 + q0*q1) * input[2]
         );
         output[2] = (
            2.f * (q1*q3 + q0*q2) * input[0] + \
            2.f * (q2*q3 - q0*q1) * input[1] + \
            (q0*q0 - q1*q1 - q2*q2 + q3*q3) * input[2]
         );

         return output;
      }

      // For a unit quaternion, performs the equivalent of rotating the vector
      // with the conjugate quaternion expressed as a rotation matrix.
      //
      //    x_prime = quat.rotationMatrix().transpose() * x
      //
      //    x_prime = quat.conjugateSandwich(x)
      //
      // This saves a conversion, several swaps, and several multiply/adds.
      Vector3 conjugateSandwich (const Vector3 & input) const
      {
         Quaternion unit = (*this)/this->magnitude();
         float q0 = unit[0];
         float q1 = unit[1];
         float q2 = unit[2];
         float q3 = unit[3];

         Vector3 output;

         output[0] = (
            (q0*q0 + q1*q1 - q2*q2 - q3*q3) * input[0] + \
            2.f * (q1*q2 - q0*q3) * input[1] + \
            2.f * (q1*q3 + q0*q2) * input[2]
         );
         output[1] = (
            2.f * (q1*q2 + q0*q3) * input[0] + \
            (q0*q0 - q1*q1 + q2*q2 - q3*q3) * input[1] + \
            2.f * (q2*q3 - q0*q1) * input[2]
         );
         output[2] = (
            2.f * (q1*q3 - q0*q2) * input[0] + \
            2.f * (q2*q3 + q0*q1) * input[1] + \
            (q0*q0 - q1*q1 - q2*q2 + q3*q3) * input[2]
         );

         return output;
      }

      Matrix33 rotationMatrix (void) const
      {
         Quaternion unit = (*this)/this->magnitude();
         float q0 = unit[0];
         float q1 = unit[1];
         float q2 = unit[2];
         float q3 = unit[3];
         Matrix33 R;
         R(0, 0) = q0*q0 + q1*q1 - q2*q2 - q3*q3;
         R(0, 1) = 2.f * (q1*q2 + q0*q3);
         R(0, 2) = 2.f * (q1*q3 - q0*q2);
         R(1, 0) = 2.f * (q1*q2 - q0*q3);
         R(1, 1) = q0*q0 - q1*q1 + q2*q2 - q3*q3;
         R(1, 2) = 2.f * (q2*q3 + q0*q1);
         R(2, 0) = 2.f * (q1*q3 + q0*q2);
         R(2, 1) = 2.f * (q2*q3 - q0*q1);
         R(2, 2) = q0*q0 - q1*q1 - q2*q2 + q3*q3;

         return R;
      }

      float & operator[] (unsigned int i)
      {
         if (i == 0)
         {
            return scalar_;
         }
         else
         {
            return vector_[i - 1];
         }
      }

      const float & operator[] (unsigned int i) const
      {
         if (i == 0)
         {
            return scalar_;
         }
         else
         {
            return vector_[i - 1];
         }
      }

      Quaternion & operator= (const Quaternion & a)
      {
         if (&a == this)
         {
            return *this;
         }

         Initialize(a[0], a[1], a[2], a[3]);
         return *this;
      }

      const Quaternion operator= (const Quaternion & a) const
      {
         if (&a == this)
         {
            return *this;
         }
         Quaternion result(a[0], a[1], a[2], a[3]);
         return result;
      }

      Quaternion operator* (float b) const
      {
         Quaternion result;
         for (unsigned int i = 0; i < 4; ++i)
         {
            result[i] = (*this)[i] * b;
         }
         return result;
      }

      Quaternion operator* (const Quaternion & b) const
      {
         Quaternion result;
         float a0 = scalar_;
         float b0 = b[0];
         Vector3 aVec = vector_;
         Vector3 bVec = b.vector();
         result[0] = a0*b0 - aVec.dot(bVec);
         result.vector() = a0*bVec + b0*aVec + aVec.crossProduct(bVec);
         return result;
      }

      Quaternion & operator*= (float b)
      {
         for (unsigned int i = 0; i < 4; ++i)
         {
            (*this)[i] *= b;
         }
         return *this;
      }

      Quaternion & operator*= (const Quaternion b)
      {
         Vector3 aVec = vector_;
         Vector3 bVec = b.vector();
         float a0 = scalar_;
         float b0 = b[0];
         scalar_ = a0*b0 - aVec.dot(bVec);
         vector_ = a0*bVec + b0*aVec + aVec.crossProduct(bVec);
         return *this;
      }

      Quaternion operator/ (const float b) const
      {
         Quaternion result(*this);
         for (unsigned int i = 0; i < 4; ++i)
         {
            result[i] /= b;
         }
         return result;
      }

      Quaternion & operator/= (const float b)
      {
         for (unsigned int i = 0; i < 4; ++i)
         {
            (*this)[i] /= b;
         }
         return *this;
      }

      Quaternion operator+ (const Quaternion & b) const
      {
         Quaternion result;
         result.scalar() = scalar_ + b.scalar();
         result.vector() = vector_ + b.vector();
         return result;
      }

      Quaternion & operator+= (const Quaternion & b)
      {
         scalar_ += b.scalar();
         vector_ += b.vector();
         return *this;
      }

      Quaternion operator- (const Quaternion & b) const
      {
         Quaternion result;
         result.scalar() = scalar_ - b.scalar();
         result.vector() = vector_ - b.vector();
         return result;
      }

      Quaternion & operator-= (const Quaternion & b)
      {
         scalar_ -= b.scalar();
         vector_ -= b.vector();
         return *this;
      }

      Quaternion operator~ (void) const
      {
         Quaternion result;
         result = conjugate()/magnitude();
         return result;
      }

   private:

      float scalar_;

      Vector3 vector_;

#ifdef MATHDEBUGGING
      Logger logger_;
#endif

};

Quaternion operator* (const float & a, const Quaternion b);

Quaternion exp (const Vector3 & u);

Quaternion exp (const Quaternion & u);

std::ostream & operator << (std::ostream & out, const Quaternion & a);

#endif
