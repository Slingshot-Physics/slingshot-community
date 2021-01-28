#ifndef ATTITUDEUTILSHEADER
#define ATTITUDEUTILSHEADER

#include "matrix33.hpp"
#include "quaternion.hpp"
#include "vector3.hpp"

// Converts a unit (w, x, y, z) quaternion into a vector of (roll, pitch, yaw)
// as a 3-2-1 Tait-Bryan sequence.
void quaternionToAttitude(const Quaternion & quatIn, Vector3 & rpyOut);

// Converts a unit (w, x, y, z) quaternion into a vector of (roll, pitch, yaw)
// as a 3-2-1 Tait-Bryan sequence, returns the vector.
Vector3 quaternionToAttitude(const Quaternion & quatIn);

// Converts a vector of (roll, pitch, yaw) into a unit (w, x, y, z) quaternion
// as a 3-2-1 Tait-Bryan sequence.
void attitudeToQuaternion(const Vector3 & rpyIn, Quaternion & quatOut);

// Converts a vector of (roll, pitch, yaw) into a unit (w, x, y, z) quaternion
// as a 3-2-1 Tait-Bryan sequence, returns the quaternion.
Quaternion attitudeToQuaternion(const Vector3 & rpyIn);

// The roll, pitch, yaw sequence is a 3-2-1 Tait Bryan set of roll, pitch, and
// yaw angles in radians. Generates a matrix that rotates from front-right-down
// to NED in body frame.
// FRD to NED.
void frd2NedMatrix(const Vector3 & rpy, Matrix33 & R_frd2Ned);

// The roll, pitch, yaw sequence is a 3-2-1 Tait Bryan set of roll, pitch, and
// yaw angles in radians. Generates a matrix that rotates from front-right-down
// to NED in body frame.
// FRD to NED.
Matrix33 frd2NedMatrix(const Vector3 & rpy);

// The roll, pitch, yaw sequence is a 3-2-1 Tait Bryan set of roll, pitch, and
// yaw angles in radians. Generates a matrix that rotates from front-right-down
// to NED in body frame.
// FRD to NED.
void frd2NedMatrix(float roll, float pitch, float yaw, Matrix33 & R_frd2Ned);

// The roll, pitch, yaw sequence is a 3-2-1 Tait Bryan set of roll, pitch, and
// yaw angles in radians. Generates a matrix that rotates from front-right-down
// to NED in body frame.
// FRD to NED.
Matrix33 frd2NedMatrix(float roll, float pitch, float yaw);

// The roll, pitch, yaw sequence is a 3-2-1 Tait Bryan set of roll, pitch, and
// yaw angles in radians. Generates a matrix that rotates from NED to
// front-right-down in body frame.
// NED to FRD.
void ned2FrdMatrix(const Vector3 & rpy, Matrix33 & R_ned2Frd);

// The roll, pitch, yaw sequence is a 3-2-1 Tait Bryan set of roll, pitch, and
// yaw angles in radians. Generates a matrix that rotates from NED to
// front-right-down in body frame.
// NED to FRD.
Matrix33 ned2FrdMatrix(const Vector3 & rpy);

// The roll, pitch, yaw sequence is a 3-2-1 Tait Bryan set of roll, pitch, and
// yaw angles in radians. Generates a matrix that rotates from NED to
// front-right-down in body frame.
// NED to FRD.
void ned2FrdMatrix(float roll, float pitch, float yaw, Matrix33 & R_ned2Frd);

// The roll, pitch, yaw sequence is a 3-2-1 Tait Bryan set of roll, pitch, and
// yaw angles in radians. Generates a matrix that rotates from NED to
// front-right-down in body frame.
// NED to FRD.
Matrix33 ned2FrdMatrix(float roll, float pitch, float yaw);

// Generates RPY sequence from a matrix that rotates from FRD to NED.
void frd2NedMatrixToRpy(const Matrix33 & R_frd2Ned, Vector3 & rpy);

// Generates RPY sequence from a matrix that rotates from FRD to NED.
void frd2NedMatrixToRpy(
   const Matrix33 & R_frd2Ned, float & roll, float & pitch, float & yaw
);

// Generates RPY sequence from a matrix that rotates from FRD to NED.
Vector3 frd2NedMatrixToRpy(const Matrix33 & R_frd2Ned);

// Generates RPY sequence from a matrix that rotates from NED to FRD.
void ned2FrdMatrixToRpy(const Matrix33 & R_ned2Frd, Vector3 & rpy);

// Generates RPY sequence from a matrix that rotates from NED to FRD.
void ned2FrdMatrixToRpy(
   const Matrix33 & R_ned2Frd, float & roll, float & pitch, float & yaw
);

// Generates RPY sequence from a matrix that rotates from NED to FRD.
Vector3 ned2FrdMatrixToRpy(const Matrix33 & R_ned2Frd);

// Generates a rotation matrix by rotating by some angle (radians) around an
// axis (does not have to be normalized).
Matrix33 rodriguesRotation(const Vector3 & axis, float angle);

// Generates a rotation matrix R that ensures that R * up = z_hat.
// The input vector does not have to be normalized.
Matrix33 makeVectorUp(const Vector3 & up);

#endif
