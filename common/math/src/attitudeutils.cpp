#include "attitudeutils.hpp"

#include <cmath>

void quaternionToAttitude(const Quaternion & quatIn, Vector3 & rpyOut)
{
    // Roll
    rpyOut[0] = std::atan2(
        2*(quatIn[0]*quatIn[1] + quatIn[2]*quatIn[3]),
        1 - 2*(quatIn[1]*quatIn[1] + quatIn[2]*quatIn[2])
    );
    // Pitch
    rpyOut[1] = std::asin(2*(quatIn[0]*quatIn[2] - quatIn[1]*quatIn[3]));
    // Yaw
    rpyOut[2] = std::atan2(
        2*(quatIn[0]*quatIn[3] + quatIn[1]*quatIn[2]),
        1 - 2*(quatIn[2]*quatIn[2] + quatIn[3]*quatIn[3])
    );
}

Vector3 quaternionToAttitude(const Quaternion & quatIn)
{
    Vector3 rpyOut;
    quaternionToAttitude(quatIn, rpyOut);
    return rpyOut;
}

void attitudeToQuaternion(const Vector3 & rpyIn, Quaternion & quatOut)
{
    // Phi - Roll
    // Theta - Pitch
    // Psi - Yaw
    const float & phi = rpyIn[0];
    const float & theta = rpyIn[1];
    const float & psi = rpyIn[2];
    float sphi = std::sin(phi/2);
    float cphi = std::cos(phi/2);
    float stheta = std::sin(theta/2);
    float ctheta = std::cos(theta/2);
    float spsi = std::sin(psi/2);
    float cpsi = std::cos(psi/2);
    quatOut[0] = cphi * ctheta * cpsi + sphi * stheta * spsi;
    quatOut[1] = sphi * ctheta * cpsi - cphi * stheta * spsi;
    quatOut[2] = cphi * stheta * cpsi + sphi * ctheta * spsi;
    quatOut[3] = cphi * ctheta * spsi - sphi * stheta * cpsi;
}

Quaternion attitudeToQuaternion(const Vector3 & rpyIn)
{
    Quaternion quatOut;
    attitudeToQuaternion(rpyIn, quatOut);
    return quatOut;
}

void frd2NedMatrix(const Vector3 & rpy, Matrix33 & R_frd2Ned)
{
    frd2NedMatrix(rpy[0], rpy[1], rpy[2], R_frd2Ned);
}

Matrix33 frd2NedMatrix(const Vector3 & rpy)
{
    Matrix33 R_frd2Ned;
    frd2NedMatrix(rpy, R_frd2Ned);

    return R_frd2Ned;
}

void frd2NedMatrix(float roll, float pitch, float yaw, Matrix33 & R_frd2Ned)
{
    float sphi = std::sin(roll);
    float cphi = std::cos(roll);
    float stheta = std::sin(pitch);
    float ctheta = std::cos(pitch);
    float spsi = std::sin(yaw);
    float cpsi = std::cos(yaw);

    R_frd2Ned(0, 0) = ctheta * cpsi;
    R_frd2Ned(0, 1) = sphi * stheta * cpsi - cphi * spsi;
    R_frd2Ned(0, 2) = cphi * stheta * cpsi + sphi * spsi;

    R_frd2Ned(1, 0) = ctheta * spsi;
    R_frd2Ned(1, 1) = sphi * stheta * spsi + cphi * cpsi;
    R_frd2Ned(1, 2) = cphi * stheta * spsi - sphi * cpsi;

    R_frd2Ned(2, 0) = -1.0f * stheta;
    R_frd2Ned(2, 1) = sphi * ctheta;
    R_frd2Ned(2, 2) = cphi * ctheta;
}

Matrix33 frd2NedMatrix(float roll, float pitch, float yaw)
{
    Matrix33 R_frd2Ned;
    frd2NedMatrix(roll, pitch, yaw, R_frd2Ned);

    return R_frd2Ned;
}

void ned2FrdMatrix(const Vector3 & rpy, Matrix33 & R_ned2Frd)
{
    ned2FrdMatrix(rpy[0], rpy[1], rpy[2], R_ned2Frd);
}

Matrix33 ned2FrdMatrix(const Vector3 & rpy)
{
    Matrix33 R_ned2Frd;
    ned2FrdMatrix(rpy[0], rpy[1], rpy[2], R_ned2Frd);

    return R_ned2Frd;
}

void ned2FrdMatrix(float roll, float pitch, float yaw, Matrix33 & R_ned2Frd)
{
    float sphi = std::sin(roll);
    float cphi = std::cos(roll);
    float stheta = std::sin(pitch);
    float ctheta = std::cos(pitch);
    float spsi = std::sin(yaw);
    float cpsi = std::cos(yaw);

    R_ned2Frd(0, 0) = ctheta * cpsi;
    R_ned2Frd(1, 0) = sphi * stheta * cpsi - cphi * spsi;
    R_ned2Frd(2, 0) = cphi * stheta * cpsi + sphi * spsi;

    R_ned2Frd(0, 1) = ctheta * spsi;
    R_ned2Frd(1, 1) = sphi * stheta * spsi + cphi * cpsi;
    R_ned2Frd(2, 1) = cphi * stheta * spsi - sphi * cpsi;

    R_ned2Frd(0, 2) = -1.0f * stheta;
    R_ned2Frd(1, 2) = sphi * ctheta;
    R_ned2Frd(2, 2) = cphi * ctheta;
}

Matrix33 ned2FrdMatrix(float roll, float pitch, float yaw)
{
    Matrix33 R_ned2Frd;
    ned2FrdMatrix(roll, pitch, yaw, R_ned2Frd);

    return R_ned2Frd;
}

void frd2NedMatrixToRpy(const Matrix33 & R_frd2Ned, Vector3 & rpy)
{
    frd2NedMatrixToRpy(R_frd2Ned, rpy[0], rpy[1], rpy[2]);
}

void frd2NedMatrixToRpy(
   const Matrix33 & R_frd2Ned, float & roll, float & pitch, float & yaw
)
{
    roll  = atan2f(R_frd2Ned(2, 1), R_frd2Ned(2, 2));
    pitch = -1.f * asinf(R_frd2Ned(2, 0));
    yaw   = atan2f(R_frd2Ned(1, 0), R_frd2Ned(0, 0));
}

Vector3 frd2NedMatrixToRpy(const Matrix33 & R_frd2Ned)
{
    Vector3 rpy;
    frd2NedMatrixToRpy(R_frd2Ned, rpy[0], rpy[1], rpy[2]);

    return rpy;
}

void ned2FrdMatrixToRpy(const Matrix33 & R_ned2Frd, Vector3 & rpy)
{
    ned2FrdMatrixToRpy(R_ned2Frd, rpy[0], rpy[1], rpy[2]);
}

void ned2FrdMatrixToRpy(
   const Matrix33 & R_ned2Frd, float & roll, float & pitch, float & yaw
)
{
    roll  = atan2f(R_ned2Frd(1, 2), R_ned2Frd(2, 2));
    pitch = -1.f * asinf(R_ned2Frd(0, 2));
    yaw   = atan2f(R_ned2Frd(0, 1), R_ned2Frd(0, 0));
}

Vector3 ned2FrdMatrixToRpy(const Matrix33 & R_ned2Frd)
{
    Vector3 rpy;
    ned2FrdMatrixToRpy(R_ned2Frd, rpy[0], rpy[1], rpy[2]);

    return rpy;
}

Matrix33 rodriguesRotation(const Vector3 & axis, float angle)
{
    Matrix33 K = crossProductMatrix(axis/axis.magnitude());
    Matrix33 R = identityMatrix() + std::sin(angle) * K + (1.0f - std::cos(angle)) * (K * K);
    return R;
}

Matrix33 makeVectorUp(const Vector3 & up)
{
    Matrix33 rotMat;

    const Vector3 zHatGlobal(up.unitVector());
    // crossVec is used to define the x-hat vector
    Vector3 crossVec(0.f, 0.f, 0.f);

    unsigned int bestIndex = 0;
    for (unsigned int i = 1; i < 3; ++i)
    {
        if (zHatGlobal[i] > zHatGlobal[bestIndex])
        {
            bestIndex = i;
        }
    }

    crossVec[bestIndex] = 1.f;
    if (fabs(crossVec.dot(zHatGlobal)) >= 1.f - 1e-7f)
    {
        crossVec[bestIndex] = 0.f;
        crossVec[(bestIndex + 1) % 3] = 1.f;
    }

    // If the z-axis is in the direction of the collision normal, just use
    // identity matrix to avoid nan's.
    Vector3 zHat(0.0f, 0.0f, 1.0f);
    if (fabs(zHatGlobal.dot(zHat)) >= (1.0f - 1e-7))
    {
        rotMat = identityMatrix();
        return rotMat;
    }

    Vector3 xHatGlobal = crossVec.crossProduct(zHatGlobal);
    // xHatGlobal /= xHatGlobal.magnitude();
    xHatGlobal.Normalize();

    Vector3 yHatGlobal = zHatGlobal.crossProduct(xHatGlobal);
    // yHatGlobal /= yHatGlobal.magnitude();
    yHatGlobal.Normalize();

    // Use the nice property that R_global/face = R_face/global.transpose()
    // to generate a rotation matrix of the form
    //    R_face/global --> rotation from global coords to face coords.
    rotMat(0, 0) = xHatGlobal[0];
    rotMat(0, 1) = xHatGlobal[1];
    rotMat(0, 2) = xHatGlobal[2];

    rotMat(1, 0) = yHatGlobal[0];
    rotMat(1, 1) = yHatGlobal[1];
    rotMat(1, 2) = yHatGlobal[2];

    rotMat(2, 0) = zHatGlobal[0];
    rotMat(2, 1) = zHatGlobal[1];
    rotMat(2, 2) = zHatGlobal[2];

    return rotMat;
}
