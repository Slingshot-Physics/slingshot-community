#include "matrix.hpp"


float determinant3x3(const Matrix<3, 3> & A)
{
   float det = \
      A(0, 0)*A(1, 1)*A(2, 2) +
      A(0, 1)*A(1, 2)*A(2, 0) +
      A(0, 2)*A(1, 0)*A(2, 1) -
      A(2, 0)*A(1, 1)*A(0, 2) -
      A(2, 1)*A(1, 2)*A(0, 0) -
      A(2, 2)*A(1, 0)*A(0, 1);
   return det;
}

Matrix<3, 3> inverse3x3(const Matrix<3, 3> & A)
{
   Matrix<3, 3> ret;
   float det = determinant3x3(A);

   ret(0, 0) =       A(1, 1)*A(2, 2) - A(1, 2)*A(2, 1);
   ret(0, 1) = -1.0*(A(1, 0)*A(2, 2) - A(1, 2)*A(2, 0));
   ret(0, 2) =       A(1, 0)*A(2, 1) - A(1, 1)*A(2, 0);
   ret(1, 0) = -1.0*(A(0, 1)*A(2, 2) - A(0, 2)*A(2, 1));
   ret(1, 1) =       A(0, 0)*A(2, 2) - A(0, 2)*A(2, 0);
   ret(1, 2) = -1.0*(A(0, 0)*A(2, 1) - A(0, 1)*A(2, 0));
   ret(2, 0) =       A(0, 1)*A(1, 2) - A(0, 2)*A(1, 1);
   ret(2, 1) = -1.0*(A(0, 0)*A(1, 2) - A(0, 2)*A(1, 0));
   ret(2, 2) =       A(0, 0)*A(1, 1) - A(0, 1)*A(1, 0);
   ret = ret.transpose();
   
   ret /= det;
   return ret;
}

Matrix<6, 6> inverse6x6 (const Matrix<6, 6> & A)
{
   Matrix<6, 6> A_inv;
   // |A B|     |E F|
   // |C D|     |G H|
   Matrix<3, 3> E(A.template getSlice<3, 3>(0, 0));
   Matrix<3, 3> F(A.template getSlice<3, 3>(0, 3));
   Matrix<3, 3> G(A.template getSlice<3, 3>(3, 0));
   Matrix<3, 3> H(A.template getSlice<3, 3>(3, 3));

   Matrix<3, 3> E_inv = inverse3x3(E);
   // std::cout << "E_inv: \n" << E_inv << "\n";
   Matrix<3, 3> HmGEinvF_inv = inverse3x3(H - G * E_inv * F);
   // std::cout << "H: \n" << H << "\n";
   // std::cout << "\tnegative part:\n" << (G * E_inv * F) << "\n";
   // std::cout << "pre-inv huge mess: \n"<< (H - G * E_inv * F) << "\n";
   // std::cout << "A huge mess: \n" << HmGEinvF_inv << "\n";
   Matrix<3, 3> F_HMGEinvF_inv = F * HmGEinvF_inv;
   A_inv.template assignSlice<3, 3>(0, 0, E_inv + E_inv * F_HMGEinvF_inv * G * E_inv);
   A_inv.template assignSlice<3, 3>(0, 3, -1.0f * E_inv * F_HMGEinvF_inv);
   A_inv.template assignSlice<3, 3>(3, 0, -1.0f * HmGEinvF_inv * G * E_inv);
   A_inv.template assignSlice<3, 3>(3, 3, HmGEinvF_inv);

   return A_inv;
}
