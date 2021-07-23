#ifndef DIHEDRAL_ANGLE_H
#define DIHEDRAL_ANGLE_H
#include <Bow/Types.h>
#include <math.h>

namespace Bow::Geometry {

namespace internal {
template <class T>
Vector<T, 3> compute_mHat(const Vector<T, 3>& xp, const Vector<T, 3>& xe0, const Vector<T, 3>& xe1)
{
    auto e = xe1 - xe0;
    Vector<T, 3> mHat = xe0 + (xp - xe0).dot(e) / e.squaredNorm() * e - xp;
    mHat /= mHat.norm();
    return mHat;
}
} // namespace internal

/**
 *             v2 --- v3
 *            /  \    /
 *           /    \  /
 *          v0 --- v1
 * \param branch 0: the angle is in (-pi, pi); +1: the angle is in (0, 2pi); -1: the angle is in (-2pi, 0)
 */
template <class T>
T dihedral_angle(const Vector<T, 3>& v0, const Vector<T, 3>& v1, const Vector<T, 3>& v2, const Vector<T, 3>& v3, const int branch = 0)
{
    auto n1 = (v1 - v0).cross(v2 - v0);
    auto n2 = (v2 - v3).cross(v1 - v3);
    T DA = std::acos(std::max(-1., std::min(1., n1.dot(n2) / std::sqrt(n1.squaredNorm() * n2.squaredNorm()))));
    if (n2.cross(n1).dot(v1 - v2) < 0)
        DA = -DA;
    // if (branch == 1 && DA < 1e-6) // if DA is in (0, 2pi) actually, especially when DA is approching pi
    //     DA += 2 * M_PI;
    // else if (branch == -1 and DA > -1e-6) // if DA is in (-2pi, 0) actually, especially when DA is approching -pi
    //     DA -= 2 * M_PI;
    return DA;
}

/**
 *             v1 --- v3
 *            /  \    /
 *           /    \  /
 *          v2 --- v0
 */
template <class T>
void dihedral_angle_gradient(const Vector<T, 3>& v2, const Vector<T, 3>& v0, const Vector<T, 3>& v1, const Vector<T, 3>& v3, Vector<T, 3 * 4>& grad)
{
    // here we map our v order to rusmas' in this function for implementation convenience
    Vector<T, 3> e0 = v1 - v0;
    Vector<T, 3> e1 = v2 - v0;
    Vector<T, 3> e2 = v3 - v0;
    Vector<T, 3> e3 = v2 - v1;
    Vector<T, 3> e4 = v3 - v1;
    Vector<T, 3> n1 = e0.cross(e1);
    Vector<T, 3> n2 = e2.cross(e0);
    T n1SqNorm = n1.squaredNorm();
    T n2SqNorm = n2.squaredNorm();
    T e0norm = e0.norm();
    Vector<T, 3> da_dv2 = -e0norm / n1SqNorm * n1;
    Vector<T, 3> da_dv0 = -e0.dot(e3) / (e0norm * n1SqNorm) * n1 - e0.dot(e4) / (e0norm * n2SqNorm) * n2;
    Vector<T, 3> da_dv1 = e0.dot(e1) / (e0norm * n1SqNorm) * n1 + e0.dot(e2) / (e0norm * n2SqNorm) * n2;
    Vector<T, 3> da_dv3 = -e0norm / n2SqNorm * n2;

    const int dim = 3;
    grad.template segment<dim>(0 * 3) = da_dv2;
    grad.template segment<dim>(1 * 3) = da_dv0;
    grad.template segment<dim>(2 * 3) = da_dv1;
    grad.template segment<dim>(3 * 3) = da_dv3;
}

/**
 *             v1 --- v3
 *            /  \    /
 *           /    \  /
 *          v2 --- v0
 */
template <class T>
void dihedral_angle_hessian(const Vector<T, 3>& v2, const Vector<T, 3>& v0, const Vector<T, 3>& v1, const Vector<T, 3>& v3, Matrix<T, 3 * 4, 3 * 4>& hess)
{
    // https://studios.disneyresearch.com/wp-content/uploads/2019/03/Discrete-Bending-Forces-and-Their-Jacobians-Paper.pdf
    const int dim = 3;
    Field<Vector<T, 3>> e({ v1 - v0, v2 - v0, v3 - v0, v2 - v1, v3 - v1 });
    std::vector<T> norm_e({ e[0].norm(), e[1].norm(), e[2].norm(), e[3].norm(), e[4].norm() });
    Vector<T, 3> n1 = e[0].cross(e[1]);
    Vector<T, 3> n2 = e[2].cross(e[0]);
    T n1norm = n1.norm();
    T n2norm = n2.norm();
    Vector<T, 3> mHat1 = internal::compute_mHat(v1, v0, v2);
    Vector<T, 3> mHat2 = internal::compute_mHat(v1, v0, v3);
    Vector<T, 3> mHat3 = internal::compute_mHat(v0, v1, v2);
    Vector<T, 3> mHat4 = internal::compute_mHat(v0, v1, v3);
    Vector<T, 3> mHat01 = internal::compute_mHat(v2, v0, v1);
    Vector<T, 3> mHat02 = internal::compute_mHat(v3, v0, v1);
    T cosalpha1 = e[0].dot(e[1]) / (norm_e[0] * norm_e[1]);
    T cosalpha2 = e[0].dot(e[2]) / (norm_e[0] * norm_e[2]);
    T cosalpha3 = -e[0].dot(e[3]) / (norm_e[0] * norm_e[3]);
    T cosalpha4 = -e[0].dot(e[4]) / (norm_e[0] * norm_e[4]);
    T h1 = n1norm / norm_e[1];
    T h2 = n2norm / norm_e[2];
    T h3 = n1norm / norm_e[3];
    T h4 = n2norm / norm_e[4];
    T h01 = n1norm / norm_e[0];
    T h02 = n2norm / norm_e[0];

    Matrix<T, dim, dim> N1_01 = n1 * (mHat01.transpose() / (h01 * h01 * n1norm));
    Matrix<T, dim, dim> N2_02 = n2 * (mHat02.transpose() / (h02 * h02 * n2norm));
    Matrix<T, dim, dim> N1_3 = n1 * (mHat3.transpose() / (h01 * h3 * n1norm));
    Matrix<T, dim, dim> N1_1 = n1 * (mHat1.transpose() / (h01 * h1 * n1norm));
    Matrix<T, dim, dim> N2_4 = n2 * (mHat4.transpose() / (h02 * h4 * n2norm));
    Matrix<T, dim, dim> N2_2 = n2 * (mHat2.transpose() / (h02 * h2 * n2norm));
    Matrix<T, dim, dim> M3_01_1 = (cosalpha3 / (h3 * h01 * n1norm) * mHat01) * n1.transpose();
    Matrix<T, dim, dim> M1_01_1 = (cosalpha1 / (h1 * h01 * n1norm) * mHat01) * n1.transpose();
    Matrix<T, dim, dim> M1_1_1 = (cosalpha1 / (h1 * h1 * n1norm) * mHat1) * n1.transpose();
    Matrix<T, dim, dim> M3_3_1 = (cosalpha3 / (h3 * h3 * n1norm) * mHat3) * n1.transpose();
    Matrix<T, dim, dim> M3_1_1 = (cosalpha3 / (h3 * h1 * n1norm) * mHat1) * n1.transpose();
    Matrix<T, dim, dim> M1_3_1 = (cosalpha1 / (h1 * h3 * n1norm) * mHat3) * n1.transpose();
    Matrix<T, dim, dim> M4_02_2 = (cosalpha4 / (h4 * h02 * n2norm) * mHat02) * n2.transpose();
    Matrix<T, dim, dim> M2_02_2 = (cosalpha2 / (h2 * h02 * n2norm) * mHat02) * n2.transpose();
    Matrix<T, dim, dim> M4_4_2 = (cosalpha4 / (h4 * h4 * n2norm) * mHat4) * n2.transpose();
    Matrix<T, dim, dim> M2_4_2 = (cosalpha2 / (h2 * h4 * n2norm) * mHat4) * n2.transpose();
    Matrix<T, dim, dim> M4_2_2 = (cosalpha4 / (h4 * h2 * n2norm) * mHat2) * n2.transpose();
    Matrix<T, dim, dim> M2_2_2 = (cosalpha2 / (h2 * h2 * n2norm) * mHat2) * n2.transpose();
    Matrix<T, dim, dim> B1 = n1 * mHat01.transpose() / (norm_e[0] * norm_e[0] * n1norm);
    Matrix<T, dim, dim> B2 = n2 * mHat02.transpose() / (norm_e[0] * norm_e[0] * n2norm);

    hess.setZero();

    Matrix<T, dim, dim> H00 = -(N1_01 + N1_01.transpose());
    hess.template block<dim, dim>(0, 0) = H00;

    Matrix<T, dim, dim> H10 = M3_01_1 - N1_3;
    hess.template block<dim, dim>(3, 0) = H10;
    hess.template block<dim, dim>(0, 3) = H10.transpose();

    Matrix<T, dim, dim> H20 = M1_01_1 - N1_1;
    hess.template block<dim, dim>(6, 0) = H20;
    hess.template block<dim, dim>(0, 6) = H20.transpose();

    Matrix<T, dim, dim> H11 = M3_3_1 + M3_3_1.transpose() - B1 + M4_4_2 + M4_4_2.transpose() - B2;
    hess.template block<dim, dim>(3, 3) = H11;

    Matrix<T, dim, dim> H12 = M3_1_1 + M1_3_1.transpose() + B1 + M4_2_2 + M2_4_2.transpose() + B2;
    hess.template block<dim, dim>(3, 6) = H12;
    hess.template block<dim, dim>(6, 3) = H12.transpose();

    Matrix<T, dim, dim> H13 = M4_02_2 - N2_4;
    hess.template block<dim, dim>(3, 9) = H13;
    hess.template block<dim, dim>(9, 3) = H13.transpose();

    Matrix<T, dim, dim> H22 = M1_1_1 + M1_1_1.transpose() - B1 + M2_2_2 + M2_2_2.transpose() - B2;
    hess.template block<dim, dim>(6, 6) = H22;

    Matrix<T, dim, dim> H23 = M2_02_2 - N2_2;
    hess.template block<dim, dim>(6, 9) = H23;
    hess.template block<dim, dim>(9, 6) = H23.transpose();

    Matrix<T, dim, dim> H33 = -(N2_02 + N2_02.transpose());
    hess.template block<dim, dim>(9, 9) = H33;
}
} // namespace Bow::Geometry

#endif