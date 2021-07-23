#include "ConstitutiveModel.h"
#include <Bow/Math/SVD.h>
#include <Bow/Types.h>
#include <Bow/Math/Utils.h>

namespace Bow {
namespace ConstitutiveModel {
template <class Scalar>
BOW_INLINE std::pair<Scalar, Scalar> lame_paramters(Scalar E, Scalar nu, bool plane_stress)
{
    Scalar mu = 0.5 * E / (1 + nu);
    Scalar lam = E * nu / (plane_stress ? (1 - nu * nu) : ((1 + nu) * (1 - 2 * nu)));
    return std::make_pair(mu, lam);
}

template <class Scalar>
BOW_INLINE std::pair<Scalar, Scalar> E_nu(Scalar mu, Scalar lam, bool plane_stress)
{
    Scalar mu_plus_lam = mu + lam * (plane_stress ? 0.5 : 1.0);
    Scalar nu = lam / (2 * mu_plus_lam);
    Scalar E = mu * (2 + lam / mu_plus_lam);
    return std::make_pair(E, nu);
}

template <class Scalar, int dim>
BOW_INLINE Scalar ConstitutiveModelBase<Scalar, dim>::psi(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar lam)
{
    Matrix<Scalar, dim, dim> U, V;
    Vector<Scalar, dim> sigma;
    Math::svd(F, U, sigma, V);
    return psi_sigma(sigma, mu, lam);
}
template <class Scalar, int dim>
BOW_INLINE void ConstitutiveModelBase<Scalar, dim>::first_piola(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar lam, Matrix<Scalar, dim, dim>& P)
{
    Matrix<Scalar, dim, dim> U, V;
    Vector<Scalar, dim> sigma;
    Math::svd(F, U, sigma, V);
    Vector<Scalar, dim> de_dsigma;
    dpsi_dsigma(sigma, mu, lam, de_dsigma);
    P = U * de_dsigma.asDiagonal() * V.transpose();
}
template <class Scalar, int dim>
BOW_INLINE void ConstitutiveModelBase<Scalar, dim>::first_piola_derivative(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar lam, Matrix<Scalar, dim * dim, dim * dim>& dPdF, bool project_pd)
{
    dPdF.setZero();
    Bow::Matrix<Scalar, dim, dim> U, V;
    Bow::Vector<Scalar, dim> sigma;
    Math::svd(F, U, sigma, V);
    Bow::Vector<Scalar, dim> de_dsigma;
    Bow::Matrix<Scalar, dim, dim> d2e_dsigma2;
    dpsi_dsigma(sigma, mu, lam, de_dsigma);
    d2psi_dsigma2(sigma, mu, lam, d2e_dsigma2);
    Bow::Vector<Scalar, 2 * dim - 3> left_coeff;
    B_left_coeff(sigma, mu, lam, left_coeff);

    if (project_pd)
        Math::make_pd(d2e_dsigma2);
    if constexpr (dim == 2) {
        Scalar left_coef = left_coeff[0];
        Scalar right_coef = de_dsigma[0] + de_dsigma[1];
        Scalar sum_sigma = std::max(sigma[0] + sigma[1], Scalar(0.000001));
        right_coef /= (Scalar(2) * sum_sigma);
        Bow::Matrix<Scalar, 2, 2> B;
        B << left_coef + right_coef, left_coef - right_coef, left_coef - right_coef, left_coef + right_coef;
        if (project_pd) {
            Math::make_pd(B);
        }
        Bow::Matrix<Scalar, dim * dim, dim * dim> M;
        M.setZero();
        M(0, 0) = d2e_dsigma2(0, 0);
        M(0, 3) = d2e_dsigma2(0, 1);
        M(1, 1) = B(0, 0);
        M(1, 2) = B(0, 1);
        M(2, 1) = B(1, 0);
        M(2, 2) = B(1, 1);
        M(3, 0) = d2e_dsigma2(1, 0);
        M(3, 3) = d2e_dsigma2(1, 1);
        for (int j = 0; j < 2; ++j)
            for (int i = 0; i < 2; ++i)
                for (int s = 0; s < 2; ++s)
                    for (int r = 0; r < 2; ++r) {
                        int ij = j * 2 + i;
                        int rs = s * 2 + r;
                        dPdF(ij, rs) = M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0)
                            + M(0, 3) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1)
                            + M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1)
                            + M(1, 2) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0)
                            + M(2, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1)
                            + M(2, 2) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0)
                            + M(3, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0)
                            + M(3, 3) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1);
                    }
    }
    else {
        Scalar left_coef = left_coeff[0];
        Scalar right_coef = de_dsigma[0] + de_dsigma[1];
        Scalar sum_sigma = std::max(sigma[0] + sigma[1], Scalar(0.000001));
        right_coef /= (Scalar(2) * sum_sigma);
        Bow::Matrix<Scalar, 2, 2> B0;
        B0 << left_coef + right_coef, left_coef - right_coef, left_coef - right_coef, left_coef + right_coef;
        if (project_pd)
            Math::make_pd(B0);

        left_coef = left_coeff[1];
        right_coef = de_dsigma[1] + de_dsigma[2];
        sum_sigma = std::max(sigma[1] + sigma[2], Scalar(0.000001));
        right_coef /= (Scalar(2) * sum_sigma);
        Bow::Matrix<Scalar, 2, 2> B1;
        B1 << left_coef + right_coef, left_coef - right_coef, left_coef - right_coef, left_coef + right_coef;
        if (project_pd)
            Math::make_pd(B1);

        left_coef = left_coeff[2];
        right_coef = de_dsigma[2] + de_dsigma[0];
        sum_sigma = std::max(sigma[2] + sigma[0], Scalar(0.000001));
        right_coef /= (Scalar(2) * sum_sigma);
        Bow::Matrix<Scalar, 2, 2> B2;
        B2 << left_coef + right_coef, left_coef - right_coef, left_coef - right_coef, left_coef + right_coef;
        if (project_pd)
            Math::make_pd(B2);

        Bow::Matrix<Scalar, dim * dim, dim * dim> M;
        M.setZero();
        M(0, 0) = d2e_dsigma2(0, 0);
        M(0, 4) = d2e_dsigma2(0, 1);
        M(0, 8) = d2e_dsigma2(0, 2);
        M(4, 0) = d2e_dsigma2(1, 0);
        M(4, 4) = d2e_dsigma2(1, 1);
        M(4, 8) = d2e_dsigma2(1, 2);
        M(8, 0) = d2e_dsigma2(2, 0);
        M(8, 4) = d2e_dsigma2(2, 1);
        M(8, 8) = d2e_dsigma2(2, 2);
        M(1, 1) = B0(0, 0);
        M(1, 3) = B0(0, 1);
        M(3, 1) = B0(1, 0);
        M(3, 3) = B0(1, 1);
        M(5, 5) = B1(0, 0);
        M(5, 7) = B1(0, 1);
        M(7, 5) = B1(1, 0);
        M(7, 7) = B1(1, 1);
        M(2, 2) = B2(1, 1);
        M(2, 6) = B2(1, 0);
        M(6, 2) = B2(0, 1);
        M(6, 6) = B2(0, 0);

        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                for (int s = 0; s < 3; ++s)
                    for (int r = 0; r < 3; ++r) {
                        int ij = j * 3 + i;
                        int rs = s * 3 + r;
                        dPdF(ij, rs) = M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0)
                            + M(0, 4) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1)
                            + M(0, 8) * U(i, 0) * V(j, 0) * U(r, 2) * V(s, 2)
                            + M(4, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0)
                            + M(4, 4) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1)
                            + M(4, 8) * U(i, 1) * V(j, 1) * U(r, 2) * V(s, 2)
                            + M(8, 0) * U(i, 2) * V(j, 2) * U(r, 0) * V(s, 0)
                            + M(8, 4) * U(i, 2) * V(j, 2) * U(r, 1) * V(s, 1)
                            + M(8, 8) * U(i, 2) * V(j, 2) * U(r, 2) * V(s, 2)
                            + M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1)
                            + M(1, 3) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0)
                            + M(3, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1)
                            + M(3, 3) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0)
                            + M(5, 5) * U(i, 1) * V(j, 2) * U(r, 1) * V(s, 2)
                            + M(5, 7) * U(i, 1) * V(j, 2) * U(r, 2) * V(s, 1)
                            + M(7, 5) * U(i, 2) * V(j, 1) * U(r, 1) * V(s, 2)
                            + M(7, 7) * U(i, 2) * V(j, 1) * U(r, 2) * V(s, 1)
                            + M(2, 2) * U(i, 0) * V(j, 2) * U(r, 0) * V(s, 2)
                            + M(2, 6) * U(i, 0) * V(j, 2) * U(r, 2) * V(s, 0)
                            + M(6, 2) * U(i, 2) * V(j, 0) * U(r, 0) * V(s, 2)
                            + M(6, 6) * U(i, 2) * V(j, 0) * U(r, 2) * V(s, 0);
                    }
    }
}

#ifdef BOW_STATIC_LIBRARY
#ifdef BOW_COMPILE_FLOAT
template std::pair<float, float> lame_paramters(float E, float nu, bool);
template std::pair<float, float> E_nu(float E, float nu, bool);
#ifdef BOW_COMPILE_2D
template class ConstitutiveModelBase<float, 2>;
#endif
#ifdef BOW_COMPILE_3D
template class ConstitutiveModelBase<float, 3>;
#endif
#endif
#ifdef BOW_COMPILE_DOUBLE
template std::pair<double, double> lame_paramters(double E, double nu, bool);
template std::pair<double, double> E_nu(double E, double nu, bool);
#ifdef BOW_COMPILE_2D
template class ConstitutiveModelBase<double, 2>;
#endif
#ifdef BOW_COMPILE_3D
template class ConstitutiveModelBase<double, 3>;
#endif
#endif
#endif
}
} // namespace Bow::ConstitutiveModel