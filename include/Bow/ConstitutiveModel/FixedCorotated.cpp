#include "FixedCorotated.h"
#include <Bow/Math/SVD.h>
#include <Bow/Math/PolarDecomposition.h>
#include "ConstitutiveModel.h"
#include <Bow/Math/Utils.h>
#include <Bow/Math/MathTools.h>

namespace Bow::ConstitutiveModel {
template <class Scalar, int dim>
BOW_INLINE Scalar FixedCorotated<Scalar, dim>::psi(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar lam)
{
    Bow::Matrix<Scalar, dim, dim> U, V;
    Bow::Vector<Scalar, dim> sigma;
    Math::svd(F, U, sigma, V);
    return mu * (sigma - Bow::Vector<Scalar, dim>::Ones()).squaredNorm() + Scalar(0.5) * lam * std::pow(sigma.prod() - Scalar(1), 2);
}
template <class Scalar, int dim>
BOW_INLINE void FixedCorotated<Scalar, dim>::first_piola(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar lam, Matrix<Scalar, dim, dim>& P)
{
    Scalar J = F.determinant();
    Eigen::Matrix<Scalar, dim, dim> JFinvT;
    Math::cofactor(F, JFinvT);
    Bow::Matrix<Scalar, dim, dim> R, S;
    Math::polar_decomposition(F, R, S);
    P = Scalar(2) * mu * (F - R) + lam * (J - 1) * JFinvT;
}

template <class Scalar, int dim>
BOW_INLINE void FixedCorotated<Scalar, dim>::dpsi_dsigma(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Vector<Scalar, dim>& de_dsigma)
{
    Scalar sigma_prod = sigma.prod();
    Vector<Scalar, dim> sigma_prod_noI;
    if constexpr (dim == 2) {
        sigma_prod_noI[0] = sigma[1];
        sigma_prod_noI[1] = sigma[0];
    }
    else {
        sigma_prod_noI[0] = sigma[1] * sigma[2];
        sigma_prod_noI[1] = sigma[0] * sigma[2];
        sigma_prod_noI[2] = sigma[0] * sigma[1];
    }
    de_dsigma = Scalar(2) * mu * (sigma - Bow::Vector<Scalar, dim>::Ones()) + lam * (sigma_prod - Scalar(1)) * sigma_prod_noI;
}

template <class Scalar, int dim>
BOW_INLINE void FixedCorotated<Scalar, dim>::d2psi_dsigma2(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Matrix<Scalar, dim, dim>& d2e_dsigma2)
{
    Scalar sigma_prod = sigma.prod();
    Scalar _2mu = mu * Scalar(2);
    if constexpr (dim == 2) {
        Bow::Vector<Scalar, dim> sigma_prod_noI;
        sigma_prod_noI[0] = sigma[1];
        sigma_prod_noI[1] = sigma[0];
        d2e_dsigma2(0, 0) = _2mu + lam * sigma_prod_noI[0] * sigma_prod_noI[0];
        d2e_dsigma2(1, 1) = _2mu + lam * sigma_prod_noI[1] * sigma_prod_noI[1];
        d2e_dsigma2(0, 1) = d2e_dsigma2(1, 0) = lam * ((sigma_prod - 1) + sigma_prod_noI[0] * sigma_prod_noI[1]);
    }
    else {
        Bow::Vector<Scalar, dim> sigma_prod_noI;
        sigma_prod_noI[0] = sigma[1] * sigma[2];
        sigma_prod_noI[1] = sigma[0] * sigma[2];
        sigma_prod_noI[2] = sigma[0] * sigma[1];
        d2e_dsigma2(0, 0) = _2mu + lam * sigma_prod_noI[0] * sigma_prod_noI[0];
        d2e_dsigma2(1, 1) = _2mu + lam * sigma_prod_noI[1] * sigma_prod_noI[1];
        d2e_dsigma2(2, 2) = _2mu + lam * sigma_prod_noI[2] * sigma_prod_noI[2];
        d2e_dsigma2(0, 1) = d2e_dsigma2(1, 0) = lam * (sigma[2] * (sigma_prod - 1) + sigma_prod_noI[0] * sigma_prod_noI[1]);
        d2e_dsigma2(0, 2) = d2e_dsigma2(2, 0) = lam * (sigma[1] * (sigma_prod - 1) + sigma_prod_noI[0] * sigma_prod_noI[2]);
        d2e_dsigma2(1, 2) = d2e_dsigma2(2, 1) = lam * (sigma[0] * (sigma_prod - 1) + sigma_prod_noI[1] * sigma_prod_noI[2]);
    }
}

template <class Scalar, int dim>
BOW_INLINE void FixedCorotated<Scalar, dim>::B_left_coeff(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Vector<Scalar, 2 * dim - 3>& left_coeff)
{
    // https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf
    // Eq 77
    if constexpr (dim == 2)
        left_coeff[0] = mu - Scalar(0.5) * lam * (sigma.prod() - Scalar(1));
    else {
        Scalar sigma_prod = sigma.prod();
        left_coeff[0] = mu - 0.5 * lam * sigma[2] * (sigma_prod - 1);
        left_coeff[1] = mu - 0.5 * lam * sigma[0] * (sigma_prod - 1);
        left_coeff[2] = mu - 0.5 * lam * sigma[1] * (sigma_prod - 1);
    }
}

#ifdef BOW_STATIC_LIBRARY
#ifdef BOW_COMPILE_FLOAT
#ifdef BOW_COMPILE_2D
template class FixedCorotated<float, 2>;
#endif
#ifdef BOW_COMPILE_3D
template class FixedCorotated<float, 3>;
#endif
#endif
#ifdef BOW_COMPILE_DOUBLE
#ifdef BOW_COMPILE_2D
template class FixedCorotated<double, 2>;
#endif
#ifdef BOW_COMPILE_3D
template class FixedCorotated<double, 3>;
#endif
#endif
#endif

} // namespace Bow::ConstitutiveModel