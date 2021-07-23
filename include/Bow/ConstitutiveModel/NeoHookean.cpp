#include "NeoHookean.h"
#include <Bow/Math/SVD.h>
#include <Bow/Math/PolarDecomposition.h>
#include "ConstitutiveModel.h"
#include <Bow/Math/Utils.h>
#include <Bow/Utils/Logging.h>

namespace Bow::ConstitutiveModel {

template <class Scalar, int dim>
BOW_INLINE void NeoHookean<Scalar, dim>::first_piola(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar lam, Matrix<Scalar, dim, dim>& P)
{
    Scalar J = F.determinant();
    Matrix<Scalar, dim, dim> FinvT;
    Math::cofactor(F, FinvT);
    FinvT /= J;
    P = mu * (F - FinvT) + lam * std::log(J) * FinvT;
}

template <class Scalar, int dim>
BOW_INLINE Scalar NeoHookean<Scalar, dim>::psi_sigma(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam)
{
    Scalar log_sigma_prod = std::log(sigma.prod());
    return mu / Scalar(2) * (sigma.squaredNorm() - dim) - (mu - lam / Scalar(2) * log_sigma_prod) * log_sigma_prod;
}

template <class Scalar, int dim>
BOW_INLINE void NeoHookean<Scalar, dim>::dpsi_dsigma(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Vector<Scalar, dim>& de_dsigma)
{
    Scalar log_sigma_prod = std::log(sigma.prod());
    Vector<Scalar, dim> sigma_prod_noI;
    const Scalar inv0 = 1.0 / sigma[0];
    de_dsigma[0] = mu * (sigma[0] - inv0) + lam * inv0 * log_sigma_prod;
    const Scalar inv1 = 1.0 / sigma[1];
    de_dsigma[1] = mu * (sigma[1] - inv1) + lam * inv1 * log_sigma_prod;
    if constexpr (dim == 3) {
        const Scalar inv2 = 1.0 / sigma[2];
        de_dsigma[2] = mu * (sigma[2] - inv2) + lam * inv2 * log_sigma_prod;
    }
}

template <class Scalar, int dim>
BOW_INLINE void NeoHookean<Scalar, dim>::d2psi_dsigma2(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Matrix<Scalar, dim, dim>& d2e_dsigma2)
{
    Scalar log_sigma_prod = std::log(sigma.prod());
    const double inv2_0 = Scalar(1) / (sigma[0] * sigma[0]);
    d2e_dsigma2(0, 0) = mu * (Scalar(1) + inv2_0) - lam * inv2_0 * (log_sigma_prod - Scalar(1));
    const double inv2_1 = Scalar(1) / (sigma[1] * sigma[1]);
    d2e_dsigma2(1, 1) = mu * (Scalar(1) + inv2_1) - lam * inv2_1 * (log_sigma_prod - Scalar(1));
    d2e_dsigma2(0, 1) = d2e_dsigma2(1, 0) = lam / sigma[0] / sigma[1];
    if constexpr (dim == 3) {
        const double inv2_2 = Scalar(1) / sigma[2] / sigma[2];
        d2e_dsigma2(2, 2) = mu * (Scalar(1) + inv2_2) - lam * inv2_2 * (log_sigma_prod - Scalar(1));
        d2e_dsigma2(1, 2) = d2e_dsigma2(2, 1) = lam / sigma[1] / sigma[2];
        d2e_dsigma2(2, 0) = d2e_dsigma2(0, 2) = lam / sigma[2] / sigma[0];
    }
}

template <class Scalar, int dim>
BOW_INLINE void NeoHookean<Scalar, dim>::B_left_coeff(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Vector<Scalar, 2 * dim - 3>& left_coeff)
{
    const Scalar sigma_prod = sigma.prod();
    if constexpr (dim == 2)
        left_coeff[0] = (mu + (mu - lam * std::log(sigma_prod)) / sigma_prod) / Scalar(2);
    else {
        const Scalar middle = mu - lam * std::log(sigma_prod);
        left_coeff[0] = (mu + middle / (sigma[0] * sigma[1])) / Scalar(2);
        left_coeff[1] = (mu + middle / (sigma[1] * sigma[2])) / Scalar(2);
        left_coeff[2] = (mu + middle / (sigma[2] * sigma[0])) / Scalar(2);
    }
}

#ifdef BOW_STATIC_LIBRARY
#ifdef BOW_COMPILE_FLOAT
#ifdef BOW_COMPILE_2D
template class NeoHookean<float, 2>;
#endif
#ifdef BOW_COMPILE_3D
template class NeoHookean<float, 3>;
#endif
#endif
#ifdef BOW_COMPILE_DOUBLE
#ifdef BOW_COMPILE_2D
template class NeoHookean<double, 2>;
#endif
#ifdef BOW_COMPILE_3D
template class NeoHookean<double, 3>;
#endif
#endif
#endif

} // namespace Bow::ConstitutiveModel
