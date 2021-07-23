#include "StvkWithHencky.h"
#include <Bow/Math/SVD.h>
#include <Bow/Math/Utils.h>
#include <Bow/Math/MathTools.h>
#include <Bow/Utils/Logging.h>

namespace Bow::ConstitutiveModel {

template <class Scalar, int dim>
BOW_INLINE Scalar StvkWithHencky<Scalar, dim>::psi_sigma(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam)
{
    Vector<Scalar, dim> log_sigma_squared = sigma.array().abs().log().square();
    Scalar trace_log_sigma = sigma.array().abs().log().sum();
    return mu * log_sigma_squared.sum() + (Scalar).5 * lam * trace_log_sigma * trace_log_sigma;
}

template <class Scalar, int dim>
BOW_INLINE void StvkWithHencky<Scalar, dim>::dpsi_dsigma(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Vector<Scalar, dim>& de_dsigma)
{
    Vector<Scalar, dim> log_sigma = sigma.array().abs().log();
    Scalar sum_log_sigma = log_sigma.sum();
    const Scalar inv0 = 1.0 / sigma[0];
    de_dsigma[0] = (2 * mu * log_sigma(0) + lam * sum_log_sigma) * inv0;
    const Scalar inv1 = 1.0 / sigma[1];
    de_dsigma[1] = (2 * mu * log_sigma(1) + lam * sum_log_sigma) * inv1;
    if constexpr (dim == 3) {
        const Scalar inv2 = 1.0 / sigma[2];
        de_dsigma[2] = (2 * mu * log_sigma(2) + lam * sum_log_sigma) * inv2;
    }
}

template <class Scalar, int dim>
BOW_INLINE void StvkWithHencky<Scalar, dim>::d2psi_dsigma2(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Matrix<Scalar, dim, dim>& d2e_dsigma2)
{
    Vector<Scalar, dim> log_sigma = sigma.array().abs().log();
    Scalar sum_log_sigma = log_sigma.sum();
    const double inv2_0 = Scalar(1) / (sigma[0] * sigma[0]);
    d2e_dsigma2(0, 0) = (2 * mu * (1 - log_sigma(0)) + lam * (1 - sum_log_sigma)) * inv2_0;
    const double inv2_1 = Scalar(1) / (sigma[1] * sigma[1]);
    d2e_dsigma2(1, 1) = (2 * mu * (1 - log_sigma(1)) + lam * (1 - sum_log_sigma)) * inv2_1;
    d2e_dsigma2(0, 1) = d2e_dsigma2(1, 0) = lam / sigma[0] / sigma[1];
    if constexpr (dim == 3) {
        const double inv2_2 = Scalar(1) / sigma[2] / sigma[2];
        d2e_dsigma2(2, 2) = (2 * mu * (1 - log_sigma(2)) + lam * (1 - sum_log_sigma)) * inv2_2;
        d2e_dsigma2(1, 2) = d2e_dsigma2(2, 1) = lam / sigma[1] / sigma[2];
        d2e_dsigma2(2, 0) = d2e_dsigma2(0, 2) = lam / sigma[2] / sigma[0];
    }
}

template <class Scalar, int dim>
BOW_INLINE void StvkWithHencky<Scalar, dim>::B_left_coeff(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Vector<Scalar, 2 * dim - 3>& left_coeff)
{
    Vector<Scalar, dim> log_sigma = sigma.array().abs().log();
    Scalar eps = 1e-6;
    if constexpr (dim == 2) {
        Scalar q = std::max(sigma(0) / sigma(1) - 1, -1 + eps);
        Scalar h = (std::abs(q) < eps) ? 1 : (std::log1p(q) / q);
        Scalar t = h / sigma(1);
        Scalar z = log_sigma(1) - t * sigma(1);
        left_coeff[0] = -(lam * (log_sigma(0) + log_sigma(1)) + 2 * mu * z) / sigma.prod() / Scalar(2);
    }
    else {
        Scalar sum_log_sigma = log_sigma.sum();
        left_coeff[0] = -(lam * sum_log_sigma + 2 * mu * MATH_TOOLS::diff_interlock_log_over_diff(sigma(0), std::abs(sigma(1)), log_sigma(1), eps)) / (sigma[0] * sigma[1]) / Scalar(2);
        left_coeff[1] = -(lam * sum_log_sigma + 2 * mu * MATH_TOOLS::diff_interlock_log_over_diff(sigma(1), std::abs(sigma(2)), log_sigma(2), eps)) / (sigma[1] * sigma[2]) / Scalar(2);
        left_coeff[2] = -(lam * sum_log_sigma + 2 * mu * MATH_TOOLS::diff_interlock_log_over_diff(sigma(0), std::abs(sigma(2)), log_sigma(2), eps)) / (sigma[0] * sigma[2]) / Scalar(2);
    }
}

#ifdef BOW_STATIC_LIBRARY
#ifdef BOW_COMPILE_FLOAT
#ifdef BOW_COMPILE_2D
template class StvkWithHencky<float, 2>;
#endif
#ifdef BOW_COMPILE_3D
template class StvkWithHencky<float, 3>;
#endif
#endif
#ifdef BOW_COMPILE_DOUBLE
#ifdef BOW_COMPILE_2D
template class StvkWithHencky<double, 2>;
#endif
#ifdef BOW_COMPILE_3D
template class StvkWithHencky<double, 3>;
#endif
#endif
#endif

} // namespace Bow::ConstitutiveModel
