#include "FibrinPoroelasticity.h"
#include <Bow/Math/SVD.h>
#include <Bow/Math/PolarDecomposition.h>
#include "ConstitutiveModel.h"
#include <Bow/Math/Utils.h>
#include <Bow/Math/MathTools.h>

namespace Bow::ConstitutiveModel {
template <class Scalar, int dim>
BOW_INLINE Scalar FibrinPoroelasticity<Scalar, dim>::psi_poro(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar c1, const Scalar c2, const Scalar phi_s0, const Scalar pi_0, const Scalar beta_1)
{
    Scalar J = F.determinant();
    T I1 = (F.transpose() * F).trace();
    T psiNet = ((phi_s0 * c1) / c2) * (exp(c2 * (I1 - dim)) - 1);
    T psiMix = (pi_0 / (beta_1 - 1)) * ((pow(1 - phi_s0, beta_1)) / (pow(J - phi_s0, beta_1 - 1)));
    T psi0 = (pi_0 * (1 - phi_s0)) / (beta_1 - 1);
    T muC = (mu * (J - phi_s0)); //C = det(F) - phi_s0
    return psiNet + psiMix - psi0 - muC;
}
template <class Scalar, int dim>
BOW_INLINE void FibrinPoroelasticity<Scalar, dim>::first_piola_poro(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar c1, const Scalar c2, const Scalar phi_s0, const Scalar pi_0, const Scalar beta_1, Matrix<Scalar, dim, dim>& P)
{
    Scalar J = F.determinant();
    T I1 = (F.transpose() * F).trace();
    Eigen::Matrix<Scalar, dim, dim> JFinvT;
    Math::cofactor(F, JFinvT);
    Eigen::Matrix<Scalar, dim, dim> Pnet = phi_s0 * 2.0 * c1 * exp(c2 * (I1 - dim)) * F;
    Eigen::Matrix<Scalar, dim, dim> Pmix = ((-pi_0 * (pow(1-phi_s0, beta_1) / pow(J - phi_s0, beta_1))) - mu) * JFinvT;
    P = Pnet + Pmix;
}

#ifdef BOW_STATIC_LIBRARY
#ifdef BOW_COMPILE_FLOAT
#ifdef BOW_COMPILE_2D
template class FibrinPoroelasticity<float, 2>;
#endif
#ifdef BOW_COMPILE_3D
template class FibrinPoroelasticity<float, 3>;
#endif
#endif
#ifdef BOW_COMPILE_DOUBLE
#ifdef BOW_COMPILE_2D
template class FibrinPoroelasticity<double, 2>;
#endif
#ifdef BOW_COMPILE_3D
template class FibrinPoroelasticity<double, 3>;
#endif
#endif
#endif

} // namespace Bow::ConstitutiveModel