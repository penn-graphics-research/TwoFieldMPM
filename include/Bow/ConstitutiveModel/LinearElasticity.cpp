#include "LinearElasticity.h"
#include <Bow/Math/SVD.h>
#include <Bow/Math/PolarDecomposition.h>
#include "ConstitutiveModel.h"
#include <Bow/Math/Utils.h>
#include <Bow/Utils/Logging.h>

namespace Bow::ConstitutiveModel {
template <class Scalar, int dim>
BOW_INLINE Scalar LinearElasticity<Scalar, dim>::psi(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar lam)
{
    Bow::Matrix<Scalar, dim, dim> R = Bow::Matrix<Scalar, dim, dim>::Identity(); // can be updated per time step for lagged corotational
    Bow::Matrix<Scalar, dim, dim> RtF = R.transpose() * F;
    Bow::Matrix<Scalar, dim, dim> smallStrain = 0.5 * (RtF + RtF.transpose()) - Bow::Matrix<Scalar, dim, dim>::Identity();
    Scalar tr_smallStrain = smallStrain.diagonal().sum();

    return mu * smallStrain.squaredNorm() + lam * 0.5 * tr_smallStrain * tr_smallStrain;
}

template <class Scalar, int dim>
BOW_INLINE void LinearElasticity<Scalar, dim>::first_piola(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar lam, Matrix<Scalar, dim, dim>& P)
{
    Bow::Matrix<Scalar, dim, dim> R = Bow::Matrix<Scalar, dim, dim>::Identity(); // can be updated per time step for lagged corotational
    Bow::Matrix<Scalar, dim, dim> RtF = R.transpose() * F;
    Bow::Matrix<Scalar, dim, dim> smallStrain = 0.5 * (RtF + RtF.transpose()) - Bow::Matrix<Scalar, dim, dim>::Identity();
    Scalar tr_smallStrain = smallStrain.diagonal().sum();

    P.noalias() = 2 * mu * R * smallStrain + lam * tr_smallStrain * R;
}

template <class Scalar, int dim>
BOW_INLINE void LinearElasticity<Scalar, dim>::first_piola_derivative(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar lam, Matrix<Scalar, dim * dim, dim * dim>& dPdF, bool project_pd)
{
    Bow::Matrix<Scalar, dim, dim> R = Bow::Matrix<Scalar, dim, dim>::Identity(); // can be updated per time step for lagged corotational
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
            int row_idx = i + j * dim;
            for (int a = 0; a < dim; ++a)
                for (int b = 0; b < dim; ++b) {
                    int col_idx = a + b * dim;
                    int ia = (i == a);
                    int jb = (j == b);
                    dPdF(row_idx, col_idx) = mu * (ia * jb + R(i, b) * R(a, j)) + lam * R(i, j) * R(a, b);
                }
        }
}

#ifdef BOW_STATIC_LIBRARY
#ifdef BOW_COMPILE_FLOAT
#ifdef BOW_COMPILE_2D
template class LinearElasticity<float, 2>;
#endif
#ifdef BOW_COMPILE_3D
template class LinearElasticity<float, 3>;
#endif
#endif
#ifdef BOW_COMPILE_DOUBLE
#ifdef BOW_COMPILE_2D
template class LinearElasticity<double, 2>;
#endif
#ifdef BOW_COMPILE_3D
template class LinearElasticity<double, 3>;
#endif
#endif
#endif

} // namespace Bow::ConstitutiveModel
