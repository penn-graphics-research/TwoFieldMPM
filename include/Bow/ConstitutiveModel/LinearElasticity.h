#ifndef LINEAR_ELASTICITY_H
#define LINEAR_ELASTICITY_H

#include <Bow/Macros.h>
#include <Bow/Types.h>
#include "ConstitutiveModel.h"

namespace Bow::ConstitutiveModel {
template <class Scalar, int dim>
class LinearElasticity : public ConstitutiveModelBase<Scalar, dim> {
public:
    BOW_INLINE Scalar psi(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar lam);
    BOW_INLINE void first_piola(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar lam, Matrix<Scalar, dim, dim>& P);
    BOW_INLINE void first_piola_derivative(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar lam, Matrix<Scalar, dim * dim, dim * dim>& dPdF, bool project_pd = true);
};
} // namespace Bow::ConstitutiveModel

#ifndef BOW_STATIC_LIBRARY
#include "LinearElasticity.cpp"
#endif

#endif