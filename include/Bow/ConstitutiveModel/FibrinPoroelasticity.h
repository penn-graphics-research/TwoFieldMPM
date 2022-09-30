#ifndef FIBRIN_POROELASTICITY_H
#define FIBRIN_POROELASTICITY_H

#include <Bow/Macros.h>
#include <Bow/Types.h>
#include "ConstitutiveModel.h"

namespace Bow::ConstitutiveModel {
template <class Scalar, int dim>
class FibrinPoroelasticity : public ConstitutiveModelBase<Scalar, dim> {
public:
    BOW_INLINE Scalar psi_poro(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar c1, const Scalar c2, const Scalar phi_s0, const Scalar pi_0, const Scalar beta_1) override;
    BOW_INLINE void first_piola_poro(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar c1, const Scalar c2, const Scalar phi_s0, const Scalar pi_0, const Scalar beta_1, Matrix<Scalar, dim, dim>& P) override;
};
} // namespace Bow::ConstitutiveModel

#ifndef BOW_STATIC_LIBRARY
#include "FibrinPoroelasticity.cpp"
#endif

#endif