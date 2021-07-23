#ifndef STVK_HENCKY_ISOTROPIC_H
#define STVK_HENCKY_ISOTROPIC_H
#include <Bow/Macros.h>
#include <Bow/Types.h>
#include "ConstitutiveModel.h"

namespace Bow::ConstitutiveModel {
template <class Scalar, int dim>
class StvkWithHencky : public ConstitutiveModelBase<Scalar, dim> {
protected:
    BOW_INLINE Scalar psi_sigma(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam) override;
    BOW_INLINE void dpsi_dsigma(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Vector<Scalar, dim>& de_dsigma) override;
    BOW_INLINE void d2psi_dsigma2(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Matrix<Scalar, dim, dim>& d2e_dsigma2) override;
    BOW_INLINE void B_left_coeff(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Vector<Scalar, 2 * dim - 3>& left_coeff) override;
};
} // namespace Bow::ConstitutiveModel

#ifndef BOW_STATIC_LIBRARY
#include "StvkWithHencky.cpp"
#endif

#endif
