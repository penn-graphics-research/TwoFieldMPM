#ifndef FIXED_COROTATED_H
#define FIXED_COROTATED_H

#include <Bow/Macros.h>
#include <Bow/Types.h>
#include "ConstitutiveModel.h"

namespace Bow::ConstitutiveModel {
template <class Scalar, int dim>
class FixedCorotated : public ConstitutiveModelBase<Scalar, dim> {
public:
    BOW_INLINE Scalar psi(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar lam) override;
    BOW_INLINE void first_piola(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar lam, Matrix<Scalar, dim, dim>& P) override;

protected:
    BOW_INLINE void dpsi_dsigma(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Vector<Scalar, dim>& de_dsigma);
    BOW_INLINE void d2psi_dsigma2(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Matrix<Scalar, dim, dim>& d2e_dsigma2);
    BOW_INLINE void B_left_coeff(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Vector<Scalar, 2 * dim - 3>& left_coeff);
};
} // namespace Bow::ConstitutiveModel

#ifndef BOW_STATIC_LIBRARY
#include "FixedCorotated.cpp"
#endif

#endif