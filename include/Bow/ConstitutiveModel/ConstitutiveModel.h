#ifndef CONSTITUTIVE_MODEL_H
#define CONSTITUTIVE_MODEL_H

#include <Bow/Macros.h>
#include <Bow/Types.h>
#include <Bow/Utils/Logging.h>

namespace Bow::ConstitutiveModel {
enum Type {
    FIXED_COROTATED,
    NEO_HOOKEAN,
    LINEAR_ELASTICITY,
    NEO_HOOKEAN_MEMBRANE, // not a subclass of ConstitutiveModelBase
    DISCRETE_HINGE_BENDING, // not a subclass of ConstitutiveModelBase
    NONE
};

template <class T>
BOW_INLINE std::pair<T, T> lame_paramters(T E, T nu, bool plane_stress = false);

template <class T>
BOW_INLINE std::pair<T, T> E_nu(T mu, T lam, bool plane_stress = false);

// TODO: not redo SVD every time
template <class Scalar, int dim>
class ConstitutiveModelBase {
public: // have default implementations, override if needed
    virtual BOW_INLINE Scalar psi(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar lam);
    virtual BOW_INLINE void first_piola(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar lam, Matrix<Scalar, dim, dim>& P);
    virtual BOW_INLINE void first_piola_derivative(const Matrix<Scalar, dim, dim>& F, const Scalar mu, const Scalar lam, Matrix<Scalar, dim * dim, dim * dim>& dPdF, bool project_pd = true);
    virtual BOW_INLINE void first_piola_differential(const Matrix<Scalar, dim, dim>& F, const Matrix<Scalar, dim, dim>& dF, const Scalar mu, const Scalar lam, Matrix<Scalar, dim, dim>& dP, bool project_pd = true) { BOW_NOT_IMPLEMENTED }

protected: // specific to different models
    virtual BOW_INLINE Scalar psi_sigma(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam) { BOW_NOT_IMPLEMENTED return 0.0; }
    virtual BOW_INLINE void dpsi_dsigma(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Vector<Scalar, dim>& de_dsigma){ BOW_NOT_IMPLEMENTED };
    virtual BOW_INLINE void d2psi_dsigma2(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Matrix<Scalar, dim, dim>& d2e_dsigma2){ BOW_NOT_IMPLEMENTED };
    /** Eq 77 in https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf
      * In 2D, left_coeff = 0.5 * [psi01]
      * In 3D, left_coeff = 0.5 * [psi01, psi12, psi02] */
    virtual BOW_INLINE void B_left_coeff(const Vector<Scalar, dim>& sigma, const Scalar mu, const Scalar lam, Vector<Scalar, 2 * dim - 3>& left_coeff){ BOW_NOT_IMPLEMENTED };
};

} // namespace Bow::ConstitutiveModel

#ifndef BOW_STATIC_LIBRARY
#include "ConstitutiveModel.cpp"
#endif

#endif