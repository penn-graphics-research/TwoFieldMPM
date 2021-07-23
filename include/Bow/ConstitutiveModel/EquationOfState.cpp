#include "EquationOfState.h"
#include <Bow/Utils/Logging.h>

namespace Bow::ConstitutiveModel {

template <class Scalar>
BOW_INLINE Scalar EquationOfState<Scalar>::psi(const Scalar& J, const Scalar bulk, const Scalar gamma)
{
    Scalar J2 = J * J;
    Scalar J6 = J2 * J2 * J2;
    BOW_ASSERT(gamma == 7);
    return -bulk * (1. / J6 / (-6.) - J);
}

template <class Scalar>
BOW_INLINE Scalar EquationOfState<Scalar>::first_piola(const Scalar& J, const Scalar bulk, const Scalar gamma)
{
    Scalar J2 = J * J;
    Scalar J4 = J2 * J2;
    Scalar J7 = J4 * J2 * J;
    BOW_ASSERT(gamma == 7);
    return -bulk * (1. / J7 - 1.);
}

template <class Scalar>
BOW_INLINE Scalar EquationOfState<Scalar>::first_piola_derivative(const Scalar& J, const Scalar bulk, const Scalar gamma)
{
    Scalar J2 = J * J;
    Scalar J4 = J2 * J2;
    Scalar J8 = J4 * J4;
    BOW_ASSERT(gamma == 7);
    return bulk * (1. / J8 * 7.);
}

#ifdef BOW_STATIC_LIBRARY
#ifdef BOW_COMPILE_FLOAT
template class EquationOfState<float>;
#endif
#ifdef BOW_COMPILE_DOUBLE
template class EquationOfState<double>;
#endif
#endif

} // namespace Bow::ConstitutiveModel