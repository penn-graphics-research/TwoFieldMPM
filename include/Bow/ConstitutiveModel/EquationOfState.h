#pragma once

#include <Bow/Macros.h>
#include <Bow/Types.h>

namespace Bow::ConstitutiveModel {
template <class Scalar>
class EquationOfState {
public:
    Scalar psi(const Scalar& J, const Scalar bulk, const Scalar gamma);
    Scalar first_piola(const Scalar& J, const Scalar bulk, const Scalar gamma);
    Scalar first_piola_derivative(const Scalar& J, const Scalar bulk, const Scalar gamma);
};
} // namespace Bow::ConstitutiveModel

#ifndef BOW_STATIC_LIBRARY
#include "EquationOfState.cpp"
#endif
