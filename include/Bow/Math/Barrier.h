#ifndef BARRIER_H
#define BARRIER_H

namespace Bow::Math {
template <class T>
inline T barrier(const T d, const T dHat2, const T kappa)
{
    T e = 0.0;
    if (d < dHat2) {
        T t2 = d - dHat2;
        e = -kappa * (t2 / dHat2) * (t2 / dHat2) * std::log(d / dHat2);
    }
    return e;
}

template <class T>
inline T barrier_gradient(const T d, const T dHat2, const T kappa)
{
    T grad = 0.0;
    if (d < dHat2) {
        T t2 = d - dHat2;
        grad = kappa * ((t2 / dHat2) * std::log(d / dHat2) * -2.0 / dHat2 - ((t2 / dHat2) * (t2 / dHat2)) / d);
    }
    return grad;
}

template <class T>
inline T barrier_hessian(const T d, const T dHat2, const T kappa)
{
    T hess = 0.0;
    if (d < dHat2) {
        T t2 = d - dHat2;
        hess = kappa * ((std::log(d / dHat2) * -2.0 - t2 * 4.0 / d) / (dHat2 * dHat2) + 1.0 / (d * d) * (t2 / dHat2) * (t2 / dHat2));
    }
    return hess;
}
} // namespace Bow::Math

#endif