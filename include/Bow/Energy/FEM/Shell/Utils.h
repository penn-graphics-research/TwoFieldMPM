#ifndef SHELL_UTILS_H
#define SHELL_UTILS_H
namespace Bow::Shell {

template <class T>
void first_fundamental_form(const Vector<T, 3>& x0, const Vector<T, 3>& x1, const Vector<T, 3>& x2, Matrix<T, 2, 2>& I)
{
    Matrix<T, 3, 2> basis;
    basis.col(0) = x1 - x0;
    basis.col(1) = x2 - x0;
    I = basis.transpose() * basis;
}

template <class T>
void deformation_gradient(const Vector<T, 3>& x0, const Vector<T, 3>& x1, const Vector<T, 3>& x2, const Matrix<T, 2, 2>& IB, Matrix<T, 2, 2>& F)
{
    Matrix<T, 2, 2> I;
    first_fundamental_form(x0, x1, x2, I);
    F = I * IB;
}
} // namespace Bow::Shell
#endif