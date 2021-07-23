#include "EulerianForceEnergy.h"

namespace Bow::MPM {
template <class T, int dim, class StorageIndex>
GravityForceEnergy<T, dim, StorageIndex>::GravityForceEnergy(MPMGrid<T, dim>& grid, const Vector<T, dim> gravity, const T dx, const T energy_scale)
    : grid(grid), gravity(gravity), dx(dx)
{
    this->energy_scale = energy_scale;
    this->name = "MPM-EulerianForce";
}

template <class T, int dim, class StorageIndex>
T GravityForceEnergy<T, dim, StorageIndex>::energy(const Field<Vector<T, dim>>& x)
{
    T total_energy = 0;
    grid.iterateGridSerial([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
        T m = g.v_and_m(dim);
        Vector<T, dim> x_n = node.template cast<T>() * dx;
        total_energy -= m * gravity.dot(x[g.idx] - x_n);
    });
    return this->energy_scale * total_energy;
}
template <class T, int dim, class StorageIndex>
void GravityForceEnergy<T, dim, StorageIndex>::gradient(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& grad)
{
    grad.resize(x.size());
    grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
        T m = g.v_and_m(dim);
        grad[g.idx] = -this->energy_scale * m * gravity;
    });
}
template <class T, int dim, class StorageIndex>
void GravityForceEnergy<T, dim, StorageIndex>::hessian(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess, bool project_pd)
{
    hess.derived().resize(x.size() * dim, x.size() * dim);
    hess.derived().setZero();
}
} // namespace Bow::MPM