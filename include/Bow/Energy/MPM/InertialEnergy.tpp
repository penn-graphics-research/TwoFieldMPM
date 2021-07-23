#include "InertialEnergy.h"
#include <Bow/Utils/Timer.h>
#include <oneapi/tbb.h>

namespace Bow::MPM {
template <class T, int dim, class StorageIndex>
InertialEnergy<T, dim, StorageIndex>::InertialEnergy(MPMGrid<T, dim>& grid, const Field<Vector<T, dim>>& x_tilde, const T energy_scale)
    : grid(grid), m_x_tilde(x_tilde)
{
    this->energy_scale = energy_scale;
    this->name = "MPM-Inertia";
}

template <class T, int dim, class StorageIndex>
T InertialEnergy<T, dim, StorageIndex>::energy(const Field<Vector<T, dim>>& x)
{
    T total_energy = 0;
    grid.iterateGridSerial([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
        total_energy += 0.5 * g.v_and_m(dim) * (x[g.idx] - m_x_tilde[g.idx]).squaredNorm();
    });
    return this->energy_scale * total_energy;
}

template <class T, int dim, class StorageIndex>
void InertialEnergy<T, dim, StorageIndex>::gradient(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& grad)
{
    grad.resize(x.size(), Vector<T, dim>::Zero());
    grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
        grad[g.idx] = this->energy_scale * g.v_and_m(dim) * (x[g.idx] - m_x_tilde[g.idx]);
    });
}

template <class T, int dim, class StorageIndex>
void InertialEnergy<T, dim, StorageIndex>::hessian(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess, bool project_pd)
{
    int nrows = x.size() * dim;
    hess.setZero();
    hess.derived().resize(nrows, nrows);
    hess.derived().reserve(nrows);
    grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
        for (int d = 0; d < dim; ++d) {
            hess.derived().outerIndexPtr()[dim * g.idx + d + 1] = dim * g.idx + d + 1;
            hess.derived().innerIndexPtr()[dim * g.idx + d] = dim * g.idx + d;
            hess.derived().valuePtr()[dim * g.idx + d] = this->energy_scale * g.v_and_m(dim);
        }
    });
    hess.derived().finalize();
}

} // namespace Bow::MPM