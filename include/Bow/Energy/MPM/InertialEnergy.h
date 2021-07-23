#ifndef MPM_INERTIAL_ENERGY_H
#define MPM_INERTIAL_ENERGY_H

#include <Bow/Macros.h>
#include <Bow/Types.h>
#include <tbb/tbb.h>
#include <Eigen/Sparse>
#include <Bow/Geometry/Hybrid/MPMGrid.h>
#include "../Prototypes.h"

namespace Bow::MPM {
template <class T, int dim, class StorageIndex = int>
class InertialEnergy : public EnergyOp<T, dim, StorageIndex> {
public:
    MPMGrid<T, dim>& grid;
    const Field<Vector<T, dim>>& m_x_tilde;
    InertialEnergy(MPMGrid<T, dim>& grid, const Field<Vector<T, dim>>& x_tilde, const T energy_scale = 1.0);
    T energy(const Field<Vector<T, dim>>& x) override;
    void gradient(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& grad) override;
    void hessian(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess, bool project_pd = true) override;
};
} // namespace Bow::MPM

#include "InertialEnergy.tpp"

#endif
