#ifndef MPM_NODAL_FORCE_ENERGY
#define MPM_NODAL_FORCE_ENERGY
#include <Bow/Macros.h>
#include <Bow/Types.h>
#include <tbb/tbb.h>
#include <Eigen/Sparse>
#include <Bow/Geometry/Hybrid/MPMGrid.h>
#include "../Prototypes.h"

namespace Bow::MPM {

template <class T, int dim, class StorageIndex = int>
class GravityForceEnergy : public EnergyOp<T, dim, StorageIndex> {
public:
    MPMGrid<T, dim>& grid;
    Vector<T, dim> gravity;
    T dx;
    GravityForceEnergy(MPMGrid<T, dim>& grid, const Vector<T, dim> gravity, const T dx, const T energy_scale = 1.0);
    T energy(const Field<Vector<T, dim>>& x) override;
    void gradient(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& grad) override;
    void hessian(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess, bool project_pd = true) override;
    void internal_force(const Field<Vector<T, dim>>& xn, const Field<Vector<T, dim>>& vn, Field<Vector<T, dim>>& force)
    {
        T energy_scale_bk = this->energy_scale;
        this->energy_scale = 1.0;
        gradient(xn, force);
        this->energy_scale = energy_scale_bk;
    }
    void callback(const Field<Vector<T, dim>>& xn) override{};
};
} // namespace Bow::MPM

#include "EulerianForceEnergy.tpp"
#endif