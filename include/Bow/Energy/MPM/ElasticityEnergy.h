#ifndef MPM_ELASTICITY_ENERGY_H
#define MPM_ELASTICITY_ENERGY_H

#include <Bow/Macros.h>
#include <Bow/Types.h>
#include <Eigen/Sparse>
#include "../Prototypes.h"
#include <Bow/Geometry/Hybrid/MPMGrid.h>
#include "ElasticityOp.h"
#include <Bow/Math/Utils.h>

namespace Bow::MPM {
template <class T, int dim, class StorageIndex = int, int interpolation_degree = 2>
class ElasticityEnergy : public EnergyOp<T, dim, StorageIndex> {
public:
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;

    MPMGrid<T, dim>& grid;
    const Field<Vector<T, dim>>& m_X;
    std::vector<std::shared_ptr<ElasticityOp<T, dim>>>& elasticity_models;
    T dx;

    static constexpr int kernel_span = 2 * interpolation_degree + 1;
    static constexpr int kernel_size = Math::constexpr_ipow(kernel_span, dim);

    ElasticityEnergy(MPMGrid<T, dim>& grid, const Field<Vector<T, dim>>& m_X, std::vector<std::shared_ptr<ElasticityOp<T, dim>>>& elasticity_models, const T dx, T energy_scale = 1.0);
    T energy(const Field<Vector<T, dim>>& x) override;
    void precompute(const Field<Vector<T, dim>>& x) override;
    void gradient(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& grad) override;
    void hessian(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess, bool project_pd = true) override;
    T stepsize_upperbound(const Field<Vector<T, dim>>& x, const Field<Vector<T, dim>>& dx) override;

    void internal_force(const Field<Vector<T, dim>>& xn, const Field<Vector<T, dim>>& vn, Field<Vector<T, dim>>& force)
    {
        T energy_scale_bk = this->energy_scale;
        this->energy_scale = 1.0;
        precompute(xn);
        gradient(xn, force);
        this->energy_scale = energy_scale_bk;
    }

    static inline int kernelOffset(const Vector<int, dim>& dnode)
    {
        if constexpr (dim == 2) {
            return (dnode(0) + interpolation_degree) * kernel_span + (dnode(1) + interpolation_degree);
        }
        else {
            return (dnode(0) + interpolation_degree) * kernel_span * kernel_span + (dnode(1) + interpolation_degree) * kernel_span + (dnode(2) + interpolation_degree);
        }
    }
};
} // namespace Bow::MPM

#include "ElasticityEnergy.tpp"

#endif
