#pragma once
#include <Bow/Macros.h>
#include <Bow/Types.h>
#include <Eigen/Sparse>
#include <Bow/Energy/Prototypes.h>
#include <Bow/Math/Utils.h>

namespace Bow::Shell {
template <class T, class StorageIndex = int>
class BendingEnergyOp : public EnergyOp<T, 3, StorageIndex> {
public:
    const static int dim = 3;
    const static int codim = 2;
    const Field<Vector<int, 4>>& m_edge_stencil;
    const Field<T>&m_e, m_h, m_rest_angle;
    const Field<T>& m_bend_stiff;

    // intermediate variables
    Field<int> angle_branch;

    BendingEnergyOp(const Field<Vector<int, 4>>& edge_stencil, const Field<T>& e, const Field<T>& h, const Field<T>& rest_angle, const Field<T>& m_bend_stiff);
    void precompute(const Field<Vector<T, dim>>& x) override;
    T energy(const Field<Vector<T, dim>>& x) override;
    void gradient(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& grad) override;
    template <bool project_pd = true>
    void hessian_impl(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess);
    void hessian(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess, const bool project_pd = true) override
    {
        if (project_pd)
            hessian_impl<true>(x, hess);
        else
            hessian_impl<false>(x, hess);
    }
};
} // namespace Bow::Shell

#include "BendingEnergy.tpp"