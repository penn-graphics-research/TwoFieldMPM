#ifndef MEMBRANE_ENERGY_H
#define MEMBRANE_ENERGY_H
#include <Bow/Macros.h>
#include <Bow/Types.h>
#include <Eigen/Sparse>
#include <Bow/Energy/Prototypes.h>
#include <Bow/Math/Utils.h>

namespace Bow::Shell {
template <class T, class StorageIndex = int>
class MembraneEnergyOp : public EnergyOp<T, 3, StorageIndex> {
public:
    const static int dim = 3;
    const static int codim = 2;
    const Field<Vector<int, 3>>& m_elem;
    const Field<T>&m_vol, m_mu, m_lam;
    const Field<Matrix<T, codim, codim>>& m_IB;
    MembraneEnergyOp(const Field<Vector<int, 3>>& elem, const Field<T>& vol, const Field<T>& mu, const Field<T>& lam, const Field<Matrix<T, codim, codim>>& IB);
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

#include "MembraneEnergy.tpp"
#endif