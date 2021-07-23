#ifndef FEM_ELASTICITY_ENERGY_H
#define FEM_ELASTICITY_ENERGY_H

#include <Bow/Macros.h>
#include <Bow/Types.h>
#include <Eigen/Sparse>
#include "../Prototypes.h"
#include <Bow/ConstitutiveModel/ConstitutiveModel.h>
#include <Bow/ConstitutiveModel/FixedCorotated.h>
#include <Bow/ConstitutiveModel/NeoHookean.h>
#include <Bow/ConstitutiveModel/LinearElasticity.h>
#include <Bow/Math/Utils.h>

namespace Bow::FEM {
template <class T, int dim, typename Model, bool inversion_free, class StorageIndex = int>
class ElasticityEnergyOp : public EnergyOp<T, dim, StorageIndex>, public Model {
public:
    using Model::first_piola;
    using Model::first_piola_derivative;
    using Model::psi;
    const Field<Vector<int, dim + 1>>& m_elem;
    const Field<T>&m_vol, m_mu, m_lam;
    const Field<Matrix<T, dim, dim>>& m_IB;
    std::vector<std::pair<int, int>>& m_offsets; // max excluded intervals
    ElasticityEnergyOp(const Field<Vector<int, dim + 1>>& elem, const Field<T>& vol, const Field<T>& mu, const Field<T>& lam, const Field<Matrix<T, dim, dim>>& IB, std::vector<std::pair<int, int>>& offsets, T energy_scale = 1.0);
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
    T stepsize_upperbound(const Field<Vector<T, dim>>& x, const Field<Vector<T, dim>>& dx) override;
    virtual void compute_cauchy(const Field<Vector<T, dim>>& x, Field<Matrix<T, dim, dim>>& cauchy);
    virtual void compute_von_mises(const Field<Vector<T, dim>>& x, Field<T>& von_mises);
};

template <class T, int dim, class StorageIndex = int>
using FixedCorotatedEnergyOp = ElasticityEnergyOp<T, dim, ConstitutiveModel::FixedCorotated<T, dim>, false, StorageIndex>;
template <class T, int dim, class StorageIndex = int>
using NeoHookeanEnergyOp = ElasticityEnergyOp<T, dim, ConstitutiveModel::NeoHookean<T, dim>, true, StorageIndex>;
template <class T, int dim, class StorageIndex = int>
using LinearElasticityEnergyOp = ElasticityEnergyOp<T, dim, ConstitutiveModel::LinearElasticity<T, dim>, false, StorageIndex>;

} // namespace Bow::FEM

#include "ElasticityEnergy.tpp"

#endif
