#ifndef TWOFIELD_HODGEPODGE_ENERGY_H
#define TWOFIELD_HODGEPODGE_ENERGY_H

#include <Bow/Macros.h>
#include <Bow/Types.h>
#include <tbb/tbb.h>
#include <Eigen/Sparse>
#include <Bow/Energy/Prototypes.h>
#include <Bow/Energy/MPM/ElasticityOp.h>

namespace Bow::DFGMPM {

template <class T, int dim, class StorageIndex = int, int interpolation_dgree = 2>
class TwoFieldHodgepodgeEnergy : public EnergyOp<T, dim, StorageIndex> {
public:
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;

    DFGMPMGrid<T, dim>& grid;
    TV gravity;
    T dx;
    T dt;

    // particles
    Field<Vector<T, dim>>& m_X;
    std::vector<std::shared_ptr<Bow::MPM::ElasticityOp<T, dim>>>& elasticity_models;

    Bow::Field<std::vector<int>>& particleAF;
    Bow::Field<std::vector<int>>& p_cached_idx;

    bool useDFG;

    TSMethod tsMethod = BE;
    T tsParam[2][3] = {
        { 1, 0.5, 1 },
        { 0.5, 0.25, 0.5 }
    };

    static const int kernel_span = 2 * interpolation_dgree + 1;
    static const int kernel_size = std::pow(kernel_span, dim);

    //Barrier Parameters -- TAG=Barrier
    bool useImplicitContact = false;
    T chat = 0.001;
    bool constraintsViolated = false;
    T delta;
    T factor = 1.0;
    T energyFactor = 1.0;
    T rhsFactor = 1.0;
    T hessianFactor = 1.0;
    T zeroingFactor = 1.0;
    T projectionThreshold = 1e-10;

    TwoFieldHodgepodgeEnergy(DFGMPMGrid<T, dim>& grid, T gravMag, T dx, T dt, Field<Vector<T, dim>>& m_X,
        std::vector<std::shared_ptr<Bow::MPM::ElasticityOp<T, dim>>>& elasticity_models, Bow::Field<std::vector<int>>& particleAF, Bow::Field<std::vector<int>>& p_cached_idx, bool useDFG, bool useImplicitContact)
        : grid(grid), dx(dx), dt(dt), m_X(m_X), elasticity_models(elasticity_models), particleAF(particleAF), p_cached_idx(p_cached_idx), useDFG(useDFG), useImplicitContact(useImplicitContact) 
        {
            gravity = TV::Unit(1) * gravMag;
        }

    static inline int kernelOffset(const Vector<int, dim>& dnode)
    {
        if constexpr (dim == 2) {
            return (dnode(0) + interpolation_dgree) * kernel_span + (dnode(1) + interpolation_dgree);
        }
        else {
            return (dnode(0) + interpolation_dgree) * kernel_span * kernel_span + (dnode(1) + interpolation_dgree) * kernel_span + (dnode(2) + interpolation_dgree);
        }
    }

    //Barrier Function Evaluations -- TAG=BARRIER
    static inline T computeB(T ci, T chat){
        T c = ci / chat;
        return (ci < chat) ? -((c - 1.0)*(c - 1.0)) * std::log(c) : 0.0;
    }

    static inline T computeBPrime(T ci, T chat){
        T c = ci / chat;
        return (ci < chat) ? -((2.0 * (c - 1.0) * std::log(c) / chat) + (((c - 1.0)*(c - 1.0)) / ci)) : 0.0;
    }

    static inline T computeBDoublePrime(T ci, T chat){
        T c = ci / chat;
        return (ci < chat) ? -(((2.0 * std::log(c) + 3.0) / (chat * chat)) - (2.0 / (ci * chat)) - (1 / (ci * ci))) : 0.0;
    }

    void precompute(const Field<Vector<T, dim>>& x) override;
    T energy(const Field<Vector<T, dim>>& x) override;
    void gradient(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& grad) override;
    void multiply(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& Ax, bool project_pd) override;
    void precondition(Field<Vector<T, dim>>& diagonal) override;
    void hessian(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess, bool project_pd) override;
    void internal_force(const Field<Vector<T, dim>>& xn, const Field<Vector<T, dim>>& vn, Field<Vector<T, dim>>& force) override;
    T stepsize_upperbound(const Field<Vector<T, dim>>& x, const Field<Vector<T, dim>>& dx) override;
};

} // namespace Bow::DFGMPM

#include "TwoFieldHodgepodgeEnergy.tpp"

#endif
