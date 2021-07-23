#ifndef FEM_TIME_INTEGRATOR_H
#define FEM_TIME_INTEGRATOR_H
#include <Bow/Macros.h>
#include <Bow/Types.h>
#include <tbb/tbb.h>
#include <Eigen/Sparse>
#include <Bow/Utils/Timer.h>
#include <Bow/Optimization/AugmentedLagrangian.h>
#include <Bow/Energy/FEM/InertialEnergy.h>
#include <Bow/Energy/FEM/ElasticityEnergy.h>
#include <Bow/Energy/FEM/ExternalForceEnergy.h>

namespace Bow {
namespace FEM {

template <class T, int dim, class _StorageIndex = int, class Optimizer = Optimization::AugmentedLagrangianNewton<T, dim, _StorageIndex>>
class TimeIntegratorUpdateOp : public Optimizer {
public:
    using StorageIndex = _StorageIndex;
    using Base = Optimizer;
    using Vec = Bow::Vector<T, Eigen::Dynamic>;
    using Mat = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;

    bool project_dirichlet = true;

    const Field<Matrix<T, dim, dim>>& BC_basis;
    const Field<int>& BC_order;
    const Field<Vector<T, dim>>& BC_target;
    const Field<uint8_t>& BC_fixed;

    Field<Vector<T, dim>>& m_x;
    Field<Vector<T, dim>>& m_v;
    Field<Vector<T, dim>>& m_a;
    Field<Vector<T, dim>>& m_x1;
    Field<Vector<T, dim>>& m_v1;
    Field<Vector<T, dim>>& m_x_tilde;
    Field<T>& m_mass;

    std::function<void(void)> lagging_callback = nullptr;
    int max_lag = 20;

    Mat m_transform_matrix;

    TSMethod tsMethod = BE;
    T tsParam[2][3] = {
        { 1, 0.5, 1 }, // Backward Euler (BE)
        { 0.5, 0.25, 0.5 } // Newmark (NM)
    };

    T dt = 0.02;

    TimeIntegratorUpdateOp(const Field<Matrix<T, dim, dim>>& BC_basis, const Field<int>& BC_order, const Field<Vector<T, dim>>& BC_target, const Field<uint8_t>& BC_fixed, Field<T>& mass, Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& v, Field<Vector<T, dim>>& a, Field<Vector<T, dim>>& x1, Field<Vector<T, dim>>& v1, Field<Vector<T, dim>>& x_tilde);
    virtual void gradient(const Vec& x_vec, Vec& grad_vec) override;
    virtual void hessian(const Vec& x_vec, Mat& hess, const bool project_pd) override;
    virtual T residual(const Vec& x, const Vec& grad, const Vec& direction) override;
    void project(Vec& x) override;
    void push_forward(Vec& direction) override;
    void set_ts_weights();
    void initialize_acceleration();
    void update_predictive_pos();
    void update_transformation_matrix();
    void callback(Vec& x_vec);
    void advance(const Vec& new_x);
    void operator()();

protected:
    T constraint_residual(const Vec& y, const Vec& cons) override;
    void set_constraint_weight(Vec& weight);
    void constraint(const Vec& x_vec, Vec& cons);
    void constraint_jacobian(const Vec& x_vec, Mat& jac);
};
}
} // namespace Bow::FEM

#include "FEMTimeIntegrator.tpp"

#endif