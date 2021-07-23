#ifndef IPC_OP_H
#define IPC_OP_H
#include <Bow/Types.h>
#include <Bow/Geometry/IpcUtils/FrictionUtils.h>
#include <Bow/Utils/SpatialHash.h>
#include "../Prototypes.h"

namespace Bow::FEM::IPC {

template <class T, int dim>
class IpcEnergyBase {
public:
    using TV = Vector<T, dim>;
    // parameters
    T dHat = 1e-3;
    T kappa = 1e4;
    bool improved_maxOp = false;

    T mu = 0, epsvh = 0, dt = 0;
    T epsv = 0;
    bool update_basis = true;
    Field<Vector<T, dim>> x_hat;
    T x_weight = 1.0; // 2.0 for Newmark

    // intermediate variables;
    SpatialHash<T, dim> sh;

    // for lagged friction
    T basis_update_error = 0;
    T normal_force_update_error = 0;

    void initialize_friction(T mu_input, T epsv_input, T dt)
    {
        mu = mu_input;
        epsvh = epsv_input * dt;
        epsv = epsv_input;
        update_basis = true;
    }

    // xn1: x^{n-1}
    void update_weight_and_xhat(const Field<Vector<T, dim>>& xn, const Field<Vector<T, dim>>& vn, const Field<Vector<T, dim>>& an, const Field<Vector<T, dim>>& xn1, const T dt, const T tsParam[], TSMethod tsMethod)
    {
        if (tsMethod == BDF2) {
            x_weight = 3.0 / 2.0;
            x_hat.resize(xn.size());
            tbb::parallel_for(size_t(0), x_hat.size(), [&](size_t i) {
                x_hat[i] = x_weight * (4.0 / 3.0 * xn[i] - xn1[i] / 3.0);
            });
        }
        else {
            x_weight = tsParam[2] / (2 * tsParam[0] * tsParam[1]);
            x_hat.resize(xn.size());
            tbb::parallel_for(size_t(0), x_hat.size(), [&](size_t i) {
                TV x_tilde = xn[i] + vn[i] * dt + tsParam[0] * (1 - 2 * tsParam[1]) * an[i] * dt * dt;
                x_hat[i] = x_weight * x_tilde - vn[i] * dt - (1 - tsParam[2]) * an[i] * dt * dt;
            });
        }
    }
};

template <class T, class StorageIndex = int>
class IpcEnergyOp2D : public EnergyOp<T, 2, StorageIndex>, public IpcEnergyBase<T, 2> {
public:
    static constexpr int dim = 2;
    using TV = Vector<T, dim>;
    using Base = IpcEnergyBase<T, 2>;

    // inputs
    const Field<int>& m_boundary_points;
    const Field<Vector<int, 2>>& m_boundary_edges;
    const Field<T>& m_mass;
    const Field<T>& m_boundary_point_area;
    const Field<std::set<int>>& m_boundary_point_nb;

    // parameters
    using Base::dHat;
    using Base::improved_maxOp;
    using Base::kappa;

    // intermediate variables;
    Field<Vector<int, 2>> PP;
    Field<T> wPP;
    Field<Vector<int, 3>> PE;
    Field<T> wPE;

    using Base::sh;
    // for lagged friction
    using Base::basis_update_error;
    using Base::dt;
    using Base::epsv;
    using Base::epsvh;
    using Base::mu;
    using Base::normal_force_update_error;
    using Base::update_basis;
    using Base::x_hat;
    using Base::x_weight;

    Field<Vector<int, 2>> PP_friction;
    Field<Vector<int, 3>> PE_friction;
    Field<T> PP_normalForce;
    Field<T> PE_normalForce;
    Field<TV> PP_tanBasis;
    Field<TV> PE_tanBasis;
    Field<T> PE_yita;

    IpcEnergyOp2D(const Field<int>& boundary_points, const Field<Vector<int, 2>>& boundary_edges, const Field<T>& mass, const Field<T>& boundary_point_area, const Field<std::set<int>>& m_boundary_point_nb, T energy_scale = 1.0);
    /* find constraint set*/
    void precompute(const Field<Vector<T, dim>>& x) override;
    /* adaptive kappa */
    void callback(const Field<Vector<T, dim>>& x) override;
    T stepsize_upperbound(const Field<Vector<T, dim>>& x, const Field<Vector<T, dim>>& dx) override;
    T energy(const Field<Vector<T, dim>>& x) override;
    void gradient(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& grad) override;
    template <bool project_pd = true>
    void hessian_impl(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess) const;
    void hessian(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess, const bool project_pd = true) override
    {
        if (project_pd)
            hessian_impl<true>(x, hess);
        else
            hessian_impl<false>(x, hess);
    }
    void internal_force(const Field<Vector<T, dim>>& xn, const Field<Vector<T, dim>>& vn, Field<Vector<T, dim>>& force) override;
};

template <class T, class StorageIndex = int>
class IpcEnergyOp3D : public EnergyOp<T, 3, StorageIndex>, public IpcEnergyBase<T, 3> {
public:
    static const int dim = 3;
    using TV = Vector<T, dim>;
    using Base = IpcEnergyBase<T, 3>;
    // inputs
    const Field<int>& m_boundary_points;
    const Field<Vector<int, 2>>& m_boundary_edges;
    const Field<Vector<int, 3>>& m_boundary_faces;
    const Field<Vector<T, dim>>& m_X;
    const Field<T>& m_mass;

    const Field<T>& m_boundary_point_area;
    const Field<int8_t>& m_boundary_point_type;
    const Field<std::set<int>>& m_boundary_point_nb;
    const Field<T>& m_boundary_edge_area;
    const Field<std::set<int>>& m_boundary_edge_pnb;

    // parameters
    using Base::dHat;
    using Base::improved_maxOp;
    using Base::kappa;

    // intermediate variables;
    Field<Vector<int, 2>> PP; // PP from PT or non mollified EE
    Field<T> wPP;
    Field<Vector<int, 3>> PE; // PE from PT or non mollified EE
    Field<T> wPE;
    Field<Vector<int, 4>> PT;
    Field<T> wPT;
    Field<Vector<int, 4>> PPM; // PP from mollified EE only， the 0st and 2nd digits are actual pair
    Field<T> wPPM;
    Field<Vector<int, 4>> PEM; // PE from mollified EE only, the 0st, 2nd, 3rd digits are actual pair
    Field<T> wPEM;
    Field<Vector<int, 4>> EEM;
    Field<T> wEEM;
    Field<Vector<int, 4>> EE;
    Field<T> wEE;

    using Base::sh;

    // friction related
    void friction_precompute(const Field<Vector<T, dim>>& x);

    using Base::basis_update_error;
    using Base::dt;
    using Base::epsv;
    using Base::epsvh;
    using Base::mu;
    using Base::normal_force_update_error;
    using Base::update_basis;
    using Base::x_hat;
    using Base::x_weight;

    Field<Vector<int, 2>> PP_friction;
    Field<Vector<int, 3>> PE_friction;
    Field<Vector<int, 4>> PT_friction;
    Field<Vector<int, 4>> EE_friction;
    Field<T> PP_normalForce;
    Field<T> PE_normalForce;
    Field<T> PT_normalForce;
    Field<T> EE_normalForce;
    Field<Matrix<T, dim, dim - 1>> PP_tanBasis;
    Field<Matrix<T, dim, dim - 1>> PE_tanBasis;
    Field<Matrix<T, dim, dim - 1>> PT_tanBasis;
    Field<Matrix<T, dim, dim - 1>> EE_tanBasis;
    Field<T> PE_yita;
    Field<Vector<T, 2>> PT_beta;
    Field<Vector<T, 2>> EE_gamma;

    IpcEnergyOp3D(const Field<int>& boundary_points, const Field<Vector<int, 2>>& boundary_edges, const Field<Vector<int, 3>>& boundary_faces,
        const Field<Vector<T, dim>>& X, const Field<T>& mass, const Field<T>& boundary_point_area, const Field<int8_t>& boundary_point_type, const Field<std::set<int>>& boundary_point_nb,
        const Field<T>& boundary_edge_area, const Field<std::set<int>>& boundary_edge_pnb, T energy_scale = 1.0);
    /* find constraint set*/
    void precompute(const Field<Vector<T, dim>>& x) override;
    /* adaptive kappa */
    void callback(const Field<Vector<T, dim>>& x) override;
    T stepsize_upperbound(const Field<Vector<T, dim>>& x, const Field<Vector<T, dim>>& dx) override;
    T energy(const Field<Vector<T, dim>>& x) override;
    void gradient(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& grad) override;
    template <bool project_pd = true>
    void hessian_impl(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess) const;
    void hessian(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess, const bool project_pd = true) override
    {
        if (project_pd)
            hessian_impl<true>(x, hess);
        else
            hessian_impl<false>(x, hess);
    }

protected:
    void point_triangle_constraints(const Field<Vector<T, 3>>& x);
    void edge_edge_constraints(const Field<Vector<T, 3>>& x);
};

template <class T, int dim, class StorageIndex = int>
using IpcEnergyOp = typename std::conditional<dim == 2, IpcEnergyOp2D<T, StorageIndex>, IpcEnergyOp3D<T, StorageIndex>>::type;
} // namespace Bow::FEM::IPC

#include "IpcEnergy2D.tpp"
#include "IpcEnergy3D.tpp"

#endif
