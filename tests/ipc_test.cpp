#include <Bow/Types.h>
#include <catch2/catch.hpp>
#include <Bow/Geometry/Primitives.h>
#include <iostream>
#include <Bow/Utils/FiniteDiff.h>
#include <Bow/Geometry/Query.h>
#include <Bow/Simulator/FEM/IPCSimulator.h>
#include <Eigen/Dense>

void barrier_test()
{
    using T = double;
    const auto f = [&](const Eigen::VectorXd& x) -> double {
        T x0 = x[0];
        return Bow::Math::barrier(x0, 1.23123, 4.1212);
    };
    const auto g = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        T x0 = x[0];
        grad.resize(1);
        grad[0] = Bow::Math::barrier_gradient(x0, 1.23123, 4.1212);
    };
    const auto h = [&](const Eigen::VectorXd& x, Eigen::MatrixXd& hess) {
        T x0 = x[0];
        hess.resize(1, 1);
        hess(0, 0) = Bow::Math::barrier_hessian(x0, 1.23123, 4.1212);
    };
    Eigen::VectorXd x = Bow::Vector<T, 1>::Random().array();
    x[0] = std::abs(x[0]);
    CHECK(Bow::FiniteDiff::check_gradient(x, f, g, 1e-4, 1e-3));
    CHECK(Bow::FiniteDiff::check_jacobian<Eigen::MatrixXd>(x, g, h, 1e-4, 1e-3));
}

template <int dim>
void pp_test()
{
    using T = double;
    const auto f = [&](const Eigen::VectorXd& x) -> double {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        return Bow::Geometry::IPC::point_point_distance(x0, x1);
    };
    const auto g = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        Bow::Vector<T, dim * 2> grad_fixed_len;
        Bow::Geometry::IPC::point_point_distance_gradient(x0, x1, grad_fixed_len);
        grad = grad_fixed_len;
    };
    const auto h = [&](const Eigen::VectorXd& x, Eigen::MatrixXd& hess) {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        Bow::Matrix<T, dim * 2, dim * 2> hess_fixed_len;
        Bow::Geometry::IPC::point_point_distance_hessian(x0, x1, hess_fixed_len);
        hess = hess_fixed_len;
    };
    Eigen::VectorXd x = Bow::Vector<T, dim * 2>::Random();
    CHECK(Bow::FiniteDiff::check_gradient(x, f, g, 1e-4, 1e-3));
    CHECK(Bow::FiniteDiff::check_jacobian<Eigen::MatrixXd>(x, g, h, 1e-4, 1e-3));
}

template <int dim>
void pe_test()
{
    using T = double;
    const auto f = [&](const Eigen::VectorXd& x) -> double {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        Bow::Vector<T, dim> x2 = x.template segment<dim>(dim * 2);
        return Bow::Geometry::IPC::point_edge_distance(x0, x1, x2);
    };
    const auto g = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        Bow::Vector<T, dim> x2 = x.template segment<dim>(dim * 2);
        Bow::Vector<T, dim * 3> grad_fixed_len;
        Bow::Geometry::IPC::point_edge_distance_gradient(x0, x1, x2, grad_fixed_len);
        grad = grad_fixed_len;
    };
    const auto h = [&](const Eigen::VectorXd& x, Eigen::MatrixXd& hess) {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        Bow::Vector<T, dim> x2 = x.template segment<dim>(dim * 2);
        Bow::Matrix<T, dim * 3, dim * 3> hess_fixed_len;
        Bow::Geometry::IPC::point_edge_distance_hessian(x0, x1, x2, hess_fixed_len);
        hess = hess_fixed_len;
    };
    Eigen::VectorXd x = Bow::Vector<T, dim * 3>::Random();
    CHECK(Bow::FiniteDiff::check_gradient(x, f, g, 1e-4, 1e-3));
    CHECK(Bow::FiniteDiff::check_jacobian<Eigen::MatrixXd>(x, g, h, 1e-4, 1e-3));
}

// MSVC cannot capture constexpr variables in lambdas, see:
// https://developercommunity.visualstudio.com/t/invalid-template-argument-expected-compile-time-co/187862
template<int dim = 3>
void edge_edge_test()
{
    using T = double;

    const auto f = [&](const Eigen::VectorXd& x) -> double {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        Bow::Vector<T, dim> x2 = x.template segment<dim>(dim * 2);
        Bow::Vector<T, dim> x3 = x.template segment<dim>(dim * 3);
        return Bow::Geometry::IPC::edge_edge_cross_norm2(x0, x1, x2, x3);
    };
    const auto g = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        Bow::Vector<T, dim> x2 = x.template segment<dim>(dim * 2);
        Bow::Vector<T, dim> x3 = x.template segment<dim>(dim * 3);
        Bow::Vector<T, dim * 4> grad_fixed_len;
        Bow::Geometry::IPC::edge_edge_cross_norm2_gradient(x0, x1, x2, x3, grad_fixed_len);
        grad = grad_fixed_len;
    };
    const auto h = [&](const Eigen::VectorXd& x, Eigen::MatrixXd& hess) {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        Bow::Vector<T, dim> x2 = x.template segment<dim>(dim * 2);
        Bow::Vector<T, dim> x3 = x.template segment<dim>(dim * 3);
        Bow::Matrix<T, dim * 4, dim * 4> hess_fixed_len;
        Bow::Geometry::IPC::edge_edge_cross_norm2_hessian(x0, x1, x2, x3, hess_fixed_len);
        hess = hess_fixed_len;
    };

    T eps = 1;
    const auto f1 = [&](const Eigen::VectorXd& x) -> double {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        Bow::Vector<T, dim> x2 = x.template segment<dim>(dim * 2);
        Bow::Vector<T, dim> x3 = x.template segment<dim>(dim * 3);
        return Bow::Geometry::IPC::edge_edge_mollifier(x0, x1, x2, x3, eps);
    };
    const auto g1 = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        Bow::Vector<T, dim> x2 = x.template segment<dim>(dim * 2);
        Bow::Vector<T, dim> x3 = x.template segment<dim>(dim * 3);
        Bow::Vector<T, dim * 4> grad_fixed_len;
        Bow::Geometry::IPC::edge_edge_mollifier_gradient(x0, x1, x2, x3, eps, grad_fixed_len);
        grad = grad_fixed_len;
    };
    const auto h1 = [&](const Eigen::VectorXd& x, Eigen::MatrixXd& hess) {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        Bow::Vector<T, dim> x2 = x.template segment<dim>(dim * 2);
        Bow::Vector<T, dim> x3 = x.template segment<dim>(dim * 3);
        Bow::Matrix<T, dim * 4, dim * 4> hess_fixed_len;
        Bow::Geometry::IPC::edge_edge_mollifier_hessian(x0, x1, x2, x3, eps, hess_fixed_len);
        hess = hess_fixed_len;
    };

    const auto f2 = [&](const Eigen::VectorXd& x) -> double {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        Bow::Vector<T, dim> x2 = x.template segment<dim>(dim * 2);
        Bow::Vector<T, dim> x3 = x.template segment<dim>(dim * 3);
        return Bow::Geometry::IPC::edge_edge_distance(x0, x1, x2, x3);
    };
    const auto g2 = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        Bow::Vector<T, dim> x2 = x.template segment<dim>(dim * 2);
        Bow::Vector<T, dim> x3 = x.template segment<dim>(dim * 3);
        Bow::Vector<T, dim * 4> grad_fixed_len;
        Bow::Geometry::IPC::edge_edge_distance_gradient(x0, x1, x2, x3, grad_fixed_len);
        grad = grad_fixed_len;
    };
    const auto h2 = [&](const Eigen::VectorXd& x, Eigen::MatrixXd& hess) {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        Bow::Vector<T, dim> x2 = x.template segment<dim>(dim * 2);
        Bow::Vector<T, dim> x3 = x.template segment<dim>(dim * 3);
        Bow::Matrix<T, dim * 4, dim * 4> hess_fixed_len;
        Bow::Geometry::IPC::edge_edge_distance_hessian(x0, x1, x2, x3, hess_fixed_len);
        hess = hess_fixed_len;
    };

    Eigen::VectorXd x = Bow::Vector<T, dim * 4>::Random();
    while (f(x) > eps)
        x = Bow::Vector<T, dim * 4>::Random();

    CHECK(Bow::FiniteDiff::check_gradient(x, f, g, 1e-4, 1e-3));
    CHECK(Bow::FiniteDiff::check_jacobian<Eigen::MatrixXd>(x, g, h, 1e-4, 1e-3));

    CHECK(Bow::FiniteDiff::check_gradient(x, f1, g1, 1e-6, 1e-3));
    CHECK(Bow::FiniteDiff::check_jacobian<Eigen::MatrixXd>(x, g1, h1, 1e-6, 1e-3));

    CHECK(Bow::FiniteDiff::check_gradient(x, f2, g2, 1e-4, 1e-3));
    CHECK(Bow::FiniteDiff::check_jacobian<Eigen::MatrixXd>(x, g2, h2, 1e-4, 1e-3));
}

template<int dim = 3>
void point_triangle_test()
{
    using T = double;

    const auto f = [&](const Eigen::VectorXd& x) -> double {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        Bow::Vector<T, dim> x2 = x.template segment<dim>(dim * 2);
        Bow::Vector<T, dim> x3 = x.template segment<dim>(dim * 3);
        return Bow::Geometry::IPC::point_triangle_distance(x0, x1, x2, x3);
    };
    const auto g = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        Bow::Vector<T, dim> x2 = x.template segment<dim>(dim * 2);
        Bow::Vector<T, dim> x3 = x.template segment<dim>(dim * 3);
        Bow::Vector<T, dim * 4> grad_fixed_len;
        Bow::Geometry::IPC::point_triangle_distance_gradient(x0, x1, x2, x3, grad_fixed_len);
        grad = grad_fixed_len;
    };
    const auto h = [&](const Eigen::VectorXd& x, Eigen::MatrixXd& hess) {
        Bow::Vector<T, dim> x0 = x.template segment<dim>(0);
        Bow::Vector<T, dim> x1 = x.template segment<dim>(dim);
        Bow::Vector<T, dim> x2 = x.template segment<dim>(dim * 2);
        Bow::Vector<T, dim> x3 = x.template segment<dim>(dim * 3);
        Bow::Matrix<T, dim * 4, dim * 4> hess_fixed_len;
        Bow::Geometry::IPC::point_triangle_distance_hessian(x0, x1, x2, x3, hess_fixed_len);
        hess = hess_fixed_len;
    };

    Eigen::VectorXd x = Bow::Vector<T, dim * 4>::Random();
    CHECK(Bow::FiniteDiff::check_gradient(x, f, g, 1e-4, 1e-3));
    CHECK(Bow::FiniteDiff::check_jacobian<Eigen::MatrixXd>(x, g, h, 1e-4, 1e-3));
}

template <int dim>
void ipc_derivative_test()
{
    using T = double;
    Bow::FEM::IPC::IPCSimulator<T, dim> fem_data;
    T dHat = 0.2;
    if constexpr (dim == 2) {
        Bow::Field<Bow::Vector<T, dim>> pos;
        Bow::Field<Bow::Vector<int, dim + 1>> elem;
        Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1), 2.0, pos, elem);
        fem_data.append(pos, elem, Bow::ConstitutiveModel::FIXED_COROTATED, 0, 0.3, 0);
        Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1), 2.0, pos, elem);
        for (size_t i = 0; i < pos.size(); ++i) {
            pos[i][0] += 2 + 0.5 * dHat;
        }
        fem_data.append(pos, elem, Bow::ConstitutiveModel::FIXED_COROTATED, 0, 0.3, 0);
    }
    else {
        Bow::Field<Bow::Vector<T, dim>> pos;
        Bow::Field<Bow::Vector<int, dim + 1>> elem;
        Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1, 1), 2.0, pos, elem);
        fem_data.append(pos, elem, Bow::ConstitutiveModel::FIXED_COROTATED, 0, 0.3, 0);
        Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1, 1), 2.0, pos, elem);
        for (size_t i = 0; i < pos.size(); ++i) {
            pos[i][0] += 2 + 0.5 * dHat;
            // pos[i] += 0.5 * dHat * Bow::Vector<T, dim>::Random();
        }
        fem_data.append(pos, elem, Bow::ConstitutiveModel::FIXED_COROTATED, 0, 0.3, 0);
    }
    Bow::FEM::InitializeOp<T, dim> initialize{ fem_data.m_X, fem_data.m_elem, fem_data.m_density,
        fem_data.m_elem_codim1, fem_data.m_thickness_codim1, fem_data.m_density_codim1,
        fem_data.m_mass, fem_data.m_vol, fem_data.m_IB, fem_data.m_vol_codim1, fem_data.m_IB_codim1 };
    initialize();
    Bow::Field<int> m_boundary_points;
    Bow::Field<Bow::Vector<int, 2>> m_boundary_edges;
    Bow::Field<Bow::Vector<int, 3>> m_boundary_faces;
    if constexpr (dim == 2) {
        Bow::Geometry::find_boundary(fem_data.m_elem, m_boundary_edges, m_boundary_points);
    }
    else {
        Bow::Geometry::find_boundary(fem_data.m_elem, m_boundary_faces, m_boundary_edges, m_boundary_points);
    }
    Bow::Field<T> boundary_point_area(m_boundary_points.size());
    Bow::Field<int8_t> boundary_point_type(m_boundary_points.size(), 2); // do not need precise value for derivative testing
    Bow::Field<std::set<int>> boundary_point_nb(m_boundary_points.size()); // do not need precise value for derivative testing
    for (size_t bpI = 0; bpI < m_boundary_points.size(); ++bpI) {
        boundary_point_area[bpI] = 0.1 * (((double)rand() / (RAND_MAX)) + 1); // do not need precise value for derivative testing
    }
    Bow::Field<T> boundary_edge_area(m_boundary_edges.size());
    Bow::Field<std::set<int>> boundary_edge_pnb(m_boundary_edges.size()); // do not need precise value for derivative testing
    for (size_t beI = 0; beI < m_boundary_edges.size(); ++beI) {
        boundary_edge_area[beI] = 0.1 * (((double)rand() / (RAND_MAX)) + 1); // do not need precise value for derivative testing
    }
    std::unique_ptr<Bow::FEM::IPC::IpcEnergyOp<T, dim>> ipc_energy;
    if constexpr (dim == 2)
        ipc_energy = std::make_unique<Bow::FEM::IPC::IpcEnergyOp<T, dim>>(m_boundary_points, m_boundary_edges, fem_data.m_mass, boundary_point_area, boundary_point_nb);
    else
        ipc_energy = std::make_unique<Bow::FEM::IPC::IpcEnergyOp<T, dim>>(m_boundary_points, m_boundary_edges, m_boundary_faces,
            fem_data.m_X, fem_data.m_mass, boundary_point_area, boundary_point_type, boundary_point_nb, boundary_edge_area, boundary_edge_pnb);
    ipc_energy->dHat = dHat;
    ipc_energy->kappa = 100;
    const auto f = [&](const Eigen::VectorXd& x_vec) -> double {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + fem_data.m_X.size());
        ipc_energy->precompute(x);
        return ipc_energy->energy(x);
    };
    const auto g = [&](const Eigen::VectorXd& x_vec, Eigen::VectorXd& grad_vec) {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + fem_data.m_X.size());
        ipc_energy->precompute(x);
        Bow::Field<Bow::Vector<T, dim>> grad;
        ipc_energy->gradient(x, grad);
        grad_vec.resize(x_vec.size());
        memcpy(grad_vec.data(), reinterpret_cast<T*>(grad.data()), sizeof(T) * x_vec.size());
    };
    const auto h = [&](const Eigen::VectorXd& x_vec, Eigen::SparseMatrix<double>& hess) {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + fem_data.m_X.size());
        ipc_energy->precompute(x);
        ipc_energy->hessian(x, hess, false);
    };
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(&(fem_data.m_x[0][0]), fem_data.m_x.size() * dim);
    ipc_energy->precompute(Bow::to_field<dim>(x));
    CHECK(Bow::FiniteDiff::check_gradient(x, f, g, 1e-8, 1e-3));
    CHECK(Bow::FiniteDiff::check_jacobian<Eigen::SparseMatrix<double>>(x, g, h, 1e-8, 1e-3));
}

template <int dim>
void ipc_friction_derivative_test()
{
    using T = double;
    Bow::FEM::IPC::IPCSimulator<T, dim> fem_data;
    T dHat = 0.02;

    Bow::Field<Bow::Vector<T, dim>> vertices;
    Bow::Field<Bow::Vector<int, dim + 1>> elements;
    // fixed cube as ground
    Bow::Field<Bow::Vector<T, dim>> vertices2;
    Bow::Field<Bow::Vector<int, dim + 1>> elements2;
    if constexpr (dim == 2) {
        Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1), 2.0, vertices, elements);
        Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1), 3.0, vertices2, elements2);
        // T dHat = 0.5 * dx;
        for (auto& i : vertices2) {
            i[0] *= 200;
            i[0] += 250;
            i[1] -= 2.5 + 0.5 * dHat;
        }
        fem_data.append(vertices, elements, Bow::ConstitutiveModel::FIXED_COROTATED, 0, 0.3, 0);
        fem_data.append(vertices2, elements2, Bow::ConstitutiveModel::FIXED_COROTATED, 0, 0.3, 0);
    }
    else {
        T L = 16.0;
        int n_seg = 8;
        int m_seg = 1;
        T dx = L / n_seg;
        Bow::Geometry::cube(Bow::Vector<int, dim>(m_seg, m_seg, m_seg), 2.0 / m_seg, vertices, elements);
        Bow::Geometry::cube(Bow::Vector<int, dim>(n_seg / 8 * 7, 1, n_seg / 8 * 3), dx, vertices2, elements2);
        for (auto& i : vertices2) {
            i[1] -= (dx + 2) / 2 + 0.9 * dHat;
        }
        fem_data.append(vertices, elements, Bow::ConstitutiveModel::FIXED_COROTATED, 0, 0.3, 0);
        fem_data.append(vertices2, elements2, Bow::ConstitutiveModel::FIXED_COROTATED, 0, 0.3, 0);
    }
    Bow::FEM::InitializeOp<T, dim> initialize{ fem_data.m_X, fem_data.m_elem, fem_data.m_density,
        fem_data.m_elem_codim1, fem_data.m_thickness_codim1, fem_data.m_density_codim1,
        fem_data.m_mass, fem_data.m_vol, fem_data.m_IB, fem_data.m_vol_codim1, fem_data.m_IB_codim1 };
    initialize();
    Bow::Field<int> m_boundary_points;
    Bow::Field<Bow::Vector<int, 2>> m_boundary_edges;
    Bow::Field<Bow::Vector<int, 3>> m_boundary_faces;
    if constexpr (dim == 2) {
        Bow::Geometry::find_boundary(fem_data.m_elem, m_boundary_edges, m_boundary_points);
    }
    else {
        Bow::Geometry::find_boundary(fem_data.m_elem, m_boundary_faces, m_boundary_edges, m_boundary_points);
    }
    Bow::Field<T> boundary_point_area(m_boundary_points.size());
    Bow::Field<std::set<int>> boundary_point_nb(m_boundary_points.size()); // do not need precise value for derivative testing
    for (size_t bpI = 0; bpI < m_boundary_points.size(); ++bpI) {
        boundary_point_area[bpI] = 0.1 * (((double)rand() / (RAND_MAX)) + 1); // do not need precise value for derivative testing
    }
    Bow::Field<T> boundary_edge_area(m_boundary_edges.size(), 1); // do not need precise value for derivative testing
    Bow::Field<std::set<int>> boundary_edge_pnb(m_boundary_edges.size());
    Bow::Field<int8_t> boundary_point_type(m_boundary_points.size(), 2);
    for (size_t bpI = 0; bpI < m_boundary_edges.size(); ++bpI) {
        boundary_edge_area[bpI] = 0.1 * (((double)rand() / (RAND_MAX)) + 1); // do not need precise value for derivative testing
    }
    std::unique_ptr<Bow::FEM::IPC::IpcEnergyOp<T, dim>> ipc_energy;
    if constexpr (dim == 2)
        ipc_energy = std::make_unique<Bow::FEM::IPC::IpcEnergyOp<T, dim>>(m_boundary_points, m_boundary_edges, fem_data.m_mass, boundary_point_area, boundary_point_nb);
    else
        ipc_energy = std::make_unique<Bow::FEM::IPC::IpcEnergyOp<T, dim>>(m_boundary_points, m_boundary_edges, m_boundary_faces, fem_data.m_X, fem_data.m_mass, boundary_point_area, boundary_point_type, boundary_point_nb, boundary_edge_area, boundary_edge_pnb);
    ipc_energy->kappa = 100;
    ipc_energy->dHat = dHat;
    T ts_param[3] = { 0.5, 0.25, 0.5 };
    ipc_energy->update_weight_and_xhat(fem_data.m_x, fem_data.m_v, fem_data.m_a, fem_data.m_x1, fem_data.suggested_dt, ts_param, Bow::NM);
    ipc_energy->initialize_friction(0.1, 100, fem_data.suggested_dt);
    const auto f = [&](const Eigen::VectorXd& x_vec) -> double {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + fem_data.m_X.size());
        ipc_energy->precompute(x);
        return ipc_energy->energy(x);
    };
    const auto g = [&](const Eigen::VectorXd& x_vec, Eigen::VectorXd& grad_vec) {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + fem_data.m_X.size());
        ipc_energy->precompute(x);
        Bow::Field<Bow::Vector<T, dim>> grad;
        ipc_energy->gradient(x, grad);
        grad_vec.resize(x_vec.size());
        memcpy(grad_vec.data(), reinterpret_cast<T*>(grad.data()), sizeof(T) * x_vec.size());
    };
    const auto h = [&](const Eigen::VectorXd& x_vec, Eigen::SparseMatrix<double>& hess) {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + fem_data.m_X.size());
        ipc_energy->precompute(x);
        ipc_energy->hessian(x, hess, false);
    };
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(&(fem_data.m_x[0][0]), fem_data.m_x.size() * dim);
    for (size_t i = 0; i < vertices.size(); ++i) {
        x(i * dim) += 1 * fem_data.suggested_dt;
    }
    CHECK(Bow::FiniteDiff::check_gradient(x, f, g, 1e-6, 1e-3));
    CHECK(Bow::FiniteDiff::check_jacobian<Eigen::SparseMatrix<double>>(x, g, h, 1e-6, 1e-3));
}

TEST_CASE("IPC Toolkit derivatives", "[IPCTool]")
{
    GENERATE(1, 2, 3, 4, 5);
    barrier_test();
    pp_test<2>();
    pp_test<3>();
    pe_test<2>();
    pe_test<3>();
    edge_edge_test();
    point_triangle_test();
}

TEST_CASE("IPC derivatives", "[IPC]")
{
    GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    ipc_derivative_test<2>();
    ipc_derivative_test<3>();
}

TEST_CASE("IPC friction derivatives", "[IPCf]")
{
    GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    ipc_friction_derivative_test<2>();
    ipc_friction_derivative_test<3>();
}
