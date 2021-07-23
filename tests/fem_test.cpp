#include <Bow/Types.h>
#include <catch2/catch.hpp>
#include <iostream>
#include <Bow/Geometry/Primitives.h>
#include <Bow/Utils/FiniteDiff.h>
#include <Bow/Types.h>
#include <Eigen/Sparse>
#include <Bow/IO/ply.h>
#include <Bow/Geometry/OrthogonalBasis.h>
#include <Bow/Simulator/FEM/FEMSimulator.h>
#include <Bow/Geometry/BoundaryConditions.h>

template <class T, int dim, Bow::ConstitutiveModel::Type type>
void fem_elasticity_energy_test()
{
    Bow::FEM::FEMSimulator<T, dim> fem_data;
    Bow::Field<Bow::Vector<T, dim>> pos;
    Bow::Field<Bow::Vector<int, dim + 1>> elem;
    if constexpr (dim == 2) {
        Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1), 0.1, pos, elem);
    }
    else
        Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1, 1), 0.1, pos, elem);

    fem_data.append(pos, elem, type, 1e4, 0.3, 10000);
    Bow::FEM::InitializeOp<T, dim> initialize{ fem_data.m_X, fem_data.m_elem, fem_data.m_density,
        fem_data.m_elem_codim1, fem_data.m_thickness_codim1, fem_data.m_density_codim1,
        fem_data.m_mass, fem_data.m_vol, fem_data.m_IB, fem_data.m_vol_codim1, fem_data.m_IB_codim1 };
    std::unique_ptr<Bow::EnergyOp<T, dim>> energy_op;
    if constexpr (type == Bow::ConstitutiveModel::FIXED_COROTATED)
        energy_op = std::make_unique<Bow::FEM::FixedCorotatedEnergyOp<T, dim>>(fem_data.m_elem, fem_data.m_vol, fem_data.m_mu, fem_data.m_lam, fem_data.m_IB, fem_data.m_obj_divider[type]);
    else if constexpr (type == Bow::ConstitutiveModel::NEO_HOOKEAN)
        energy_op = std::make_unique<Bow::FEM::NeoHookeanEnergyOp<T, dim>>(fem_data.m_elem, fem_data.m_vol, fem_data.m_mu, fem_data.m_lam, fem_data.m_IB, fem_data.m_obj_divider[type]);
    initialize();
    const auto f = [&](const Eigen::VectorXd& x_vec) -> double {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + fem_data.m_X.size());
        return energy_op->energy(x);
    };
    const auto g = [&](const Eigen::VectorXd& x_vec, Eigen::VectorXd& grad_vec) {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + fem_data.m_X.size());
        Bow::Field<Bow::Vector<T, dim>> grad;
        energy_op->gradient(x, grad);
        grad_vec.resize(x_vec.size());
        memcpy(grad_vec.data(), reinterpret_cast<T*>(grad.data()), sizeof(T) * x_vec.size());
    };
    const auto h = [&](const Eigen::VectorXd& x_vec, Eigen::SparseMatrix<double>& hess) {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + fem_data.m_X.size());
        energy_op->hessian(x, hess, false);
    };
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(reinterpret_cast<T*>(fem_data.m_x.data()), fem_data.m_x.size() * dim);
    x *= 2;
    CHECK(Bow::FiniteDiff::check_gradient(x, f, g, 1e-4, 1e-3));
    CHECK(Bow::FiniteDiff::check_jacobian<Eigen::SparseMatrix<double>>(x, g, h, 1e-4, 1e-3));
}

template <class T, int dim>
void fem_inertial_energy_test()
{
    Bow::FEM::FEMSimulator<T, dim> fem_data;
    Bow::Field<Bow::Vector<T, dim>> pos;
    Bow::Field<Bow::Vector<int, dim + 1>> elem;
    if constexpr (dim == 2) {
        Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1), 0.1, pos, elem);
    }
    else
        Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1, 1), 0.1, pos, elem);

    fem_data.append(pos, elem, 0, 1e4, 0.3, 10000);
    Bow::FEM::InitializeOp<T, dim> initialize{ fem_data.m_X, fem_data.m_elem, fem_data.m_density,
        fem_data.m_elem_codim1, fem_data.m_thickness_codim1, fem_data.m_density_codim1,
        fem_data.m_mass, fem_data.m_vol, fem_data.m_IB, fem_data.m_vol_codim1, fem_data.m_IB_codim1 };
    Bow::FEM::InertialEnergyOp<T, dim> inertial_energy(fem_data.m_mass, fem_data.m_x_tilde);
    initialize();
    const auto f = [&](const Eigen::VectorXd& x_vec) -> double {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + fem_data.m_X.size());
        return inertial_energy.energy(x);
    };
    const auto g = [&](const Eigen::VectorXd& x_vec, Eigen::VectorXd& grad_vec) {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + fem_data.m_X.size());
        Bow::Field<Bow::Vector<T, dim>> grad;
        inertial_energy.gradient(x, grad);
        grad_vec.resize(x_vec.size());
        memcpy(grad_vec.data(), reinterpret_cast<T*>(grad.data()), sizeof(T) * x_vec.size());
    };
    const auto h = [&](const Eigen::VectorXd& x_vec, Eigen::SparseMatrix<double>& hess) {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + fem_data.m_X.size());
        inertial_energy.hessian(x, hess, false);
    };
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(reinterpret_cast<T*>(fem_data.m_x.data()), fem_data.m_x.size() * dim);
    x *= 2;
    CHECK(Bow::FiniteDiff::check_gradient(x, f, g, 1e-4, 1e-3));
    CHECK(Bow::FiniteDiff::check_jacobian<Eigen::SparseMatrix<double>>(x, g, h, 1e-4, 1e-3));
}

template <class T, int dim>
void diff_test_with_BC()
{
    // Bow::FEM::FEMSimulator<T, dim> simulator;
    // simulator.gravity = Bow::Vector<T, dim>(10, -98);
    // // generate data
    // Bow::Field<Bow::Vector<T, dim>> vertices;
    // Bow::Field<Bow::Vector<int, dim + 1>> elements;
    // Bow::Geometry::cube(Bow::Vector<int, dim>(10, 10), 0.1, vertices, elements);
    // // transformation
    // Bow::Matrix<T, 2, 2> transformation = Bow::Geometry::extend_basis(Bow::Vector<T, dim>(1.0 / sqrt(2.0), 1.0 / sqrt(2.0)));
    // for (size_t i = 0; i < vertices.size(); ++i) {
    //     vertices[i] = transformation * vertices[i];
    // }
    // simulator.append(vertices, elements, Bow::ConstitutiveModel::FIXED_COROTATED, 1e3, 0.3, 1000);
    // simulator.add_boundary_condition(new Bow::Geometry::HalfSpaceLevelSet<T, dim>(Bow::Geometry::SLIP, Eigen::Vector2d(-0.5 * sqrt(2) + 0.0001, 0), Eigen::Vector2d(1, 1)));
    // simulator.initialize();
    // for (int i = 0; i < 5; ++i)
    //     simulator.advance(0.02);
    // const auto f = [&](const Eigen::VectorXd& y) -> double {
    //     Eigen::VectorXd x = simulator.m_advance_op->m_transform_matrix * y;
    //     return simulator.m_advance_op->energy(x);
    // };
    // const auto g = [&](const Eigen::VectorXd& y, Eigen::VectorXd& grad) {
    //     Eigen::VectorXd x = simulator.m_advance_op->m_transform_matrix * y;
    //     simulator.m_advance_op->gradient(x, grad);
    // };
    // const auto h = [&](const Eigen::VectorXd& y, Eigen::SparseMatrix<double>& hess) {
    //     Eigen::VectorXd x = simulator.m_advance_op->m_transform_matrix * y;
    //     simulator.m_advance_op->hessian(x, hess, false);
    // };
    // Eigen::VectorXd y = simulator.m_advance_op->m_transform_matrix.transpose() * Eigen::Map<Eigen::VectorXd>(reinterpret_cast<T*>(simulator.m_x.data()), simulator.m_x.size() * dim);
    // Eigen::VectorXd dy = Eigen::VectorXd::Random(y.size());
    // dy /= dy.norm();
    // dy *= 1e-4;
    // for (size_t i = 0; i < simulator.m_x.size(); ++i) {
    //     for (int d = 0; d < simulator.BC_order[i]; ++d)
    //         dy[dim * i + d] = 0;
    // }

    // CHECK(Bow::FiniteDiff::check_gradient(y, f, g, dy, 1e-3));
    // CHECK(Bow::FiniteDiff::check_jacobian<Eigen::SparseMatrix<double>>(y, g, h, dy, 1e-3));
}

TEST_CASE("Test FEM Derivative", "[FEM]")
{
    GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    fem_elasticity_energy_test<double, 2, Bow::ConstitutiveModel::FIXED_COROTATED>();
    fem_elasticity_energy_test<double, 3, Bow::ConstitutiveModel::FIXED_COROTATED>();
    fem_elasticity_energy_test<double, 2, Bow::ConstitutiveModel::NEO_HOOKEAN>();
    fem_elasticity_energy_test<double, 3, Bow::ConstitutiveModel::NEO_HOOKEAN>();
    fem_inertial_energy_test<double, 2>();
    fem_inertial_energy_test<double, 3>();
    diff_test_with_BC<double, 2>();
}