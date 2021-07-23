#include <Bow/Types.h>
#include <catch2/catch.hpp>
#include <iostream>
#include <Bow/Utils/FiniteDiff.h>
#include <Eigen/Sparse>
#include <Bow/Energy/FEM/Shell/MembraneEnergy.h>
#include <Bow/Energy/FEM/Shell/BendingEnergy.h>
#include <Bow/Simulator/FEM/InitializeShellOp.h>

template <int dim = 3>
void membrane_energy_test()
{
    using namespace Bow;
    using namespace Shell;
    using T = double;
    Field<Vector<T, 3>> m_X(4);
    Eigen::Map<Vector<T, 12>>(&(m_X[0](0))) = Vector<T, 12>::Random();
    Field<Vector<int, 3>> m_elem(2);
    m_elem[0] << 0, 1, 2;
    m_elem[1] << 1, 2, 3;
    Field<T> mu(2, 1);
    Field<T> lam(2, 1);
    T thickness = 1.0;

    Field<Matrix<T, 2, 2>> m_IB;
    Field<T> m_vol;

    InitializeMembraneOp<double> initialize(m_X, m_elem, thickness, m_IB, m_vol);
    initialize();

    MembraneEnergyOp<double> energy_op(m_elem, m_vol, mu, lam, m_IB);

    const auto f = [&](const Eigen::VectorXd& x_vec) -> double {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + m_X.size());
        return energy_op.energy(x);
    };
    const auto g = [&](const Eigen::VectorXd& x_vec, Eigen::VectorXd& grad_vec) {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + m_X.size());
        Bow::Field<Bow::Vector<T, dim>> grad;
        energy_op.gradient(x, grad);
        grad_vec.resize(x_vec.size());
        memcpy(grad_vec.data(), reinterpret_cast<T*>(grad.data()), sizeof(T) * x_vec.size());
    };
    const auto h = [&](const Eigen::VectorXd& x_vec, Eigen::SparseMatrix<double>& hess) {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + m_X.size());
        energy_op.hessian(x, hess, false);
    };
    Eigen::VectorXd x = to_vec(m_X);
    x *= 2;
    CHECK(Bow::FiniteDiff::check_gradient(x, f, g, 1e-4, 1e-3));
    CHECK(Bow::FiniteDiff::check_jacobian<Eigen::SparseMatrix<double>>(x, g, h, 1e-4, 1e-3));
}

template <int dim = 3>
void dihedral_angle_test()
{
    using T = double;
    using namespace Bow;
    using namespace Shell;
    using namespace Geometry;

    int branch = 0;

    const auto f = [&](const Eigen::VectorXd& x_vec) -> double {
        Vector<T, dim> v0 = x_vec.template segment<dim>(0);
        Vector<T, dim> v1 = x_vec.template segment<dim>(dim);
        Vector<T, dim> v2 = x_vec.template segment<dim>(2 * dim);
        Vector<T, dim> v3 = x_vec.template segment<dim>(3 * dim);
        return dihedral_angle(v0, v1, v2, v3, branch);
    };
    const auto g = [&](const Eigen::VectorXd& x_vec, Eigen::VectorXd& grad_vec) {
        Vector<T, dim> v0 = x_vec.template segment<dim>(0);
        Vector<T, dim> v1 = x_vec.template segment<dim>(dim);
        Vector<T, dim> v2 = x_vec.template segment<dim>(2 * dim);
        Vector<T, dim> v3 = x_vec.template segment<dim>(3 * dim);
        Vector<T, dim * 4> grad;
        dihedral_angle_gradient(v0, v1, v2, v3, grad);
        grad_vec = grad;
    };
    const auto h = [&](const Eigen::VectorXd& x_vec, Eigen::MatrixXd& hess) {
        Vector<T, dim> v0 = x_vec.template segment<dim>(0);
        Vector<T, dim> v1 = x_vec.template segment<dim>(dim);
        Vector<T, dim> v2 = x_vec.template segment<dim>(2 * dim);
        Vector<T, dim> v3 = x_vec.template segment<dim>(3 * dim);
        Matrix<T, dim * 4, dim * 4> hess_;
        dihedral_angle_hessian(v0, v1, v2, v3, hess_);
        hess = hess_;
    };

    Vector<T, 12> v = Vector<T, 12>::Random();
    CHECK(Bow::FiniteDiff::check_gradient(v, f, g, 1e-4, 1e-3));
    CHECK(Bow::FiniteDiff::check_jacobian<Eigen::MatrixXd>(v, g, h, 1e-4, 1e-3));

    v << 0, 0, -1,
        0, 0, 0,
        0, 1, 0,
        0, 0, 1;
    branch = 0;
    CHECK(Bow::FiniteDiff::check_gradient(v, f, g, 1e-4, 1e-3));
    CHECK(Bow::FiniteDiff::check_jacobian<Eigen::MatrixXd>(v, g, h, 1e-4, 1e-3));
}

template <int dim = 3>
void bending_energy_test()
{
    using namespace Bow;
    using namespace Shell;
    using T = double;
    Field<Vector<T, 3>> m_X(4);
    m_X[0] << 0, 0, 0;
    m_X[1] << 0, 1, 0;
    m_X[2] << 0, 0, -1;
    m_X[3] << 0, 0, 1;
    Field<Vector<int, 4>> m_edge_stencil({ Vector<int, 4>(0, 1, 2, 3) });
    Field<T> m_edge_weight(1, 1);
    Field<T> m_rest_angle;
    Field<T> m_e;
    Field<T> m_h;

    InitializeBendingOp<double> initialize(m_X, m_edge_stencil, m_e, m_h, m_rest_angle);
    initialize();

    BendingEnergyOp<double> energy_op(m_edge_stencil, m_e, m_h, m_rest_angle, m_edge_weight);

    const auto f = [&](const Eigen::VectorXd& x_vec) -> double {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + m_X.size());
        energy_op.precompute(x);
        return energy_op.energy(x);
    };
    const auto g = [&](const Eigen::VectorXd& x_vec, Eigen::VectorXd& grad_vec) {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + m_X.size());
        Bow::Field<Bow::Vector<T, dim>> grad;
        energy_op.gradient(x, grad);
        grad_vec.resize(x_vec.size());
        energy_op.precompute(x);
        memcpy(grad_vec.data(), reinterpret_cast<T*>(grad.data()), sizeof(T) * x_vec.size());
    };
    const auto h = [&](const Eigen::VectorXd& x_vec, Eigen::SparseMatrix<double>& hess) {
        const Bow::Vector<T, dim>* p_x = reinterpret_cast<const Bow::Vector<T, dim>*>(x_vec.data());
        Bow::Field<Bow::Vector<T, dim>> x(p_x, p_x + m_X.size());
        energy_op.precompute(x);
        energy_op.hessian(x, hess, false);
    };

    Field<Vector<T, 3>> m_x = m_X;
    T delta_theta = 3. * M_PI / 2. / 20.;
    std::fill(energy_op.angle_branch.begin(), energy_op.angle_branch.end(), 0);
    for (int i = 1; i <= 20; ++i) {
        m_x[3] << std::sin(delta_theta * i), 0, std::cos(delta_theta * i);
        Eigen::VectorXd x = to_vec(m_x);
        CHECK(Bow::FiniteDiff::check_gradient(x, f, g, 1e-4, 1e-3));
        CHECK(Bow::FiniteDiff::check_jacobian<Eigen::SparseMatrix<double>>(x, g, h, 1e-4, 1e-3));
    }

    std::fill(energy_op.angle_branch.begin(), energy_op.angle_branch.end(), 0);
    for (int i = 1; i <= 20; ++i) {
        m_x[3] << std::sin(-delta_theta * i), 0, std::cos(-delta_theta * i);
        Eigen::VectorXd x = to_vec(m_x);
        CHECK(Bow::FiniteDiff::check_gradient(x, f, g, 1e-4, 1e-3));
        CHECK(Bow::FiniteDiff::check_jacobian<Eigen::SparseMatrix<double>>(x, g, h, 1e-4, 1e-3));
    }
}

TEST_CASE("Test Membrane", "[MEM]")
{
    GENERATE(1, 2, 3, 4, 5);
    membrane_energy_test();
}

TEST_CASE("Test Dihedral", "[DA]")
{
    GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    dihedral_angle_test();
}

TEST_CASE("Test Bending", "[BEND]")
{
    bending_energy_test();
}
