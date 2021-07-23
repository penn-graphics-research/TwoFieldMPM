#include <Bow/Types.h>
#include <catch2/catch.hpp>
#include <iostream>
#include <Bow/Utils/FiniteDiff.h>
#include <Bow/Types.h>
#include <Eigen/Sparse>
#include <Bow/IO/ply.h>
#include <Bow/Geometry/OrthogonalBasis.h>
#include <Bow/Simulator/MPM/MPMSimulator.h>
#include <Bow/Geometry/BoundaryConditions.h>
#include <Bow/Utils/Logging.h>

template <class T, int dim>
void grid_kernel_test()
{
    Bow::MPM::MPMSimulator<T, dim> mpm;
    mpm.suggested_dt = 1e-2;
    if constexpr (dim == 2)
        mpm.dx = 0.1;
    else
        mpm.dx = 0.5;
    auto material1 = mpm.create_elasticity(new Bow::MPM::NeoHookeanOp<T, dim>(1000, 0.3));
    if constexpr (dim == 3) {
        mpm.add_particles(material1, Bow::Vector<T, dim>(0, 0, 0), Bow::Vector<T, dim>(1.0, 1.0, 1.0), Bow::Vector<T, dim>(0, 0, 0));
    }
    else {
        mpm.add_particles(material1, Bow::Vector<T, dim>(0, 0), Bow::Vector<T, dim>(1.0, 1.0), Bow::Vector<T, dim>(0, 0));
    }
    mpm.grid.sortParticles(mpm.m_X, mpm.dx);
    Bow::MPM::ParticlesToGridOp<T, dim, false> p2g{ {}, mpm.m_X, mpm.m_V, mpm.m_mass, mpm.m_C, mpm.stress, mpm.grid, mpm.dx, mpm.suggested_dt };
    p2g();

    for (size_t i = 0; i < mpm.m_X.size(); ++i) {
        Bow::BSplineWeights<T, dim> spline(mpm.m_X[i], mpm.dx);
        Bow::Vector<T, dim> X_new = Bow::Vector<T, dim>::Zero();
        Bow::Matrix<T, dim, dim> dxn = Bow::Matrix<T, dim, dim>::Zero();
        mpm.grid.iterateKernel(spline, [&](const Bow::Vector<int, dim>& node, T w, Bow::Vector<T, dim> dw, Bow::MPM::GridState<T, dim>& g) {
            Bow::Vector<T, dim> xn = node.template cast<T>() * mpm.dx;
            X_new += xn * w;
            dxn += xn * dw.transpose();
        });
        CHECK((X_new - mpm.m_X[i]).norm() < 1e-13);
        CHECK((dxn - Bow::Matrix<T, dim, dim>::Identity()).norm() < 1e-13);
    }
}

template <class T, int dim, bool Fbased = true>
void mpm_test()
{
    Bow::MPM::MPMSimulator<T, dim> mpm;
    mpm.suggested_dt = 0.001;
    mpm.dx = 1.0;
    std::shared_ptr<Bow::MPM::ElasticityOp<T, dim>> material1;
    if constexpr (Fbased)
        material1 = mpm.create_elasticity(new Bow::MPM::FixedCorotatedOp<T, dim>(10000, 0.4));
    else
        material1 = mpm.create_elasticity(new Bow::MPM::EquationOfStateOp<T, dim>(10000, 7));

    mpm.add_particles(material1, Bow::Vector<T, dim>::Zero(), Bow::Vector<T, dim>::Ones() * mpm.dx, Bow::Vector<T, dim>::Zero());

    mpm.grid.sortParticles(mpm.m_X, mpm.dx);
    if constexpr (Fbased)
        dynamic_cast<Bow::MPM::FixedCorotatedOp<T, dim>*>(material1.get())->m_F *= 0.5;
    else {
        auto& m_J = dynamic_cast<Bow::MPM::EquationOfStateOp<T, dim>*>(material1.get())->m_J;
        for (size_t i = 0; i < m_J.size(); ++i)
            m_J[i] *= std::abs(Bow::Vector<T, 1>::Random()(0)) * 10 + 1e-3;
    }

    Bow::MPM::ParticlesToGridOp<T, dim, false> p2g{ {}, mpm.m_X, mpm.m_V, mpm.m_mass, mpm.m_C, mpm.stress, mpm.grid, mpm.dx, mpm.suggested_dt };
    p2g();
    Bow::MPM::ImplicitBoundaryConditionUpdateOp<T, dim> bc_update{ {}, mpm.grid, mpm.BC, mpm.BC_basis, mpm.BC_order, mpm.dx };
    bc_update();
    mpm.gravity = Bow::Vector<T, dim>::Random();
    Bow::MPM::InertialEnergy<T, dim, int> inertial_energy(mpm.grid, mpm.m_x_tilde);
    Bow::MPM::GravityForceEnergy<T, dim, int> gravity_energy(mpm.grid, mpm.gravity, mpm.dx);
    Bow::MPM::ElasticityEnergy<T, dim, int> elasticity_energy(mpm.grid, mpm.m_X, mpm.elasticity_models, mpm.dx);
    Bow::MPM::TimeIntegratorUpdateOp<T, dim, int> implicit_mpm(mpm.grid, mpm.m_X, mpm.BC_basis, mpm.BC_order, mpm.m_x_tilde, mpm.dx, mpm.suggested_dt);
    implicit_mpm.m_energy_terms.push_back(&inertial_energy);
    implicit_mpm.m_energy_terms.push_back(&gravity_energy);
    implicit_mpm.m_energy_terms.push_back(&elasticity_energy);
    implicit_mpm.update_predictive_pos();
    implicit_mpm.update_transformation_matrix();
    const auto f = [&](const Eigen::VectorXd& x) -> double {
        implicit_mpm.precompute(x);
        return implicit_mpm.energy(x);
    };
    const auto g = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        implicit_mpm.precompute(x);
        implicit_mpm.gradient(x, grad);
    };
    const auto h = [&](const Eigen::VectorXd& x, Eigen::SparseMatrix<double>& hess) {
        implicit_mpm.precompute(x);
        implicit_mpm.hessian(x, hess, false);
    };

    Bow::Field<Bow::Vector<T, dim>> x_hat(mpm.grid.num_nodes, Bow::Vector<T, dim>::Zero());
    // modify initial v_n to satisfy BC
    mpm.grid.iterateGrid([&](const Bow::Vector<int, dim>& node, Bow::MPM::GridState<T, dim>& g) {
        Bow::Vector<T, dim> x_n = node.template cast<T>() * mpm.dx;
        x_hat[g.idx] = x_n;
    });
    Eigen::VectorXd x = Bow::to_vec(x_hat);
    CHECK(Bow::FiniteDiff::check_gradient(x, f, g, 1e-4, 1e-3));
    CHECK(Bow::FiniteDiff::check_jacobian<Eigen::SparseMatrix<double>>(x, g, h, 1e-4, 1e-3));
}

template <class T, int dim, int interpolation_degree = 2>
void mpm_check_momentum()
{
    Bow::MPM::MPMSimulator<T, dim> mpm;
    mpm.suggested_dt = 1e-2;
    mpm.verbose = false;
    if constexpr (dim == 2)
        mpm.dx = 0.1;
    else
        mpm.dx = 0.5;
    auto material = mpm.create_elasticity(new Bow::MPM::FixedCorotatedOp<T, dim>(0, 0.3));
    mpm.add_particles(material, Bow::Vector<T, dim>::Zero(), Bow::Vector<T, dim>::Ones(), Bow::Vector<T, dim>::Zero());
    for (size_t i = 0; i < mpm.m_X.size(); ++i) {
        mpm.m_V[i] = 100 * (Bow::Vector<T, dim>::Random());
    }

    mpm.template p2g<interpolation_degree>(mpm.suggested_dt);
    Bow::Field<Bow::Matrix<T, dim, dim>> D(mpm.m_X.size(), Bow::Matrix<T, dim, dim>::Zero());
    // Somehow MSVC gets confused when trying to use default capture &, so we capture each var explicitly
    auto compute_D = [&mpm, &D]() {
        tbb::parallel_for(size_t(0), mpm.m_X.size(), [&](int i) {
            const Bow::Vector<T, dim> pos = mpm.m_X[i];
            Bow::BSplineWeights<T, dim, interpolation_degree> spline(pos, mpm.dx);
            Bow::Matrix<T, dim, dim> Dp = Bow::Matrix<T, dim, dim>::Zero();
            mpm.grid.iterateKernel(spline, [&pos, &mpm, &Dp, &D](const auto& node, T w, const auto& dw, auto& g) {
                Bow::Vector<T, dim> xi_minus_xp = node.template cast<T>() * mpm.dx - pos;
                Dp += w * xi_minus_xp * xi_minus_xp.transpose();
            });
            D[i] = Dp;
        });
    };

    if constexpr (dim == 3) {
        Bow::Vector<T, dim> LP_old = Bow::Vector<T, dim>::Zero();
        Bow::Vector<T, dim> LP_new = Bow::Vector<T, dim>::Zero();
        Bow::Vector<T, dim> LG_old = Bow::Vector<T, dim>::Zero();
        Bow::Vector<T, dim> LG_new = Bow::Vector<T, dim>::Zero();

        Bow::Vector<T, dim> PP_old = Bow::Vector<T, dim>::Zero();
        Bow::Vector<T, dim> PP_new = Bow::Vector<T, dim>::Zero();
        Bow::Vector<T, dim> PG_old = Bow::Vector<T, dim>::Zero();
        Bow::Vector<T, dim> PG_new = Bow::Vector<T, dim>::Zero();

        for (size_t i = 0; i < mpm.m_X.size(); ++i) {
            LP_new += mpm.m_mass[i] * mpm.m_X[i].cross(mpm.m_V[i]);
            PP_new += mpm.m_mass[i] * mpm.m_V[i];
        }

        for (int i = 0; i < 10; ++i) {
            LP_old.setZero();
            compute_D();
            for (size_t i = 0; i < mpm.m_X.size(); ++i) {
                Bow::Matrix<T, dim, dim> Bp = mpm.m_C[i] * D[i];
                LP_old += mpm.m_mass[i] * mpm.m_X[i].cross(mpm.m_V[i]);
                LP_old[2] -= mpm.m_mass[i] * (Bp(0, 1) - Bp(1, 0));
                LP_old[1] -= mpm.m_mass[i] * (Bp(2, 0) - Bp(0, 2));
                LP_old[0] -= mpm.m_mass[i] * (Bp(1, 2) - Bp(2, 1));
            }

            LP_new = Bow::Vector<T, dim>::Zero();
            LG_old = Bow::Vector<T, dim>::Zero();
            LG_new = Bow::Vector<T, dim>::Zero();
            PP_old = PP_new;
            PP_new = Bow::Vector<T, dim>::Zero();
            PG_old = Bow::Vector<T, dim>::Zero();
            PG_new = Bow::Vector<T, dim>::Zero();

            T dt = mpm.suggested_dt;
            mpm.template p2g<interpolation_degree>(dt);
            mpm.template grid_update<interpolation_degree>(dt);
            compute_D();
            mpm.template g2p<interpolation_degree>(dt);

            mpm.grid.iterateGridSerial([&](const Bow::Vector<int, dim>& node, Bow::MPM::GridState<T, dim>& g) {
                Bow::Vector<T, dim> new_v = g.v_and_m.template segment<dim>(0);
                Bow::Vector<T, dim> old_v = g.old_v;
                Bow::Vector<T, dim> old_x = node.template cast<T>() * mpm.dx;
                LG_new += g.v_and_m(dim) * g.x.cross(new_v);
                LG_old += g.v_and_m(dim) * old_x.cross(old_v);
                PG_new += g.v_and_m(dim) * (new_v);
                PG_old += g.v_and_m(dim) * (old_v);
            });
            for (size_t i = 0; i < mpm.m_X.size(); ++i) {
                Bow::Matrix<T, dim, dim> Bp = mpm.m_C[i] * D[i];
                LP_new += mpm.m_mass[i] * mpm.m_X[i].cross(mpm.m_V[i]);
                LP_new[2] -= mpm.m_mass[i] * (Bp(0, 1) - Bp(1, 0));
                LP_new[1] -= mpm.m_mass[i] * (Bp(2, 0) - Bp(0, 2));
                LP_new[0] -= mpm.m_mass[i] * (Bp(1, 2) - Bp(2, 1));
                PP_new += mpm.m_mass[i] * (mpm.m_V[i]);
            }
            CHECK((PP_old - PG_old).norm() < 1e-8);
            // Bow::Logging::info("Linear Momentum Change in Grid Update: ", (PG_new - PG_old).transpose());
            CHECK((PP_new - PG_new).norm() < 1e-8);
            CHECK((LP_old - LG_old).norm() < 1e-8);
            // Bow::Logging::info("Angular Momentum Change in Grid Update: ", (LG_new - LG_old).transpose());
            CHECK((LP_new - LG_new).norm() < 1e-8);
        }
    }
    else { // dim == 2
        T LP_old = 0;
        T LP_new = 0;
        T LG_old = 0;
        T LG_new = 0;
        Bow::Vector<T, dim> PP_old = Bow::Vector<T, dim>::Zero();
        Bow::Vector<T, dim> PP_new = Bow::Vector<T, dim>::Zero();
        Bow::Vector<T, dim> PG_old = Bow::Vector<T, dim>::Zero();
        Bow::Vector<T, dim> PG_new = Bow::Vector<T, dim>::Zero();

        for (size_t i = 0; i < mpm.m_X.size(); ++i) {
            LP_new += mpm.m_mass[i] * (mpm.m_X[i](0) * mpm.m_V[i](1) - mpm.m_X[i](1) * mpm.m_V[i](0));
            PP_new += mpm.m_mass[i] * mpm.m_V[i];
        }

        for (int i = 0; i < 10; ++i) {
            LP_old = 0;
            compute_D();
            for (size_t i = 0; i < mpm.m_X.size(); ++i) {
                Bow::Matrix<T, dim, dim> Bp = mpm.m_C[i] * D[i];
                LP_old += mpm.m_mass[i] * (mpm.m_X[i](0) * mpm.m_V[i](1) - mpm.m_X[i](1) * mpm.m_V[i](0));
                LP_old -= mpm.m_mass[i] * (Bp(0, 1) - Bp(1, 0));
            }
            LP_new = 0;
            LG_old = 0;
            LG_new = 0;
            PP_old = PP_new;
            PP_new = Bow::Vector<T, dim>::Zero();
            PG_old = Bow::Vector<T, dim>::Zero();
            PG_new = Bow::Vector<T, dim>::Zero();

            T dt = mpm.suggested_dt;
            mpm.template p2g<interpolation_degree>(dt);
            mpm.template grid_update<interpolation_degree>(dt);
            compute_D();
            mpm.template g2p<interpolation_degree>(dt);

            mpm.grid.iterateGridSerial([&](const Bow::Vector<int, dim>& node, Bow::MPM::GridState<T, dim>& g) {
                Bow::Vector<T, dim> new_v = g.v_and_m.template segment<dim>(0);
                Bow::Vector<T, dim> old_v = g.old_v;
                Bow::Vector<T, dim> old_x = node.template cast<T>() * mpm.dx;
                LG_new += g.v_and_m(dim) * (g.x(0) * new_v(1) - g.x(1) * new_v(0));
                LG_old += g.v_and_m(dim) * (old_x(0) * old_v(1) - old_x(1) * old_v(0));
                PG_new += g.v_and_m(dim) * (new_v);
                PG_old += g.v_and_m(dim) * (old_v);
            });
            for (size_t i = 0; i < mpm.m_X.size(); ++i) {
                Bow::Matrix<T, dim, dim> Bp = mpm.m_C[i] * D[i];
                LP_new += mpm.m_mass[i] * (mpm.m_X[i](0) * mpm.m_V[i](1) - mpm.m_X[i](1) * mpm.m_V[i](0));
                LP_new -= mpm.m_mass[i] * (Bp(0, 1) - Bp(1, 0));
                PP_new += mpm.m_mass[i] * (mpm.m_V[i]);
            }
            CHECK((PP_old - PG_old).norm() < 1e-8);
            // Bow::Logging::info("Linear Momentum Change in Grid Update: ", (PG_new - PG_old).transpose());
            CHECK((PP_new - PG_new).norm() < 1e-8);
            CHECK(std::abs(LP_old - LG_old) < 1e-8);
            // Bow::Logging::info("Angular Momentum Change in Grid Update: ", (LG_new - LG_old));
            CHECK(std::abs(LP_new - LG_new) < 1e-8);
        }
    }
}

TEST_CASE("MPM grid test", "[MPM-grid]")
{
    grid_kernel_test<double, 2>();
}

TEST_CASE("MPM implicit diff test", "[MPM]")
{
    GENERATE(1, 2, 3, 4, 5);
    mpm_test<double, 2>();
    mpm_test<double, 3>();
    mpm_test<double, 2, false>();
    mpm_test<double, 3, false>();
}

TEST_CASE("Check momentum", "[MPM-momentum]")
{
    mpm_check_momentum<double, 2, 1>();
    mpm_check_momentum<double, 3, 1>();
    mpm_check_momentum<double, 2, 2>();
    mpm_check_momentum<double, 3, 2>();
    mpm_check_momentum<double, 2, 3>();
    // mpm_check_momentum<double, 3, 3>();
}
