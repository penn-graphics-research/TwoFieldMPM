#pragma once

#include <Bow/Geometry/Hybrid/MPMTransfer.h>
#include <Bow/Geometry/Hybrid/MPMGrid.h>
#include <Bow/Math/LinearSolver/SparseQR.h>
#include <Bow/Math/LinearSolver/SparseCholesky.h>
#include <Bow/Utils/Timer.h>
#include <Bow/Utils/Logging.h>
#include <Bow/Utils/FiniteDiff.h>
#include <Bow/Optimization/Newton.h>
#include <tbb/tbb.h>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

using namespace SPGrid;

namespace Bow {
namespace MPM {

template <class T, int dim, class StorageIndex = int, int interpolation_degree = 2>
class TimeIntegratorUpdateOp : public Optimization::Newton<T, dim, StorageIndex> {
public:
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    using Vec = Bow::Vector<T, Eigen::Dynamic>;
    using Mat = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;
    using Base = Optimization::Newton<T, dim, StorageIndex>;

    MPMGrid<T, dim>& grid;
    Field<Vector<T, dim>>& particle_X;

    // https://bow.readthedocs.io/en/latest/boundary_condition.html
    // BC_basis is V in doc, which is [v_n, v_m, v_l]
    Field<Matrix<T, dim, dim>>& BC_basis;
    Field<int>& BC_order;

    Field<Vector<T, dim>>& m_x_tilde;

    T dx;
    T dt;
    TV gravity = TV::Zero();

    Mat m_transform_matrix;
    bool project_dirichlet = true;

    TSMethod tsMethod = BE;
    T tsParam[2][3] = {
        { 1, 0.5, 1 },
        { 0.5, 0.25, 0.5 }
    };

    TimeIntegratorUpdateOp(MPMGrid<T, dim>& grid, Field<Vector<T, dim>>& particle_X, Field<Matrix<T, dim, dim>>& BC_basis, Field<int>& BC_order, Field<Vector<T, dim>>& m_x_tilde, T dx, T dt)
        : grid(grid), particle_X(particle_X), BC_basis(BC_basis), BC_order(BC_order), m_x_tilde(m_x_tilde), dx(dx), dt(dt) {}

    void update_transformation_matrix()
    {
        // construct transform matrix
        std::vector<StorageIndex> ptr(BC_basis.size() * dim + 1);
        ptr[0] = 0;
        std::vector<StorageIndex> row(BC_basis.size() * dim * dim);
        for (size_t i = 0; i < BC_basis.size(); ++i) {
            for (int d1 = 0; d1 < dim; ++d1) {
                ptr[i * dim + d1 + 1] = ptr[i * dim + d1] + dim;
                for (int d2 = 0; d2 < dim; ++d2)
                    row[i * dim * dim + d1 * dim + d2] = i * dim + d2;
            }
        }
        std::vector<T> val(dim * dim * BC_basis.size());
        memcpy(val.data(), reinterpret_cast<const T*>(BC_basis.data()), sizeof(T) * dim * dim * BC_basis.size());
        m_transform_matrix.setZero();
        Math::sparse_from_csr(ptr, row, val, BC_basis.size() * dim, BC_basis.size() * dim, m_transform_matrix);
    }

    void update_predictive_pos()
    {
        m_x_tilde.resize(grid.num_nodes);
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            auto idx = g.idx;
            TV x_n = node.template cast<T>() * dx;
            TV v_n = g.v_and_m.template topLeftCorner<dim, 1>();
            TV a_n = g.a;
            TV x_tilde;
            if (tsMethod == BDF2) {
                BOW_NOT_IMPLEMENTED
            }
            else {
                x_tilde = x_n + v_n * dt + tsParam[tsMethod][0] * (1 - 2 * tsParam[tsMethod][1]) * a_n * dt * dt;
            }
            m_x_tilde[idx] = x_tilde;
        });
    }

    void apply_newton_result(const Field<Vector<T, dim>>& x_hat)
    {
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            g.x = x_hat[g.idx];
            TV x_n = node.template cast<T>() * dx;
            TV v_n = g.v_and_m.template topLeftCorner<dim, 1>();
            TV a_n = g.a;
            TV x_tilde = x_n + v_n * dt + tsParam[tsMethod][0] * (1 - 2 * tsParam[tsMethod][1]) * a_n * dt * dt;
            Bow::Vector<T, dim> new_ai = (g.x - x_tilde) / (2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt);
            g.v_and_m.template topLeftCorner<dim, 1>() += dt * ((1 - tsParam[tsMethod][2]) * a_n + tsParam[tsMethod][2] * new_ai);
        });
    }

    T residual(const Vec& x, const Vec& grad, const Vec& direction)
    {
        Vec grid_residual = direction;
        Vec particle_residual(dim * particle_X.size());
        particle_residual.setZero();
        tbb::parallel_for(0, (int)particle_X.size(), [&](int i) {
            Vector<T, dim>& Xp = particle_X[i];
            BSplineWeights<T, dim, interpolation_degree> spline(Xp, dx);
            Vector<T, dim> rp = Vector<T, dim>::Zero();
            grid.iterateKernel(spline, [&](const Vector<int, dim>& node, T w, Vector<T, dim> dw, MPM::GridState<T, dim>& g) {
                rp += grid_residual.template segment<dim>(dim * g.idx) * w;
            });
            particle_residual.template segment<dim>(dim * i) = rp;
        });
        return particle_residual.cwiseAbs().maxCoeff() / dt;
    }

    void initialize_acceleration()
    {
        Field<Vector<T, dim>> xn(grid.num_nodes, Vector<T, dim>::Zero());
        Field<Vector<T, dim>> vn(grid.num_nodes, Vector<T, dim>::Zero());
        Field<Vector<T, dim>> force(grid.num_nodes, Vector<T, dim>::Zero());
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            xn[g.idx] = node.template cast<T>() * dx;
            vn[g.idx] = g.v_and_m.template segment<dim>(0);
        });

        for (auto energy : this->m_energy_terms) {
            Field<Vector<T, dim>> sub_force;
            energy->internal_force(xn, vn, sub_force);
            force -= sub_force;
        }

        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            g.a = force[g.idx] / g.v_and_m(dim);
        });
    }

    virtual void gradient(const Vec& x, Vec& grad) override
    {
        Base::gradient(x, grad);
        grad = m_transform_matrix.transpose() * grad;
        if (project_dirichlet) {
            tbb::parallel_for((size_t)0, BC_order.size(), [&](size_t i) {
                if (BC_order[i] > 0) {
                    for (int d = 0; d < BC_order[i]; ++d)
                        grad(i * dim + d) = 0;
                }
            });
        }
    }
    virtual void hessian(const Vec& x, Mat& hess, const bool project_pd) override
    {
        Base::hessian(x, hess, project_pd);
        hess = m_transform_matrix.transpose() * hess * m_transform_matrix;
        if (project_dirichlet) {
            tbb::parallel_for(0, (int)BC_order.size(), [&](int i) {
                for (int d = 0; d < dim; ++d) {
                    bool clear_col = d < BC_order[i];
                    for (typename Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>::InnerIterator it(hess, dim * i + d); it; ++it) {
                        bool clear_row = (it.row() % dim) < BC_order[it.row() / dim];
                        if (clear_row || clear_col) {
                            if (it.col() == it.row())
                                it.valueRef() = 1;
                            else
                                it.valueRef() = 0;
                        }
                    }
                }
            });
        }
    }

    void project(Vec& b) override
    {
        for (int i = 0; i < grid.num_nodes; ++i)
            for (int d = 0; d < BC_order[i]; ++d)
                b(i * dim + d) = 0;
    }

    void push_forward(Vec& direction) override
    {
        direction = m_transform_matrix * direction;
    }

    void set_ts_weights()
    {
        for (auto energy : this->m_energy_terms) {
            std::string method = energy->name_hierarchy()[0];
            std::string name = energy->name_hierarchy()[1];
            if (name.compare("Inertia") == 0)
                energy->energy_scale = T(1);
            else {
                if (tsMethod == BDF2) {
                    BOW_NOT_IMPLEMENTED
                }
                else {
                    energy->energy_scale = 2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt;
                }
            }
        }
    }

    Vec initial_direction_yielding_to_BC()
    {
        Field<Vector<T, dim>> direction(grid.num_nodes);
        grid.iterateGrid([&](const Vector<int, dim>& node, MPM::GridState<T, dim>& g) {
            direction[g.idx] = gravity * dt;
            for (int d = 0; d < BC_order[g.idx]; ++d) {
                TV n = BC_basis[g.idx].col(d);
                direction[g.idx] -= direction[g.idx].dot(n) * n;
            }
        });
        return to_vec(direction);
    }

    void operator()()
    {
        BOW_TIMER_FLAG("newton one step");
        Field<Vector<T, dim>> x_hat(grid.num_nodes, Vector<T, dim>::Zero());
        Field<Vector<T, dim>> xn(grid.num_nodes, Vector<T, dim>::Zero());
        update_transformation_matrix();
        set_ts_weights();
        initialize_acceleration();
        update_predictive_pos();
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            TV x_n = node.template cast<T>() * dx;
            x_hat[g.idx] = x_n;
        });
        Vec x = to_vec(x_hat);
        this->precompute(x);
        Vec initial_direction = initial_direction_yielding_to_BC();
        x += initial_direction * this->initial_stepsize(x, initial_direction);
        this->optimize(x);
        x_hat = to_field<dim>(x);
        apply_newton_result(x_hat);
    }
};
}
} // namespace Bow::MPM