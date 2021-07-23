#pragma once

#include <Bow/Math/LinearSolver/SparseQR.h>
#include <Bow/Math/LinearSolver/SparseCholesky.h>
#include <Bow/Utils/Timer.h>
#include <Bow/Utils/Logging.h>
#include <Bow/Utils/FiniteDiff.h>
#include <Bow/Optimization/Newton.h>
#include <tbb/tbb.h>
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

using namespace SPGrid;

namespace Bow {
namespace DFGMPM {

template <class T, int dim, class StorageIndex = int>
class TwoFieldBackwardEulerUpdateOp : public Optimization::Newton<T, dim, StorageIndex> {
public:
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    using Vec = Bow::Vector<T, Eigen::Dynamic>;
    using Mat = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;
    using Base = Optimization::Newton<T, dim, StorageIndex>;
    using Base::direct_solver;

    DFGMPMGrid<T, dim>& grid;
    // https://bow.readthedocs.io/en/latest/boundary_condition.html
    // BC_basis is V in doc, which is [v_n, v_m, v_l]
    Field<Matrix<T, dim, dim>>& BC_basis;
    Field<int>& BC_order;

    T dx;
    T dt;

    bool useDFG;

    TV gravity = TV::Zero();

    Mat m_transform_matrix;
    bool project_dirichlet = true;

    TSMethod tsMethod = BE;
    T tsParam[2][3] = {
        { 1, 0.5, 1 },
        { 0.5, 0.25, 0.5 }
    };

    TwoFieldBackwardEulerUpdateOp(DFGMPMGrid<T, dim>& grid, Field<Matrix<T, dim, dim>>& BC_basis, Field<int>& BC_order, T dx, T dt, bool useDFG)
        : grid(grid), BC_basis(BC_basis), BC_order(BC_order), dx(dx), dt(dt), useDFG(useDFG) {}

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

    void applyNewtonResult(const Field<Vector<T, dim>>& x_hat)
    {
        int ndof = grid.num_nodes;
        int sdof = useDFG ? grid.separable_nodes : 0;
        
        //Field 1
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            g.x1 = x_hat[g.idx];
            TV x_n = node.template cast<T>() * dx;
            TV v_n = g.v1;
            TV a_n = g.a1;
            TV x_tilde = x_n + v_n * dt + tsParam[tsMethod][0] * (1 - 2 * tsParam[tsMethod][1]) * a_n * dt * dt;
            Bow::Vector<T, dim> new_ai = (g.x1 - x_tilde) / (2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt);
            g.v1 += dt * ((1 - tsParam[tsMethod][2]) * a_n + tsParam[tsMethod][2] * new_ai);
        });
        //Field 2
        if(sdof > 0){
            grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
                g.x2 = x_hat[ndof + g.sep_idx];
                TV x_n = node.template cast<T>() * dx;
                TV v_n = g.v2;
                TV a_n = g.a2;
                TV x_tilde = x_n + v_n * dt + tsParam[tsMethod][0] * (1 - 2 * tsParam[tsMethod][1]) * a_n * dt * dt;
                Bow::Vector<T, dim> new_ai = (g.x2 - x_tilde) / (2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt);
                g.v2 += dt * ((1 - tsParam[tsMethod][2]) * a_n + tsParam[tsMethod][2] * new_ai);
            });
        }
    }

    T residual(const Vec& x, const Vec& grad, const Vec& direction)
    {
        return direction.cwiseAbs().maxCoeff() / this->dt;
    }

    // Temporarily remove
    void diff_test_with_matrix(const Vec& x)
    {
        initialize_acceleration();
        update_transformation_matrix();
        const auto f = [&](const Eigen::VectorXd& y) -> double {
            Vec x = m_transform_matrix * y;
            this->precompute(x);
            return this->energy(x);
        };
        const auto g = [&](const Eigen::VectorXd& y, Eigen::VectorXd& grad) {
            Vec x = m_transform_matrix * y;
            this->precompute(x);
            this->gradient(x, grad);
        };
        const auto h = [&](const Eigen::VectorXd& y, Eigen::SparseMatrix<T>& hess) {
            Vec x = m_transform_matrix * y;
            this->precompute(x);
            this->hessian(x, hess, false);
        };
        Eigen::VectorXd y = m_transform_matrix.transpose() * x;
        Eigen::VectorXd dy = Eigen::VectorXd::Random(y.size());
        FiniteDiff::ziran_check_false(y, f, g, h, this->project, nullptr);
    }

    void diff_test_matrix_free(const Vec& x)
    {
        const auto f = [&](const Eigen::VectorXd& y) -> double {
            Vec x = m_transform_matrix * y;
            this->precompute(x);
            return this->energy(x);
        };
        const auto g = [&](const Eigen::VectorXd& y, Eigen::VectorXd& grad) {
            Vec x = m_transform_matrix * y;
            this->precompute(x);
            this->gradient(x, grad);
        };

        const auto h = [&](const Eigen::VectorXd& y, const Eigen::VectorXd& dx, Eigen::VectorXd& Adx) {
            Vec x = m_transform_matrix * y;
            this->precompute(x);
            this->multiply(dx, Adx, false);
            this->project(Adx);
        };
        Eigen::VectorXd y = m_transform_matrix.transpose() * x;
        FiniteDiff::ziran_check_true(y, f, g, h, this->project, nullptr);
    }

    void initialize_acceleration()
    {
        int ndof = grid.num_nodes;
        int sdof = useDFG ? grid.separable_nodes : 0;
        Field<Vector<T, dim>> xn(ndof + sdof, Vector<T, dim>::Zero());
        Field<Vector<T, dim>> vn(ndof + sdof, Vector<T, dim>::Zero());
        Field<Vector<T, dim>> force(ndof + sdof, Vector<T, dim>::Zero());
        
        //Set grid positions and velocities (field 1)
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            xn[g.idx] = node.template cast<T>() * dx;
            vn[g.idx] = g.v1;
        });
        //Set grid positions and velocities (field 2)
        if(sdof > 0){
            grid.iterateSeparableNodes([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
                if (ndof + g.sep_idx < 0) return;
                xn[ndof + g.sep_idx] = node.template cast<T>() * dx;
                vn[ndof + g.sep_idx] = g.v2;
            });
        }
        
        //Compute forces based on our energies
        for (auto energy : this->m_energy_terms) {
            Field<Vector<T, dim>> sub_force;
            energy->internal_force(xn, vn, sub_force);
            force -= sub_force;
        }

        //Set grid acceleration (field 1)
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            g.a1 = force[g.idx] / g.m1;
        });
        //Set grid acceleration (field 2)
        if(sdof > 0){
            grid.iterateSeparableNodes([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
                if (ndof + g.sep_idx < 0) return;
                g.a2 = force[ndof + g.sep_idx] / g.m2;
            });
        }
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

    void multiply(const Vec& x_vec, Vec& Ax_vec, const bool project_pd = true) const
    {
        Vec y = m_transform_matrix * x_vec;
        Base::multiply(x_vec, Ax_vec, project_pd);
        Ax_vec = m_transform_matrix.transpose() * Ax_vec;
    }

    void project(Vec& b) override
    {
        int ndof = grid.num_nodes;
        int sdof = useDFG ? grid.separable_nodes : 0;
        for (int i = 0; i < (ndof + sdof); ++i)
            for (int d = 0; d < BC_order[i]; ++d)
                b(i * dim + d) = 0;
    }

    void push_forward(Vec& direction) override
    {
        direction = m_transform_matrix * direction;
    }

    Vec initial_direction_yielding_to_BC()
    {
        int ndof = grid.num_nodes;
        int sdof = useDFG ? grid.separable_nodes : 0;
        Field<Vector<T, dim>> direction(ndof + sdof);
        grid.iterateGrid([&](const Vector<int, dim>& node, DFGMPM::GridState<T, dim>& g) {
            direction[g.idx] = gravity * dt;
            for (int d = 0; d < BC_order[g.idx]; ++d) {
                TV n = BC_basis[g.idx].col(d);
                direction[g.idx] -= direction[g.idx].dot(n) * n;
            }
        });
        //Now for the separable DOFs
        if(sdof > 0){
            grid.iterateSeparableNodes([&](const Vector<int, dim>& node, DFGMPM::GridState<T, dim>& g) {
                direction[ndof + g.sep_idx] = gravity * dt;
                for (int d = 0; d < BC_order[ndof + g.sep_idx]; ++d) {
                    TV n = BC_basis[ndof + g.sep_idx].col(d);
                    direction[ndof + g.sep_idx] -= direction[ndof + g.sep_idx].dot(n) * n;
                }
            });
        }
        return to_vec(direction);
    }

    void operator()()
    {
        BOW_TIMER_FLAG("Single Newton Iteration");
        int ndof = grid.num_nodes;
        int sdof = useDFG ? grid.separable_nodes : 0;
        Field<Vector<T, dim>> x_hat(ndof + sdof, Vector<T, dim>::Zero());
        Field<Vector<T, dim>> xn(ndof + sdof, Vector<T, dim>::Zero());
        update_transformation_matrix();
        initialize_acceleration();
        
        //Set xHat (field 1)
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            TV x_n = node.template cast<T>() * dx;
            x_hat[g.idx] = x_n;
        });
        //Set xHat (field 2)
        if(sdof > 0){
            grid.iterateSeparableNodes([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
                TV x_n = node.template cast<T>() * dx;
                x_hat[ndof + g.sep_idx] = x_n;
            });
        }

        Vec x = to_vec(x_hat);
        this->precompute(x);
        Vec initial_direction = initial_direction_yielding_to_BC();
        x += initial_direction * this->initial_stepsize(x, initial_direction);
        this->optimize(x);
        x_hat = to_field<dim>(x);
        applyNewtonResult(x_hat);
    }
};
}
} // namespace Bow::MPM