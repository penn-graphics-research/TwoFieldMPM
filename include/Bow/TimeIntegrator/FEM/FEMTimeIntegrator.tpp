#include "FEMTimeIntegrator.h"
#include <Bow/Optimization/Newton.h>
#include <iostream>
#include <Bow/Utils/Timer.h>
#include <oneapi/tbb.h>
#include <fstream>
#include <Bow/Math/Utils.h>

namespace Bow {
namespace FEM {

/* TimeIntegratorUpdateOp */

template <class T, int dim, class _StorageIndex, class Optimizer>
TimeIntegratorUpdateOp<T, dim, _StorageIndex, Optimizer>::TimeIntegratorUpdateOp(const Field<Matrix<T, dim, dim>>& BC_basis, const Field<int>& BC_order, const Field<Vector<T, dim>>& BC_target, const Field<uint8_t>& BC_fixed, Field<T>& mass, Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& v, Field<Vector<T, dim>>& a, Field<Vector<T, dim>>& x1, Field<Vector<T, dim>>& v1, Field<Vector<T, dim>>& x_tilde)
    : BC_basis(BC_basis)
    , BC_order(BC_order)
    , BC_target(BC_target)
    , BC_fixed(BC_fixed)
    , m_x(x)
    , m_v(v)
    , m_a(a)
    , m_x1(x1)
    , m_v1(v1)
    , m_x_tilde(x_tilde)
    , m_mass(mass)
{
    update_transformation_matrix();
}

template <class T, int dim, class StorageIndex, class Optimizer>
void TimeIntegratorUpdateOp<T, dim, StorageIndex, Optimizer>::update_transformation_matrix()
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

template <class T, int dim, class StorageIndex, class Optimizer>
void TimeIntegratorUpdateOp<T, dim, StorageIndex, Optimizer>::initialize_acceleration()
{
    Field<Vector<T, dim>> force(m_x.size(), Vector<T, dim>::Zero());
    for (auto energy : this->m_energy_terms) {
        Field<Vector<T, dim>> sub_force;
        energy->internal_force(m_x, m_v, sub_force);
        force -= sub_force;
    }
    tbb::parallel_for((size_t)0, m_x.size(), [&](size_t i) {
        m_a[i] = force[i] / m_mass[i];
    });
}

template <class T, int dim, class StorageIndex, class Optimizer>
void TimeIntegratorUpdateOp<T, dim, StorageIndex, Optimizer>::update_predictive_pos()
{
    m_x_tilde.resize(m_x.size());
    if (tsMethod == BDF2) {
        tbb::parallel_for((size_t)0, m_x_tilde.size(), [&](size_t i) {
            m_x_tilde[i] = m_x[i] * 4.0 / 3.0 - m_x1[i] / 3.0 + (m_v[i] * 4 - m_v1[i]) * dt * 2.0 / 9.0;
        });
    }
    else {
        tbb::parallel_for((size_t)0, m_x_tilde.size(), [&](size_t i) {
            m_x_tilde[i] = m_x[i] + m_v[i] * dt + tsParam[tsMethod][0] * (1 - 2 * tsParam[tsMethod][1]) * m_a[i] * dt * dt;
        });
    }
}

template <class T, int dim, class StorageIndex, class Optimizer>
void TimeIntegratorUpdateOp<T, dim, StorageIndex, Optimizer>::project(Vec& x)
{
    x = m_transform_matrix.transpose() * x;
    tbb::parallel_for((size_t)0, m_x.size(), [&](size_t i) {
        if (BC_order[i] > 0) {
            for (int d = 0; d < BC_order[i]; ++d)
                x(i * dim + d) = 0;
        }
    });
    x = m_transform_matrix * x;
}

template <class T, int dim, class StorageIndex, class Optimizer>
void TimeIntegratorUpdateOp<T, dim, StorageIndex, Optimizer>::set_ts_weights()
{
    for (auto energy : this->m_energy_terms) {
        std::string name = energy->name_hierarchy()[1];
        if (name.compare("Inertia") == 0)
            energy->energy_scale = T(1);
        else {
            if (tsMethod == BDF2) {
                energy->energy_scale = dt * dt * 4.0 / 9.0;
            }
            else {
                energy->energy_scale = 2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt;
            }
        }
    }
}

template <class T, int dim, class StorageIndex, class Optimizer>
void TimeIntegratorUpdateOp<T, dim, StorageIndex, Optimizer>::gradient(const Vec& x, Vec& grad)
{
    Base::gradient(x, grad);
    grad = m_transform_matrix.transpose() * grad;
    if (project_dirichlet) {
        tbb::parallel_for((size_t)0, m_x.size(), [&](size_t i) {
            if (BC_order[i] > 0) {
                for (int d = 0; d < BC_order[i]; ++d)
                    grad(i * dim + d) = 0;
            }
        });
    }
    else {
        // only project DBC with 0 velocity
        tbb::parallel_for((size_t)0, m_x.size(), [&](size_t i) {
            if (BC_fixed[i] > 0) {
                for (int d = 0; d < BC_order[i]; ++d)
                    grad(i * dim + d) = 0;
            }
        });
    }
}

template <class T, int dim, class StorageIndex, class Optimizer>
void TimeIntegratorUpdateOp<T, dim, StorageIndex, Optimizer>::hessian(const Vec& x, Mat& hess, const bool project_pd)
{
    Base::hessian(x, hess, project_pd);
    hess = m_transform_matrix.transpose() * hess * m_transform_matrix;
    if (project_dirichlet) {
        tbb::parallel_for(0, (int)m_x.size(), [&](int i) {
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
    else {
        tbb::parallel_for(0, (int)m_x.size(), [&](int i) {
            if (BC_fixed[i] > 0) {
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
            }
        });
    }
}

template <class T, int dim, class StorageIndex, class Optimizer>
T TimeIntegratorUpdateOp<T, dim, StorageIndex, Optimizer>::residual(const Vec& x, const Vec& grad, const Vec& direction)
{
    return direction.cwiseAbs().maxCoeff() / this->dt;
}

template <class T, int dim, class StorageIndex, class Optimizer>
void TimeIntegratorUpdateOp<T, dim, StorageIndex, Optimizer>::push_forward(Vec& y)
{
    y = m_transform_matrix * y;
}

template <class T, int dim, class StorageIndex, class Optimizer>
void TimeIntegratorUpdateOp<T, dim, StorageIndex, Optimizer>::advance(const Vec& new_x)
{
    if (tsMethod == BDF2) {
        tbb::parallel_for(0, (int)m_x.size(), [&](int i) {
            m_x1[i] = m_x[i];
            m_x[i] = new_x.template segment<dim>(i * dim);
            m_a[i] = (m_x[i] - m_x_tilde[i]) / (dt * dt * 4.0 / 9.0);
            Bow::Vector<T, dim> temp = m_v[i];
            m_v[i] = m_v[i] * 4.0 / 3.0 - m_v1[i] / 3.0 + m_a[i] * dt * 2.0 / 3.0;
            m_v1[i] = temp;
        });
    }
    else {
        tbb::parallel_for(0, (int)m_x.size(), [&](int i) {
            m_x[i] = new_x.template segment<dim>(i * dim);
        });

        tbb::parallel_for(0, (int)m_x.size(), [&](int i) {
            Bow::Vector<T, dim> new_ai = (m_x[i] - m_x_tilde[i]) / (2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt);
            m_v[i] += dt * ((1 - tsParam[tsMethod][2]) * m_a[i] + tsParam[tsMethod][2] * new_ai);
            m_a[i] = new_ai;
        });
    }
}

template <class T, int dim, class StorageIndex, class Optimizer>
T TimeIntegratorUpdateOp<T, dim, StorageIndex, Optimizer>::constraint_residual(const Vec& y, const Vec& cons)
{
    if (!project_dirichlet) {
        // https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf Eq (5)
        T num = 0;
        T den = 0;
        int index = 0;
        for (size_t i = 0; i < BC_order.size(); ++i)
            if (BC_order[i] > 0)
                for (int d = 0; d < BC_order[i]; ++d) {
                    num += std::pow(cons(index), 2);
                    den += std::pow(BC_basis[i].col(d).dot(m_x[i]) - BC_target[i](d), 2);
                    index++;
                }
        if (den == 0)
            return std::sqrt(num);
        else
            return std::sqrt(num / den);
    }
    return 0;
}

template <class T, int dim, class StorageIndex, class Optimizer>
void TimeIntegratorUpdateOp<T, dim, StorageIndex, Optimizer>::callback(Vec& x)
{
    Base::callback(x);
    Vec cons;
    constraint(x, cons);
    T res = constraint_residual(x, cons);
    if (res < 1e-2) {
        project_dirichlet = true;
        this->BC_satisfied = true;
        //NOTE: don't need to set x according to BC_target here
        //because it can cause infeasibility due to numerical error!
        Logging::info("Dirichlet BC is satisfied.");
    }
}

template <class T, int dim, class StorageIndex, class Optimizer>
void TimeIntegratorUpdateOp<T, dim, StorageIndex, Optimizer>::set_constraint_weight(Vec& weight)
{
    std::vector<T> weight_array;
    for (size_t i = 0; i < BC_order.size(); ++i)
        if (BC_order[i] > 0)
            for (int d = 0; d < BC_order[i]; ++d)
                weight_array.push_back(std::sqrt(m_mass[i]));
    weight = Eigen::Map<Vec>(weight_array.data(), weight_array.size());
}

template <class T, int dim, class StorageIndex, class Optimizer>
void TimeIntegratorUpdateOp<T, dim, StorageIndex, Optimizer>::constraint(const Vec& x, Vec& cons)
{
    Vec y = m_transform_matrix.transpose() * x;
    std::vector<T> cons_array;
    for (size_t i = 0; i < BC_order.size(); ++i)
        if (BC_order[i] > 0)
            for (int d = 0; d < BC_order[i]; ++d)
                cons_array.push_back(y(i * dim + d) - BC_target[i](d));
    cons = Eigen::Map<Vec>(cons_array.data(), cons_array.size());
}

template <class T, int dim, class StorageIndex, class Optimizer>
void TimeIntegratorUpdateOp<T, dim, StorageIndex, Optimizer>::constraint_jacobian(const Vec& x, Mat& jac)
{
    using IJK = Eigen::Triplet<T, StorageIndex>;
    std::vector<IJK> triplets;
    int index = 0;
    for (size_t i = 0; i < BC_order.size(); ++i)
        if (BC_order[i] > 0)
            for (int d = 0; d < BC_order[i]; ++d) {
                triplets.emplace_back(index, i * dim + d, 1);
                index++;
            }
    jac.resize(index, x.size());
    jac.setFromTriplets(triplets.begin(), triplets.end());
}

template <class T, int dim, class _StorageIndex, class Optimizer>
inline void TimeIntegratorUpdateOp<T, dim, _StorageIndex, Optimizer>::operator()()
{
    update_transformation_matrix();
    set_ts_weights();
    initialize_acceleration();
    update_predictive_pos();
    if (this->requires_feasible_bc_init()) {
        // modify initial x so that it satisfied the constraint.
        tbb::parallel_for((size_t)0, m_x.size(), [&](size_t i) {
            if (BC_order[i] > 0) {
                m_x[i] = BC_basis[i].transpose() * m_x[i];
                for (int d = 0; d < BC_order[i]; ++d)
                    m_x[i](d) = BC_target[i](d);
                m_x[i] = BC_basis[i] * m_x[i];
            }
        });
    }
    Vector<T, Eigen::Dynamic> new_x = Eigen::Map<Vector<T, Eigen::Dynamic>>(reinterpret_cast<double*>(m_x.data()), dim * m_x.size());
    this->optimize(new_x);
    if (lagging_callback) {
        for (int i = 0; i < max_lag; ++i) {
            lagging_callback();
            if (this->optimize(new_x) == 0) break;
        }
    }
    advance(new_x);
}
}
} // namespace Bow::FEM