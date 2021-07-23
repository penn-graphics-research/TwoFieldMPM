#ifndef AUGMENTED_LAGRANGIAN_H
#define AUGMENTED_LAGRANGIAN_H

#include <Bow/Macros.h>
#include <Eigen/Eigen>
#include <Bow/Utils/Timer.h>
#include <Eigen/SparseCholesky>
#include <Bow/Utils/Logging.h>
#include <Bow/Math/LinearSolver/SparseCholesky.h>
#include "Newton.h"
#include <functional>
#include <Bow/Types.h>

namespace Bow::Optimization {
template <class Scalar, int dim, class StorageIndex, class Newton>
class AugmentedLagrangianExtension : public Newton {
public:
    using Base = Newton;
    using Vec = Bow::Vector<Scalar, Eigen::Dynamic>;
    using Mat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;

    Vec constraint_weight;
    Vec lambda;
    Scalar kappa = 1e6;

    bool BC_satisfied = false;

    virtual void search_direction(const Vec& x_vec, const Vec& original_grad, Vec& direction) override
    {
        Vec cons;
        Mat cons_jac;
        constraint(x_vec, cons);
        constraint_jacobian(x_vec, cons_jac);
        cons = constraint_weight.asDiagonal() * cons;
        cons_jac = constraint_weight.asDiagonal() * cons_jac;
        BOW_ASSERT(constraint_weight.size() == lambda.size());
        BOW_ASSERT(cons.size() == lambda.size());
        Vec grad = original_grad;
        if (!BC_satisfied && lambda.size() > 0)
            grad += -cons_jac.transpose() * lambda + kappa * cons_jac.transpose() * cons;
        Mat hess;
        {
            BOW_TIMER_FLAG("Compute Hessian");
            this->hessian(x_vec, hess, this->project_pd);
            if (!BC_satisfied && lambda.size() > 0)
                hess += kappa * cons_jac.transpose() * cons_jac; // drop 3-order tensors
        }
        direction = -this->linear_system(hess, grad);
    }

    virtual Scalar backtracking(const Vec& direction, Scalar& alpha, Vec& xn)
    {
        auto aug_lag_energy = [&](Vec& x) -> Scalar {
            if (lambda.size() == 0) return 0;
            Vec cons;
            constraint(x, cons);
            cons = constraint_weight.asDiagonal() * cons;
            if (!BC_satisfied)
                return -lambda.dot(cons) + 0.5 * kappa * cons.squaredNorm();
            else
                return 0;
        };
        Scalar E0 = this->energy(xn) + aug_lag_energy(xn);
        Vec new_x = xn + alpha * direction;
        this->precompute(new_x);
        Scalar E = this->energy(new_x) + aug_lag_energy(new_x);
        int line_search_it = 0;
        if (this->line_search) {
            BOW_TIMER_FLAG(this->method + " Line Search");
            while (E > E0) {
                alpha *= 0.5;
                new_x = xn + alpha * direction;
                this->precompute(new_x);
                E = this->energy(new_x) + aug_lag_energy(new_x);
                line_search_it++;
                if (line_search_it > 20) {
                    Logging::warn("Backtracked too many times! Current line_search_it: ", line_search_it);
                    if (line_search_it > 100)
                        BOW_ASSERT_INFO(false, "Not a descent direction. Go debug your math!")
                }
            }
        }
        xn = new_x;
        return E;
    }

    virtual void update_rule(const Vec& x, const Scalar& res)
    {
        Vec cons;
        constraint(x, cons);
        if (cons.size() == 0) return;
        Scalar cons_res = constraint_residual(x, cons);
        Logging::info("Constraint residual: ", cons_res);
        if (res < 1e-1 && cons_res > 1e-2) {
            // when optimization residual is small so augLag update won't explode,
            // and when cons_res is still not well satisfied:
            if (kappa < 1e8) { // if kappa is still not too large, make it stiffer
                kappa *= 2;
            }
            else { // if kappa is already very large, make dual updates
                lambda -= kappa * constraint_weight.asDiagonal() * cons;
            }
        }
    }

    virtual void initialize_optimizer(const Vec& x)
    {
        set_constraint_weight(constraint_weight);
        Vec cons;
        constraint(x, cons);
        if (cons.size() == 0) return;
        if (constraint_weight.size() != cons.size()) {
            constraint_weight = cons;
            constraint_weight.setOnes();
            Logging::warn("Constraint weights are set to ones.");
        }
        lambda.resize(constraint_weight.size());
        lambda.setZero();
    }

    template <class DerivedX>
    int optimize(Eigen::MatrixBase<DerivedX>& x)
    {
        BOW_TIMER_FLAG(this->method);
        initialize_optimizer(x);
        this->iter_num = 0;
        Bow::Vector<Scalar, Eigen::Dynamic> xn = x;
        this->precompute(xn);
        bool tol_satisfied = false;
        for (; this->iter_num < this->max_iter; ++this->iter_num) {
            this->callback(xn);
            Vec direction, grad;
            this->gradient(xn, grad);
            this->search_direction(xn, grad, direction);
            this->push_forward(direction);
            Scalar res = this->residual(xn, grad, direction);
            Vec cons;
            constraint(xn, cons);
            Scalar constraint_res = constraint_residual(xn, cons);
            if (res < this->tol && constraint_res == 0) {
                static int total_iter = 0;
                total_iter += this->iter_num;
                Bow::Logging::info(this->method + " Converged iter: ", this->iter_num, ",\tGrad_inf: ", grad.cwiseAbs().maxCoeff(), ",\tResidual: ", res, ",\tTotal iter: ", total_iter);
                if (res < this->tol)
                    tol_satisfied = true;
                break;
            }
            Scalar alpha0 = this->initial_stepsize(xn, direction);
            Scalar alpha = alpha0;
            Vec xn0 = xn;
            Scalar E = backtracking(direction, alpha, xn);
            if (std::isnan(std::abs(E)))
                BOW_ASSERT_INFO(false, "FPE!")
            if (std::log2(alpha0 / alpha) > 30) {
                Bow::Logging::warn(this->method + " tiny stepsize!");
                if (this->try_fixed_stepsize) {
                    xn = xn0 + 0.1 * alpha0 * direction;
                    this->precompute(xn);
                }
                else if (this->return_if_backtracking_fails)
                    break;
            }
            if (this->verbose)
                Bow::Logging::info(this->method + " Iter: ", this->iter_num, ",\t E: ", E, ", \tResidual: ", res, ",\tStep size: ", alpha, ",\tInitial step size: ", alpha0);
            update_rule(xn, res);
        }
        if (!tol_satisfied)
            Bow::Logging::warn(this->method + " tolerance is not reached!");
        x = xn;
        return this->iter_num;
    }

    bool requires_feasible_bc_init()
    {
        return false;
    }

protected: // to be defined by user
    virtual Scalar constraint_residual(const Vec& x, const Vec& cons)
    {
        if (cons.cwiseAbs().maxCoeff() < 1e-6)
            return 0;
        else
            return cons.cwiseAbs().maxCoeff();
    }
    virtual void set_constraint_weight(Vec& weight) {}
    virtual void constraint(const Vec& x_vec, Vec& cons) {}
    virtual void constraint_jacobian(const Vec& x_vec, Mat& jac) {}
};

template <class Scalar, int dim, class StorageIndex>
using AugmentedLagrangianNewton = AugmentedLagrangianExtension<Scalar, dim, StorageIndex, Newton<Scalar, dim, StorageIndex>>;

} // namespace Bow::Optimization

#endif