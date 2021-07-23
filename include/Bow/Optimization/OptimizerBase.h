#ifndef OPTIMIZER_BASE
#define OPTIMIZER_BASE
#include <Bow/Macros.h>
#include <Eigen/Eigen>
#include <Bow/Utils/Timer.h>
#include <Eigen/SparseCholesky>
#include <Bow/Utils/Logging.h>
#include <Bow/Math/LinearSolver/SparseCholesky.h>
#include <Bow/Math/LinearSolver/Amgcl.h>
#include <Bow/Energy/Prototypes.h>
#include <functional>
#include <Bow/Types.h>
#include <Bow/Utils/ResultRecorder.h>
#include <Bow/Utils/FiniteDiff.h>

namespace Bow::Optimization {
template <class Scalar, int dim, class StorageIndex = int>
class OptimizerBase {
public:
    using Vec = Bow::Vector<Scalar, Eigen::Dynamic>;
    using Mat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;

    bool line_search = true;
    bool project_pd = true;
    bool BC_satisfied = true;
    bool verbose = true;
    int max_iter = 1000;
    int iter_num = 0;
    std::string method = "GD";
    Scalar tol = 1e-3;
    bool return_if_backtracking_fails = false;
    bool try_fixed_stepsize = false;
    bool enforce_tolerance = false;
    bool direct_solver = true;
    std::vector<EnergyOp<Scalar, dim>*> m_energy_terms;

    virtual Scalar energy(const Vec& x_vec)
    {
        Scalar total_energy = 0.0;
        auto x = to_field<dim>(x_vec);
        for (auto e : m_energy_terms)
            total_energy += e->energy(x);
        return total_energy;
    }

    virtual void gradient(const Vec& x_vec, Vec& grad_vec)
    {
        grad_vec.resize(x_vec.size());
        grad_vec.setZero();
        auto x = to_field<dim>(x_vec);
        for (auto e : m_energy_terms) {
            Field<Vector<Scalar, dim>> sub_grad;
            e->gradient(x, sub_grad);
            grad_vec += to_vec(sub_grad);
        }
    }

    virtual void hessian(const Vec& x_vec, Mat& hess, const bool project_pd)
    {
        hess.derived().resize(x_vec.size(), x_vec.size());
        hess.derived().setZero();
        auto x = to_field<dim>(x_vec);
        for (auto e : m_energy_terms) {
            Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex> sub_hess;
            e->hessian(x, sub_hess, project_pd);
            {
                BOW_TIMER_FLAG("Merge Hessians");
                hess += sub_hess;
            }
        }
    }

    virtual Vec linear_system(const Mat& A, const Vec& b)
    {
        if (direct_solver) {
#ifdef BOW_SUITESPARSE
            BOW_TIMER_FLAG("Cholmod Solve");
            Math::LinearSolver::CholmodLLT<Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>> solver;
#else
            BOW_TIMER_FLAG("Eigen LDLT Solve");
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>> solver;
#endif
            solver.compute(A);
            return solver.solve(b);
        }
        else {
#ifdef BOW_AMGCL
            BOW_TIMER_FLAG("AMG CG Solve");
            Bow::Math::LinearSolver::AmgclSolver<Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>, true> solver;
            solver.prm.put("solver.type", "cg");
            solver.prm.get_child("solver").erase("M");
            return solver.solve(A, b);
#else
            BOW_TIMER_FLAG("Eigen CG Solve");
            Eigen::ConjugateGradient<Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>, Eigen::Lower | Eigen::Upper> solver;
            solver.compute(A);
            return solver.solve(b);
#endif
        }
    }

    virtual Scalar residual(const Vec& x, const Vec& grad, const Vec& direction) { return direction.cwiseAbs().maxCoeff(); };

    /** 
     * we may want to optimize f(x) by optimizing f(g(y)). 
     * pull_back maps x to y.
     * push_forward maps y to x.
     * for example, in slip boundary conditions, x = Ay, 
     *      where A is a diagonal block matrix.
     */
    virtual void pull_back(Vec& x) {}
    virtual void push_forward(Vec& y) {}
    virtual void project(Vec& b_vec) {}

    virtual void callback(Vec& x_vec)
    {
        auto x = to_field<dim>(x_vec);
        for (auto e : m_energy_terms)
            e->callback(x);
    }

    virtual Scalar initial_stepsize(const Vec& x_vec, const Vec& dx_vec)
    {
        auto x = to_field<dim>(x_vec);
        auto dx = to_field<dim>(dx_vec);
        Scalar upper_bound = 1.0;
        for (auto e : m_energy_terms)
            upper_bound = std::min(upper_bound, e->stepsize_upperbound(x, dx));
        return upper_bound;
    }

    virtual void precompute(const Vec& x_vec)
    {
        auto x = to_field<dim>(x_vec);
        for (auto e : m_energy_terms)
            e->precompute(x);
    }

    virtual void search_direction(const Vec& x_vec, const Vec& grad, Vec& direction)
    {
        direction = -grad;
    }

    virtual void initialize_optimizer(const Vec& x) {}

    virtual Scalar backtracking(const Vec& direction, Scalar& alpha, Vec& xn)
    {
        Scalar E0 = energy(xn);
        Vec new_x = xn + alpha * direction;
        precompute(new_x);
        Scalar E = energy(new_x);
        int line_search_it = 0;
        if (line_search) {
            BOW_TIMER_FLAG(this->method + " Line Search");
            while (E > E0) {
                alpha *= 0.5;
                new_x = xn + alpha * direction;
                precompute(new_x);
                E = energy(new_x);
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
    /**
     * x is modified in place.
     */
    template <class DerivedX>
    int optimize(Eigen::MatrixBase<DerivedX>& x)
    {
        BOW_TIMER_FLAG(this->method);
        initialize_optimizer(x);
        iter_num = 0;
        Bow::Vector<Scalar, Eigen::Dynamic> xn = x;
        precompute(xn);
        bool tol_satisfied = false;
        for (; iter_num < max_iter; ++iter_num) {
            callback(xn);
            Vec direction, grad;
            gradient(xn, grad);
            search_direction(xn, grad, direction);
            push_forward(direction);
            Scalar res = residual(xn, grad, direction);
            if (res < tol) {
                static int total_iter = 0;
                total_iter += iter_num;
                Bow::Logging::info(this->method + " Converged iter: ", iter_num, ",\tGrad_inf: ", grad.cwiseAbs().maxCoeff(), ",\tResidual: ", res, ",\tTotal iter: ", total_iter);
                if (res < tol)
                    tol_satisfied = true;
                break;
            }
            Vec xn0 = xn;
            Scalar alpha0 = initial_stepsize(xn, direction);
            Scalar alpha = alpha0;
            Scalar E = backtracking(direction, alpha, xn);
            if (std::isnan(std::abs(E)))
                BOW_ASSERT_INFO(false, "FPE!")
            if (verbose)
                Bow::Logging::info(this->method + " Iter: ", iter_num, ",\t E: ", E, ", \tResidual: ", res, ",\tStep size: ", alpha, ",\tInitial step size: ", alpha0);

            if (std::log2(alpha0 / alpha) > 30) {
                Bow::Logging::warn(this->method + " tiny stepsize!");
                if (try_fixed_stepsize) {
                    xn = xn0 + 0.1 * alpha0 * direction;
                    precompute(xn);
                }
                else if (return_if_backtracking_fails)
                    break;
            }
        }
        if (!tol_satisfied) {
            Bow::Logging::warn(this->method + " tolerance is not reached!");
            if (enforce_tolerance)
                BOW_ASSERT_INFO(false, this->method + " tolerance is not reached!");
        }
        x = xn;
        return iter_num;
    }

    bool requires_feasible_bc_init()
    {
        return true;
    }

    void diff_test(Vec x)
    {
        if constexpr (std::is_same<Scalar, double>::value) {
            const auto f = [&](const Eigen::VectorXd& x) -> double {
                this->precompute(x);
                return this->energy(x);
            };
            const auto g = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
                this->precompute(x);
                this->gradient(x, grad);
            };
            const auto h = [&](const Eigen::VectorXd& x, Eigen::SparseMatrix<double>& hess) {
                this->precompute(x);
                this->hessian(x, hess, false);
            };
            std::random_device rd;
            Eigen::VectorXd direction = Eigen::VectorXd::Random(x.size());
            direction /= direction.norm();
            Scalar eps = 1e-6;
            Eigen::VectorXd dx = eps * direction;
            this->project(dx);
            // FiniteDiff::ziran_check_false(x, f, g, h, [&](Vec& x) { this->project(x); }, nullptr, 1);
            FiniteDiff::check_gradient(x, f, g, dx, 1e-3);
            FiniteDiff::check_jacobian<Eigen::SparseMatrix<double>>(x, g, h, dx, 1e-3);
            getchar();
        }
    }
};
} // namespace Bow::Optimization

#endif