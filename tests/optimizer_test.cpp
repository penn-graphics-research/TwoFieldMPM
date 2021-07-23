#include <Bow/Types.h>
#include <Bow/Optimization/Newton.h>
#include <Bow/Optimization/LBFGS.h>
#include <Bow/Optimization/AugmentedLagrangian.h>
#include <Bow/Math/Utils.h>
#include <Bow/Utils/FiniteDiff.h>
#include <catch2/catch.hpp>
#include <iostream>

template <class Optimizer>
class Himmelblau : public Optimizer {
    /**
     * https://en.wikipedia.org/wiki/Test_functions_for_optimization
     */
public:
    double energy(const Bow::Vector<double, -1>& m) override
    {
        const double x = m(0);
        const double y = m(1);
        return std::pow(x * x + y - 11, 2) + std::pow(x + y * y - 7, 2);
    }
    void gradient(const Bow::Vector<double, -1>& m, Bow::Vector<double, -1>& res) override
    {
        res.derived().resize(2);
        const double x = m(0);
        const double y = m(1);
        res(0) = 4 * (x * x * x + y * x - 11 * x) + 2 * (x + y * y - 7); // derivative of rosen() with respect to x
        res(1) = 2 * (x * x + y - 11) + 4 * (x * y + y * y * y - 7 * y); // derivative of rosen() with respect to y
    }
    void hessian(const Bow::Vector<double, -1>& m, Eigen::SparseMatrix<double>& hess, bool project_pd = true) override
    {
        hess.derived().resize(2, 2);
        const double x = m(0);
        const double y = m(1);
        Bow::Matrix<double, 2, 2> res1, res2;
        res1(0, 0) = 4 * (3 * x * x + y - 11);
        res1(0, 1) = res1(1, 0) = 4 * x;
        res1(1, 1) = 2;

        res2(0, 0) = 2;
        res2(0, 1) = res2(1, 0) = 4 * y;
        res2(1, 1) = 4 * (x + 3 * y * y - 7);
        if (project_pd) {
            Bow::Math::make_pd(res1);
            Bow::Math::make_pd(res2);
        }
        hess = (res1 + res2).sparseView();
    }
    double residual(const Bow::Vector<double, -1>& x_vec, const Bow::Vector<double, -1>& grad, const Bow::Vector<double, -1>& direction) override
    {
        return grad.norm();
    }
};

TEST_CASE("Test Newton", "[Newton]")
{
    using Vec = Bow::Vector<double, Eigen::Dynamic>;
    Vec x = Vec::Zero(2);
    x << -0.27084, -0.92303; // search from a point near the local maximum;
    Himmelblau<Bow::Optimization::Newton<double, 1, int>> opt;
    opt.tol = 1e-10;
    opt.optimize(x);
    std::cout << "optimum: " << x.transpose() << std::endl;
    double eps = 1e-10;
    CHECK(std::abs(opt.energy(x)) < eps);
}

TEST_CASE("Test LBFGS", "[LBFGS]")
{
    using Vec = Bow::Vector<double, Eigen::Dynamic>;
    Vec x = Vec::Zero(2);
    x << -0.27084, -0.92303;
    Himmelblau<Bow::Optimization::LBFGS<double, 1, int>> opt;
    opt.tol = 1e-10;
    opt.history_size = 10;
    opt.return_if_backtracking_fails = false;
    opt.optimize(x);
    std::cout << "optimum: " << x.transpose() << std::endl;
    double eps = 1e-8;
    CHECK(std::abs(opt.energy(x)) < eps);
}

class AugNewtonToy : public Bow::Optimization::AugmentedLagrangianNewton<double, 1, int> {
public:
    double energy(const Bow::Vector<double, -1>& m) override
    {
        const double x = m(0);
        const double y = m(1);
        return -x * y;
    }
    void gradient(const Bow::Vector<double, -1>& m, Bow::Vector<double, -1>& res) override
    {
        res.derived().resize(2);
        const double x = m(0);
        const double y = m(1);
        res(0) = -y; // derivative of rosen() with respect to x
        res(1) = -x; // derivative of rosen() with respect to y
    }
    void hessian(const Bow::Vector<double, -1>& m, Eigen::SparseMatrix<double>& hess, bool project_pd = true) override
    {
        hess.derived().resize(2, 2);
        Bow::Matrix<double, 2, 2> res;
        res(0, 0) = 0;
        res(0, 1) = res(1, 0) = -1;
        res(1, 1) = 0;
        if (project_pd) {
            Bow::Math::make_pd(res);
        }
        hess = res.sparseView();
    }

protected:
    virtual void constraint(const Bow::Vector<double, -1>& x_vec, Bow::Vector<double, -1>& cons)
    {
        cons.resize(1);
        cons(0) = x_vec.sum() - 6;
    }
    virtual void constraint_jacobian(const Bow::Vector<double, -1>& x_vec, Eigen::SparseMatrix<double>& jac)
    {
        jac.resize(1, 2);
        jac.coeffRef(0, 0) = 1;
        jac.coeffRef(0, 1) = 1;
    }
    virtual double constraint_residual(const Bow::Vector<double, -1>& x, const Bow::Vector<double, -1>& cons)
    {
        if (cons.cwiseAbs().maxCoeff() < 1e-4)
            return 0;
        else
            return cons.cwiseAbs().maxCoeff();
    }
};

TEST_CASE("Test AugmentedLagrangianNewton", "[AugNewton]")
{
    using Vec = Bow::Vector<double, Eigen::Dynamic>;
    Vec x = Vec::Constant(2, -100);
    AugNewtonToy opt;
    opt.tol = 1e-6;
    opt.optimize(x);
    std::cout << "optimum: " << x.transpose() << std::endl;
    CHECK((x.array() - 3).square().sum() <= 1e-10);
}
