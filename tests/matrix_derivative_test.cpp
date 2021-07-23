#include <Bow/Types.h>
#include <catch2/catch.hpp>
#include <iostream>
#include <Bow/Utils/FiniteDiff.h>
#include <Bow/Types.h>
#include <Bow/Math/MatrixDerivative.h>

TEST_CASE("dAX/dX", "[Matrix]")
{
    GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(2, 3);
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> f = [&](const Eigen::VectorXd& x, Eigen::VectorXd& value) {
        Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(x.data(), 3, 4);
        Eigen::MatrixXd AX = A * X;
        value = Eigen::Map<Eigen::VectorXd>(AX.data(), 8);
    };
    std::function<void(const Eigen::VectorXd&, Eigen::MatrixXd&)> g = [&](const Eigen::VectorXd& x, Eigen::MatrixXd& jacobian) {
        Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(x.data(), 3, 4);
        Bow::Math::dAX(A, X, jacobian);
    };
    Eigen::VectorXd x = Eigen::VectorXd::Random(12);
    CHECK(Bow::FiniteDiff::check_jacobian(x, f, g));
}

TEST_CASE("dXA/dX", "[Matrix]")
{
    GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(3, 4);
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> f = [&](const Eigen::VectorXd& x, Eigen::VectorXd& value) {
        Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(x.data(), 2, 3);
        Eigen::MatrixXd XA = X * A;
        value = Eigen::Map<Eigen::VectorXd>(XA.data(), 8);
    };
    std::function<void(const Eigen::VectorXd&, Eigen::MatrixXd&)> g = [&](const Eigen::VectorXd& x, Eigen::MatrixXd& jacobian) {
        Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(x.data(), 2, 3);
        Bow::Math::dXA(A, X, jacobian);
    };
    Eigen::VectorXd x = Eigen::VectorXd::Random(6);
    CHECK(Bow::FiniteDiff::check_jacobian(x, f, g));
}

TEST_CASE("dX^{-1}/dX", "[Matrix]")
{
    GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> f = [&](const Eigen::VectorXd& x, Eigen::VectorXd& value) {
        Eigen::MatrixXd Xinv = Eigen::Map<const Eigen::MatrixXd>(x.data(), 4, 4).inverse();
        value = Eigen::Map<Eigen::VectorXd>(Xinv.data(), 16);
    };
    std::function<void(const Eigen::VectorXd&, Eigen::MatrixXd&)> g = [&](const Eigen::VectorXd& x, Eigen::MatrixXd& jacobian) {
        Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(x.data(), 4, 4);
        Bow::Math::dXinv(X, jacobian);
    };

    Eigen::MatrixXd X = Eigen::MatrixXd::Random(4, 4);
    while (std::abs(X.determinant()) < 1e-5)
        X = Eigen::MatrixXd::Random(4, 4);
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(X.data(), X.size());
    CHECK(Bow::FiniteDiff::check_jacobian(x, f, g));
}

TEST_CASE("df(AX^{-1})/dX", "[Matrix]")
{
    GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(3, 4);
    Eigen::MatrixXd df = Eigen::MatrixXd::Random(3, 4);
    std::function<double(const Eigen::VectorXd&)> f = [&](const Eigen::VectorXd& x) -> double {
        Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(x.data(), 4, 4);
        Eigen::MatrixXd F = A * X.inverse();
        return df.cwiseProduct(F).sum();
    };
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> g = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(x.data(), 4, 4);
        Eigen::MatrixXd jacobian;
        Bow::Math::df_AXinv(A, X, df, jacobian);
        grad = Eigen::Map<Eigen::VectorXd>(jacobian.data(), 16);
    };

    Eigen::MatrixXd X = Eigen::MatrixXd::Random(4, 4);
    while (std::abs(X.determinant()) < 1e-5)
        X = Eigen::MatrixXd::Random(4, 4);
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(X.data(), X.size());
    CHECK(Bow::FiniteDiff::check_gradient(x, f, g));
}