#include <Bow/Types.h>
#include <catch2/catch.hpp>
#include <iostream>
#include <Bow/Utils/FiniteDiff.h>
#include <Bow/Types.h>
#include <Bow/Geometry/GeometryDerivative.h>

template <int dim>
void test_angle()
{
    const auto f = [&](const Eigen::VectorXd& x) -> double {
        Bow::Vector<double, dim> x0 = x.segment<dim>(0);
        Bow::Vector<double, dim> x1 = x.segment<dim>(dim);
        return Bow::Geometry::angle(x0, x1);
    };
    const auto g = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        Bow::Vector<double, dim> x0 = x.segment<dim>(0);
        Bow::Vector<double, dim> x1 = x.segment<dim>(dim);
        Bow::Vector<double, dim * 2> grad_fixed;
        Bow::Geometry::angle_gradient(x0, x1, grad_fixed);
        grad = grad_fixed;
    };
    Eigen::VectorXd x = Eigen::VectorXd::Random(dim * 2);
    CHECK(Bow::FiniteDiff::check_gradient(x, f, g));
}

TEST_CASE("Test Simplex Volume 2D", "[Simplex]")
{
    GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    const auto f = [&](const Eigen::VectorXd& x) -> double {
        Bow::Vector<double, 2> x0 = x.segment<2>(0);
        Bow::Vector<double, 2> x1 = x.segment<2>(2);
        Bow::Vector<double, 2> x2 = x.segment<2>(4);
        return Bow::Geometry::simplex_volume(x0, x1, x2);
    };
    const auto g = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        Bow::Vector<double, 2> x0 = x.segment<2>(0);
        Bow::Vector<double, 2> x1 = x.segment<2>(2);
        Bow::Vector<double, 2> x2 = x.segment<2>(4);
        Bow::Vector<double, 6> grad_fixed;
        Bow::Geometry::simplex_volume_gradient(x0, x1, x2, grad_fixed);
        grad = grad_fixed;
    };
    Eigen::VectorXd x = Eigen::VectorXd::Random(6);
    CHECK(Bow::FiniteDiff::check_gradient(x, f, g));
}

TEST_CASE("Test Simplex Volume 3D", "[Simplex]")
{
    GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    const auto f = [&](const Eigen::VectorXd& x) -> double {
        Bow::Vector<double, 3> x0 = x.segment<3>(0);
        Bow::Vector<double, 3> x1 = x.segment<3>(3);
        Bow::Vector<double, 3> x2 = x.segment<3>(6);
        Bow::Vector<double, 3> x3 = x.segment<3>(9);
        return Bow::Geometry::simplex_volume(x0, x1, x2, x3);
    };
    const auto g = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        Bow::Vector<double, 3> x0 = x.segment<3>(0);
        Bow::Vector<double, 3> x1 = x.segment<3>(3);
        Bow::Vector<double, 3> x2 = x.segment<3>(6);
        Bow::Vector<double, 3> x3 = x.segment<3>(9);
        Bow::Vector<double, 12> grad_fixed;
        Bow::Geometry::simplex_volume_gradient(x0, x1, x2, x3, grad_fixed);
        grad = grad_fixed;
    };
    Eigen::VectorXd x = Eigen::VectorXd::Random(12);
    CHECK(Bow::FiniteDiff::check_gradient(x, f, g));
}

TEST_CASE("Test Angle", "[Angle]")
{
    GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    test_angle<2>();
    test_angle<3>();
}
