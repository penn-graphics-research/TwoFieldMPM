#include <Bow/Types.h>
#include <catch2/catch.hpp>
#include <iostream>
#include <Bow/Utils/FiniteDiff.h>
#include <Bow/ConstitutiveModel/StvkWithHencky.h>
#include <Bow/ConstitutiveModel/EquationOfState.h>
#include <Bow/ConstitutiveModel/FixedCorotated.h>
#include <Bow/ConstitutiveModel/NeoHookean.h>
#include <Bow/ConstitutiveModel/LinearElasticity.h>
#include <Bow/Math/SVD.h>

TEST_CASE("Test (E nu) - (mu lam) conversion", "[Param]")
{
    GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    Eigen::Vector2d E_nu = Eigen::Vector2d::Random();
    double mu, lam;
    std::tie(mu, lam) = Bow::ConstitutiveModel::lame_paramters(E_nu[0], E_nu[1]);
    Eigen::Vector2d E_nu_convert;
    std::tie(E_nu_convert[0], E_nu_convert[1]) = Bow::ConstitutiveModel::E_nu(mu, lam);
    CHECK((E_nu - E_nu_convert).norm() < 1e-6);
}

template <int dim, class Model>
void constitutive_model_test()
{
    Model model;
    double mu = 0.1523;
    double lam = 0.5234;
    const auto f = [&](const Eigen::VectorXd x) -> double {
        Bow::Matrix<double, dim, dim> F = Eigen::Map<const Bow::Matrix<double, dim, dim>>(x.data(), dim, dim);
        return model.psi(F, mu, lam);
    };
    const auto g = [&](const Eigen::VectorXd x, Eigen::VectorXd& grad) {
        Bow::Matrix<double, dim, dim> F = Eigen::Map<const Bow::Matrix<double, dim, dim>>(x.data(), dim, dim);
        grad.resize(dim, dim);
        Bow::Matrix<double, dim, dim> grad_mat;
        model.first_piola(F, mu, lam, grad_mat);
        Eigen::Map<Bow::Matrix<double, dim, dim>>(grad.data(), dim, dim) = grad_mat;
    };
    const auto h = [&](const Eigen::VectorXd x, Eigen::MatrixXd& hess) {
        Bow::Matrix<double, dim, dim> F = Eigen::Map<const Bow::Matrix<double, dim, dim>>(x.data(), dim, dim);
        hess.resize(dim * dim, dim * dim);
        Bow::Matrix<double, dim * dim, dim * dim> hess_mat;
        model.first_piola_derivative(F, mu, lam, hess_mat, false);
        Eigen::Map<Bow::Matrix<double, dim * dim, dim * dim>>(hess.data(), dim * dim, dim * dim) = hess_mat;
    };

    Eigen::VectorXd x = Eigen::VectorXd::Random(dim * dim);
    Bow::Matrix<double, dim, dim> F = Eigen::Map<Bow::Matrix<double, dim, dim>>(x.data());
    Bow::Matrix<double, dim, dim> U, V;
    Bow::Vector<double, dim> sigma;
    Bow::Math::svd(F, U, sigma, V);
    sigma = sigma.cwiseAbs().array() + 1e-6;
    Eigen::Map<Bow::Matrix<double, dim, dim>>(x.data()) = U * sigma.asDiagonal() * V.transpose();
    CHECK(Bow::FiniteDiff::check_gradient(x, f, g, 1e-6, 1e-3));
    CHECK(Bow::FiniteDiff::check_jacobian<Eigen::MatrixXd>(x, g, h, 1e-6, 1e-3));
}

template <int dim>
void test_eos()
{
    Bow::ConstitutiveModel::EquationOfState<double> model;
    const auto f = [&](const Eigen::VectorXd x) -> double {
        return model.psi(x(0), 1000., 7.);
    };
    const auto g = [&](const Eigen::VectorXd x, Eigen::VectorXd& grad) {
        grad.resize(1);
        grad(0) = model.first_piola(x(0), 1000., 7.);
    };
    const auto h = [&](const Eigen::VectorXd x, Eigen::MatrixXd& hess) {
        hess.resize(1, 1);
        hess(0, 0) = model.first_piola_derivative(x(0), 1000., 7.);
    };

    Eigen::VectorXd x = Eigen::VectorXd::Random(dim * dim);
    Bow::Matrix<double, dim, dim> F = Eigen::Map<Bow::Matrix<double, dim, dim>>(x.data());
    Bow::Matrix<double, dim, dim> U, V;
    Bow::Vector<double, dim> sigma;
    Bow::Math::svd(F, U, sigma, V);
    sigma = sigma.cwiseAbs().array() + 1e-6;
    Eigen::VectorXd J(1);
    J(0) = sigma.prod();
    CHECK(Bow::FiniteDiff::check_gradient(J, f, g, 1e-6, 1e-3));
    CHECK(Bow::FiniteDiff::check_jacobian<Eigen::MatrixXd>(J, g, h, 1e-6, 1e-3));
}

TEST_CASE("Test Consititutive Models", "[CM]")
{
    GENERATE(1, 2, 3, 4, 5);
    Bow::Logging::info("check FixedCorotated");
    constitutive_model_test<2, Bow::ConstitutiveModel::FixedCorotated<double, 2>>();
    constitutive_model_test<3, Bow::ConstitutiveModel::FixedCorotated<double, 3>>();
    Bow::Logging::info("check NeoHookean");
    constitutive_model_test<2, Bow::ConstitutiveModel::NeoHookean<double, 2>>();
    constitutive_model_test<3, Bow::ConstitutiveModel::NeoHookean<double, 3>>();
    Bow::Logging::info("check LinearElasticity");
    constitutive_model_test<2, Bow::ConstitutiveModel::LinearElasticity<double, 2>>();
    constitutive_model_test<3, Bow::ConstitutiveModel::LinearElasticity<double, 3>>();
    Bow::Logging::info("check StvkWithHencky");
    constitutive_model_test<2, Bow::ConstitutiveModel::StvkWithHencky<double, 2>>();
    constitutive_model_test<3, Bow::ConstitutiveModel::StvkWithHencky<double, 3>>();
    test_eos<2>();
    test_eos<3>();
}