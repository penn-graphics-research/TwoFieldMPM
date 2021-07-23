#pragma once

#include <Bow/Math/SVD.h>
#include <Bow/Energy/MPM/ElasticityOp.h>

namespace Bow::MPM {

template <class T, int dim>
class PlasticityOp {
public:
    virtual void project_strain() = 0;
};

template <class T, int dim, class FBasedMaterial = StvkWithHenckyOp<T, dim>>
class VonMisesStvkHencky : public PlasticityOp<T, dim> {
    using TM = Matrix<T, dim, dim>;
    using TV = Vector<T, dim>;

    std::shared_ptr<FBasedMaterial> stvk;
    T yield_stress, fail_stress, xi;

public:
    VonMisesStvkHencky(std::shared_ptr<ElasticityOp<T, dim>> stvk, T yield_stress = 500, T fail_stress = std::numeric_limits<T>::max(), T xi = 0)
        : stvk(std::dynamic_pointer_cast<FBasedMaterial>(stvk)), yield_stress(yield_stress), fail_stress(fail_stress), xi(xi) {}

    void project_strain() override
    {
        tbb::parallel_for(size_t(0), stvk->m_F.size(), [&](size_t i) {
            TM& F = stvk->m_F[i];
            TM U, V;
            TV sigma;

            // TODO: this is inefficient because next time step updateState will do the svd again!
            Math::svd(F, U, sigma, V);

            //TV epsilon = sigma.array().log();
            TV epsilon = sigma.array().max(1e-4).log(); //TODO: need the max part?
            T trace_epsilon = epsilon.sum();
            TV epsilon_hat = epsilon - (trace_epsilon / (T)dim) * TV::Ones();
            T epsilon_hat_squared_norm = epsilon_hat.squaredNorm();
            T epsilon_hat_norm = std::sqrt(epsilon_hat_squared_norm);
            T delta_gamma = epsilon_hat_norm - yield_stress / (2 * stvk->mu);
            if (delta_gamma <= 0) // case I
            {
                return;
            }
            //hardening
            yield_stress -= xi * delta_gamma; //supposed to only increase yield_stress
            //yield_stress = std::max((T)0, yield_stress);

            TV H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat; // case II
            TV exp_H = H.array().exp();
            F = U * exp_H.asDiagonal() * V.transpose();
        });
    }
};

template <class T, int dim, class FBasedMaterial = StvkWithHenckyOp<T, dim>>
class DruckerPragerStvkHencky : public PlasticityOp<T, dim> {
public:
    using TM = Matrix<T, dim, dim>;
    using TV = Vector<T, dim>;

    std::shared_ptr<FBasedMaterial> stvk;
    T alpha; // friction_coeff
    T beta; // hardening coeff
    T cohesion;
    bool volume_correction;
    Field<T> logJp;

    DruckerPragerStvkHencky(std::shared_ptr<ElasticityOp<T, dim>> stvk, const T friction_angle = 30, const T beta = 1, const T cohesion = 0, const bool volume_correction = false)
        : stvk(std::dynamic_pointer_cast<FBasedMaterial>(stvk))
        , beta(beta)
        , cohesion(cohesion)
        , volume_correction(volume_correction)
    {
        T sin_phi = std::sin(friction_angle / (T)180 * M_PI);
        alpha = std::sqrt((T)2 / (T)3) * (T)2 * sin_phi / ((T)3 - sin_phi);
    }

    // strain s is deformation F
    void project_strain()
    {
        if (!stvk) return;
        if (stvk->mu == 0)
            return;
        logJp.resize(stvk->m_F.size(), 0);
        tbb::parallel_for(size_t(0), stvk->m_F.size(), [&](size_t i) {
            TM& F = stvk->m_F[i];
            TM U, V;
            TV sigma;

            // TODO: this is inefficient because next time step updateState will do the svd again!
            Math::svd(F, U, sigma, V);

            TV epsilon = sigma.array().max(1e-4).log(); //TODO: need the max part?
            T trace_epsilon = epsilon.sum();
            TV epsilon_hat = epsilon - (trace_epsilon / (T)dim) * TV::Ones();
            T epsilon_hat_squared_norm = epsilon_hat.squaredNorm();
            T epsilon_hat_norm = std::sqrt(epsilon_hat_squared_norm);
            T delta_gamma = epsilon_hat_norm + (dim * stvk->lambda + 2 * stvk->mu) / (2 * stvk->mu) * trace_epsilon * alpha;
            if (trace_epsilon >= (T)0) // case II: project to tip
            {
                F = U * std::exp(cohesion) * V.transpose();
                if (volume_correction)
                    logJp[i] = beta * epsilon.sum() + logJp[i];
                return;
            }

            logJp[i] = 0;
            TV H;
            if (delta_gamma <= 0) // case I: inside yield surface
            {
                H = epsilon + TV::Constant(cohesion);
            }
            else {
                H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat + TV::Constant(cohesion); // case III: projection
            }
            TV exp_H = H.array().exp();
            F = U * exp_H.asDiagonal() * V.transpose();
        });
    }
};

template <class T, int dim, class FBasedMaterial = FixedCorotatedOp<T, dim>>
class SnowPlasticity : public PlasticityOp<T, dim> {
    using TM = Matrix<T, dim, dim>;
    using TV = Vector<T, dim>;

    std::shared_ptr<FBasedMaterial> fcr;
    T Jp, psi, theta_c, theta_s, min_Jp, max_Jp;

public:
    SnowPlasticity(std::shared_ptr<ElasticityOp<T, dim>> fcr, T psi_in = 10, T theta_c_in = 2e-2, T theta_s_in = 7.5e-3, T min_Jp_in = 0.6, T max_Jp_in = 20)
        : fcr(std::dynamic_pointer_cast<FBasedMaterial>(fcr)), Jp(1), psi(psi_in), theta_c(theta_c_in), theta_s(theta_s_in), min_Jp(min_Jp_in), max_Jp(max_Jp_in) {}

    void project_strain() override
    {
        tbb::parallel_for(size_t(0), fcr->m_F.size(), [&](size_t i) {
            TM& F = fcr->m_F[i];
            T& Jp = fcr->m_J[i];
            TM U, V;
            TV sigma;

            // TODO: this is inefficient because next time step updateState will do the svd again!
            Math::svd(F, U, sigma, V);

            T Fe_det = (T)1;
            for (int i = 0; i < dim; i++) {
                sigma(i) = std::max(std::min(sigma(i), (T)1 + theta_s), (T)1 - theta_c);
                Fe_det *= sigma(i);
            }

            Eigen::DiagonalMatrix<T, dim, dim> sigma_m(sigma);
            TM Fe = U * sigma_m * V.transpose();
            // T Jp_new = std::max(std::min(Jp * strain.determinant() / Fe_det, max_Jp), min_Jp);
            T Jp_new = Jp * F.determinant() / Fe_det;
            if (!(Jp_new <= max_Jp))
                Jp_new = max_Jp;
            if (!(Jp_new >= min_Jp))
                Jp_new = min_Jp;

            F = Fe;
            fcr->m_mu[i] *= std::exp(psi * (Jp - Jp_new));
            fcr->m_lambda[i] *= std::exp(psi * (Jp - Jp_new));
            Jp = Jp_new;
        });
    }
};

} // namespace Bow::MPM