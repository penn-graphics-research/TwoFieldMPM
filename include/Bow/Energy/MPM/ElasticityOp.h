#ifndef ELASTICITY_OP_H
#define ELASTICITY_OP_H

#include <Bow/Types.h>
#include <Bow/ConstitutiveModel/FixedCorotated.h>
#include <Bow/ConstitutiveModel/NeoHookean.h>
#include <Bow/ConstitutiveModel/StvkWithHencky.h>
#include <Bow/ConstitutiveModel/LinearElasticity.h>
#include <Bow/ConstitutiveModel/EquationOfState.h>
#include <Bow/Geometry/Hybrid/MPMTransfer.h>
#include <Bow/Math/Utils.h>
#include <Bow/Utils/Serialization.h>
#include <Bow/Math/SVD.h>

namespace Bow::MPM {

template <class T, int dim>
class ElasticityOp {
public:
    Field<Matrix<T, dim, dim>> m_F;
    Field<T> m_J;
    Field<T> m_mu, m_lambda;
    Field<T> m_vol;
    Field<T> m_chemPotential; //used only for poroelasticity
    std::vector<int> m_global_index;
    virtual void append(int start, int end, T vol) = 0;
    virtual void compute_stress(Field<Matrix<T, dim, dim>>& stress) {}
    virtual void compute_piola(Field<Matrix<T, dim, dim>>& piola) {}
    virtual void compute_cauchy(Field<Matrix<T, dim, dim>>& stress) {}
    virtual void compute_volume(Field<T>& volume) {}
    virtual void compute_von_mises(Field<T>& stress) {}
    virtual void compute_criticalStress(T percent, Field<Matrix<T, dim, dim>>& stretchedCauchy) { BOW_NOT_IMPLEMENTED }
    virtual void evolve_strain(const Field<Matrix<T, dim, dim>>& m_gradXp) = 0;
    virtual void trial_strain(const Field<Matrix<T, dim, dim>>& m_gradXp) { BOW_NOT_IMPLEMENTED }
    virtual void trial_energy(Field<T>& t_energy) { BOW_NOT_IMPLEMENTED }
    virtual void strain_energy(Field<T>& m_energy) { }
    virtual void trial_gradient(Field<Matrix<T, dim, dim>>& t_gradient) { BOW_NOT_IMPLEMENTED }
    virtual void trial_differential(const Field<Matrix<T, dim, dim>>& d_F, Field<Matrix<T, dim, dim>>& t_differential, bool project_pd) { BOW_NOT_IMPLEMENTED }
    virtual void trial_hessian(Field<Matrix<T, dim * dim, dim * dim>>& t_hessian, bool project_pd) { BOW_NOT_IMPLEMENTED }
    virtual T stepsize_upperbound(const Field<Matrix<T, dim, dim>>& m_gradDXp) { return 1.0; }
    virtual void set_dt(T _dt) { BOW_NOT_IMPLEMENTED }
    virtual void collect_mu(Field<T>& _m_mu) {}
    virtual void collect_la(Field<T>& _m_la) {}
    virtual void collect_strain(Field<Matrix<T, dim, dim>>& _m_F) {}
    virtual void collect_initialVolume(Field<T>& m_initialVolume) {}
    virtual void collect_chemPotential(Field<T>& m_chemPotential) {}
    virtual void update_chemPotential(Field<T>& m_chemPotential) {}
};

template <class T, int dim, class Model, bool inversion_free>
class FBasedElastiticityOp : public ElasticityOp<T, dim>, public Model {
public:
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    using ElasticityOp<T, dim>::m_F;
    using ElasticityOp<T, dim>::m_J;
    using ElasticityOp<T, dim>::m_global_index;
    using ElasticityOp<T, dim>::m_vol;
    using ElasticityOp<T, dim>::m_mu;
    using ElasticityOp<T, dim>::m_lambda;
    using Model::first_piola;
    using Model::first_piola_derivative;
    using Model::first_piola_differential;
    using Model::psi;

    T mu, lambda; // TODO: passed by append

    Field<Matrix<T, dim, dim>> t_F; // only used in implicit

    SERIALIZATION_REGISTER(m_global_index)
    SERIALIZATION_REGISTER(m_F)
    SERIALIZATION_REGISTER(m_vol)
    SERIALIZATION_REGISTER(m_J)
    SERIALIZATION_REGISTER(m_mu)
    SERIALIZATION_REGISTER(m_lambda)

    FBasedElastiticityOp(T E, T nu)
    {
        std::tie(mu, lambda) = Bow::ConstitutiveModel::lame_paramters(E, nu);
    }
    void append(int start, int end, T vol) override
    {
        for (int i = start; i < end; ++i) {
            m_F.push_back(Matrix<T, dim, dim>::Identity());
            m_vol.push_back(vol);
            m_global_index.push_back(i);
            m_J.push_back(1.0);
            m_mu.push_back(mu);
            m_lambda.push_back(lambda);
        }
    }

    void evolve_strain(const Field<Matrix<T, dim, dim>>& m_gradXp) override
    {
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            m_F[i] = (m_gradXp[m_global_index[i]]) * m_F[i];
        });
    }

    void trial_strain(const Field<Matrix<T, dim, dim>>& m_gradXp) override
    {
        t_F = m_F;
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            t_F[i] = (m_gradXp[m_global_index[i]]) * m_F[i];
        });
    }

    void trial_energy(Field<T>& t_energy) override
    {
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            T energy = m_vol[i] * psi(t_F[i], m_mu[i], m_lambda[i]);
            t_energy[m_global_index[i]] += energy;
        });
    }

    void strain_energy(Field<T>& m_energy) override
    {
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            T energy = m_vol[i] * psi(m_F[i], m_mu[i], m_lambda[i]);
            m_energy[m_global_index[i]] = energy;
        });
    }

    void trial_gradient(Field<TM>& t_gradient) override
    {
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            Matrix<T, dim, dim> P;
            first_piola(t_F[i], m_mu[i], m_lambda[i], P);
            // Eqn 194. https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf
            Matrix<T, dim, dim> stress = m_vol[i] * P * m_F[i].transpose();
            t_gradient[m_global_index[i]] += stress;
        });
    }

    void trial_differential(const Field<Matrix<T, dim, dim>>& d_F, Field<Matrix<T, dim, dim>>& t_differential, bool project_pd) override
    {
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            TM A = TM::Zero();
            TM D = d_F[m_global_index[i]] * m_F[i];
            first_piola_differential(t_F[i], D, m_mu[i], m_lambda[i], A, project_pd);
            t_differential[m_global_index[i]] = m_vol[i] * A * m_F[i].transpose();
        });
    }

    void trial_hessian(Field<Matrix<T, dim * dim, dim * dim>>& t_hessian, bool project_pd) override
    {
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            Matrix<T, dim * dim, dim * dim> dPdF;
            first_piola_derivative(t_F[i], m_mu[i], m_lambda[i], dPdF, project_pd);
            TM FT = m_F[i].transpose();
            Matrix<T, dim * dim, dim* dim> deformed_dPdF = Matrix<T, dim * dim, dim * dim>::Zero();
            for (int u = 0; u < dim; ++u)
                for (int v = 0; v < dim; ++v)
                    for (int x = 0; x < dim; ++x)
                        for (int p = 0; p < dim; ++p)
                            for (int q = 0; q < dim; ++q)
                                for (int y = 0; y < dim; ++y)
                                    deformed_dPdF(u + x * dim, p + y * dim) += dPdF(u + v * dim, p + q * dim) * FT(v, x) * FT(q, y);
            t_hessian[m_global_index[i]] += m_vol[i] * deformed_dPdF;
        });
    }

    void compute_cauchy(Field<Matrix<T, dim, dim>>& cauchy)
    {
        BOW_TIMER_FLAG("compute cauchy (FBased)");
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            Matrix<T, dim, dim> F = m_F[i];
            Matrix<T, dim, dim> P;
            T J = F.determinant();
            first_piola(F, m_mu[i], m_lambda[i], P);
            cauchy[m_global_index[i]] = (1.0 / J) * P * F.transpose();
        });
    }

    void compute_volume(Field<T>& volume){
        BOW_TIMER_FLAG("compute current volume (Fbased)");
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            Matrix<T, dim, dim> F = m_F[i];
            T J = F.determinant();
            volume[m_global_index[i]] = J * m_vol[i]; //current volume VpN = J * Vp0
        });
    }

    void compute_von_mises(Field<T>& stress)
    {
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            Matrix<T, dim, dim> F = m_F[i];
            Matrix<T, dim, dim> U, V;
            Vector<T, dim> sigma;
            Math::svd(F, U, sigma, V);
            Matrix<T, dim, dim> Sigma = sigma.asDiagonal();
            Matrix<T, dim, dim> tau;
            this->first_piola(Sigma, m_mu[i], m_lambda[i], tau);
            tau = tau * Sigma.transpose();
            if constexpr (dim == 2)
                stress[m_global_index[i]] = std::abs(tau(0, 0) - tau(1, 1));
            else {
                Vector<T, dim> shifted_tau(tau(1, 1), tau(2, 2), tau(0, 0));
                stress[m_global_index[i]] = std::sqrt(0.5 * (tau.diagonal() - shifted_tau).squaredNorm());
            }
        });
    }

    void compute_stress(Field<Matrix<T, dim, dim>>& stress) override
    {
        BOW_TIMER_FLAG("compute elasticity");
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            Matrix<T, dim, dim> F = m_F[i];
            Matrix<T, dim, dim> P;
            first_piola(F, m_mu[i], m_lambda[i], P);
            stress[m_global_index[i]] = m_vol[i] * P * F.transpose();
        });
    }

    void compute_piola(Field<Matrix<T, dim, dim>>& piola) override
    {
        BOW_TIMER_FLAG("compute piola");
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            Matrix<T, dim, dim> F = m_F[i];
            Matrix<T, dim, dim> P;
            first_piola(F, m_mu[i], m_lambda[i], P);
            piola[m_global_index[i]] = P;
        });
    }

    //this will have to be written for each elasticity model we want to use with MPM damage
    void compute_criticalStress(T percent, Field<Matrix<T, dim, dim>>& stretchedCauchy)
    {
        BOW_TIMER_FLAG("compute sigmaC (FBased)");
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            TM F = TM::Identity() * (1.0 + percent); //stretched F
            T J = F.determinant();
            Matrix<T, dim, dim> P;
            first_piola(F, m_mu[i], m_lambda[i], P);
            stretchedCauchy[m_global_index[i]] = (1.0 / J) * P * F.transpose();
        });
    }

    T stepsize_upperbound(const Field<Matrix<T, dim, dim>>& m_gradDXp) override
    {
        if (m_F.size() == 0) return 1.0;
        if constexpr (!inversion_free) return 1.0;
        Vector<T, Eigen::Dynamic> alphas(m_F.size());
        alphas.setOnes();
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            Matrix<T, dim, dim> A = t_F[i].transpose().partialPivLu().solve((m_gradDXp[m_global_index[i]] * m_F[i]).transpose());
            T a, b, c, d;
            if constexpr (dim == 2) {
                a = 0;
                b = A.determinant();
            }
            else {
                a = A.determinant();
                b = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0) + A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0) + A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
            }
            c = A.diagonal().sum();
            d = 0.9;

            T t = Math::get_smallest_positive_real_cubic_root(a, b, c, d);
            if (t < 0 || t > 1) t = 1;
            alphas(i) = t;
        });
        return alphas.minCoeff();
    }

    void set_dt(T _dt) override
    {
        return;
    }

    void collect_mu(Field<T>& _m_mu) override
    {
        tbb::parallel_for(size_t(0), m_mu.size(), [&](size_t i) {
            _m_mu[m_global_index[i]] = m_mu[i];
        });
    }

    void collect_la(Field<T>& _m_la) override
    {
        tbb::parallel_for(size_t(0), m_lambda.size(), [&](size_t i) {
            _m_la[m_global_index[i]] = m_lambda[i];
        });
    }

    void collect_strain(Field<Matrix<T, dim, dim>>& _m_F) override
    {
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            _m_F[m_global_index[i]] = m_F[i];
        });
    }

    void collect_initialVolume(Field<T>& m_initialVolume) override
    {
        tbb::parallel_for(size_t(0), m_vol.size(), [&](size_t i) {
            m_initialVolume[m_global_index[i]] = m_vol[i];
        });
    }

    void collect_chemPotential(Field<T>& _m_chemPotential) override
    {
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            _m_chemPotential[m_global_index[i]] = 0.0;
        });
    }

    void update_chemPotential(Field<T>& _m_chemPotential) override
    {
        return;
    }
};

template <class T, int dim>
using FixedCorotatedOp = FBasedElastiticityOp<T, dim, ConstitutiveModel::FixedCorotated<T, dim>, false>;

template <class T, int dim>
using NeoHookeanOp = FBasedElastiticityOp<T, dim, ConstitutiveModel::NeoHookean<T, dim>, true>;

template <class T, int dim>
using StvkWithHenckyOp = FBasedElastiticityOp<T, dim, ConstitutiveModel::StvkWithHencky<T, dim>, true>;

template <class T, int dim>
using LinearElasticityOp = FBasedElastiticityOp<T, dim, ConstitutiveModel::LinearElasticity<T, dim>, false>;


template <class T, int dim, class Model, bool inversion_free>
class FBasedPoroelasticityOp : public ElasticityOp<T, dim>, public Model {
public:
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    using ElasticityOp<T, dim>::m_F;
    using ElasticityOp<T, dim>::m_global_index;
    using ElasticityOp<T, dim>::m_vol;
    using ElasticityOp<T, dim>::m_mu;
    using ElasticityOp<T, dim>::m_lambda;
    using ElasticityOp<T, dim>::m_chemPotential;

    T mu, lambda; // TODO: passed by append
    T c1, c2;
    T phi_s0;
    T pi_0;
    //T mu_0;
    T beta_1;
    //T r_f;
    T a1; //for correcting initial stress state

    Field<Matrix<T, dim, dim>> t_F; // only used in implicit
    

    SERIALIZATION_REGISTER(m_global_index)
    SERIALIZATION_REGISTER(m_F)
    SERIALIZATION_REGISTER(m_vol)
    SERIALIZATION_REGISTER(m_mu)
    SERIALIZATION_REGISTER(m_lambda)

    FBasedPoroelasticityOp(T _c1, T _c2, T _phi_s0, T _pi_0, T _beta_1)
    {
        mu = 0;
        lambda = 0;
        c1 = _c1;
        c2 = _c2;
        phi_s0 = _phi_s0;
        pi_0 = _pi_0;
        //mu_0 = _mu_0;
        beta_1 = _beta_1;
        a1 = (pi_0 / phi_s0) - (2.0 * c1);
    }
    void append(int start, int end, T vol) override
    {
        for (int i = start; i < end; ++i) {
            m_F.push_back(Matrix<T, dim, dim>::Identity());
            m_vol.push_back(vol);
            m_global_index.push_back(i);
            m_mu.push_back(mu);
            m_lambda.push_back(lambda);
            m_chemPotential.push_back(0.0);
        }
    }

    void evolve_strain(const Field<Matrix<T, dim, dim>>& m_gradXp) override
    {
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            m_F[i] = (m_gradXp[m_global_index[i]]) * m_F[i];
        });
    }

    void strain_energy(Field<T>& m_energy) override
    {
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            T mu = 0.0; // TODO: grab actual chemical potential, mu
            T energy = m_vol[i] * psi_poro(m_F[i], mu, c1, c2, phi_s0, pi_0, beta_1, a1);
            m_energy[m_global_index[i]] = energy;
        });
    }

    void compute_cauchy(Field<Matrix<T, dim, dim>>& cauchy)
    {
        BOW_TIMER_FLAG("compute cauchy (FBased)");
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            Matrix<T, dim, dim> F = m_F[i];
            Matrix<T, dim, dim> P;
            T J = F.determinant();
            T mu = m_chemPotential[i];
            first_piola_poro(F, mu, c1, c2, phi_s0, pi_0, beta_1, a1, P);
            cauchy[m_global_index[i]] = (1.0 / J) * P * F.transpose();
        });
    }

    void compute_volume(Field<T>& volume){
        BOW_TIMER_FLAG("compute current volume (Fbased)");
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            Matrix<T, dim, dim> F = m_F[i];
            T J = F.determinant();
            volume[m_global_index[i]] = J * m_vol[i]; //current volume VpN = J * Vp0
        });
    }

    void compute_stress(Field<Matrix<T, dim, dim>>& stress) override
    {
        BOW_TIMER_FLAG("compute elasticity");
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            Matrix<T, dim, dim> F = m_F[i];
            Matrix<T, dim, dim> P;
            T mu = m_chemPotential[i];
            first_piola_poro(F, mu, c1, c2, phi_s0, pi_0, beta_1, a1, P);
            stress[m_global_index[i]] = m_vol[i] * P * F.transpose();
        });
    }

    void compute_piola(Field<Matrix<T, dim, dim>>& piola) override
    {
        BOW_TIMER_FLAG("compute piola");
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            Matrix<T, dim, dim> F = m_F[i];
            Matrix<T, dim, dim> P;
            T mu = m_chemPotential[i];
            first_piola_poro(F, mu, c1, c2, phi_s0, pi_0, beta_1, a1, P);
            piola[m_global_index[i]] = P;
        });
    }

    //this will have to be written for each elasticity model we want to use with MPM damage
    // void compute_criticalStress(T percent, Field<Matrix<T, dim, dim>>& stretchedCauchy)
    // {
    //     BOW_TIMER_FLAG("compute sigmaC (FBased)");
    //     tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
    //         TM F = TM::Identity() * (1.0 + percent); //stretched F
    //         T J = F.determinant();
    //         Matrix<T, dim, dim> P;
    //         T mu = 0.0; //TODO: grab actual chemical potential, mu
    //         first_piola_poro(F, mu, c1, c2, phi_s0, pi_0, beta_1, P);
    //         stretchedCauchy[m_global_index[i]] = (1.0 / J) * P * F.transpose();
    //     });
    // }

    void set_dt(T _dt) override
    {
        return;
    }

    void collect_mu(Field<T>& _m_mu) override
    {
        tbb::parallel_for(size_t(0), m_mu.size(), [&](size_t i) {
            _m_mu[m_global_index[i]] = m_mu[i];
        });
    }

    void collect_la(Field<T>& _m_la) override
    {
        tbb::parallel_for(size_t(0), m_lambda.size(), [&](size_t i) {
            _m_la[m_global_index[i]] = m_lambda[i];
        });
    }

    void collect_strain(Field<Matrix<T, dim, dim>>& _m_F) override
    {
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            _m_F[m_global_index[i]] = m_F[i];
        });
    }

    void collect_initialVolume(Field<T>& m_initialVolume) override
    {
        tbb::parallel_for(size_t(0), m_vol.size(), [&](size_t i) {
            m_initialVolume[m_global_index[i]] = m_vol[i];
        });
    }

    void collect_chemPotential(Field<T>& _m_chemPotential) override
    {
        tbb::parallel_for(size_t(0), m_vol.size(), [&](size_t i) {
            _m_chemPotential[m_global_index[i]] = m_chemPotential[i];
        });
    }

    void update_chemPotential(Field<T>& _m_chemPotential) override
    {
        tbb::parallel_for(size_t(0), m_vol.size(), [&](size_t i) {
            m_chemPotential[i] = _m_chemPotential[m_global_index[i]];
        });
    }

  private:
    T psi_poro(const Matrix<T, dim, dim>& F, const T mu, const T c1, const T c2, const T phi_s0, const T pi_0, const T beta_1, const T a1)
    {
        T J = F.determinant();
        T I1 = (F.transpose() * F).trace();
        T psiNet = ((phi_s0 * c1) / c2) * (exp(c2 * (I1 - dim)) - 1);
        T psiMix = (pi_0 / (beta_1 - 1)) * ((pow(1 - phi_s0, beta_1)) / (pow(J - phi_s0, beta_1 - 1)));
        T psi0 = (pi_0 * (1 - phi_s0)) / (beta_1 - 1);
        T muC = (mu * (J - phi_s0)); //C = det(F) - phi_s0
        T correction = phi_s0 * a1 * log(J); // psi += a1 * ln(J), log here = ln
        return psiNet + psiMix - psi0 - muC + correction;
    }

    void first_piola_poro(const Matrix<T, dim, dim>& F, const T mu, const T c1, const T c2, const T phi_s0, const T pi_0, const T beta_1, const T a1, Matrix<T, dim, dim>& P)
    {
        T J = F.determinant();
        T I1 = (F.transpose() * F).trace();
        Eigen::Matrix<T, dim, dim> JFinvT;
        Math::cofactor(F, JFinvT);
        Eigen::Matrix<T, dim, dim> Pnet = phi_s0 * 2.0 * c1 * exp(c2 * (I1 - dim)) * F;
        Eigen::Matrix<T, dim, dim> Pmix = ((-pi_0 * (pow(1-phi_s0, beta_1) / pow(J - phi_s0, beta_1))) - mu) * JFinvT;
        Eigen::Matrix<T, dim, dim> Pcorrection = phi_s0 * a1 * (JFinvT / J); // P += a1 * F^-T
        P = Pnet + Pmix + Pcorrection;
    }

};

template <class T, int dim>
using FibrinPoroelasticityOp = FBasedPoroelasticityOp<T, dim, ConstitutiveModel::FixedCorotated<T, dim>, false>;





template <class T, int dim>
class EquationOfStateOp : public ElasticityOp<T, dim>, public ConstitutiveModel::EquationOfState<T> {
public:
    using ConstitutiveModel::EquationOfState<T>::psi;
    using ConstitutiveModel::EquationOfState<T>::first_piola;
    using ConstitutiveModel::EquationOfState<T>::first_piola_derivative;
    using ElasticityOp<T, dim>::m_J;
    using ElasticityOp<T, dim>::m_global_index;
    using ElasticityOp<T, dim>::m_vol;
    Field<T> t_J; // only used in implicit
    T bulk, gamma;

    SERIALIZATION_REGISTER(m_J)
    SERIALIZATION_REGISTER(m_vol)
    SERIALIZATION_REGISTER(m_global_index)

    EquationOfStateOp(T bulk, T gamma)
        : bulk(bulk), gamma(gamma) {}

    void append(int start, int end, T vol) override
    {
        for (int i = start; i < end; ++i) {
            m_J.push_back(1);
            m_vol.push_back(vol);
            m_global_index.push_back(i);
        }
    }

    void evolve_strain(const Field<Matrix<T, dim, dim>>& m_gradXp) override
    {
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            m_J[i] = (1 + (m_gradXp[m_global_index[i]].trace() - dim)) * m_J[i];
        });
    }

    void trial_strain(const Field<Matrix<T, dim, dim>>& m_gradXp) override
    {
        t_J = m_J;
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            t_J[i] = (1 + (m_gradXp[m_global_index[i]].trace() - dim)) * m_J[i];
        });
    }

    void trial_energy(Field<T>& t_energy) override
    {
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            T energy = m_vol[i] * psi(t_J[i], bulk, gamma);
            t_energy[m_global_index[i]] += energy;
        });
    }

    void strain_energy(Field<T>& m_energy) override
    {
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            T energy = m_vol[i] * psi(m_J[i], bulk, gamma);
            m_energy[m_global_index[i]] = energy;
        });
    }

    void trial_gradient(Field<Matrix<T, dim, dim>>& t_gradient) override
    {
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            T P = first_piola(t_J[i], bulk, gamma);
            // Eqn 194. https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf
            Matrix<T, dim, dim> stress = m_vol[i] * P * m_J[i] * Matrix<T, dim, dim>::Identity();
            t_gradient[m_global_index[i]] += stress;
        });
    }

    void trial_hessian(Field<Matrix<T, dim * dim, dim * dim>>& t_hessian, bool project_pd) override
    {
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            T dPdJ = first_piola_derivative(t_J[i], bulk, gamma);
            Matrix<T, dim * dim, dim* dim> deformed_dPdF = Matrix<T, dim * dim, dim * dim>::Zero();
            for (int u = 0; u < dim; ++u)
                for (int x = 0; x < dim; ++x)
                    for (int p = 0; p < dim; ++p)
                        for (int y = 0; y < dim; ++y)
                            if (u == x && p == y)
                                deformed_dPdF(u + x * dim, p + y * dim) += dPdJ * m_J[i] * m_J[i];
            t_hessian[m_global_index[i]] += m_vol[i] * deformed_dPdF;
        });
    }

    void compute_stress(Field<Matrix<T, dim, dim>>& stress) override
    {
        BOW_TIMER_FLAG("compute elasticity");
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            T J = m_J[i];
            T P = first_piola(J, bulk, gamma);
            stress[m_global_index[i]] = m_vol[i] * P * J * Matrix<T, dim, dim>::Identity();
        });
    }

    void compute_volume(Field<T>& volume)
    {
        BOW_TIMER_FLAG("compute current volume (EOS)");
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            T J = m_J[i];
            volume[m_global_index[i]] = m_vol[i] * J;
        });
    }

    void compute_cauchy(Field<Matrix<T, dim, dim>>& stress)
    {
        BOW_TIMER_FLAG("compute cauchy stress (J-Based Fluid)");
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            T J = m_J[i];
            T P = first_piola(J, bulk, gamma);
            stress[m_global_index[i]] = P * Matrix<T, dim, dim>::Identity(); //piola is pressure here
        });
    }

    T stepsize_upperbound(const Field<Matrix<T, dim, dim>>& m_gradDXp) override
    {
        if (m_J.size() == 0) return 1.0;
        Vector<T, Eigen::Dynamic> alphas(m_J.size());
        alphas.setOnes();
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            T A = m_gradDXp[m_global_index[i]].trace() * m_J[i];
            T d = -0.9 * t_J[i];
            T alpha = d / A;
            if (alpha <= 0 || alpha > 1) alpha = 1;
            alphas(i) = alpha;
        });
        return alphas.minCoeff();
    }

    void set_dt(T _dt) override
    {
        return;
    }

    void collect_mu(Field<T>& _m_mu) override
    {
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            _m_mu[m_global_index[i]] = 0.0; //fluid has no mu
        });
    }

    void collect_la(Field<T>& _m_la) override
    {
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            _m_la[m_global_index[i]] = 0.0; //fluid has no la
        });
    }

    void collect_strain(Field<Matrix<T, dim, dim>>& _m_F) override
    {
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            _m_F[m_global_index[i]] = Matrix<T,dim,dim>::Identity();
        });
    }

    void compute_criticalStress(T percent, Field<Matrix<T, dim, dim>>& stretchedCauchy)
    {
        BOW_TIMER_FLAG("compute sigmaC (EOS)");
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            stretchedCauchy[m_global_index[i]] = Matrix<T, dim, dim>::Zero();
        });
    }

    void collect_initialVolume(Field<T>& m_initialVolume) override
    {
        tbb::parallel_for(size_t(0), m_vol.size(), [&](size_t i) {
            m_initialVolume[m_global_index[i]] = m_vol[i];
        });
    }

    void collect_chemPotential(Field<T>& _m_chemPotential) override
    {
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            _m_chemPotential[m_global_index[i]] = 0.0;
        });
    }

    void update_chemPotential(Field<T>& _m_chemPotential) override
    {
        return;
    }
};

template <class T, int dim>
class ViscousEquationOfStateOp : public ElasticityOp<T, dim>, public ConstitutiveModel::EquationOfState<T> {
public:
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    using ConstitutiveModel::EquationOfState<T>::psi;
    using ConstitutiveModel::EquationOfState<T>::first_piola;
    using ConstitutiveModel::EquationOfState<T>::first_piola_derivative;
    using ElasticityOp<T, dim>::m_F;
    using ElasticityOp<T, dim>::m_J;
    using ElasticityOp<T, dim>::m_global_index;
    using ElasticityOp<T, dim>::m_vol;
    Field<T> t_J; // only used in implicit
    Field<TM> m_Fprevious; //for computing Fdot
    T bulk, gamma, viscosity, dt;

    SERIALIZATION_REGISTER(m_J)
    SERIALIZATION_REGISTER(m_F)
    SERIALIZATION_REGISTER(m_Fprevious)
    SERIALIZATION_REGISTER(m_vol)
    SERIALIZATION_REGISTER(m_global_index)

    ViscousEquationOfStateOp(T bulk, T gamma, T viscosity)
        : bulk(bulk), gamma(gamma), viscosity(viscosity), dt(1.0) {}

    void append(int start, int end, T vol) override
    {
        for (int i = start; i < end; ++i) {
            m_J.push_back(1);
            m_F.push_back(Matrix<T, dim, dim>::Identity());
            m_Fprevious.push_back(Matrix<T, dim, dim>::Identity());
            m_vol.push_back(vol);
            m_global_index.push_back(i);
        }
    }

    void evolve_strain(const Field<Matrix<T, dim, dim>>& m_gradXp) override
    {
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            m_Fprevious[i] = m_F[i]; //save Fprevious
            m_F[i] = (m_gradXp[m_global_index[i]]) * m_F[i]; //also evolve F for viscosity!
            m_J[i] = (1 + (m_gradXp[m_global_index[i]].trace() - dim)) * m_J[i];
        });
    }

    // void trial_strain(const Field<Matrix<T, dim, dim>>& m_gradXp) override
    // {
    //     t_J = m_J;
    //     tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
    //         t_J[i] = (1 + (m_gradXp[m_global_index[i]].trace() - dim)) * m_J[i];
    //     });
    // }

    void strain_energy(Field<T>& m_energy) override  //TODO: eventually this needs to include the energy from the viscous term too!
    {
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            T energy = m_vol[i] * psi(m_J[i], bulk, gamma);
            m_energy[m_global_index[i]] = energy;
        });
    }

    // void trial_gradient(Field<Matrix<T, dim, dim>>& t_gradient) override
    // {
    //     tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
    //         T P = first_piola(t_J[i], bulk, gamma);
    //         // Eqn 194. https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf
    //         Matrix<T, dim, dim> stress = m_vol[i] * P * m_J[i] * Matrix<T, dim, dim>::Identity();
    //         t_gradient[m_global_index[i]] += stress;
    //     });
    // }

    // void trial_hessian(Field<Matrix<T, dim * dim, dim * dim>>& t_hessian, bool project_pd) override
    // {
    //     tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
    //         T dPdJ = first_piola_derivative(t_J[i], bulk, gamma);
    //         Matrix<T, dim * dim, dim* dim> deformed_dPdF = Matrix<T, dim * dim, dim * dim>::Zero();
    //         for (int u = 0; u < dim; ++u)
    //             for (int x = 0; x < dim; ++x)
    //                 for (int p = 0; p < dim; ++p)
    //                     for (int y = 0; y < dim; ++y)
    //                         if (u == x && p == y)
    //                             deformed_dPdF(u + x * dim, p + y * dim) += dPdJ * m_J[i] * m_J[i];
    //         t_hessian[m_global_index[i]] += m_vol[i] * deformed_dPdF;
    //     });
    // }

    void compute_volume(Field<T>& volume)
    {
        BOW_TIMER_FLAG("Compute Current Volume (V_p^n), (ViscousEOS)");
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            T J = m_J[i];
            volume[m_global_index[i]] = m_vol[i] * J;
        });
    }

    void compute_cauchy(Field<Matrix<T, dim, dim>>& stress)
    {
        BOW_TIMER_FLAG("Compute Cauchy Stress (Viscous Equation of State)");
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            TM F = m_F[i];
            TM Fprev = m_Fprevious[i];
            T J = m_J[i];
            T pressure = first_piola(J, bulk, gamma); //for EoS, P = cauchy, so this is actully just pressure!
            
            //Now compute viscous effects by computing the velocity gradient
            TM Fdot = (F - Fprev) / dt;
            TM Finv = F.inverse();
            TM L = Fdot * Finv; //velocity gradient, nabla v = L = Fdot * Finv

            stress[m_global_index[i]] = (pressure * Matrix<T, dim, dim>::Identity()) + (viscosity * (L + L.transpose())); //Cauchy Stress for Viscous Fluid: Cauchy = -pI + mu(L + L^T)
        });
    }

    void compute_stress(Field<Matrix<T, dim, dim>>& stress) override
    {
        BOW_TIMER_FLAG("Compute Stress (Viscous Equation of State)");
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            TM F = m_F[i];
            TM Fprev = m_Fprevious[i];
            T J = m_J[i];
            T pressure = first_piola(J, bulk, gamma); //for EoS, P = cauchy, so this is actully just pressure!
            
            //Now compute viscous effects by computing the velocity gradient
            TM Fdot = (F - Fprev) / dt;
            TM Finv = F.inverse();
            TM L = Fdot * Finv; //velocity gradient, nabla v = L = Fdot * Finv

            TM cauchy = (pressure * Matrix<T, dim, dim>::Identity()) + (viscosity * (L + L.transpose())); //Cauchy Stress for Viscous Fluid: Cauchy = -pI + mu(L + L^T)
            stress[m_global_index[i]] = m_vol[i] * J * cauchy; //force to integrate = VpN * cauchy = Vp0 * J * cauchy!
        });
    }

    void set_dt(T _dt){
        dt = _dt;
        return;
    }

    void collect_mu(Field<T>& _m_mu) override
    {
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            _m_mu[m_global_index[i]] = 0.0; //fluid has no mu
        });
    }

    void collect_la(Field<T>& _m_la) override
    {
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            _m_la[m_global_index[i]] = 0.0; //fluid has no la
        });
    }

    //for viscous Equation of State, collect J here and store it in F_11, pressure in F_22
    void collect_strain(Field<Matrix<T, dim, dim>>& _m_F) override 
    {
        tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
            T J = m_J[i];
            _m_F[m_global_index[i]](0,0) = J; //J (volumetric def grad)
            _m_F[m_global_index[i]](1,1) = -1 * first_piola(J, bulk, gamma); //pressure
        });
    }

    void compute_criticalStress(T percent, Field<Matrix<T, dim, dim>>& stretchedCauchy)
    {
        BOW_TIMER_FLAG("compute sigmaC (ViscousEOS)");
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            stretchedCauchy[m_global_index[i]] = Matrix<T, dim, dim>::Zero();
        });
    }

    void collect_initialVolume(Field<T>& m_initialVolume) override
    {
        tbb::parallel_for(size_t(0), m_vol.size(), [&](size_t i) {
            m_initialVolume[m_global_index[i]] = m_vol[i];
        });
    }

    void collect_chemPotential(Field<T>& _m_chemPotential) override
    {
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            _m_chemPotential[m_global_index[i]] = 0.0;
        });
    }

    void update_chemPotential(Field<T>& _m_chemPotential) override
    {
        return;
    }

    // T stepsize_upperbound(const Field<Matrix<T, dim, dim>>& m_gradDXp) override
    // {
    //     if (m_J.size() == 0) return 1.0;
    //     Vector<T, Eigen::Dynamic> alphas(m_J.size());
    //     alphas.setOnes();
    //     tbb::parallel_for(size_t(0), m_J.size(), [&](size_t i) {
    //         T A = m_gradDXp[m_global_index[i]].trace() * m_J[i];
    //         T d = -0.9 * t_J[i];
    //         T alpha = d / A;
    //         if (alpha <= 0 || alpha > 1) alpha = 1;
    //         alphas(i) = alpha;
    //     });
    //     return alphas.minCoeff();
    // }
};

} // namespace Bow::MPM

#endif
