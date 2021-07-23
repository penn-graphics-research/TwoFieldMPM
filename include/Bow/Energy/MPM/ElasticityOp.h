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
    std::vector<int> m_global_index;
    virtual void append(int start, int end, T vol) = 0;
    virtual void compute_stress(Field<Matrix<T, dim, dim>>& stress) {}
    virtual void compute_cauchy(Field<Matrix<T, dim, dim>>& stress) {}
    virtual void compute_von_mises(Field<T>& stress) {}
    virtual void compute_criticalStress(T percent, Field<Matrix<T, dim, dim>>& stretchedCauchy) { BOW_NOT_IMPLEMENTED }
    virtual void evolve_strain(const Field<Matrix<T, dim, dim>>& m_gradXp) = 0;
    virtual void trial_strain(const Field<Matrix<T, dim, dim>>& m_gradXp) { BOW_NOT_IMPLEMENTED }
    virtual void trial_energy(Field<T>& t_energy) { BOW_NOT_IMPLEMENTED }
    virtual void trial_gradient(Field<Matrix<T, dim, dim>>& t_gradient) { BOW_NOT_IMPLEMENTED }
    virtual void trial_differential(const Field<Matrix<T, dim, dim>>& d_F, Field<Matrix<T, dim, dim>>& t_differential, bool project_pd) { BOW_NOT_IMPLEMENTED }
    virtual void trial_hessian(Field<Matrix<T, dim * dim, dim * dim>>& t_hessian, bool project_pd) { BOW_NOT_IMPLEMENTED }
    virtual T stepsize_upperbound(const Field<Matrix<T, dim, dim>>& m_gradDXp) { return 1.0; }
    virtual void collect_strain(Field<Matrix<T, dim, dim>>& m_Fs) { BOW_NOT_IMPLEMENTED }
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
        BOW_TIMER_FLAG("compute cauchy");
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            Matrix<T, dim, dim> F = m_F[i];
            Matrix<T, dim, dim> P;
            T J = F.determinant();
            first_piola(F, m_mu[i], m_lambda[i], P);
            cauchy[m_global_index[i]] = (1.0 / J) * P * F.transpose();
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

    //this will have to be written for each elasticity model we want to use with MPM damage
    void compute_criticalStress(T percent, Field<Matrix<T, dim, dim>>& stretchedCauchy)
    {
        BOW_TIMER_FLAG("compute sigmaC");
        tbb::parallel_for(size_t(0), stretchedCauchy.size(), [&](size_t i) {
            TM F = TM::Identity() * (1.0 + percent); //stretched F
            T J = F.determinant();
            Matrix<T, dim, dim> P;
            first_piola(F, m_mu[i], m_lambda[i], P);
            stretchedCauchy[i] = (1.0 / J) * P * F.transpose();
        });
    }

    void collect_strain(Field<Matrix<T, dim, dim>>& m_Fs) override
    {
        tbb::parallel_for(size_t(0), m_F.size(), [&](size_t i) {
            m_Fs[m_global_index[i]] = m_F[i];
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
};

template <class T, int dim>
using FixedCorotatedOp = FBasedElastiticityOp<T, dim, ConstitutiveModel::FixedCorotated<T, dim>, false>;

template <class T, int dim>
using NeoHookeanOp = FBasedElastiticityOp<T, dim, ConstitutiveModel::NeoHookean<T, dim>, true>;

template <class T, int dim>
using StvkWithHenckyOp = FBasedElastiticityOp<T, dim, ConstitutiveModel::StvkWithHencky<T, dim>, true>;

template <class T, int dim>
using LinearElasticityOp = FBasedElastiticityOp<T, dim, ConstitutiveModel::LinearElasticity<T, dim>, false>;

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
};

} // namespace Bow::MPM

#endif