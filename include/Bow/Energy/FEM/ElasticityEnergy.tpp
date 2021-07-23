#include "ElasticityEnergy.h"
#include "Utils.h"
#include <Bow/Utils/Timer.h>
#include <oneapi/tbb.h>
#include <Bow/Math/SVD.h>

namespace Bow::FEM {
template <class T, int dim, class Model, bool inversion_free, class StorageIndex>
ElasticityEnergyOp<T, dim, Model, inversion_free, StorageIndex>::ElasticityEnergyOp(const Field<Vector<int, dim + 1>>& elem, const Field<T>& vol, const Field<T>& mu, const Field<T>& lam, const Field<Matrix<T, dim, dim>>& IB, std::vector<std::pair<int, int>>& offsets, T energy_scale)
    : m_elem(elem)
    , m_vol(vol)
    , m_mu(mu)
    , m_lam(lam)
    , m_IB(IB)
    , m_offsets(offsets)
{
    this->energy_scale = energy_scale;
    this->name = "FEM-Elasticity";
}

template <class T, int dim, class Model, bool inversion_free, class StorageIndex>
T ElasticityEnergyOp<T, dim, Model, inversion_free, StorageIndex>::energy(const Field<Vector<T, dim>>& x)
{
    T total_energy = 0.0;
    for (auto range : m_offsets) {
        Field<T> elem_energy(range.second - range.first);
        tbb::parallel_for(range.first, range.second, [&](int e) {
            Matrix<T, dim, dim> F;
            const auto& vertices = m_elem[e];
            if constexpr (dim == 2)
                deformation_gradient(x[vertices[0]], x[vertices[1]], x[vertices[2]], m_IB[e], F);
            else
                deformation_gradient(x[vertices[0]], x[vertices[1]], x[vertices[2]], x[vertices[3]], m_IB[e], F);
            elem_energy[e - range.first] = m_vol[e] * this->psi(F, m_mu[e], m_lam[e]);
        });
        total_energy += this->energy_scale * std::accumulate(elem_energy.begin(), elem_energy.end(), T(0));
    }
    return total_energy;
}

template <class T, int dim, class Model, bool inversion_free, class StorageIndex>
void ElasticityEnergyOp<T, dim, Model, inversion_free, StorageIndex>::gradient(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& grad)
{
    grad.resize(x.size());
    std::fill(grad.begin(), grad.end(), Vector<T, dim>::Zero());

    for (auto range : m_offsets) {
        // compute local gradients in parallel
        Field<Vector<T, dim*(dim + 1)>> local_grad(range.second - range.first);
        tbb::parallel_for(range.first, range.second, [&](int e) {
            // deformation gradient
            Matrix<T, dim, dim> F;
            const auto& vertices = m_elem[e];
            if constexpr (dim == 2)
                deformation_gradient(x[vertices[0]], x[vertices[1]], x[vertices[2]], m_IB[e], F);
            else
                deformation_gradient(x[vertices[0]], x[vertices[1]], x[vertices[2]], x[vertices[3]], m_IB[e], F);
            // first piola
            Matrix<T, dim, dim> P;
            this->first_piola(F, m_mu[e], m_lam[e], P);
            // backpropagate
            backpropagate_element_gradient(m_IB[e], P, local_grad[e - range.first]);
            local_grad[e - range.first] *= this->energy_scale * m_vol[e];
        });

        // assemble global gradient
        for (int e = range.first; e < range.second; ++e) {
            const auto& vertices = m_elem[e];
            for (int local_index = 0; local_index < dim + 1; ++local_index) {
                grad[vertices(local_index)] += local_grad[e - range.first].template segment<dim>(local_index * dim);
            }
        }
    }
}

template <class T, int dim, class Model, bool inversion_free, class StorageIndex>
void ElasticityEnergyOp<T, dim, Model, inversion_free, StorageIndex>::hessian(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess, bool project_pd)
{
    BOW_TIMER_FLAG("Elasticity");

    using IJK = Eigen::Triplet<T, StorageIndex>;
    static const int nrow = (dim + 1) * dim;
    static const int nelem = nrow * nrow;
    std::vector<IJK> coeffs;
    for (auto range : m_offsets) {
        const int coeff_ind0 = coeffs.size();
        coeffs.resize(coeff_ind0 + (range.second - range.first) * nelem);
        tbb::parallel_for(range.first, range.second, [&](int e) {
            // deformation gradient
            Matrix<T, dim, dim> F;
            const auto& vertices = m_elem[e];
            if constexpr (dim == 2)
                deformation_gradient(x[vertices[0]], x[vertices[1]], x[vertices[2]], m_IB[e], F);
            else
                deformation_gradient(x[vertices[0]], x[vertices[1]], x[vertices[2]], x[vertices[3]], m_IB[e], F);
            // first piola derivative
            Matrix<T, dim * dim, dim * dim> dPdF;
            this->first_piola_derivative(F, m_mu[e], m_lam[e], dPdF, project_pd);
            Matrix<T, nrow, nrow> local_hessian;
            backpropagate_element_hessian(m_IB[e], dPdF, local_hessian);
            local_hessian *= this->energy_scale * m_vol[e];
            int indMap[nrow];
            for (int i = 0; i < dim + 1; ++i)
                for (int d = 0; d < dim; ++d)
                    indMap[i * dim + d] = vertices[i] * dim + d;
            for (int row = 0; row < nrow; ++row)
                for (int col = 0; col < nrow; ++col) {
                    coeffs[coeff_ind0 + (e - range.first) * nelem + row * nrow + col] = std::move(IJK(indMap[row], indMap[col], local_hessian(row, col)));
                }
        });
    }
    hess.derived().resize(x.size() * dim, x.size() * dim);
    hess.derived().setZero();
    hess.derived().setFromTriplets(coeffs.begin(), coeffs.end());
}

template <class T, int dim, class Model, bool inversion_free, class StorageIndex>
void ElasticityEnergyOp<T, dim, Model, inversion_free, StorageIndex>::compute_cauchy(const Field<Vector<T, dim>>& x, Field<Matrix<T, dim, dim>>& cauchy)
{
    for (auto range : m_offsets) {
        tbb::parallel_for(range.first, range.second, [&](int e) {
            Matrix<T, dim, dim> F;
            const auto& vertices = m_elem[e];
            if constexpr (dim == 2)
                deformation_gradient(x[vertices[0]], x[vertices[1]], x[vertices[2]], m_IB[e], F);
            else
                deformation_gradient(x[vertices[0]], x[vertices[1]], x[vertices[2]], x[vertices[3]], m_IB[e], F);
            Matrix<T, dim, dim> P;
            this->first_piola(F, m_mu[e], m_lam[e], P);
            T J = F.determinant();
            cauchy[e] = (1.0 / J) * P * F.transpose();
        });
    }
}

template <class T, int dim, class Model, bool inversion_free, class StorageIndex>
void ElasticityEnergyOp<T, dim, Model, inversion_free, StorageIndex>::compute_von_mises(const Field<Vector<T, dim>>& x, Field<T>& von_mises)
{
    for (auto range : m_offsets) {
        tbb::parallel_for(range.first, range.second, [&](int e) {
            Matrix<T, dim, dim> F;
            const auto& vertices = m_elem[e];
            if constexpr (dim == 2)
                deformation_gradient(x[vertices[0]], x[vertices[1]], x[vertices[2]], m_IB[e], F);
            else
                deformation_gradient(x[vertices[0]], x[vertices[1]], x[vertices[2]], x[vertices[3]], m_IB[e], F);
            Matrix<T, dim, dim> U, V;
            Vector<T, dim> sigma;
            Math::svd(F, U, sigma, V);
            Matrix<T, dim, dim> Sigma = sigma.asDiagonal();
            Matrix<T, dim, dim> tau;
            this->first_piola(Sigma, m_mu[e], m_lam[e], tau);
            tau = tau * Sigma.transpose();
            if constexpr (dim == 2)
                von_mises[e] = std::abs(tau(0, 0) - tau(1, 1));
            else {
                Vector<T, dim> shifted_tau(tau(1, 1), tau(2, 2), tau(0, 0));
                von_mises[e] = std::sqrt(0.5 * (tau.diagonal() - shifted_tau).squaredNorm());
            }
        });
    }
}

template <class T, int dim, class Model, bool inversion_free, class StorageIndex>
T ElasticityEnergyOp<T, dim, Model, inversion_free, StorageIndex>::stepsize_upperbound(const Field<Vector<T, dim>>& x, const Field<Vector<T, dim>>& dx)
{
    if constexpr (!inversion_free) return 1.0;
    // https://www.dropbox.com/scl/fi/pdlkk5uzvlrpp9bd85ygo/Non-invertable-stepsize.paper?dl=0&rlkey=pm8gy8wqxset4rnt22rrfw1my
    Vector<T, Eigen::Dynamic> alphas(m_elem.size());
    alphas.setOnes();
    for (auto range : m_offsets)
        for (int e = range.first; e < range.second; ++e) {
            const auto& vert = m_elem[e];
            Matrix<T, dim, dim> basis, dbasis;
            basis.row(0) = x[vert[1]] - x[vert[0]];
            dbasis.row(0) = dx[vert[1]] - dx[vert[0]];
            basis.row(1) = x[vert[2]] - x[vert[0]];
            dbasis.row(1) = dx[vert[2]] - dx[vert[0]];
            if constexpr (dim == 3) {
                basis.row(2) = x[vert[3]] - x[vert[0]];
                dbasis.row(2) = dx[vert[3]] - dx[vert[0]];
            }
            if (dbasis.norm() == 0) continue;
            Matrix<T, dim, dim> A = basis.partialPivLu().solve(dbasis);

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
            alphas(e) = t;
        }
    if (alphas.size() == 0)
        return 1.0;
    else
        return alphas.minCoeff();
}
} // namespace Bow::FEM