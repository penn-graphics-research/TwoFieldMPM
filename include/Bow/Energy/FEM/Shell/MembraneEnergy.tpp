#include "Utils.h"
#include <Bow/Utils/Timer.h>

namespace Bow::Shell {
template <class T, class StorageIndex>
MembraneEnergyOp<T, StorageIndex>::MembraneEnergyOp(const Field<Vector<int, 3>>& elem, const Field<T>& vol, const Field<T>& mu, const Field<T>& lam, const Field<Matrix<T, codim, codim>>& IB)
    : m_elem(elem), m_vol(vol), m_mu(mu), m_lam(lam), m_IB(IB)
{
    this->name = "FEM-Membrane";
}

template <class T, class StorageIndex>
T MembraneEnergyOp<T, StorageIndex>::energy(const Field<Vector<T, dim>>& x)
{
    //TODO: parallelize
    T total_energy = 0.0;
    for (size_t i = 0; i < m_elem.size(); ++i) {
        const auto& vertices = m_elem[i];
        Matrix<T, codim, codim> F;
        deformation_gradient(x[vertices[0]], x[vertices[1]], x[vertices[2]], m_IB[i], F);
        T lnJ = 0.5 * std::log(F.determinant());
        total_energy += m_vol[i] * (0.5 * m_mu[i] * (F.trace() - codim - 2 * lnJ) + 0.5 * m_lam[i] * lnJ * lnJ);
    }
    return this->energy_scale * total_energy;
}

template <class T, class StorageIndex>
void MembraneEnergyOp<T, StorageIndex>::gradient(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& grad)
{
    //TODO: parallelize
    grad.resize(x.size());
    std::fill(grad.begin(), grad.end(), Vector<T, dim>::Zero());
    for (size_t e = 0; e < m_elem.size(); ++e) {
        const auto& vertices = m_elem[e];
        const Vector<T, dim>& x1 = x[vertices[0]];
        const Vector<T, dim>& x2 = x[vertices[1]];
        const Vector<T, dim>& x3 = x[vertices[2]];
        Matrix<T, codim, codim> A;
        first_fundamental_form(x1, x2, x3, A);
        Matrix<T, codim, codim> IA = A.inverse();
        T lnJ = 0.5 * std::log(A.determinant() * m_IB[e].determinant());
        Vector<T, codim * codim> de_div_dA;
        for (int i = 0; i < codim; ++i)
            for (int j = 0; j < codim; ++j)
                de_div_dA(j * codim + i) = m_vol[e] * ((0.5 * m_mu[e] * m_IB[e](i, j) + 0.5 * (-m_mu[e] + m_lam[e] * lnJ) * IA(i, j)));
        Matrix<T, 4, 9> dA_div_dx;
        dA_div_dx.setZero();
        dA_div_dx.template block<1, dim>(0, 3) += 2.0 * (x2 - x1).transpose();
        dA_div_dx.template block<1, dim>(0, 0) -= 2.0 * (x2 - x1).transpose();
        dA_div_dx.template block<1, dim>(1, 6) += (x2 - x1).transpose();
        dA_div_dx.template block<1, dim>(1, 3) += (x3 - x1).transpose();
        dA_div_dx.template block<1, dim>(1, 0) += -(x2 - x1).transpose() - (x3 - x1).transpose();
        dA_div_dx.template block<1, dim>(2, 6) += (x2 - x1).transpose();
        dA_div_dx.template block<1, dim>(2, 3) += (x3 - x1).transpose();
        dA_div_dx.template block<1, dim>(2, 0) += -(x2 - x1).transpose() - (x3 - x1).transpose();
        dA_div_dx.template block<1, dim>(3, 6) += 2.0 * (x3 - x1).transpose();
        dA_div_dx.template block<1, dim>(3, 0) -= 2.0 * (x3 - x1).transpose();

        Vector<T, 9> local_grad = this->energy_scale * dA_div_dx.transpose() * de_div_dA;
        // assemble global gradient
        for (int local_index = 0; local_index < dim; ++local_index) {
            grad[vertices(local_index)] += local_grad.template segment<dim>(local_index * dim);
        }
    }
}

template <class T, class StorageIndex>
template <bool project_pd>
void MembraneEnergyOp<T, StorageIndex>::hessian_impl(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess)
{
    BOW_TIMER_FLAG("Shell Membrane");
    //TODO: parallelize

    using IJK = Eigen::Triplet<T, StorageIndex>;
    std::vector<IJK> coeffs;

    Field<Matrix<T, 9, 9>> ahess(4, Matrix<T, 9, 9>::Zero());
    ahess[0].template block<dim, dim>(0, 0) += 2.0 * Matrix<T, dim, dim>::Identity();
    ahess[0].template block<dim, dim>(3, 3) += 2.0 * Matrix<T, dim, dim>::Identity();
    ahess[0].template block<dim, dim>(0, 3) -= 2.0 * Matrix<T, dim, dim>::Identity();
    ahess[0].template block<dim, dim>(3, 0) -= 2.0 * Matrix<T, dim, dim>::Identity();

    ahess[1].template block<dim, dim>(3, 6) += 1.0 * Matrix<T, dim, dim>::Identity();
    ahess[1].template block<dim, dim>(6, 3) += 1.0 * Matrix<T, dim, dim>::Identity();
    ahess[1].template block<dim, dim>(0, 3) -= 1.0 * Matrix<T, dim, dim>::Identity();
    ahess[1].template block<dim, dim>(0, 6) -= 1.0 * Matrix<T, dim, dim>::Identity();
    ahess[1].template block<dim, dim>(3, 0) -= 1.0 * Matrix<T, dim, dim>::Identity();
    ahess[1].template block<dim, dim>(6, 0) -= 1.0 * Matrix<T, dim, dim>::Identity();
    ahess[1].template block<dim, dim>(0, 0) += 2.0 * Matrix<T, dim, dim>::Identity();

    ahess[2].template block<dim, dim>(3, 6) += 1.0 * Matrix<T, dim, dim>::Identity();
    ahess[2].template block<dim, dim>(6, 3) += 1.0 * Matrix<T, dim, dim>::Identity();
    ahess[2].template block<dim, dim>(0, 3) -= 1.0 * Matrix<T, dim, dim>::Identity();
    ahess[2].template block<dim, dim>(0, 6) -= 1.0 * Matrix<T, dim, dim>::Identity();
    ahess[2].template block<dim, dim>(3, 0) -= 1.0 * Matrix<T, dim, dim>::Identity();
    ahess[2].template block<dim, dim>(6, 0) -= 1.0 * Matrix<T, dim, dim>::Identity();
    ahess[2].template block<dim, dim>(0, 0) += 2.0 * Matrix<T, dim, dim>::Identity();

    ahess[3].template block<dim, dim>(0, 0) += 2.0 * Matrix<T, dim, dim>::Identity();
    ahess[3].template block<dim, dim>(6, 6) += 2.0 * Matrix<T, dim, dim>::Identity();
    ahess[3].template block<dim, dim>(0, 6) -= 2.0 * Matrix<T, dim, dim>::Identity();
    ahess[3].template block<dim, dim>(6, 0) -= 2.0 * Matrix<T, dim, dim>::Identity();

    for (size_t e = 0; e < m_elem.size(); ++e) {
        const auto& vertices = m_elem[e];
        const Vector<T, dim>& x1 = x[vertices[0]];
        const Vector<T, dim>& x2 = x[vertices[1]];
        const Vector<T, dim>& x3 = x[vertices[2]];
        Matrix<T, codim, codim> A;
        first_fundamental_form(x1, x2, x3, A);
        Matrix<T, 4, 9> dA_div_dx;
        dA_div_dx.setZero();
        dA_div_dx.template block<1, dim>(0, 3) += 2.0 * (x2 - x1).transpose();
        dA_div_dx.template block<1, dim>(0, 0) -= 2.0 * (x2 - x1).transpose();
        dA_div_dx.template block<1, dim>(1, 6) += (x2 - x1).transpose();
        dA_div_dx.template block<1, dim>(1, 3) += (x3 - x1).transpose();
        dA_div_dx.template block<1, dim>(1, 0) += -(x2 - x1).transpose() - (x3 - x1).transpose();
        dA_div_dx.template block<1, dim>(2, 6) += (x2 - x1).transpose();
        dA_div_dx.template block<1, dim>(2, 3) += (x3 - x1).transpose();
        dA_div_dx.template block<1, dim>(2, 0) += -(x2 - x1).transpose() - (x3 - x1).transpose();
        dA_div_dx.template block<1, dim>(3, 6) += 2.0 * (x3 - x1).transpose();
        dA_div_dx.template block<1, dim>(3, 0) -= 2.0 * (x3 - x1).transpose();
        Matrix<T, codim, codim> IA = A.inverse();
        Vector<T, 9> ainvda = Vector<T, 9>::Zero();
        for (int endI = 0; endI < codim + 1; ++endI)
            for (int dimI = 0; dimI < dim; ++dimI)
                ainvda[endI * dim + dimI] = dA_div_dx(0, endI * dim + dimI) * IA(0, 0)
                    + dA_div_dx(1, endI * dim + dimI) * IA(1, 0)
                    + dA_div_dx(2, endI * dim + dimI) * IA(0, 1)
                    + dA_div_dx(3, endI * dim + dimI) * IA(1, 1);
        T deta = A.determinant();
        T lnJ = 0.5 * std::log(deta * m_IB[e].determinant());
        T term1 = (-m_mu[e] + m_lam[e] * lnJ) * 0.5;
        Matrix<T, 9, 9> local_hess = (-term1 + 0.25 * m_lam[e]) * (ainvda * ainvda.transpose());
        Matrix<T, 4, 9> aderivadj = Matrix<T, 4, 9>::Zero();
        aderivadj.row(0) = dA_div_dx.row(3);
        aderivadj.row(1) = -dA_div_dx.row(1);
        aderivadj.row(2) = -dA_div_dx.row(2);
        aderivadj.row(3) = dA_div_dx.row(0);
        local_hess += term1 / deta * aderivadj.transpose() * dA_div_dx;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                local_hess += (term1 * IA(i, j) + 0.5 * m_mu[e] * m_IB[e](i, j)) * ahess[i + j * 2];
        local_hess *= m_vol[e];
        if constexpr (project_pd)
            Math::make_pd(local_hess);
        local_hess *= this->energy_scale;

        int indMap[dim * (codim + 1)];
        for (int i = 0; i < codim + 1; ++i)
            for (int d = 0; d < dim; ++d)
                indMap[i * dim + d] = vertices[i] * dim + d;
        for (int row = 0; row < (codim + 1) * dim; ++row)
            for (int col = 0; col < (codim + 1) * dim; ++col) {
                coeffs.push_back(IJK(indMap[row], indMap[col], local_hess(row, col)));
            }
    }
    hess.resize(x.size() * dim, x.size() * dim);
    hess.setZero();
    hess.setFromTriplets(coeffs.begin(), coeffs.end());
}

} // namespace Bow::Shell