#include "BendingEnergy.h"
#include <Bow/Geometry/DihedralAngle.h>
#include <oneapi/tbb.h>
#include <Bow/IO/ply.h>
#include <Bow/Utils/Timer.h>

namespace Bow::Shell {

template <class T, class StorageIndex>
BendingEnergyOp<T, StorageIndex>::BendingEnergyOp(const Field<Vector<int, 4>>& edge_stencil, const Field<T>& e, const Field<T>& h, const Field<T>& rest_angle, const Field<T>& bend_stiff)
    : m_edge_stencil(edge_stencil), m_e(e), m_h(h), m_rest_angle(rest_angle), m_bend_stiff(bend_stiff)
{
    this->name = "FEM-Bending";
}

template <class T, class StorageIndex>
T BendingEnergyOp<T, StorageIndex>::energy(const Field<Vector<T, dim>>& x)
{
    //TODO: parallelize
    using namespace Geometry;
    T total_energy = 0.0;
    for (size_t e = 0; e < m_edge_stencil.size(); ++e) {
        if (m_edge_stencil[e](3) < 0) continue;
        T theta = dihedral_angle(x[m_edge_stencil[e](2)], x[m_edge_stencil[e](0)], x[m_edge_stencil[e](1)], x[m_edge_stencil[e](3)], angle_branch[e]);
        T ben = ((theta - m_rest_angle[e]) * (theta - m_rest_angle[e])) * m_e[e] / m_h[e];
        total_energy += m_bend_stiff[e] * ben;
    }
    return this->energy_scale * total_energy;
}

template <class T, class StorageIndex>
void BendingEnergyOp<T, StorageIndex>::gradient(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& grad)
{
    //TODO: parallelize
    using namespace Geometry;
    grad.resize(x.size());
    std::fill(grad.begin(), grad.end(), Vector<T, dim>::Zero());
    for (size_t e = 0; e < m_edge_stencil.size(); ++e) {
        if (m_edge_stencil[e](3) < 0) continue;
        T theta = dihedral_angle(x[m_edge_stencil[e](2)], x[m_edge_stencil[e](0)], x[m_edge_stencil[e](1)], x[m_edge_stencil[e](3)], angle_branch[e]);
        Vector<T, 12> local_grad;
        dihedral_angle_gradient(x[m_edge_stencil[e](2)], x[m_edge_stencil[e](0)], x[m_edge_stencil[e](1)], x[m_edge_stencil[e](3)], local_grad);
        local_grad *= this->energy_scale * m_bend_stiff[e] * 2 * (theta - m_rest_angle[e]) * m_e[e] / m_h[e];
        grad[m_edge_stencil[e](2)] += local_grad.template segment<dim>(0);
        grad[m_edge_stencil[e](0)] += local_grad.template segment<dim>(3);
        grad[m_edge_stencil[e](1)] += local_grad.template segment<dim>(6);
        grad[m_edge_stencil[e](3)] += local_grad.template segment<dim>(9);
    }
}

template <class T, class StorageIndex>
template <bool project_pd>
void BendingEnergyOp<T, StorageIndex>::hessian_impl(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess)
{
    BOW_TIMER_FLAG("Shell Bending");
    //TODO: parallelize

    using namespace Geometry;
    using IJK = Eigen::Triplet<T, StorageIndex>;
    std::vector<IJK> coeffs;
    for (size_t e = 0; e < m_edge_stencil.size(); ++e) {
        if (m_edge_stencil[e](3) < 0) continue;
        T theta = dihedral_angle(x[m_edge_stencil[e](2)], x[m_edge_stencil[e](0)], x[m_edge_stencil[e](1)], x[m_edge_stencil[e](3)], angle_branch[e]);
        Vector<T, 12> local_grad;
        dihedral_angle_gradient(x[m_edge_stencil[e](2)], x[m_edge_stencil[e](0)], x[m_edge_stencil[e](1)], x[m_edge_stencil[e](3)], local_grad);
        Matrix<T, 12, 12> local_hessian;
        dihedral_angle_hessian(x[m_edge_stencil[e](2)], x[m_edge_stencil[e](0)], x[m_edge_stencil[e](1)], x[m_edge_stencil[e](3)], local_hessian);
        local_hessian *= m_bend_stiff[e] * 2.0 * (theta - m_rest_angle[e]) * m_e[e] / m_h[e];
        local_hessian += (m_bend_stiff[e] * 2.0 * m_e[e] / m_h[e]) * local_grad * local_grad.transpose();
        if constexpr (project_pd)
            Math::make_pd(local_hessian);
        local_hessian *= this->energy_scale;
        int indMap[dim * 4] = { m_edge_stencil[e](2) * dim, m_edge_stencil[e](2) * dim + 1, m_edge_stencil[e](2) * dim + 2,
            m_edge_stencil[e](0) * dim, m_edge_stencil[e](0) * dim + 1, m_edge_stencil[e](0) * dim + 2,
            m_edge_stencil[e](1) * dim, m_edge_stencil[e](1) * dim + 1, m_edge_stencil[e](1) * dim + 2,
            m_edge_stencil[e](3) * dim, m_edge_stencil[e](3) * dim + 1, m_edge_stencil[e](3) * dim + 2 };
        for (int row = 0; row < 12; ++row)
            for (int col = 0; col < 12; ++col) {
                coeffs.push_back(IJK(indMap[row], indMap[col], local_hessian(row, col)));
            }
    }
    hess.resize(x.size() * dim, x.size() * dim);
    hess.setZero();
    hess.setFromTriplets(coeffs.begin(), coeffs.end());
}

template <class T, class StorageIndex>
void BendingEnergyOp<T, StorageIndex>::precompute(const Field<Vector<T, dim>>& x)
{
    using namespace Geometry;
    angle_branch.resize(m_edge_stencil.size(), 0);
    tbb::parallel_for(size_t(0), m_edge_stencil.size(), [&](size_t e) {
        if (m_edge_stencil[e](3) < 0) return;
        T theta = dihedral_angle(x[m_edge_stencil[e](2)], x[m_edge_stencil[e](0)], x[m_edge_stencil[e](1)], x[m_edge_stencil[e](3)], angle_branch[e]);
        if (theta >= M_PI / 2)
            angle_branch[e] = 1;
        else if (theta <= -M_PI / 2)
            angle_branch[e] = -1;
        else
            angle_branch[e] = 0;
    });
}
} // namespace Bow::Shell