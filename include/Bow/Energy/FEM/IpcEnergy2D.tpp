#include "IpcEnergy.h"
#include <Bow/Geometry/IpcUtils/CCD.h>
#include <Bow/Geometry/IpcUtils/PointPointDistance.h>
#include <Bow/Geometry/IpcUtils/PointEdgeDistance.h>
#include <Bow/Geometry/IpcUtils/PointTriangleDistance.h>
#include <Bow/Geometry/IpcUtils/EdgeEdgeDistance.h>
#include <Bow/Geometry/IpcUtils/DistanceType.h>
#include <Bow/Math/Barrier.h>
#include <Bow/Math/Utils.h>
#include <Bow/Utils/Logging.h>
#include <oneapi/tbb.h>
#include <Bow/Utils/Timer.h>

namespace Bow::FEM::IPC {

template <class T, class StorageIndex>
IpcEnergyOp2D<T, StorageIndex>::IpcEnergyOp2D(const Field<int>& boundary_points, const Field<Vector<int, 2>>& boundary_edges, const Field<T>& mass,
    const Field<T>& boundary_point_area, const Field<std::set<int>>& boundary_point_nb, T energy_scale)
    : m_boundary_points(boundary_points)
    , m_boundary_edges(boundary_edges)
    , m_mass(mass)
    , m_boundary_point_area(boundary_point_area)
    , m_boundary_point_nb(boundary_point_nb)
{
    this->energy_scale = energy_scale;
    this->name = "FEM-IPC";
}

template <class T, class StorageIndex>
void IpcEnergyOp2D<T, StorageIndex>::precompute(const Field<Vector<T, IpcEnergyOp2D<T, StorageIndex>::dim>>& x)
{
    BOW_TIMER_FLAG("IPC Constraint Set");

    PP.resize(0);
    wPP.resize(0);
    PE.resize(0);
    wPE.resize(0);

    {
        BOW_TIMER_FLAG("Build Hash");
        sh.build(x, m_boundary_points, m_boundary_edges, Field<Vector<int, 3>>(), 1.0);
    }

    T dHat2 = dHat * dHat;
    int bpI = 0;
    for (const auto p : m_boundary_points) {
        const T area_weight = m_boundary_point_area[bpI] / 2; // 1/2 to handle double counting for correct integration
        Field<int> edgeInds;
        sh.query_point_for_edges(x[p], dHat, edgeInds);
        for (const auto& beI : edgeInds) {
            const auto& edge = m_boundary_edges[beI];
            if (p == edge[0] || p == edge[1])
                continue;
            const auto& e0 = x[edge[0]];
            const auto& e1 = x[edge[1]];
            if (!Geometry::IPC::point_edge_cd_broadphase(x[p], e0, e1, dHat))
                continue;
            switch (Geometry::IPC::point_edge_distance_type(x[p], e0, e1)) {
            case 0:
                if (Geometry::IPC::point_point_distance(x[p], e0) < dHat2) {
                    PP.emplace_back(p, edge[0]);
                    wPP.emplace_back(area_weight);
                }
                break;
            case 1:
                if (Geometry::IPC::point_point_distance(x[p], e1) < dHat2) {
                    PP.emplace_back(p, edge[1]);
                    wPP.emplace_back(area_weight);
                }
                break;
            case 2:
                if (Geometry::IPC::point_edge_distance(x[p], e0, e1) < dHat2) {
                    PE.emplace_back(p, edge[0], edge[1]);
                    wPE.emplace_back(area_weight);
                }
                break;
            }
        }
        ++bpI;
    }
    if (improved_maxOp) {
        // better max() approximation via duplication subtraction
        for (size_t bpI = 0; bpI < m_boundary_points.size(); ++bpI) {
            int p = m_boundary_points[bpI];
            const T area_weight = m_boundary_point_area[bpI] / 2; // 1/2 to handle double counting for correct integration

            Field<int> pointInds;
            sh.query_point_for_points(x[p], dHat, pointInds);
            for (const auto& bpJ : pointInds) {
                int q = m_boundary_points[bpJ];
                if (q != p) {
                    if (bpJ >= (int)m_boundary_point_nb.size()) continue; // same as no neighbor
                    int incEdgeAmt = m_boundary_point_nb[bpJ].size() - (m_boundary_point_nb[bpJ].find(p) != m_boundary_point_nb[bpJ].end());
                    if (incEdgeAmt > 1) {
                        if (Geometry::IPC::point_point_distance(x[p], x[q]) < dHat2) {
                            PP.emplace_back(p, q);
                            wPP.emplace_back((1 - incEdgeAmt) * area_weight);
                        }
                    }
                }
            }
        }
    }
    //    Logging::info("# PP constraint: ", PP.size());
    //    Logging::info("# PE constraint: ", PE.size());
    if (mu > 0 && update_basis) {
        update_basis = false;

        // for error computation:
        Field<T> PP_normalForce_old = PP_normalForce;
        Field<T> PE_normalForce_old = PE_normalForce;
        Field<TV> PP_tanBasis_old = PP_tanBasis;
        Field<TV> PE_tanBasis_old = PE_tanBasis;

        PP_friction = PP;
        PE_friction = PE;
        PP_normalForce.clear();
        PE_normalForce.clear();
        PP_tanBasis.clear();
        PE_tanBasis.clear();
        PE_yita.clear();

        int ppI = 0;
        for (const auto& pp_pair : PP_friction) {
            TV p0 = x[pp_pair[0]], p1 = x[pp_pair[1]];
            T dist2 = Geometry::IPC::point_point_distance(p0, p1);
            T bGrad = Math::barrier_gradient(dist2, dHat2, kappa);
            PP_normalForce.push_back(-bGrad * 2 * wPP[ppI] * dHat * std::sqrt(dist2));

            TV m = (p1 - p0).normalized();
            PP_tanBasis.push_back(TV(m(1), -m(0)));

            ++ppI;
        }
        int peI = 0;
        for (const auto& pe_pair : PE_friction) {
            TV p = x[pe_pair[0]], e0 = x[pe_pair[1]], e1 = x[pe_pair[2]];
            T dist2 = Geometry::IPC::point_edge_distance(p, e0, e1);
            T bGrad = Math::barrier_gradient(dist2, dHat2, kappa);
            PE_normalForce.push_back(-bGrad * 2 * wPE[peI] * dHat * std::sqrt(dist2));

            TV m = (e1 - e0).normalized();
            PE_tanBasis.push_back(m);
            PE_yita.push_back(m.dot(p - e0) / (e1 - e0).norm());

            ++peI;
        }

        if (PP_normalForce.size() != PP_normalForce_old.size() || PE_normalForce.size() != PE_normalForce_old.size() || PP_normalForce.size() == 0 || PE_normalForce.size() == 0) {
            basis_update_error = 1e30;
            normal_force_update_error = 1e30;
            Logging::info("Friction pairs changed");
        }
        else {
            basis_update_error = (to_vec(PE_tanBasis_old) - to_vec(PE_tanBasis)).cwiseAbs().maxCoeff();
            basis_update_error = std::max(basis_update_error, (to_vec(PP_tanBasis_old) - to_vec(PP_tanBasis)).cwiseAbs().maxCoeff());
            normal_force_update_error = (to_vec(PE_normalForce_old) - to_vec(PE_normalForce)).cwiseAbs().maxCoeff();
            normal_force_update_error = std::max(normal_force_update_error, (to_vec(PP_normalForce_old) - to_vec(PP_normalForce_old)).cwiseAbs().maxCoeff());
            Logging::info("Friction basis update error: ", basis_update_error);
            Logging::info("Normal force update error: ", normal_force_update_error);
        }
    }
}

template <class T, class StorageIndex>
void IpcEnergyOp2D<T, StorageIndex>::internal_force(const Field<Vector<T, dim>>& x, const Field<Vector<T, dim>>& vn, Field<Vector<T, dim>>& force)
{
    bool update_basis_bk = update_basis;
    update_basis = true;
    precompute(x);
    update_basis = update_basis_bk;
    T dHat2 = dHat * dHat;
    force.resize(x.size());
    std::fill(force.begin(), force.end(), Vector<T, dim>::Zero());
    int ppI = 0;
    for (const auto& pp_pair : PP) {
        Vector<T, 2 * 2> PP_grad;
        Geometry::IPC::point_point_distance_gradient(x[pp_pair[0]], x[pp_pair[1]], PP_grad);
        T dist2 = Geometry::IPC::point_point_distance(x[pp_pair[0]], x[pp_pair[1]]);
        PP_grad *= wPP[ppI] * dHat * Math::barrier_gradient(dist2, dHat2, kappa);
        force[pp_pair[0]] += PP_grad.template segment<2>(0);
        force[pp_pair[1]] += PP_grad.template segment<2>(2);
        ++ppI;
    }
    int peI = 0;
    for (const auto& pe_pair : PE) {
        Vector<T, 2 * 3> PE_grad;
        Geometry::IPC::point_edge_distance_gradient(x[pe_pair[0]], x[pe_pair[1]], x[pe_pair[2]], PE_grad);
        T dist2 = Geometry::IPC::point_edge_distance(x[pe_pair[0]], x[pe_pair[1]], x[pe_pair[2]]);
        PE_grad *= wPE[peI] * dHat * Math::barrier_gradient(dist2, dHat2, kappa);
        force[pe_pair[0]] += PE_grad.template segment<2>(0);
        force[pe_pair[1]] += PE_grad.template segment<2>(2);
        force[pe_pair[2]] += PE_grad.template segment<2>(4);
        ++peI;
    }
    if (mu > 0) {
        Field<Vector<T, dim>> friction(x.size(), Vector<T, dim>::Zero());
        for (size_t i = 0; i < PP_friction.size(); ++i) {
            TV p0 = vn[PP_friction[i][0]], p1 = vn[PP_friction[i][1]];
            TV relDX2D;
            Point_Point_RelDX_2D(p0, p1, relDX2D);
            T relDX = PP_tanBasis[i].transpose() * relDX2D;
            T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDX * relDX, epsv);
            relDX *= f1_div_relDXNorm * mu * PP_normalForce[i];
            Eigen::Matrix<T, 4, 1> TTTDX;
            Point_Point_RelDXTan_To_Mesh_2D(relDX, PP_tanBasis[i], TTTDX);
            force[PP_friction[i][0]] += TTTDX.template segment<2>(0);
            force[PP_friction[i][1]] += TTTDX.template segment<2>(2);
        }
        for (size_t i = 0; i < PE_friction.size(); ++i) {
            TV p = vn[PE_friction[i][0]], e0 = vn[PE_friction[i][1]], e1 = vn[PE_friction[i][2]];
            TV relDX2D;
            Point_Edge_RelDX(p, e0, e1, PE_yita[i], relDX2D);
            T relDX = PE_tanBasis[i].transpose() * relDX2D;
            T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDX * relDX, epsv);
            relDX *= f1_div_relDXNorm * mu * PE_normalForce[i];
            Eigen::Matrix<T, 6, 1> TTTDX;
            Point_Edge_RelDXTan_To_Mesh_2D(relDX, PE_tanBasis[i], PE_yita[i], TTTDX);
            force[PE_friction[i][0]] += TTTDX.template segment<2>(0);
            force[PE_friction[i][1]] += TTTDX.template segment<2>(2);
            force[PE_friction[i][2]] += TTTDX.template segment<2>(4);
        }
    }
}

template <class T, class StorageIndex>
void IpcEnergyOp2D<T, StorageIndex>::callback(const Field<Vector<T, dim>>& x)
{
    //NOTE: kappa lower bound (adaptive kappa) is not necessary for the new constitutive IPC,
    // instead kappa can simply be set like a Young's modulus in elasticity (~10x that of density should be great)
    // T dHat2 = dHat * dHat;
    // T H_b = Math::barrier_hessian(1.0e-16, dHat2, 1.0);
    // T total_mass = Eigen::Map<const Vector<T, Eigen::Dynamic>>(m_mass.data(), m_mass.size()).sum();
    // // kappa = 1.0e16 * total_mass / T(m_mass.size()) / (4.0e-16 * H_b) / 0.516693 * 400;

    //TODO: implement kappa adjustment when distances become too tiny for numerical fail-safe
}

template <class T, class StorageIndex>
T IpcEnergyOp2D<T, StorageIndex>::stepsize_upperbound(const Field<Vector<T, dim>>& xn, const Field<Vector<T, dim>>& dx)
{
    BOW_TIMER_FLAG("CCD");

    T alpha = 1.0;
    {
        BOW_TIMER_FLAG("Build Hash");
        sh.build(xn, m_boundary_points, m_boundary_edges, Field<Vector<int, 3>>(), dx, alpha, 1.0, 0.0);
    }

    Field<T> alpha_PE(m_boundary_points.size(), alpha);
    tbb::parallel_for(size_t(0), m_boundary_points.size(), [&](size_t bpI) {
        const int p = m_boundary_points[bpI];
        Field<int> edgeInds;
        sh.query_point_for_edges(bpI, edgeInds);
        for (const auto& beI : edgeInds) {
            const auto& edge = m_boundary_edges[beI];
            if (p == edge[0] || p == edge[1])
                continue;
            if (Geometry::IPC::point_edge_ccd_broadphase(xn[p], xn[edge[0]], xn[edge[1]], dx[p], dx[edge[0]], dx[edge[1]], dHat))
                alpha_PE[bpI] = std::min(alpha_PE[bpI], Geometry::IPC::point_edge_ccd(xn[p], xn[edge[0]], xn[edge[1]], dx[p], dx[edge[0]], dx[edge[1]], 0.1));
        }
    });
    alpha = std::min(alpha, *std::min_element(alpha_PE.begin(), alpha_PE.end()));
    return alpha;
}

template <class T, class StorageIndex>
T IpcEnergyOp2D<T, StorageIndex>::energy(const Field<Vector<T, dim>>& x)
{
    T energy = 0;
    T dHat2 = dHat * dHat;
    int ppI = 0;
    for (const auto& pp_pair : PP) {
        T dist2 = Geometry::IPC::point_point_distance(x[pp_pair[0]], x[pp_pair[1]]);
        energy += this->energy_scale * wPP[ppI] * dHat * Math::barrier(dist2, dHat2, kappa);
        ++ppI;
    }
    int peI = 0;
    for (const auto& pe_pair : PE) {
        T dist2 = Geometry::IPC::point_edge_distance(x[pe_pair[0]], x[pe_pair[1]], x[pe_pair[2]]);
        energy += this->energy_scale * wPE[peI] * dHat * Math::barrier(dist2, dHat2, kappa);
        ++peI;
    }

    if (mu > 0) {
        for (size_t i = 0; i < PP_friction.size(); ++i) {
            TV p0 = x_weight * x[PP_friction[i][0]] - x_hat[PP_friction[i][0]], p1 = x_weight * x[PP_friction[i][1]] - x_hat[PP_friction[i][1]];
            TV relDX2D;
            Point_Point_RelDX_2D(p0, p1, relDX2D);
            T relDX = PP_tanBasis[i].transpose() * relDX2D;
            energy += this->energy_scale * f0_SF(relDX * relDX, epsvh) * mu * PP_normalForce[i] / x_weight;
        }
        for (size_t i = 0; i < PE_friction.size(); ++i) {
            TV p = x_weight * x[PE_friction[i][0]] - x_hat[PE_friction[i][0]], e0 = x_weight * x[PE_friction[i][1]] - x_hat[PE_friction[i][1]], e1 = x_weight * x[PE_friction[i][2]] - x_hat[PE_friction[i][2]];
            TV relDX2D;
            Point_Edge_RelDX(p, e0, e1, PE_yita[i], relDX2D);
            T relDX = PE_tanBasis[i].transpose() * relDX2D;
            energy += this->energy_scale * f0_SF(relDX * relDX, epsvh) * mu * PE_normalForce[i] / x_weight;
        }
    }
    return energy;
}

template <class T, class StorageIndex>
void IpcEnergyOp2D<T, StorageIndex>::gradient(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& grad)
{
    T dHat2 = dHat * dHat;
    grad.resize(x.size());
    std::fill(grad.begin(), grad.end(), Vector<T, dim>::Zero());
    int ppI = 0;
    for (const auto& pp_pair : PP) {
        Vector<T, 2 * 2> PP_grad;
        Geometry::IPC::point_point_distance_gradient(x[pp_pair[0]], x[pp_pair[1]], PP_grad);
        T dist2 = Geometry::IPC::point_point_distance(x[pp_pair[0]], x[pp_pair[1]]);
        PP_grad *= this->energy_scale * wPP[ppI] * dHat * Math::barrier_gradient(dist2, dHat2, kappa);
        grad[pp_pair[0]] += PP_grad.template segment<2>(0);
        grad[pp_pair[1]] += PP_grad.template segment<2>(2);
        ++ppI;
    }
    int peI = 0;
    for (const auto& pe_pair : PE) {
        Vector<T, 2 * 3> PE_grad;
        Geometry::IPC::point_edge_distance_gradient(x[pe_pair[0]], x[pe_pair[1]], x[pe_pair[2]], PE_grad);
        T dist2 = Geometry::IPC::point_edge_distance(x[pe_pair[0]], x[pe_pair[1]], x[pe_pair[2]]);
        PE_grad *= this->energy_scale * wPE[peI] * dHat * Math::barrier_gradient(dist2, dHat2, kappa);
        grad[pe_pair[0]] += PE_grad.template segment<2>(0);
        grad[pe_pair[1]] += PE_grad.template segment<2>(2);
        grad[pe_pair[2]] += PE_grad.template segment<2>(4);
        ++peI;
    }
    if (mu > 0) {
        for (size_t i = 0; i < PP_friction.size(); ++i) {
            TV p0 = x_weight * x[PP_friction[i][0]] - x_hat[PP_friction[i][0]], p1 = x_weight * x[PP_friction[i][1]] - x_hat[PP_friction[i][1]];
            TV relDX2D;
            Point_Point_RelDX_2D(p0, p1, relDX2D);
            T relDX = PP_tanBasis[i].transpose() * relDX2D;
            T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDX * relDX, epsvh);
            relDX *= f1_div_relDXNorm * mu * PP_normalForce[i];
            Eigen::Matrix<T, 4, 1> TTTDX;
            Point_Point_RelDXTan_To_Mesh_2D(relDX, PP_tanBasis[i], TTTDX);
            TTTDX *= this->energy_scale;
            grad[PP_friction[i][0]] += TTTDX.template segment<2>(0);
            grad[PP_friction[i][1]] += TTTDX.template segment<2>(2);
        }
        for (size_t i = 0; i < PE_friction.size(); ++i) {
            TV p = x_weight * x[PE_friction[i][0]] - x_hat[PE_friction[i][0]], e0 = x_weight * x[PE_friction[i][1]] - x_hat[PE_friction[i][1]], e1 = x_weight * x[PE_friction[i][2]] - x_hat[PE_friction[i][2]];
            TV relDX2D;
            Point_Edge_RelDX(p, e0, e1, PE_yita[i], relDX2D);
            T relDX = PE_tanBasis[i].transpose() * relDX2D;
            T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDX * relDX, epsvh);
            relDX *= f1_div_relDXNorm * mu * PE_normalForce[i];
            Eigen::Matrix<T, 6, 1> TTTDX;
            Point_Edge_RelDXTan_To_Mesh_2D(relDX, PE_tanBasis[i], PE_yita[i], TTTDX);
            TTTDX *= this->energy_scale;
            grad[PE_friction[i][0]] += TTTDX.template segment<2>(0);
            grad[PE_friction[i][1]] += TTTDX.template segment<2>(2);
            grad[PE_friction[i][2]] += TTTDX.template segment<2>(4);
        }
    }
}

template <class T, class StorageIndex>
template <bool project_pd>
void IpcEnergyOp2D<T, StorageIndex>::hessian_impl(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess) const
{
    BOW_TIMER_FLAG("IPC");

    T dHat2 = dHat * dHat;
    using IJK = Eigen::Triplet<T, StorageIndex>;
    std::vector<IJK> coeffs;
    int ppI = 0;
    for (const auto& pp_pair : PP) {
        Matrix<T, 2 * 2, 2 * 2> PP_hess;
        Geometry::IPC::point_point_distance_hessian(x[pp_pair[0]], x[pp_pair[1]], PP_hess);
        Vector<T, 2 * 2> PP_grad;
        Geometry::IPC::point_point_distance_gradient(x[pp_pair[0]], x[pp_pair[1]], PP_grad);
        T dist2 = Geometry::IPC::point_point_distance(x[pp_pair[0]], x[pp_pair[1]]);
        PP_hess = wPP[ppI] * dHat * Math::barrier_hessian(dist2, dHat2, kappa) * PP_grad * PP_grad.transpose() + wPP[ppI] * dHat * Math::barrier_gradient(dist2, dHat2, kappa) * PP_hess;
        if constexpr (project_pd)
            Math::make_pd(PP_hess);
        int indMap[] = { 2 * pp_pair[0], 2 * pp_pair[0] + 1, 2 * pp_pair[1], 2 * pp_pair[1] + 1 };
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                coeffs.push_back(IJK(indMap[i], indMap[j], this->energy_scale * PP_hess(i, j)));
        ++ppI;
    }
    int peI = 0;
    for (const auto& pe_pair : PE) {
        Matrix<T, 2 * 3, 2 * 3> PE_hess;
        Geometry::IPC::point_edge_distance_hessian(x[pe_pair[0]], x[pe_pair[1]], x[pe_pair[2]], PE_hess);
        Vector<T, 2 * 3> PE_grad;
        Geometry::IPC::point_edge_distance_gradient(x[pe_pair[0]], x[pe_pair[1]], x[pe_pair[2]], PE_grad);
        T dist2 = Geometry::IPC::point_edge_distance(x[pe_pair[0]], x[pe_pair[1]], x[pe_pair[2]]);
        PE_hess = wPE[peI] * dHat * Math::barrier_hessian(dist2, dHat2, kappa) * PE_grad * PE_grad.transpose() + wPE[peI] * dHat * Math::barrier_gradient(dist2, dHat2, kappa) * PE_hess;
        if constexpr (project_pd)
            Math::make_pd(PE_hess);
        int indMap[] = { 2 * pe_pair[0], 2 * pe_pair[0] + 1, 2 * pe_pair[1], 2 * pe_pair[1] + 1, 2 * pe_pair[2], 2 * pe_pair[2] + 1 };
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                coeffs.push_back(IJK(indMap[i], indMap[j], this->energy_scale * PE_hess(i, j)));
        ++peI;
    }

    if (mu > 0) {
        for (size_t i = 0; i < PP_friction.size(); ++i) {
            TV p0 = x_weight * x[PP_friction[i][0]] - x_hat[PP_friction[i][0]], p1 = x_weight * x[PP_friction[i][1]] - x_hat[PP_friction[i][1]];
            TV relDX2D;
            Point_Point_RelDX_2D(p0, p1, relDX2D);
            T relDX = PP_tanBasis[i].transpose() * relDX2D;

            Eigen::Matrix<T, 1, 4> TT;
            Point_Point_TT_2D(PP_tanBasis[i], TT);
            T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDX * relDX, epsvh);
            T f2_term = f2_SF_Term(relDX * relDX, epsvh);
            Eigen::Matrix<T, 4, 4> HessianI = Eigen::Matrix<T, 4, 4>::Zero();
            if (relDX * relDX >= epsvh * epsvh) {
                // zero
            }
            else {
                if (relDX == 0) {
                    if (PP_normalForce[i] > 0) {
                        // no SPD projection needed
                        HessianI = ((mu * PP_normalForce[i] * f1_div_relDXNorm) * TT.transpose()) * TT;
                    }
                    // if PP_normalForce[i] <= 0 due to max() approximation, then this SND hessian can be ignored.
                }
                else {
                    // only need to project the inner 2x2 matrix to SPD
                    T innerMtr = ((f2_term / std::abs(relDX)) * relDX) * relDX;
                    innerMtr += f1_div_relDXNorm;
                    innerMtr *= mu * PP_normalForce[i];
                    if constexpr (project_pd)
                        innerMtr = std::max(innerMtr, (T)0);
                    // tensor product:
                    HessianI = TT.transpose() * innerMtr * TT;
                }
            }
            HessianI *= x_weight * this->energy_scale;
            int cIVInd[2] = { PP_friction[i][0], PP_friction[i][1] };
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    for (int idI = 0; idI < 2; ++idI)
                        for (int jdI = 0; jdI < 2; ++jdI)
                            coeffs.emplace_back(
                                cIVInd[i] * 2 + idI,
                                cIVInd[j] * 2 + jdI,
                                HessianI(i * 2 + idI, j * 2 + jdI));
        }
        for (size_t i = 0; i < PE_friction.size(); ++i) {
            TV p = x_weight * x[PE_friction[i][0]] - x_hat[PE_friction[i][0]], e0 = x_weight * x[PE_friction[i][1]] - x_hat[PE_friction[i][1]], e1 = x_weight * x[PE_friction[i][2]] - x_hat[PE_friction[i][2]];
            TV relDX2D;
            Point_Edge_RelDX(p, e0, e1, PE_yita[i], relDX2D);
            T relDX = PE_tanBasis[i].transpose() * relDX2D;

            Eigen::Matrix<T, 1, 6> TT;
            Point_Edge_TT_2D(PE_tanBasis[i], PE_yita[i], TT);
            T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDX * relDX, epsvh);
            T f2_term = f2_SF_Term(relDX * relDX, epsvh);
            Eigen::Matrix<T, 6, 6> HessianI = Eigen::Matrix<T, 6, 6>::Zero();
            if (relDX * relDX >= epsvh * epsvh) {
                // zero
            }
            else {
                if (relDX == 0) {
                    if (PE_normalForce[i] > 0) {
                        // no SPD projection needed
                        HessianI = ((mu * PE_normalForce[i] * f1_div_relDXNorm) * TT.transpose()) * TT;
                    }
                    // if PE_normalForce[i] <= 0 due to max() approximation, then this SND hessian can be ignored.
                }
                else {
                    // only need to project the inner 2x2 matrix to SPD
                    T innerMtr = ((f2_term / std::abs(relDX)) * relDX) * relDX;
                    innerMtr += f1_div_relDXNorm;
                    innerMtr *= mu * PE_normalForce[i];
                    if constexpr (project_pd)
                        innerMtr = std::max(innerMtr, (T)0);
                    // tensor product:
                    HessianI = TT.transpose() * innerMtr * TT;
                }
            }
            HessianI *= x_weight * this->energy_scale;
            int cIVInd[3] = { PE_friction[i][0], PE_friction[i][1], PE_friction[i][2] };
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    for (int idI = 0; idI < 2; ++idI)
                        for (int jdI = 0; jdI < 2; ++jdI)
                            coeffs.emplace_back(
                                cIVInd[i] * 2 + idI,
                                cIVInd[j] * 2 + jdI,
                                HessianI(i * 2 + idI, j * 2 + jdI));
        }
    }

    hess.resize(x.size() * dim, x.size() * dim);
    hess.setZero();
    hess.setFromTriplets(coeffs.begin(), coeffs.end());
}
} // namespace Bow::FEM::IPC
