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
/** ############### IPC 3D ################# */

template <class T, class StorageIndex>
void IpcEnergyOp3D<T, StorageIndex>::friction_precompute(const Field<Vector<T, dim>>& x)
{
    update_basis = false;
    PP_friction = PP;
    PE_friction = PE;
    PT_friction = PT;
    EE_friction = EE;
    PP_normalForce.clear();
    PE_normalForce.clear();
    PT_normalForce.clear();
    EE_normalForce.clear();
    PP_tanBasis.clear();
    PE_tanBasis.clear();
    PT_tanBasis.clear();
    EE_tanBasis.clear();
    PE_yita.clear();
    PT_beta.clear();
    EE_gamma.clear();
    T dHat2 = dHat * dHat;
    int ppI = 0;
    for (const auto& pp_pair : PP_friction) {
        const auto &p0 = x[pp_pair[0]], p1 = x[pp_pair[1]];
        T dist2 = Geometry::IPC::point_point_distance(p0, p1);
        T bGrad = Math::barrier_gradient(dist2, dHat2, kappa);
        PP_normalForce.push_back(-bGrad * 2 * wPP[ppI] * dHat * std::sqrt(dist2));
        Matrix<T, dim, dim - 1> basis;
        Point_Point_Tangent_Basis(p0, p1, basis);
        PP_tanBasis.push_back(basis);
        ++ppI;
    }
    int peI = 0;
    for (const auto& pe_pair : PE_friction) {
        const auto &p = x[pe_pair[0]], e0 = x[pe_pair[1]], e1 = x[pe_pair[2]];
        T dist2 = Geometry::IPC::point_edge_distance(p, e0, e1);
        T bGrad = Math::barrier_gradient(dist2, dHat2, kappa);
        PE_normalForce.push_back(-bGrad * 2 * wPE[peI] * dHat * std::sqrt(dist2));
        Matrix<T, dim, dim - 1> basis;
        Point_Edge_Tangent_Basis(p, e0, e1, basis);
        PE_tanBasis.push_back(basis);
        PE_yita.push_back((p - e0).dot(e1 - e0) / (e1 - e0).squaredNorm());
        ++peI;
    }
    int ptI = 0;
    for (const auto& pt_pair : PT_friction) {
        const auto &p = x[pt_pair[0]], v0 = x[pt_pair[1]], v1 = x[pt_pair[2]], v2 = x[pt_pair[3]];
        T dist2 = Geometry::IPC::point_triangle_distance(p, v0, v1, v2);
        T bGrad = Math::barrier_gradient(dist2, dHat2, kappa);
        PT_normalForce.push_back(-bGrad * 2 * wPT[ptI] * dHat * std::sqrt(dist2));
        Matrix<T, dim, dim - 1> basis;
        Point_Triangle_Tangent_Basis(p, v0, v1, v2, basis);

        PT_tanBasis.push_back(basis);
        Eigen::Matrix<T, 2, 3> triangle_basis;
        triangle_basis.row(0) = (v1 - v0).transpose();
        triangle_basis.row(1) = (v2 - v0).transpose();
        PT_beta.push_back((triangle_basis * triangle_basis.transpose()).ldlt().solve(triangle_basis * (p - v0)));
        ++ptI;
    }
    int eeI = 0;
    for (const auto& ee_pair : EE_friction) {
        const auto &v0 = x[ee_pair[0]], v1 = x[ee_pair[1]], v2 = x[ee_pair[2]], v3 = x[ee_pair[3]];
        T dist2 = Geometry::IPC::edge_edge_distance(v0, v1, v2, v3);
        T bGrad = Math::barrier_gradient(dist2, dHat2, kappa);
        EE_normalForce.push_back(-bGrad * 2 * wEE[eeI] * dHat * std::sqrt(dist2));
        Matrix<T, dim, dim - 1> basis;
        Edge_Edge_Tangent_Basis(v0, v1, v2, v3, basis);
        EE_tanBasis.push_back(basis);

        Eigen::Matrix<T, 1, 3> e20 = (v0 - v2).transpose();
        Eigen::Matrix<T, 1, 3> e01 = (v1 - v0).transpose();
        Eigen::Matrix<T, 1, 3> e23 = (v3 - v2).transpose();

        Eigen::Matrix<T, 2, 2> coefMtr;
        coefMtr(0, 0) = e01.squaredNorm();
        coefMtr(0, 1) = coefMtr(1, 0) = -e23.dot(e01);
        coefMtr(1, 1) = e23.squaredNorm();

        Eigen::Matrix<T, 2, 1> rhs;
        rhs[0] = -e20.dot(e01);
        rhs[1] = e20.dot(e23);
        EE_gamma.push_back(coefMtr.ldlt().solve(rhs));
        ++eeI;
    }
}

template <class T, class StorageIndex>
IpcEnergyOp3D<T, StorageIndex>::IpcEnergyOp3D(const Field<int>& boundary_points, const Field<Vector<int, 2>>& boundary_edges, const Field<Vector<int, 3>>& boundary_faces,
    const Field<Vector<T, dim>>& X, const Field<T>& mass, const Field<T>& boundary_point_area, const Field<int8_t>& boundary_point_type, const Field<std::set<int>>& boundary_point_nb,
    const Field<T>& boundary_edge_area, const Field<std::set<int>>& boundary_edge_pnb, T energy_scale)
    : m_boundary_points(boundary_points)
    , m_boundary_edges(boundary_edges)
    , m_boundary_faces(boundary_faces)
    , m_X(X)
    , m_mass(mass)
    , m_boundary_point_area(boundary_point_area)
    , m_boundary_point_type(boundary_point_type)
    , m_boundary_point_nb(boundary_point_nb)
    , m_boundary_edge_area(boundary_edge_area)
    , m_boundary_edge_pnb(boundary_edge_pnb)
{
    this->energy_scale = energy_scale;
    this->name = "FEM-IPC";
}

template <class T, class StorageIndex>
void IpcEnergyOp3D<T, StorageIndex>::precompute(const Field<Vector<T, dim>>& x)
{
    BOW_TIMER_FLAG("IPC Constraint Set");

    PP.resize(0);
    wPP.resize(0);
    PE.resize(0);
    wPE.resize(0);
    PT.resize(0);
    wPT.resize(0);
    PPM.resize(0);
    wPPM.resize(0);
    PEM.resize(0);
    wPEM.resize(0);
    EEM.resize(0);
    wEEM.resize(0);
    EE.resize(0);
    wEE.resize(0);

    {
        BOW_TIMER_FLAG("Build Hash");
        sh.build(x, m_boundary_points, m_boundary_edges, m_boundary_faces, 1.0);
    }
    point_triangle_constraints(x);
    edge_edge_constraints(x);

    // Logging::info("# PP constraint: ", PP.size());
    // Logging::info("# PE constraint: ", PE.size());
    // Logging::info("# PT constraint: ", PT.size());
    // Logging::info("# PPM constraint: ", PPM.size());
    // Logging::info("# PEM constraint: ", PEM.size());
    // Logging::info("# EEM constraint: ", EEM.size());
    // Logging::info("# EE constraint: ", EE.size());
    if (mu > 0 && update_basis) {
        friction_precompute(x);
    }
}

template <class T, class StorageIndex>
void IpcEnergyOp3D<T, StorageIndex>::callback(const Field<Vector<T, dim>>& x)
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
T IpcEnergyOp3D<T, StorageIndex>::stepsize_upperbound(const Field<Vector<T, dim>>& xn, const Field<Vector<T, dim>>& dx)
{
    BOW_TIMER_FLAG("CCD");
    T alpha = 1.0;
    {
        BOW_TIMER_FLAG("Build Hash");
        sh.build(xn, m_boundary_points, m_boundary_edges, m_boundary_faces, dx, alpha, 1.0, 0.0);
    }
    {
        BOW_TIMER_FLAG("PT");
        Field<T> alpha_PT(m_boundary_points.size(), alpha);
        tbb::parallel_for(size_t(0), m_boundary_points.size(), [&](size_t bpI) {
            const auto& p = m_boundary_points[bpI];
            Field<int> pointInds;
            Field<int> edgeInds;
            Field<int> triInds;
            sh.query_point_for_primitives(bpI, pointInds, edgeInds, triInds);
            for (const auto& bfI : triInds) {
                const auto face = m_boundary_faces[bfI];
                if (p == face[0] || p == face[1] || p == face[2])
                    continue;
                if (Geometry::IPC::point_triangle_ccd_broadphase(xn[p], xn[face[0]], xn[face[1]], xn[face[2]], dx[p], dx[face[0]], dx[face[1]], dx[face[2]], dHat))
                    alpha_PT[bpI] = std::min(alpha_PT[bpI], Geometry::IPC::point_triangle_ccd(xn[p], xn[face[0]], xn[face[1]], xn[face[2]], dx[p], dx[face[0]], dx[face[1]], dx[face[2]], 0.1, 0.0));
            }
            //TODO: PE and PP for C-IPC
        });
        alpha = std::min(alpha, *std::min_element(alpha_PT.begin(), alpha_PT.end()));
    }
    {
        BOW_TIMER_FLAG("EE");
        Field<T> alpha_EE(m_boundary_edges.size(), alpha);
        tbb::parallel_for(size_t(0), m_boundary_edges.size() - 1, [&](size_t i) {
            auto edge0 = m_boundary_edges[i];
            Field<int> edgeInds;
            sh.query_edge_for_edges(i, edgeInds);
            for (const auto& j : edgeInds) {
                auto edge1 = m_boundary_edges[j];
                if (edge0[0] == edge1[0] || edge1[1] == edge0[1] || edge0[0] == edge1[1] || edge1[0] == edge0[1])
                    continue;
                if (Geometry::IPC::edge_edge_ccd_broadphase(xn[edge0[0]], xn[edge0[1]], xn[edge1[0]], xn[edge1[1]], dx[edge0[0]], dx[edge0[1]], dx[edge1[0]], dx[edge1[1]], dHat))
                    alpha_EE[i] = std::min(alpha_EE[i], Geometry::IPC::edge_edge_ccd(xn[edge0[0]], xn[edge0[1]], xn[edge1[0]], xn[edge1[1]], dx[edge0[0]], dx[edge0[1]], dx[edge1[0]], dx[edge1[1]], 0.1, 0.0));
            }
        });
        alpha = std::min(alpha, *std::min_element(alpha_EE.begin(), alpha_EE.end()));
    }
    return alpha;
}

template <class T, class StorageIndex>
T IpcEnergyOp3D<T, StorageIndex>::energy(const Field<Vector<T, dim>>& x)
{
    T energy = 0;
    T dHat2 = dHat * dHat;

    // PP
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
    int ptI = 0;
    for (const auto& pt_pair : PT) {
        T dist2 = Geometry::IPC::point_triangle_distance(x[pt_pair[0]], x[pt_pair[1]], x[pt_pair[2]], x[pt_pair[3]]);
        energy += this->energy_scale * wPT[ptI] * dHat * Math::barrier(dist2, dHat2, kappa);
        ++ptI;
    }

    // EE
    auto mollifier_info = [&](const Vector<int, 4>& pair_info) {
        const Vector<T, dim>& ea0_rest = m_X[pair_info[0]];
        const Vector<T, dim>& ea1_rest = m_X[pair_info[1]];
        const Vector<T, dim>& eb0_rest = m_X[pair_info[2]];
        const Vector<T, dim>& eb1_rest = m_X[pair_info[3]];
        T eps_x = Geometry::IPC::edge_edge_mollifier_threshold(ea0_rest, ea1_rest, eb0_rest, eb1_rest);
        return Geometry::IPC::edge_edge_mollifier(x[pair_info[0]], x[pair_info[1]], x[pair_info[2]], x[pair_info[3]], eps_x);
    };
    int ppmI = 0;
    for (const auto& pp_pair : PPM) {
        T dist2 = Geometry::IPC::point_point_distance(x[pp_pair[0]], x[pp_pair[2]]);
        energy += this->energy_scale * wPPM[ppmI] * dHat * mollifier_info(pp_pair) * Math::barrier(dist2, dHat2, kappa);
        ++ppmI;
    }
    int pemI = 0;
    for (const auto& pe_pair : PEM) {
        T dist2 = Geometry::IPC::point_edge_distance(x[pe_pair[0]], x[pe_pair[2]], x[pe_pair[3]]);
        energy += this->energy_scale * wPEM[pemI] * dHat * mollifier_info(pe_pair) * Math::barrier(dist2, dHat2, kappa);
        ++pemI;
    }
    int eemI = 0;
    for (const auto& ee_pair : EEM) {
        T dist2 = Geometry::IPC::edge_edge_distance(x[ee_pair[0]], x[ee_pair[1]], x[ee_pair[2]], x[ee_pair[3]]);
        energy += this->energy_scale * wEEM[eemI] * dHat * mollifier_info(ee_pair) * Math::barrier(dist2, dHat2, kappa);
        ++eemI;
    }
    int eeI = 0;
    for (const auto& ee_pair : EE) {
        T dist2 = Geometry::IPC::edge_edge_distance(x[ee_pair[0]], x[ee_pair[1]], x[ee_pair[2]], x[ee_pair[3]]);
        energy += this->energy_scale * wEE[eeI] * dHat * Math::barrier(dist2, dHat2, kappa);
        ++eeI;
    }

    if (mu > 0) {
        for (size_t i = 0; i < PP_friction.size(); ++i) {
            TV p0 = x_weight * x[PP_friction[i][0]] - x_hat[PP_friction[i][0]], p1 = x_weight * x[PP_friction[i][1]] - x_hat[PP_friction[i][1]];
            TV relDX3D;
            Point_Point_RelDX_3D(p0, p1, relDX3D);
            T relDXSqNorm = (PP_tanBasis[i].transpose() * relDX3D).squaredNorm();
            energy += this->energy_scale * f0_SF(relDXSqNorm, epsvh) * mu * PP_normalForce[i] / x_weight;
        }
        for (size_t i = 0; i < PE_friction.size(); ++i) {
            TV p = x_weight * x[PE_friction[i][0]] - x_hat[PE_friction[i][0]], e0 = x_weight * x[PE_friction[i][1]] - x_hat[PE_friction[i][1]], e1 = x_weight * x[PE_friction[i][2]] - x_hat[PE_friction[i][2]];
            TV relDX3D;
            Point_Edge_RelDX(p, e0, e1, PE_yita[i], relDX3D);
            T relDXSqNorm = (PE_tanBasis[i].transpose() * relDX3D).squaredNorm();
            energy += this->energy_scale * f0_SF(relDXSqNorm, epsvh) * mu * PE_normalForce[i] / x_weight;
        }
        for (size_t i = 0; i < PT_friction.size(); ++i) {
            TV p = x_weight * x[PT_friction[i][0]] - x_hat[PT_friction[i][0]];
            TV v0 = x_weight * x[PT_friction[i][1]] - x_hat[PT_friction[i][1]];
            TV v1 = x_weight * x[PT_friction[i][2]] - x_hat[PT_friction[i][2]];
            TV v2 = x_weight * x[PT_friction[i][3]] - x_hat[PT_friction[i][3]];
            TV relDX3D;
            Point_Triangle_RelDX(p, v0, v1, v2, PT_beta[i](0), PT_beta[i](1), relDX3D);
            T relDXSqNorm = (PT_tanBasis[i].transpose() * relDX3D).squaredNorm();
            energy += this->energy_scale * f0_SF(relDXSqNorm, epsvh) * mu * PT_normalForce[i] / x_weight;
        }
        for (size_t i = 0; i < EE_friction.size(); ++i) {
            TV e0 = x_weight * x[EE_friction[i][0]] - x_hat[EE_friction[i][0]];
            TV e1 = x_weight * x[EE_friction[i][1]] - x_hat[EE_friction[i][1]];
            TV e2 = x_weight * x[EE_friction[i][2]] - x_hat[EE_friction[i][2]];
            TV e3 = x_weight * x[EE_friction[i][3]] - x_hat[EE_friction[i][3]];
            TV relDX3D;
            Edge_Edge_RelDX(e0, e1, e2, e3, EE_gamma[i](0), EE_gamma[i](1), relDX3D);
            T relDXSqNorm = (EE_tanBasis[i].transpose() * relDX3D).squaredNorm();
            energy += this->energy_scale * f0_SF(relDXSqNorm, epsvh) * mu * EE_normalForce[i] / x_weight;
        }
    }

    return energy;
}

template <class T, class StorageIndex>
void IpcEnergyOp3D<T, StorageIndex>::gradient(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& grad)
{
    T dHat2 = dHat * dHat;
    grad.resize(x.size());
    std::fill(grad.begin(), grad.end(), Vector<T, dim>::Zero());

    // PP
    int ppI = 0;
    for (const auto& pp_pair : PP) {
        Vector<T, 2 * 3> PP_grad;
        Geometry::IPC::point_point_distance_gradient(x[pp_pair[0]], x[pp_pair[1]], PP_grad);
        T dist2 = Geometry::IPC::point_point_distance(x[pp_pair[0]], x[pp_pair[1]]);
        PP_grad *= this->energy_scale * wPP[ppI] * dHat * Math::barrier_gradient(dist2, dHat2, kappa);
        grad[pp_pair[0]] += PP_grad.template segment<3>(0);
        grad[pp_pair[1]] += PP_grad.template segment<3>(3);
        ++ppI;
    }
    int peI = 0;
    for (const auto& pe_pair : PE) {
        Vector<T, 3 * 3> PE_grad;
        Geometry::IPC::point_edge_distance_gradient(x[pe_pair[0]], x[pe_pair[1]], x[pe_pair[2]], PE_grad);
        T dist2 = Geometry::IPC::point_edge_distance(x[pe_pair[0]], x[pe_pair[1]], x[pe_pair[2]]);
        PE_grad *= this->energy_scale * wPE[peI] * dHat * Math::barrier_gradient(dist2, dHat2, kappa);
        grad[pe_pair[0]] += PE_grad.template segment<3>(0);
        grad[pe_pair[1]] += PE_grad.template segment<3>(3);
        grad[pe_pair[2]] += PE_grad.template segment<3>(6);
        ++peI;
    }
    int ptI = 0;
    for (const auto& pt_pair : PT) {
        Vector<T, 4 * 3> PT_grad;
        Geometry::IPC::point_triangle_distance_gradient(x[pt_pair[0]], x[pt_pair[1]], x[pt_pair[2]], x[pt_pair[3]], PT_grad);
        T dist2 = Geometry::IPC::point_triangle_distance(x[pt_pair[0]], x[pt_pair[1]], x[pt_pair[2]], x[pt_pair[3]]);
        PT_grad *= this->energy_scale * wPT[ptI] * dHat * Math::barrier_gradient(dist2, dHat2, kappa);
        grad[pt_pair[0]] += PT_grad.template segment<3>(0);
        grad[pt_pair[1]] += PT_grad.template segment<3>(3);
        grad[pt_pair[2]] += PT_grad.template segment<3>(6);
        grad[pt_pair[3]] += PT_grad.template segment<3>(9);
        ++ptI;
    }

    // EE
    auto mollifier_info = [&](const Vector<int, 4>& pair_info, T& m, Vector<T, dim * 4>& gm) {
        const Vector<T, dim>& ea0_rest = m_X[pair_info[0]];
        const Vector<T, dim>& ea1_rest = m_X[pair_info[1]];
        const Vector<T, dim>& eb0_rest = m_X[pair_info[2]];
        const Vector<T, dim>& eb1_rest = m_X[pair_info[3]];
        T eps_x = Geometry::IPC::edge_edge_mollifier_threshold(ea0_rest, ea1_rest, eb0_rest, eb1_rest);
        m = Geometry::IPC::edge_edge_mollifier(x[pair_info[0]], x[pair_info[1]], x[pair_info[2]], x[pair_info[3]], eps_x);
        Geometry::IPC::edge_edge_mollifier_gradient(x[pair_info[0]], x[pair_info[1]], x[pair_info[2]], x[pair_info[3]], eps_x, gm);
    };
    int ppmI = 0;
    for (const auto& pp_pair : PPM) {
        Vector<T, 2 * 3> PP_grad;
        Geometry::IPC::point_point_distance_gradient(x[pp_pair[0]], x[pp_pair[2]], PP_grad);
        T dist2 = Geometry::IPC::point_point_distance(x[pp_pair[0]], x[pp_pair[2]]);
        T barrier_dist2 = Math::barrier(dist2, dHat2, kappa);
        T mollifier;
        Vector<T, 4 * 3> mollifier_grad;
        mollifier_info(pp_pair, mollifier, mollifier_grad);
        PP_grad *= Math::barrier_gradient(dist2, dHat2, kappa);
        T scale = this->energy_scale * wPPM[ppmI] * dHat;
        grad[pp_pair[0]] += scale * (barrier_dist2 * mollifier_grad.template segment<3>(0) + mollifier * PP_grad.template segment<3>(0));
        grad[pp_pair[1]] += scale * (barrier_dist2 * mollifier_grad.template segment<3>(3));
        grad[pp_pair[2]] += scale * (barrier_dist2 * mollifier_grad.template segment<3>(6) + mollifier * PP_grad.template segment<3>(3));
        grad[pp_pair[3]] += scale * (barrier_dist2 * mollifier_grad.template segment<3>(9));
        ++ppmI;
    }
    int pemI = 0;
    for (const auto& pe_pair : PEM) {
        Vector<T, 3 * 3> PE_grad;
        Geometry::IPC::point_edge_distance_gradient(x[pe_pair[0]], x[pe_pair[2]], x[pe_pair[3]], PE_grad);
        T dist2 = Geometry::IPC::point_edge_distance(x[pe_pair[0]], x[pe_pair[2]], x[pe_pair[3]]);
        T barrier_dist2 = Math::barrier(dist2, dHat2, kappa);
        T mollifier;
        Vector<T, 4 * 3> mollifier_grad;
        mollifier_info(pe_pair, mollifier, mollifier_grad);
        PE_grad *= Math::barrier_gradient(dist2, dHat2, kappa);
        T scale = this->energy_scale * wPEM[pemI] * dHat;
        grad[pe_pair[0]] += scale * (barrier_dist2 * mollifier_grad.template segment<3>(0) + mollifier * PE_grad.template segment<3>(0));
        grad[pe_pair[1]] += scale * (barrier_dist2 * mollifier_grad.template segment<3>(3));
        grad[pe_pair[2]] += scale * (barrier_dist2 * mollifier_grad.template segment<3>(6) + mollifier * PE_grad.template segment<3>(3));
        grad[pe_pair[3]] += scale * (barrier_dist2 * mollifier_grad.template segment<3>(9) + mollifier * PE_grad.template segment<3>(6));
        ++pemI;
    }
    int eemI = 0;
    for (const auto& ee_pair : EEM) {
        Vector<T, 4 * 3> EE_grad;
        Geometry::IPC::edge_edge_distance_gradient(x[ee_pair[0]], x[ee_pair[1]], x[ee_pair[2]], x[ee_pair[3]], EE_grad);
        T dist2 = Geometry::IPC::edge_edge_distance(x[ee_pair[0]], x[ee_pair[1]], x[ee_pair[2]], x[ee_pair[3]]);
        T barrier_dist2 = Math::barrier(dist2, dHat2, kappa);
        T mollifier;
        Vector<T, 4 * 3> mollifier_grad;
        mollifier_info(ee_pair, mollifier, mollifier_grad);
        EE_grad *= Math::barrier_gradient(dist2, dHat2, kappa);
        T scale = this->energy_scale * wEEM[eemI] * dHat;
        grad[ee_pair[0]] += scale * (barrier_dist2 * mollifier_grad.template segment<3>(0) + mollifier * EE_grad.template segment<3>(0));
        grad[ee_pair[1]] += scale * (barrier_dist2 * mollifier_grad.template segment<3>(3) + mollifier * EE_grad.template segment<3>(3));
        grad[ee_pair[2]] += scale * (barrier_dist2 * mollifier_grad.template segment<3>(6) + mollifier * EE_grad.template segment<3>(6));
        grad[ee_pair[3]] += scale * (barrier_dist2 * mollifier_grad.template segment<3>(9) + mollifier * EE_grad.template segment<3>(9));
        ++eemI;
    }
    int eeI = 0;
    for (const auto& ee_pair : EE) {
        Vector<T, 4 * 3> EE_grad;
        Geometry::IPC::edge_edge_distance_gradient(x[ee_pair[0]], x[ee_pair[1]], x[ee_pair[2]], x[ee_pair[3]], EE_grad);
        T dist2 = Geometry::IPC::edge_edge_distance(x[ee_pair[0]], x[ee_pair[1]], x[ee_pair[2]], x[ee_pair[3]]);
        EE_grad *= Math::barrier_gradient(dist2, dHat2, kappa);
        T scale = this->energy_scale * wEE[eeI] * dHat;
        grad[ee_pair[0]] += scale * EE_grad.template segment<3>(0);
        grad[ee_pair[1]] += scale * EE_grad.template segment<3>(3);
        grad[ee_pair[2]] += scale * EE_grad.template segment<3>(6);
        grad[ee_pair[3]] += scale * EE_grad.template segment<3>(9);
        ++eeI;
    }

    if (mu > 0) {
        for (size_t i = 0; i < PP_friction.size(); ++i) {
            TV p0 = x_weight * x[PP_friction[i][0]] - x_hat[PP_friction[i][0]], p1 = x_weight * x[PP_friction[i][1]] - x_hat[PP_friction[i][1]];
            TV relDX3D;
            Point_Point_RelDX_3D(p0, p1, relDX3D);
            Vector<T, 2> relDX = PP_tanBasis[i].transpose() * relDX3D;
            T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDX.squaredNorm(), epsvh);
            relDX *= f1_div_relDXNorm * mu * PP_normalForce[i];
            Eigen::Matrix<T, 6, 1> TTTDX;
            Point_Point_RelDXTan_To_Mesh_3D(relDX, PP_tanBasis[i], TTTDX);
            TTTDX *= this->energy_scale;
            grad[PP_friction[i][0]] += TTTDX.template segment<3>(0);
            grad[PP_friction[i][1]] += TTTDX.template segment<3>(3);
        }
        for (size_t i = 0; i < PE_friction.size(); ++i) {
            TV p = x_weight * x[PE_friction[i][0]] - x_hat[PE_friction[i][0]], e0 = x_weight * x[PE_friction[i][1]] - x_hat[PE_friction[i][1]], e1 = x_weight * x[PE_friction[i][2]] - x_hat[PE_friction[i][2]];
            TV relDX3D;
            Point_Edge_RelDX(p, e0, e1, PE_yita[i], relDX3D);
            Vector<T, 2> relDX = PE_tanBasis[i].transpose() * relDX3D;
            T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDX.squaredNorm(), epsvh);
            relDX *= f1_div_relDXNorm * mu * PE_normalForce[i];
            Eigen::Matrix<T, 9, 1> TTTDX;
            Point_Edge_RelDXTan_To_Mesh_3D(relDX, PE_tanBasis[i], PE_yita[i], TTTDX);
            TTTDX *= this->energy_scale;
            grad[PE_friction[i][0]] += TTTDX.template segment<3>(0);
            grad[PE_friction[i][1]] += TTTDX.template segment<3>(3);
            grad[PE_friction[i][2]] += TTTDX.template segment<3>(6);
        }
        for (size_t i = 0; i < PT_friction.size(); ++i) {
            TV p = x_weight * x[PT_friction[i][0]] - x_hat[PT_friction[i][0]];
            TV v0 = x_weight * x[PT_friction[i][1]] - x_hat[PT_friction[i][1]];
            TV v1 = x_weight * x[PT_friction[i][2]] - x_hat[PT_friction[i][2]];
            TV v2 = x_weight * x[PT_friction[i][3]] - x_hat[PT_friction[i][3]];
            TV relDX3D;
            Point_Triangle_RelDX(p, v0, v1, v2, PT_beta[i](0), PT_beta[i](1), relDX3D);
            Vector<T, 2> relDX = PT_tanBasis[i].transpose() * relDX3D;
            T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDX.squaredNorm(), epsvh);
            relDX *= f1_div_relDXNorm * mu * PT_normalForce[i];
            Eigen::Matrix<T, 12, 1> TTTDX;
            Point_Triangle_RelDXTan_To_Mesh(relDX, PT_tanBasis[i], PT_beta[i](0), PT_beta[i](1), TTTDX);
            TTTDX *= this->energy_scale;
            grad[PT_friction[i][0]] += TTTDX.template segment<3>(0);
            grad[PT_friction[i][1]] += TTTDX.template segment<3>(3);
            grad[PT_friction[i][2]] += TTTDX.template segment<3>(6);
            grad[PT_friction[i][3]] += TTTDX.template segment<3>(9);
        }
        for (size_t i = 0; i < EE_friction.size(); ++i) {
            TV e0 = x_weight * x[EE_friction[i][0]] - x_hat[EE_friction[i][0]];
            TV e1 = x_weight * x[EE_friction[i][1]] - x_hat[EE_friction[i][1]];
            TV e2 = x_weight * x[EE_friction[i][2]] - x_hat[EE_friction[i][2]];
            TV e3 = x_weight * x[EE_friction[i][3]] - x_hat[EE_friction[i][3]];
            TV relDX3D;
            Edge_Edge_RelDX(e0, e1, e2, e3, EE_gamma[i](0), EE_gamma[i](1), relDX3D);
            Vector<T, 2> relDX = EE_tanBasis[i].transpose() * relDX3D;
            T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDX.squaredNorm(), epsvh);
            relDX *= f1_div_relDXNorm * mu * EE_normalForce[i];
            Eigen::Matrix<T, 12, 1> TTTDX;
            Edge_Edge_RelDXTan_To_Mesh(relDX, EE_tanBasis[i], EE_gamma[i](0), EE_gamma[i](1), TTTDX);
            TTTDX *= this->energy_scale;
            grad[EE_friction[i][0]] += TTTDX.template segment<3>(0);
            grad[EE_friction[i][1]] += TTTDX.template segment<3>(3);
            grad[EE_friction[i][2]] += TTTDX.template segment<3>(6);
            grad[EE_friction[i][3]] += TTTDX.template segment<3>(9);
        }
    }
}

template <class T, class StorageIndex>
template <bool project_pd>
void IpcEnergyOp3D<T, StorageIndex>::hessian_impl(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess) const
{
    BOW_TIMER_FLAG("IPC");

    T dHat2 = dHat * dHat;
    using IJK = Eigen::Triplet<T, StorageIndex>;
    std::vector<IJK> coeffs;

    // PT
    int coeff_start = coeffs.size();
    coeffs.resize(coeffs.size() + PP.size() * 36);
    tbb::parallel_for(size_t(0), PP.size(), [&](size_t ppI) {
        const auto& pp_pair = PP[ppI];
        Matrix<T, 2 * 3, 2 * 3> PP_hess;
        Geometry::IPC::point_point_distance_hessian(x[pp_pair[0]], x[pp_pair[1]], PP_hess);
        Vector<T, 2 * 3> PP_grad;
        Geometry::IPC::point_point_distance_gradient(x[pp_pair[0]], x[pp_pair[1]], PP_grad);
        T dist2 = Geometry::IPC::point_point_distance(x[pp_pair[0]], x[pp_pair[1]]);
        PP_hess = this->energy_scale * wPP[ppI] * dHat * Math::barrier_hessian(dist2, dHat2, kappa) * PP_grad * PP_grad.transpose() + this->energy_scale * wPP[ppI] * dHat * Math::barrier_gradient(dist2, dHat2, kappa) * PP_hess;
        if constexpr (project_pd)
            Math::make_pd(PP_hess);
        int indMap[] = { 3 * pp_pair[0], 3 * pp_pair[0] + 1, 3 * pp_pair[0] + 2,
            3 * pp_pair[1], 3 * pp_pair[1] + 1, 3 * pp_pair[1] + 2 };
        for (int i = 0; i < 2 * 3; ++i)
            for (int j = 0; j < 2 * 3; ++j)
                coeffs[coeff_start + ppI * 36 + i * 6 + j] = std::move(IJK(indMap[i], indMap[j], PP_hess(i, j)));
    });

    coeff_start = coeffs.size();
    coeffs.resize(coeffs.size() + PE.size() * 81);
    tbb::parallel_for(size_t(0), PE.size(), [&](size_t peI) {
        const auto& pe_pair = PE[peI];
        Matrix<T, 3 * 3, 3 * 3> PE_hess;
        Geometry::IPC::point_edge_distance_hessian(x[pe_pair[0]], x[pe_pair[1]], x[pe_pair[2]], PE_hess);
        Vector<T, 3 * 3> PE_grad;
        Geometry::IPC::point_edge_distance_gradient(x[pe_pair[0]], x[pe_pair[1]], x[pe_pair[2]], PE_grad);
        T dist2 = Geometry::IPC::point_edge_distance(x[pe_pair[0]], x[pe_pair[1]], x[pe_pair[2]]);
        PE_hess = this->energy_scale * wPE[peI] * dHat * Math::barrier_hessian(dist2, dHat2, kappa) * PE_grad * PE_grad.transpose() + this->energy_scale * wPE[peI] * dHat * Math::barrier_gradient(dist2, dHat2, kappa) * PE_hess;
        if constexpr (project_pd)
            Math::make_pd(PE_hess);
        int indMap[] = { 3 * pe_pair[0], 3 * pe_pair[0] + 1, 3 * pe_pair[0] + 2,
            3 * pe_pair[1], 3 * pe_pair[1] + 1, 3 * pe_pair[1] + 2,
            3 * pe_pair[2], 3 * pe_pair[2] + 1, 3 * pe_pair[2] + 2 };
        for (int i = 0; i < 3 * 3; ++i)
            for (int j = 0; j < 3 * 3; ++j)
                coeffs[coeff_start + peI * 81 + i * 9 + j] = std::move(IJK(indMap[i], indMap[j], PE_hess(i, j)));
    });

    coeff_start = coeffs.size();
    coeffs.resize(coeffs.size() + PT.size() * 144);
    tbb::parallel_for(size_t(0), PT.size(), [&](size_t ptI) {
        const auto& pt_pair = PT[ptI];
        Matrix<T, 4 * 3, 4 * 3> PT_hess;
        Geometry::IPC::point_triangle_distance_hessian(x[pt_pair[0]], x[pt_pair[1]], x[pt_pair[2]], x[pt_pair[3]], PT_hess);
        Vector<T, 4 * 3> PT_grad;
        Geometry::IPC::point_triangle_distance_gradient(x[pt_pair[0]], x[pt_pair[1]], x[pt_pair[2]], x[pt_pair[3]], PT_grad);
        T dist2 = Geometry::IPC::point_triangle_distance(x[pt_pair[0]], x[pt_pair[1]], x[pt_pair[2]], x[pt_pair[3]]);
        PT_hess = this->energy_scale * wPT[ptI] * dHat * Math::barrier_hessian(dist2, dHat2, kappa) * PT_grad * PT_grad.transpose() + this->energy_scale * wPT[ptI] * dHat * Math::barrier_gradient(dist2, dHat2, kappa) * PT_hess;
        if constexpr (project_pd)
            Math::make_pd(PT_hess);
        int indMap[] = { 3 * pt_pair[0], 3 * pt_pair[0] + 1, 3 * pt_pair[0] + 2,
            3 * pt_pair[1], 3 * pt_pair[1] + 1, 3 * pt_pair[1] + 2,
            3 * pt_pair[2], 3 * pt_pair[2] + 1, 3 * pt_pair[2] + 2,
            3 * pt_pair[3], 3 * pt_pair[3] + 1, 3 * pt_pair[3] + 2 };
        for (int i = 0; i < 4 * 3; ++i)
            for (int j = 0; j < 4 * 3; ++j)
                coeffs[coeff_start + ptI * 144 + i * 12 + j] = std::move(IJK(indMap[i], indMap[j], PT_hess(i, j)));
    });

    // EE
    auto mollifier_info = [&](const Vector<int, 4>& pair_info, T& m, Vector<T, dim * 4>& gm, Matrix<T, dim * 4, dim * 4>& hm) {
        const Vector<T, dim>& ea0_rest = m_X[pair_info[0]];
        const Vector<T, dim>& ea1_rest = m_X[pair_info[1]];
        const Vector<T, dim>& eb0_rest = m_X[pair_info[2]];
        const Vector<T, dim>& eb1_rest = m_X[pair_info[3]];
        T eps_x = Geometry::IPC::edge_edge_mollifier_threshold(ea0_rest, ea1_rest, eb0_rest, eb1_rest);
        m = Geometry::IPC::edge_edge_mollifier(x[pair_info[0]], x[pair_info[1]], x[pair_info[2]], x[pair_info[3]], eps_x);
        Geometry::IPC::edge_edge_mollifier_gradient(x[pair_info[0]], x[pair_info[1]], x[pair_info[2]], x[pair_info[3]], eps_x, gm);
        Geometry::IPC::edge_edge_mollifier_hessian(x[pair_info[0]], x[pair_info[1]], x[pair_info[2]], x[pair_info[3]], eps_x, hm);
    };

    coeff_start = coeffs.size();
    coeffs.resize(coeffs.size() + PPM.size() * 144);
    tbb::parallel_for(size_t(0), PPM.size(), [&](size_t ppmI) {
        const auto& pp_pair = PPM[ppmI];
        T dist2 = Geometry::IPC::point_point_distance(x[pp_pair[0]], x[pp_pair[2]]);
        T barrier_dist2 = Math::barrier(dist2, dHat2, kappa);
        Vector<T, 2 * 3> dist2_grad;
        Geometry::IPC::point_point_distance_gradient(x[pp_pair[0]], x[pp_pair[2]], dist2_grad);
        Vector<T, 2 * 3> barrier_dist2_grad;
        barrier_dist2_grad = dist2_grad * Math::barrier_gradient(dist2, dHat2, kappa);
        Vector<T, 4 * 3> barrier_dist2_grad_extended;
        barrier_dist2_grad_extended.setZero();
        barrier_dist2_grad_extended.template segment<dim>(0) = barrier_dist2_grad.template segment<dim>(0);
        barrier_dist2_grad_extended.template segment<dim>(6) = barrier_dist2_grad.template segment<dim>(3);
        Matrix<T, 2 * 3, 2 * 3> barrier_dist2_hess;
        Geometry::IPC::point_point_distance_hessian(x[pp_pair[0]], x[pp_pair[2]], barrier_dist2_hess);
        barrier_dist2_hess = Math::barrier_hessian(dist2, dHat2, kappa) * dist2_grad * dist2_grad.transpose() + Math::barrier_gradient(dist2, dHat2, kappa) * barrier_dist2_hess;

        T mollifier;
        Vector<T, 4 * 3> mollifier_grad;
        Matrix<T, 4 * 3, 4 * 3> mollifier_hess;
        mollifier_info(pp_pair, mollifier, mollifier_grad, mollifier_hess);

        Matrix<T, 4 * 3, 4 * 3> PP_hess = barrier_dist2 * mollifier_hess
            + mollifier_grad * barrier_dist2_grad_extended.transpose()
            + barrier_dist2_grad_extended * mollifier_grad.transpose();
        PP_hess.template block<dim, dim>(0, 0) += mollifier * barrier_dist2_hess.template block<dim, dim>(0, 0);
        PP_hess.template block<dim, dim>(0, 6) += mollifier * barrier_dist2_hess.template block<dim, dim>(0, 3);
        PP_hess.template block<dim, dim>(6, 0) += mollifier * barrier_dist2_hess.template block<dim, dim>(3, 0);
        PP_hess.template block<dim, dim>(6, 6) += mollifier * barrier_dist2_hess.template block<dim, dim>(3, 3);

        PP_hess *= this->energy_scale * wPPM[ppmI] * dHat;

        if constexpr (project_pd)
            Math::make_pd(PP_hess);

        int indMap[] = { 3 * pp_pair[0], 3 * pp_pair[0] + 1, 3 * pp_pair[0] + 2,
            3 * pp_pair[1], 3 * pp_pair[1] + 1, 3 * pp_pair[1] + 2,
            3 * pp_pair[2], 3 * pp_pair[2] + 1, 3 * pp_pair[2] + 2,
            3 * pp_pair[3], 3 * pp_pair[3] + 1, 3 * pp_pair[3] + 2 };
        for (int i = 0; i < 4 * 3; ++i)
            for (int j = 0; j < 4 * 3; ++j)
                coeffs[coeff_start + ppmI * 144 + i * 12 + j] = std::move(IJK(indMap[i], indMap[j], PP_hess(i, j)));
    });

    coeff_start = coeffs.size();
    coeffs.resize(coeffs.size() + PEM.size() * 144);
    tbb::parallel_for(size_t(0), PEM.size(), [&](size_t pemI) {
        const auto& pe_pair = PEM[pemI];
        T dist2 = Geometry::IPC::point_edge_distance(x[pe_pair[0]], x[pe_pair[2]], x[pe_pair[3]]);
        T barrier_dist2 = Math::barrier(dist2, dHat2, kappa);
        Vector<T, 3 * 3> dist2_grad;
        Geometry::IPC::point_edge_distance_gradient(x[pe_pair[0]], x[pe_pair[2]], x[pe_pair[3]], dist2_grad);
        Vector<T, 3 * 3> barrier_dist2_grad;
        barrier_dist2_grad = dist2_grad * Math::barrier_gradient(dist2, dHat2, kappa);
        Vector<T, 4 * 3> barrier_dist2_grad_extended;
        barrier_dist2_grad_extended.setZero();
        barrier_dist2_grad_extended.template segment<dim>(0) = barrier_dist2_grad.template segment<dim>(0);
        barrier_dist2_grad_extended.template segment<2 * dim>(6) = barrier_dist2_grad.template segment<2 * dim>(3);
        Matrix<T, 3 * 3, 3 * 3> barrier_dist2_hess;
        Geometry::IPC::point_edge_distance_hessian(x[pe_pair[0]], x[pe_pair[2]], x[pe_pair[3]], barrier_dist2_hess);
        barrier_dist2_hess = Math::barrier_hessian(dist2, dHat2, kappa) * dist2_grad * dist2_grad.transpose() + Math::barrier_gradient(dist2, dHat2, kappa) * barrier_dist2_hess;

        T mollifier;
        Vector<T, 4 * 3> mollifier_grad;
        Matrix<T, 4 * 3, 4 * 3> mollifier_hess;
        mollifier_info(pe_pair, mollifier, mollifier_grad, mollifier_hess);
        Matrix<T, 4 * 3, 4 * 3> PE_hess = barrier_dist2 * mollifier_hess
            + mollifier_grad * barrier_dist2_grad_extended.transpose()
            + barrier_dist2_grad_extended * mollifier_grad.transpose();
        PE_hess.template block<dim, dim>(0, 0) += mollifier * barrier_dist2_hess.template block<dim, dim>(0, 0);
        PE_hess.template block<dim, 2 * dim>(0, 6) += mollifier * barrier_dist2_hess.template block<dim, 2 * dim>(0, 3);
        PE_hess.template block<2 * dim, dim>(6, 0) += mollifier * barrier_dist2_hess.template block<2 * dim, dim>(3, 0);
        PE_hess.template block<2 * dim, 2 * dim>(6, 6) += mollifier * barrier_dist2_hess.template block<2 * dim, 2 * dim>(3, 3);

        PE_hess *= this->energy_scale * wPEM[pemI] * dHat;
        if constexpr (project_pd)
            Math::make_pd(PE_hess);

        int indMap[] = { 3 * pe_pair[0], 3 * pe_pair[0] + 1, 3 * pe_pair[0] + 2,
            3 * pe_pair[1], 3 * pe_pair[1] + 1, 3 * pe_pair[1] + 2,
            3 * pe_pair[2], 3 * pe_pair[2] + 1, 3 * pe_pair[2] + 2,
            3 * pe_pair[3], 3 * pe_pair[3] + 1, 3 * pe_pair[3] + 2 };
        for (int i = 0; i < 4 * 3; ++i)
            for (int j = 0; j < 4 * 3; ++j)
                coeffs[coeff_start + pemI * 144 + i * 12 + j] = std::move(IJK(indMap[i], indMap[j], PE_hess(i, j)));
    });

    coeff_start = coeffs.size();
    coeffs.resize(coeffs.size() + EEM.size() * 144);
    tbb::parallel_for(size_t(0), EEM.size(), [&](size_t eemI) {
        const auto& ee_pair = EEM[eemI];
        T dist2 = Geometry::IPC::edge_edge_distance(x[ee_pair[0]], x[ee_pair[1]], x[ee_pair[2]], x[ee_pair[3]]);
        T barrier_dist2 = Math::barrier(dist2, dHat2, kappa);
        Vector<T, 4 * 3> dist2_grad;
        Geometry::IPC::edge_edge_distance_gradient(x[ee_pair[0]], x[ee_pair[1]], x[ee_pair[2]], x[ee_pair[3]], dist2_grad);
        Vector<T, 4 * 3> barrier_dist2_grad;
        barrier_dist2_grad = dist2_grad * Math::barrier_gradient(dist2, dHat2, kappa);
        Matrix<T, 4 * 3, 4 * 3> barrier_dist2_hess;
        Geometry::IPC::edge_edge_distance_hessian(x[ee_pair[0]], x[ee_pair[1]], x[ee_pair[2]], x[ee_pair[3]], barrier_dist2_hess);
        barrier_dist2_hess = Math::barrier_hessian(dist2, dHat2, kappa) * dist2_grad * dist2_grad.transpose() + Math::barrier_gradient(dist2, dHat2, kappa) * barrier_dist2_hess;

        T mollifier;
        Vector<T, 4 * 3> mollifier_grad;
        Matrix<T, 4 * 3, 4 * 3> mollifier_hess;
        mollifier_info(ee_pair, mollifier, mollifier_grad, mollifier_hess);

        Matrix<T, 4 * 3, 4 * 3> EE_hess = barrier_dist2 * mollifier_hess + mollifier * barrier_dist2_hess
            + mollifier_grad * barrier_dist2_grad.transpose()
            + barrier_dist2_grad * mollifier_grad.transpose();

        EE_hess *= this->energy_scale * wEEM[eemI] * dHat;

        if constexpr (project_pd)
            Math::make_pd(EE_hess);

        int indMap[] = { 3 * ee_pair[0], 3 * ee_pair[0] + 1, 3 * ee_pair[0] + 2,
            3 * ee_pair[1], 3 * ee_pair[1] + 1, 3 * ee_pair[1] + 2,
            3 * ee_pair[2], 3 * ee_pair[2] + 1, 3 * ee_pair[2] + 2,
            3 * ee_pair[3], 3 * ee_pair[3] + 1, 3 * ee_pair[3] + 2 };
        for (int i = 0; i < 4 * 3; ++i)
            for (int j = 0; j < 4 * 3; ++j)
                coeffs[coeff_start + eemI * 144 + i * 12 + j] = std::move(IJK(indMap[i], indMap[j], EE_hess(i, j)));
    });

    coeff_start = coeffs.size();
    coeffs.resize(coeffs.size() + EE.size() * 144);
    tbb::parallel_for(size_t(0), EE.size(), [&](size_t eeI) {
        const auto& ee_pair = EE[eeI];
        T dist2 = Geometry::IPC::edge_edge_distance(x[ee_pair[0]], x[ee_pair[1]], x[ee_pair[2]], x[ee_pair[3]]);
        Vector<T, 4 * 3> dist2_grad;
        Geometry::IPC::edge_edge_distance_gradient(x[ee_pair[0]], x[ee_pair[1]], x[ee_pair[2]], x[ee_pair[3]], dist2_grad);
        Vector<T, 4 * 3> barrier_dist2_grad;
        barrier_dist2_grad = dist2_grad * Math::barrier_gradient(dist2, dHat2, kappa);
        Matrix<T, 4 * 3, 4 * 3> EE_hess;
        Geometry::IPC::edge_edge_distance_hessian(x[ee_pair[0]], x[ee_pair[1]], x[ee_pair[2]], x[ee_pair[3]], EE_hess);
        EE_hess = Math::barrier_hessian(dist2, dHat2, kappa) * dist2_grad * dist2_grad.transpose() + Math::barrier_gradient(dist2, dHat2, kappa) * EE_hess;

        EE_hess *= this->energy_scale * wEE[eeI] * dHat;

        if constexpr (project_pd)
            Math::make_pd(EE_hess);

        int indMap[] = { 3 * ee_pair[0], 3 * ee_pair[0] + 1, 3 * ee_pair[0] + 2,
            3 * ee_pair[1], 3 * ee_pair[1] + 1, 3 * ee_pair[1] + 2,
            3 * ee_pair[2], 3 * ee_pair[2] + 1, 3 * ee_pair[2] + 2,
            3 * ee_pair[3], 3 * ee_pair[3] + 1, 3 * ee_pair[3] + 2 };
        for (int i = 0; i < 4 * 3; ++i)
            for (int j = 0; j < 4 * 3; ++j)
                coeffs[coeff_start + eeI * 144 + i * 12 + j] = std::move(IJK(indMap[i], indMap[j], EE_hess(i, j)));
    });

    if (mu > 0) {
        for (size_t i = 0; i < PP_friction.size(); ++i) {
            TV p0 = x_weight * x[PP_friction[i][0]] - x_hat[PP_friction[i][0]], p1 = x_weight * x[PP_friction[i][1]] - x_hat[PP_friction[i][1]];
            TV relDX3D;
            Point_Point_RelDX_3D(p0, p1, relDX3D);
            Vector<T, dim - 1> relDX = PP_tanBasis[i].transpose() * relDX3D;
            T relDXSqNorm = relDX.squaredNorm();
            T relDXNorm = std::sqrt(relDXSqNorm);
            Eigen::Matrix<T, 2, 6> TT;
            Point_Point_TT_3D(PP_tanBasis[i], TT);
            T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDXSqNorm, epsvh);
            T f2_term = f2_SF_Term(relDXSqNorm, epsvh);
            Eigen::Matrix<T, 6, 6> HessianI = Eigen::Matrix<T, 6, 6>::Zero();
            if (relDXSqNorm >= epsvh * epsvh) {
                Vector<T, dim - 1> ubar(-relDX(1), relDX(0));
                HessianI = (TT.transpose() * ((mu * PP_normalForce[i] * f1_div_relDXNorm / relDXSqNorm) * ubar)) * (ubar.transpose() * TT);
            }
            else {
                if (relDXNorm == 0) {
                    if (PP_normalForce[i] > 0) {
                        // no SPD projection needed
                        HessianI = ((mu * PP_normalForce[i] * f1_div_relDXNorm) * TT.transpose()) * TT;
                    }
                    // if PP_normalForce[i] <= 0 due to max() approximation, then this SND hessian can be ignored.
                }
                else {
                    // only need to project the inner 2x2 matrix to SPD
                    Matrix<T, 2, 2> innerMtr = ((f2_term / relDXNorm) * relDX) * relDX.transpose();
                    innerMtr.diagonal().array() += f1_div_relDXNorm;
                    innerMtr *= mu * PP_normalForce[i];
                    if constexpr (project_pd)
                        Math::make_pd(innerMtr);
                    // tensor product:
                    HessianI = TT.transpose() * innerMtr * TT;
                }
            }
            HessianI *= x_weight * this->energy_scale;
            int cIVInd[2] = { PP_friction[i][0], PP_friction[i][1] };
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    for (int idI = 0; idI < 3; ++idI)
                        for (int jdI = 0; jdI < 3; ++jdI)
                            coeffs.emplace_back(
                                cIVInd[i] * 3 + idI,
                                cIVInd[j] * 3 + jdI,
                                HessianI(i * 3 + idI, j * 3 + jdI));
        }
        for (size_t i = 0; i < PE_friction.size(); ++i) {
            TV p = x_weight * x[PE_friction[i][0]] - x_hat[PE_friction[i][0]], e0 = x_weight * x[PE_friction[i][1]] - x_hat[PE_friction[i][1]], e1 = x_weight * x[PE_friction[i][2]] - x_hat[PE_friction[i][2]];
            TV relDX3D;
            Point_Edge_RelDX(p, e0, e1, PE_yita[i], relDX3D);
            Vector<T, dim - 1> relDX = PE_tanBasis[i].transpose() * relDX3D;
            T relDXSqNorm = relDX.squaredNorm();
            T relDXNorm = std::sqrt(relDXSqNorm);
            Eigen::Matrix<T, 2, 9> TT;
            Point_Edge_TT_3D(PE_tanBasis[i], PE_yita[i], TT);
            T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDXSqNorm, epsvh);
            T f2_term = f2_SF_Term(relDXSqNorm, epsvh);
            Eigen::Matrix<T, 9, 9> HessianI = Eigen::Matrix<T, 9, 9>::Zero();
            if (relDXSqNorm >= epsvh * epsvh) {
                Vector<T, dim - 1> ubar(-relDX(1), relDX(0));
                HessianI = (TT.transpose() * ((mu * PE_normalForce[i] * f1_div_relDXNorm / relDXSqNorm) * ubar)) * (ubar.transpose() * TT);
            }
            else {
                if (relDXNorm == 0) {
                    if (PE_normalForce[i] > 0) {
                        // no SPD projection needed
                        HessianI = ((mu * PE_normalForce[i] * f1_div_relDXNorm) * TT.transpose()) * TT;
                    }
                    // if PE_normalForce[i] <= 0 due to max() approximation, then this SND hessian can be ignored.
                }
                else {
                    // only need to project the inner 2x2 matrix to SPD
                    Matrix<T, 2, 2> innerMtr = ((f2_term / relDXNorm) * relDX) * relDX.transpose();
                    innerMtr.diagonal().array() += f1_div_relDXNorm;
                    innerMtr *= mu * PE_normalForce[i];
                    if constexpr (project_pd)
                        Math::make_pd(innerMtr);
                    // tensor product:
                    HessianI = TT.transpose() * innerMtr * TT;
                }
            }
            HessianI *= x_weight * this->energy_scale;
            int cIVInd[3] = { PE_friction[i][0], PE_friction[i][1], PE_friction[i][2] };
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    for (int idI = 0; idI < 3; ++idI)
                        for (int jdI = 0; jdI < 3; ++jdI)
                            coeffs.emplace_back(
                                cIVInd[i] * 3 + idI,
                                cIVInd[j] * 3 + jdI,
                                HessianI(i * 3 + idI, j * 3 + jdI));
        }
        for (size_t i = 0; i < PT_friction.size(); ++i) {
            TV p = x_weight * x[PT_friction[i][0]] - x_hat[PT_friction[i][0]];
            TV v0 = x_weight * x[PT_friction[i][1]] - x_hat[PT_friction[i][1]];
            TV v1 = x_weight * x[PT_friction[i][2]] - x_hat[PT_friction[i][2]];
            TV v2 = x_weight * x[PT_friction[i][3]] - x_hat[PT_friction[i][3]];
            TV relDX3D;
            Point_Triangle_RelDX(p, v0, v1, v2, PT_beta[i](0), PT_beta[i](1), relDX3D);
            Vector<T, 2> relDX = PT_tanBasis[i].transpose() * relDX3D;
            T relDXSqNorm = relDX.squaredNorm();
            T relDXNorm = std::sqrt(relDXSqNorm);
            Eigen::Matrix<T, 2, 12> TT;
            Point_Triangle_TT(PT_tanBasis[i], PT_beta[i](0), PT_beta[i](1), TT);
            T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDXSqNorm, epsvh);
            T f2_term = f2_SF_Term(relDXSqNorm, epsvh);
            Eigen::Matrix<T, 12, 12> HessianI = Eigen::Matrix<T, 12, 12>::Zero();
            if (relDXSqNorm >= epsvh * epsvh) {
                Vector<T, dim - 1> ubar(-relDX(1), relDX(0));
                HessianI = (TT.transpose() * ((mu * PT_normalForce[i] * f1_div_relDXNorm / relDXSqNorm) * ubar)) * (ubar.transpose() * TT);
            }
            else {
                if (relDXNorm == 0) {
                    if (PT_normalForce[i] > 0) {
                        // no SPD projection needed
                        HessianI = ((mu * PT_normalForce[i] * f1_div_relDXNorm) * TT.transpose()) * TT;
                    }
                }
                else {
                    // only need to project the inner 2x2 matrix to SPD
                    Matrix<T, 2, 2> innerMtr = ((f2_term / relDXNorm) * relDX) * relDX.transpose();
                    innerMtr.diagonal().array() += f1_div_relDXNorm;
                    innerMtr *= mu * PT_normalForce[i];
                    if constexpr (project_pd)
                        Math::make_pd(innerMtr);
                    // tensor product:
                    HessianI = TT.transpose() * innerMtr * TT;
                }
            }
            HessianI *= x_weight * this->energy_scale;
            int cIVInd[4] = { PT_friction[i][0], PT_friction[i][1], PT_friction[i][2], PT_friction[i][3] };
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    for (int idI = 0; idI < 3; ++idI)
                        for (int jdI = 0; jdI < 3; ++jdI)
                            coeffs.emplace_back(
                                cIVInd[i] * 3 + idI,
                                cIVInd[j] * 3 + jdI,
                                HessianI(i * 3 + idI, j * 3 + jdI));
        }
        for (size_t i = 0; i < EE_friction.size(); ++i) {
            TV e0 = x_weight * x[EE_friction[i][0]] - x_hat[EE_friction[i][0]];
            TV e1 = x_weight * x[EE_friction[i][1]] - x_hat[EE_friction[i][1]];
            TV e2 = x_weight * x[EE_friction[i][2]] - x_hat[EE_friction[i][2]];
            TV e3 = x_weight * x[EE_friction[i][3]] - x_hat[EE_friction[i][3]];
            TV relDX3D;
            Edge_Edge_RelDX(e0, e1, e2, e3, EE_gamma[i](0), EE_gamma[i](1), relDX3D);
            Vector<T, 2> relDX = EE_tanBasis[i].transpose() * relDX3D;
            T relDXSqNorm = relDX.squaredNorm();
            T relDXNorm = std::sqrt(relDXSqNorm);
            Eigen::Matrix<T, 2, 12> TT;
            Edge_Edge_TT(EE_tanBasis[i], EE_gamma[i](0), EE_gamma[i](1), TT);
            T f1_div_relDXNorm = f1_SF_Div_RelDXNorm(relDXSqNorm, epsvh);
            T f2_term = f2_SF_Term(relDXSqNorm, epsvh);
            Eigen::Matrix<T, 12, 12> HessianI = Eigen::Matrix<T, 12, 12>::Zero();
            if (relDXSqNorm >= epsvh * epsvh) {
                Vector<T, dim - 1> ubar(-relDX(1), relDX(0));
                HessianI = (TT.transpose() * ((mu * EE_normalForce[i] * f1_div_relDXNorm / relDXSqNorm) * ubar)) * (ubar.transpose() * TT);
            }
            else {
                if (relDXNorm == 0) {
                    if (EE_normalForce[i] > 0) {
                        // no SPD projection needed
                        HessianI = ((mu * EE_normalForce[i] * f1_div_relDXNorm) * TT.transpose()) * TT;
                    }
                }
                else {
                    // only need to project the inner 2x2 matrix to SPD
                    Matrix<T, 2, 2> innerMtr = ((f2_term / relDXNorm) * relDX) * relDX.transpose();
                    innerMtr.diagonal().array() += f1_div_relDXNorm;
                    innerMtr *= mu * EE_normalForce[i];
                    if constexpr (project_pd)
                        Math::make_pd(innerMtr);
                    // tensor product:
                    HessianI = TT.transpose() * innerMtr * TT;
                }
            }
            HessianI *= x_weight * this->energy_scale;
            int cIVInd[4] = { EE_friction[i][0], EE_friction[i][1], EE_friction[i][2], EE_friction[i][3] };
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    for (int idI = 0; idI < 3; ++idI)
                        for (int jdI = 0; jdI < 3; ++jdI)
                            coeffs.emplace_back(
                                cIVInd[i] * 3 + idI,
                                cIVInd[j] * 3 + jdI,
                                HessianI(i * 3 + idI, j * 3 + jdI));
        }
    }

    hess.resize(x.size() * dim, x.size() * dim);
    hess.setZero();
    hess.setFromTriplets(coeffs.begin(), coeffs.end());
}

template <class T, class StorageIndex>
void IpcEnergyOp3D<T, StorageIndex>::point_triangle_constraints(const Field<Vector<T, 3>>& x)
{
    BOW_TIMER_FLAG("PT");
    using namespace Geometry::IPC;
    T dHat2 = dHat * dHat;
    int bpI = 0;
    for (const auto p : m_boundary_points) {
        Vector<T, 3> p0 = x[p];
        const T area_weight = m_boundary_point_area[bpI] / 4; // 1/2 * 1/2 to handle double counting and PT+EE for correct integration
        Field<int> triInds;
        sh.query_point_for_triangles(p0, dHat, triInds);
        for (const auto& bfI : triInds) {
            const auto& face = m_boundary_faces[bfI];
            if (p == face[0] || p == face[1] || p == face[2])
                continue;
            const Vector<T, 3>& t0 = x[face[0]];
            const Vector<T, 3>& t1 = x[face[1]];
            const Vector<T, 3>& t2 = x[face[2]];
            if (!Geometry::IPC::point_triangle_cd_broadphase(p0, t0, t1, t2, dHat))
                continue;
            switch (Geometry::IPC::point_triangle_distance_type(p0, t0, t1, t2)) {
            case 0:
                if (Geometry::IPC::point_point_distance(p0, t0) < dHat2) {
                    PP.emplace_back(p, face[0]);
                    wPP.emplace_back(area_weight);
                }
                break;
            case 1:
                if (Geometry::IPC::point_point_distance(p0, t1) < dHat2) {
                    PP.emplace_back(p, face[1]);
                    wPP.emplace_back(area_weight);
                }
                break;
            case 2:
                if (Geometry::IPC::point_point_distance(p0, t2) < dHat2) {
                    PP.emplace_back(p, face[2]);
                    wPP.emplace_back(area_weight);
                }
                break;
            case 3:
                if (Geometry::IPC::point_edge_distance(p0, t0, t1) < dHat2) {
                    PE.emplace_back(p, face[0], face[1]);
                    wPE.emplace_back(area_weight);
                }
                break;
            case 4:
                if (Geometry::IPC::point_edge_distance(p0, t1, t2) < dHat2) {
                    PE.emplace_back(p, face[1], face[2]);
                    wPE.emplace_back(area_weight);
                }
                break;
            case 5:
                if (Geometry::IPC::point_edge_distance(p0, t2, t0) < dHat2) {
                    PE.emplace_back(p, face[2], face[0]);
                    wPE.emplace_back(area_weight);
                }
                break;
            case 6:
                if (Geometry::IPC::point_triangle_distance(p0, t0, t1, t2) < dHat2) {
                    PT.emplace_back(p, face[0], face[1], face[2]);
                    wPT.emplace_back(area_weight);
                }
                break;
            default:
                break;
            }
        }
        ++bpI;
    }
    if (improved_maxOp) {
        for (int bpI = 0; bpI < (int)m_boundary_points.size(); ++bpI) {
            int p = m_boundary_points[bpI];
            const T area_weight = m_boundary_point_area[bpI] / 4; // 1/2 * 1/2 to handle double counting and PT+EE for correct integration

            Field<int> edgeInds;
            sh.query_point_for_edges(x[p], dHat, edgeInds);
            for (const auto& beI : edgeInds) {
                if (m_boundary_edge_pnb[beI].size() < 2)
                    continue;

                const auto edge = m_boundary_edges[beI];
                if (p == edge[0] || p == edge[1])
                    continue;

                int incTriAmt = m_boundary_edge_pnb[beI].size();
                for (const auto& pnbI : m_boundary_edge_pnb[beI]) {
                    if (pnbI == bpI) {
                        --incTriAmt;
                        break;
                    }
                }
                if (incTriAmt > 1) {
                    const Vector<T, 3>& e0 = x[edge[0]];
                    const Vector<T, 3>& e1 = x[edge[1]];
                    if (!Geometry::IPC::point_edge_cd_broadphase(x[p], e0, e1, dHat))
                        continue;

                    switch (Geometry::IPC::point_edge_distance_type(x[p], e0, e1)) {
                    case 0:
                        if (Geometry::IPC::point_point_distance(x[p], e0) < dHat2) {
                            PP.emplace_back(p, edge[0]);
                            wPP.emplace_back((1 - incTriAmt) * area_weight);
                        }
                        break;

                    case 1:
                        if (Geometry::IPC::point_point_distance(x[p], e1) < dHat2) {
                            PP.emplace_back(p, edge[1]);
                            wPP.emplace_back((1 - incTriAmt) * area_weight);
                        }
                        break;

                    case 2:
                        if (Geometry::IPC::point_edge_distance(x[p], e0, e1) < dHat2) {
                            PE.emplace_back(p, edge[0], edge[1]);
                            wPE.emplace_back((1 - incTriAmt) * area_weight);
                        }
                        break;
                    }
                }
            }

            Field<int> pointInds;
            sh.query_point_for_points(x[p], dHat, pointInds);
            for (const auto& bpJ : pointInds) {
                int q = m_boundary_points[bpJ];
                if (bpJ == bpI)
                    continue;
                if (m_boundary_point_type[bpJ] != 2)
                    continue;
                if (bpI < (int)m_boundary_point_nb.size() && m_boundary_point_nb[bpI].find(q) != m_boundary_point_nb[bpI].end())
                    continue;

                if (Geometry::IPC::point_point_distance(x[p], x[q]) < dHat2) {
                    PP.emplace_back(p, q);
                    wPP.emplace_back(area_weight);
                }
            }
        }
    }
}

template <class T, class StorageIndex>
void IpcEnergyOp3D<T, StorageIndex>::edge_edge_constraints(const Field<Vector<T, 3>>& x)
{
    BOW_TIMER_FLAG("EE");
    using namespace Geometry::IPC;
    T dHat2 = dHat * dHat;
    for (size_t i = 0; i < m_boundary_edges.size() - 1; ++i) {
        auto edge0 = m_boundary_edges[i];
        const Vector<T, dim>& ea0 = x[edge0[0]];
        const Vector<T, dim>& ea1 = x[edge0[1]];
        Field<int> edgeInds;
        sh.query_edge_for_edges(ea0, ea1, dHat, edgeInds, i);
        for (const auto& j : edgeInds) {
            auto edge1 = m_boundary_edges[j];
            if (edge0[0] == edge1[0] || edge1[1] == edge0[1] || edge0[0] == edge1[1] || edge1[0] == edge0[1])
                continue;
            const Vector<T, dim>& eb0 = x[edge1[0]];
            const Vector<T, dim>& eb1 = x[edge1[1]];
            if (!edge_edge_cd_broadphase(ea0, ea1, eb0, eb1, dHat))
                continue;

            T area_weight = (m_boundary_edge_area[i] + m_boundary_edge_area[j]) / 4; // 1/2 * 1/2 to handle double counting and PT+EE for correct integration
            T eps_x = Geometry::IPC::edge_edge_mollifier_threshold(m_X[edge0[0]], m_X[edge0[1]], m_X[edge1[0]], m_X[edge1[1]]);
            T m = Geometry::IPC::edge_edge_mollifier(ea0, ea1, eb0, eb1, eps_x);
            switch (edge_edge_distance_type(ea0, ea1, eb0, eb1)) {
            case 0:
                if (point_point_distance(ea0, eb0) < dHat2) {
                    if (m == 1) {
                        PP.emplace_back(edge0[0], edge1[0]);
                        wPP.emplace_back(area_weight);
                    }
                    else {
                        PPM.emplace_back(edge0[0], edge0[1], edge1[0], edge1[1]);
                        wPPM.emplace_back(area_weight);
                    }
                }
                break;
            case 1:
                if (point_point_distance(ea0, eb1) < dHat2) {
                    if (m == 1) {
                        PP.emplace_back(edge0[0], edge1[1]);
                        wPP.emplace_back(area_weight);
                    }
                    else {
                        PPM.emplace_back(edge0[0], edge0[1], edge1[1], edge1[0]);
                        wPPM.emplace_back(area_weight);
                    }
                }
                break;
            case 2:
                if (point_edge_distance(ea0, eb0, eb1) < dHat2) {
                    if (m == 1) {
                        PE.emplace_back(edge0[0], edge1[0], edge1[1]);
                        wPE.emplace_back(area_weight);
                    }
                    else {
                        PEM.emplace_back(edge0[0], edge0[1], edge1[0], edge1[1]);
                        wPEM.emplace_back(area_weight);
                    }
                }
                break;
            case 3:
                if (point_point_distance(ea1, eb0) < dHat2) {
                    if (m == 1) {
                        PP.emplace_back(edge0[1], edge1[0]);
                        wPP.emplace_back(area_weight);
                    }
                    else {
                        PPM.emplace_back(edge0[1], edge0[0], edge1[0], edge1[1]);
                        wPPM.emplace_back(area_weight);
                    }
                }
                break;
            case 4:
                if (point_point_distance(ea1, eb1) < dHat2) {
                    if (m == 1) {
                        PP.emplace_back(edge0[1], edge1[1]);
                        wPP.emplace_back(area_weight);
                    }
                    else {
                        PPM.emplace_back(edge0[1], edge0[0], edge1[1], edge1[0]);
                        wPPM.emplace_back(area_weight);
                    }
                }
                break;
            case 5:
                if (point_edge_distance(ea1, eb0, eb1) < dHat2) {
                    if (m == 1) {
                        PE.emplace_back(edge0[1], edge1[0], edge1[1]);
                        wPE.emplace_back(area_weight);
                    }
                    else {
                        PEM.emplace_back(edge0[1], edge0[0], edge1[0], edge1[1]);
                        wPEM.emplace_back(area_weight);
                    }
                }
                break;
            case 6:
                if (point_edge_distance(eb0, ea0, ea1) < dHat2) {
                    if (m == 1) {
                        PE.emplace_back(edge1[0], edge0[0], edge0[1]);
                        wPE.emplace_back(area_weight);
                    }
                    else {
                        PEM.emplace_back(edge1[0], edge1[1], edge0[0], edge0[1]);
                        wPEM.emplace_back(area_weight);
                    }
                }
                break;
            case 7:
                if (point_edge_distance(eb1, ea0, ea1) < dHat2) {
                    if (m == 1) {
                        PE.emplace_back(edge1[1], edge0[0], edge0[1]);
                        wPE.emplace_back(area_weight);
                    }
                    else {
                        PEM.emplace_back(edge1[1], edge1[0], edge0[0], edge0[1]);
                        wPEM.emplace_back(area_weight);
                    }
                }
                break;
            case 8:
                if (edge_edge_distance(ea0, ea1, eb0, eb1) < dHat2) {
                    if (m == 1) {
                        EE.emplace_back(edge0[0], edge0[1], edge1[0], edge1[1]);
                        wEE.emplace_back(area_weight);
                    }
                    else {
                        EEM.emplace_back(edge0[0], edge0[1], edge1[0], edge1[1]);
                        wEEM.emplace_back(area_weight);
                    }
                }
                break;
            default:
                break;
            }
        }
    }
    if (improved_maxOp) {
        for (size_t beI = 0; beI < m_boundary_edges.size(); ++beI) { //NOTE: E-P traversing, needs to consider the last edge
            auto edge = m_boundary_edges[beI];
            const Vector<T, dim>& e0 = x[edge[0]];
            const Vector<T, dim>& e1 = x[edge[1]];
            T area_weight = m_boundary_edge_area[beI] / 4; // 1/2 * 1/2 to handle double counting and PT+EE for correct integration
            //NOTE: E-P traversing, so only one edge area here
            Field<int> pointInds;
            sh.query_edge_for_points(e0, e1, dHat, pointInds);
            for (const auto& bpI : pointInds) {
                int p = m_boundary_points[bpI];
                if (p == edge[0] || p == edge[1])
                    continue;
                if (!Geometry::IPC::point_edge_cd_broadphase(x[p], e0, e1, dHat))
                    continue;
                // first, subtract E-P from every mollified E-E
                int nonmollified_EE_count = 0;
                for (const auto& q : m_boundary_point_nb[bpI]) {
                    if (q == edge[0] || q == edge[1])
                        continue;
                    T eps_x = Geometry::IPC::edge_edge_mollifier_threshold(m_X[p], m_X[q], m_X[edge[0]], m_X[edge[1]]);
                    if (Geometry::IPC::edge_edge_mollifier(x[p], x[q], e0, e1, eps_x) == 1) {
                        ++nonmollified_EE_count;
                        continue;
                    }
                    switch (Geometry::IPC::point_edge_distance_type(x[p], e0, e1)) {
                    case 0:
                        if (Geometry::IPC::point_point_distance(x[p], e0) < dHat2) {
                            PPM.emplace_back(p, q, edge[0], edge[1]);
                            wPPM.emplace_back(-area_weight);
                        }
                        break;
                    case 1:
                        if (Geometry::IPC::point_point_distance(x[p], e1) < dHat2) {
                            PPM.emplace_back(p, q, edge[1], edge[0]);
                            wPPM.emplace_back(-area_weight);
                        }
                        break;
                    case 2:
                        if (Geometry::IPC::point_edge_distance(x[p], e0, e1) < dHat2) {
                            PEM.emplace_back(p, q, edge[0], edge[1]);
                            wPEM.emplace_back(-area_weight);
                        }
                        break;
                    }
                }
                // then, leave only one E-P (non-mollified)
                switch (Geometry::IPC::point_edge_distance_type(x[p], e0, e1)) {
                case 0:
                    if (Geometry::IPC::point_point_distance(x[p], e0) < dHat2) {
                        PP.emplace_back(p, edge[0]);
                        wPP.emplace_back((1 - nonmollified_EE_count) * area_weight);
                    }
                    break;
                case 1:
                    if (Geometry::IPC::point_point_distance(x[p], e1) < dHat2) {
                        PP.emplace_back(p, edge[1]);
                        wPP.emplace_back((1 - nonmollified_EE_count) * area_weight);
                    }
                    break;
                case 2:
                    if (Geometry::IPC::point_edge_distance(x[p], e0, e1) < dHat2) {
                        PE.emplace_back(p, edge[0], edge[1]);
                        wPE.emplace_back((1 - nonmollified_EE_count) * area_weight);
                    }
                    break;
                }
            }
        }
    }
}

} // namespace Bow::FEM::IPC