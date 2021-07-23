#pragma once

#include <Bow/Simulator/PhysicallyBasedSimulator.h>
#include <Bow/Energy/FEM/FEMEnergies.h>
#include <Bow/TimeIntegrator/FEM/FEMTimeIntegrator.h>
#include <Bow/IO/vtk.h>
#include <Bow/IO/ply.h>
#include <Bow/Utils/Serialization.h>
#include <Bow/Simulator/BoundaryConditionManager.h>
#include <Bow/Geometry/Query.h>
#include "InitializeOp.h"
#include "InitializeShellOp.h"
#include "BoundaryConditionUpdateOp.h"

namespace Bow::FEM {

template <class T, int dim, class Optimizer = Optimization::AugmentedLagrangianNewton<T, dim, int>>
class FEMSimulator : virtual public PhysicallyBasedSimulator<T, dim> {
public:
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    using TimeIntegrator = TimeIntegratorUpdateOp<T, dim, int, Optimizer>;

    // constant
    TV gravity = TV::Zero();
    Field<TV> m_X; // material coordinate;
    Field<T> m_mass; // node mass

    // codim0 elements
    Field<T> m_density; // element density
    Field<T> m_vol; // element volume
    Field<Vector<int, dim + 1>> m_elem; // element vertex indices
    Field<TM> m_IB;
    Field<T> m_mu;
    Field<T> m_lam;
    std::unordered_map<int, std::vector<std::pair<int, int>>> m_obj_divider;

    // codim1 membrane elements
    Field<T> m_density_codim1; // element density
    Field<T> m_thickness_codim1; // element thickness
    Field<T> m_vol_codim1; // element volume
    Field<Vector<int, dim>> m_elem_codim1; // element vertex indices
    Field<Matrix<T, dim - 1, dim - 1>> m_IB_codim1; // membrane energy inverse bases
    Field<T> m_mu_codim1;
    Field<T> m_lam_codim1;
    std::unordered_map<int, std::vector<std::pair<int, int>>> m_obj_divider_codim1;

    // codim1 bending elements
    Field<Vector<int, 4>> m_edge_stencil_codim1;
    Field<T> m_e_codim1, m_h_codim1, m_rest_angle_codim1;
    Field<T> m_bend_stiff_codim1;

    // state
    Field<TV> m_x;
    Field<TV> m_v;
    Field<TV> m_a;

    // control
    Field<TM> BC_basis;
    Field<int> BC_order;
    Field<TV> BC_target;
    Field<uint8_t> BC_fixed;
    Field<TV> m_f;

    // intermediate
    Field<TV> m_x1, m_v1; // x and v in the previous previous time step
    Field<TV> m_x_tilde;

    SERIALIZATION_REGISTER(m_x)
    SERIALIZATION_REGISTER(m_v)
    SERIALIZATION_REGISTER(m_a)
    SERIALIZATION_REGISTER(m_x1)
    SERIALIZATION_REGISTER(m_v1)

    TSMethod tsMethod = BE;
    bool static_sim = false;
    T tol = 1e-3;
    int max_iter = 200;

    // level-set based boundary condition
    BoundaryConditionManager<T, dim> BC;

    // internal helper function
    virtual void append_nodes(const Field<TV>& x)
    {
        m_X.insert(m_X.end(), x.begin(), x.end());
        m_x.insert(m_x.end(), x.begin(), x.end());
        m_v.resize(m_v.size() + x.size(), Vector<T, dim>::Zero());
        m_a.resize(m_a.size() + x.size(), Vector<T, dim>::Zero());
        m_x1.insert(m_x1.end(), x.begin(), x.end());
        m_v1.resize(m_v1.size() + x.size(), Vector<T, dim>::Zero());
        m_x_tilde.insert(m_x_tilde.end(), x.begin(), x.end());
        BC_basis.resize(m_x.size(), TM::Identity());
        BC_order.resize(m_x.size(), 0);
        BC_target.resize(m_x.size(), TV::Zero());
        BC_fixed.resize(m_x.size(), 0);
        m_f.resize(m_x.size(), TV::Zero());
    }

    // codim0 elements
    virtual void append(const Field<TV>& x, const Field<Vector<int, dim + 1>>& elem, const int type, const T E, const T nu, const T density)
    {
        Field<Vector<int, dim + 1>> shifted_elem = elem;
        Eigen::Map<Matrix<int, dim + 1, Eigen::Dynamic>>(&(shifted_elem[0][0]), dim + 1, shifted_elem.size()).array() += m_X.size();
        T mu, lam;
        std::tie(mu, lam) = ConstitutiveModel::lame_paramters(E, nu);

        m_density.resize(m_density.size() + elem.size(), density);
        m_elem.insert(m_elem.end(), shifted_elem.begin(), shifted_elem.end());
        m_mu.resize(m_mu.size() + elem.size(), mu);
        m_lam.resize(m_lam.size() + elem.size(), lam);
        m_obj_divider[type].push_back(std::make_pair(m_elem.size() - elem.size(), m_elem.size()));

        append_nodes(x);
    }

    // codim1 elements
    virtual void append(const Field<TV>& x, const Field<Vector<int, dim>>& elem_codim1, const T thickness,
        const int type_memb, const T E_memb, const T nu_memb,
        const int type_bend, const T E_bend, const T nu_bend,
        const T density)
    {
        Field<Vector<int, dim>> shifted_elem_codim1 = elem_codim1;
        Eigen::Map<Matrix<int, dim, Eigen::Dynamic>>(&(shifted_elem_codim1[0][0]), dim, shifted_elem_codim1.size()).array() += m_X.size();
        T mu, lam;
        std::tie(mu, lam) = ConstitutiveModel::lame_paramters(E_memb, nu_memb, true);

        m_thickness_codim1.resize(m_thickness_codim1.size() + elem_codim1.size(), thickness);
        m_density_codim1.resize(m_density_codim1.size() + elem_codim1.size(), density);
        m_elem_codim1.insert(m_elem_codim1.end(), shifted_elem_codim1.begin(), shifted_elem_codim1.end());
        m_mu_codim1.resize(m_mu_codim1.size() + elem_codim1.size(), mu);
        m_lam_codim1.resize(m_lam_codim1.size() + elem_codim1.size(), lam);
        m_obj_divider_codim1[type_memb].push_back(std::make_pair(m_elem_codim1.size() - elem_codim1.size(), m_elem_codim1.size()));

        //TODO: move to geometry
        Field<Vector<int, 4>> edge_stencil_codim1;
        {
            const auto& elements = elem_codim1;
            Bow::Field<Bow::Vector<int, 4>> edge_stencil;
            std::unordered_map<Bow::Vector<int, 2>, int, VectorHash<2>> half_edge;
            for (const auto& triVInd : elements) {
                for (int i = 0; i < 3; ++i) {
                    const auto finder = half_edge.find(Bow::Vector<int, 2>(triVInd[(i + 1) % 3], triVInd[i]));
                    if (finder == half_edge.end()) {
                        half_edge[Bow::Vector<int, 2>(triVInd[i], triVInd[(i + 1) % 3])] = triVInd[(i + 2) % 3];
                    }
                    else {
                        edge_stencil.emplace_back(triVInd[i], triVInd[(i + 1) % 3], triVInd[(i + 2) % 3], finder->second);
                    }
                }
            }
            edge_stencil_codim1 = edge_stencil;
        }
        Field<Vector<int, 4>> shifted_edge_stencil_codim1 = edge_stencil_codim1;
        Eigen::Map<Matrix<int, 4, Eigen::Dynamic>>(&(shifted_edge_stencil_codim1[0][0]), 4, shifted_edge_stencil_codim1.size()).array() += m_X.size();
        T k_bend = E_bend / (24 * (1.0 - nu_bend * nu_bend)) * thickness * thickness * thickness;

        m_edge_stencil_codim1.insert(m_edge_stencil_codim1.end(), shifted_edge_stencil_codim1.begin(), shifted_edge_stencil_codim1.end());
        m_bend_stiff_codim1.resize(m_bend_stiff_codim1.size() + edge_stencil_codim1.size(), k_bend);
        m_obj_divider_codim1[type_bend].push_back(std::make_pair(m_edge_stencil_codim1.size() - edge_stencil_codim1.size(), edge_stencil_codim1.size()));

        append_nodes(x);
    }

    void compute_cauchy(Field<TM>& cauchy)
    {
        cauchy.resize(m_elem.size());
        if (m_obj_divider.find(Bow::ConstitutiveModel::FIXED_COROTATED) != m_obj_divider.end()) {
            auto energy_ptr = std::make_shared<FixedCorotatedEnergyOp<T, dim>>(m_elem, m_vol, m_mu, m_lam, m_IB, m_obj_divider[Bow::ConstitutiveModel::FIXED_COROTATED]);
            energy_ptr->compute_cauchy(m_x, cauchy);
        }
        if (m_obj_divider.find(Bow::ConstitutiveModel::NEO_HOOKEAN) != m_obj_divider.end()) {
            auto energy_ptr = std::make_shared<NeoHookeanEnergyOp<T, dim>>(m_elem, m_vol, m_mu, m_lam, m_IB, m_obj_divider[Bow::ConstitutiveModel::NEO_HOOKEAN]);
            energy_ptr->compute_cauchy(m_x, cauchy);
        }
        if (m_obj_divider.find(Bow::ConstitutiveModel::LINEAR_ELASTICITY) != m_obj_divider.end()) {
            auto energy_ptr = std::make_shared<LinearElasticityEnergyOp<T, dim>>(m_elem, m_vol, m_mu, m_lam, m_IB, m_obj_divider[Bow::ConstitutiveModel::LINEAR_ELASTICITY]);
            energy_ptr->compute_cauchy(m_x, cauchy);
        }
    }

    void compute_von_mises(Field<T>& von_mises)
    {
        von_mises.resize(m_elem.size());
        if (m_obj_divider.find(Bow::ConstitutiveModel::FIXED_COROTATED) != m_obj_divider.end()) {
            auto energy_ptr = std::make_shared<FixedCorotatedEnergyOp<T, dim>>(m_elem, m_vol, m_mu, m_lam, m_IB, m_obj_divider[Bow::ConstitutiveModel::FIXED_COROTATED]);
            energy_ptr->compute_von_mises(m_x, von_mises);
        }
        if (m_obj_divider.find(Bow::ConstitutiveModel::NEO_HOOKEAN) != m_obj_divider.end()) {
            auto energy_ptr = std::make_shared<NeoHookeanEnergyOp<T, dim>>(m_elem, m_vol, m_mu, m_lam, m_IB, m_obj_divider[Bow::ConstitutiveModel::NEO_HOOKEAN]);
            energy_ptr->compute_von_mises(m_x, von_mises);
        }
        if (m_obj_divider.find(Bow::ConstitutiveModel::LINEAR_ELASTICITY) != m_obj_divider.end()) {
            auto energy_ptr = std::make_shared<LinearElasticityEnergyOp<T, dim>>(m_elem, m_vol, m_mu, m_lam, m_IB, m_obj_divider[Bow::ConstitutiveModel::LINEAR_ELASTICITY]);
            energy_ptr->compute_von_mises(m_x, von_mises);
        }
    }

    /** initialize energy terms and operators */
    virtual void initialize() override
    {
        Bow::FEM::InitializeOp<T, dim> fem_initialize{ m_X, m_elem, m_density,
            m_elem_codim1, m_thickness_codim1, m_density_codim1,
            m_mass, m_vol, m_IB, m_vol_codim1, m_IB_codim1 };
        fem_initialize();

        if constexpr (dim == 3) {
            Bow::Shell::InitializeBendingOp<T> shellBend_initialize{ m_X, m_edge_stencil_codim1, m_e_codim1, m_h_codim1, m_rest_angle_codim1 };
            shellBend_initialize();
        }
    }

    template <class BOUNDARY_CONDITION>
    void add_boundary_condition(BOUNDARY_CONDITION* bc)
    {
        BC.add(std::shared_ptr<BOUNDARY_CONDITION>(bc));
    }

    bool update_index_based_bc_velocity(int node, const TV& velocity)
    {
        return BC.update_index_based_bc_velocity(node, velocity);
    }

    virtual void additional_options(TimeIntegrator& update_op, std::vector<std::shared_ptr<EnergyOp<T, dim>>>& energy_terms, const T dt) {}

    virtual void advance(T dt) override
    {
        BoundaryConditionUpdateOp<T, dim> bc_update{ BC, m_x, BC_basis, BC_order, BC_target, BC_fixed, dt };
        bc_update();
        std::vector<std::shared_ptr<EnergyOp<T, dim>>> energy_terms;
        if (!static_sim)
            energy_terms.push_back(std::make_shared<InertialEnergyOp<T, dim>>(m_mass, m_x_tilde));
        energy_terms.push_back(std::make_shared<StaticForceEnergy<T, dim>>(m_f, m_mass, gravity));
        if (m_obj_divider.find(Bow::ConstitutiveModel::FIXED_COROTATED) != m_obj_divider.end()) {
            auto energy_ptr = std::make_shared<FixedCorotatedEnergyOp<T, dim>>(m_elem, m_vol, m_mu, m_lam, m_IB, m_obj_divider[Bow::ConstitutiveModel::FIXED_COROTATED]);
            energy_terms.push_back(energy_ptr);
        }
        if (m_obj_divider.find(Bow::ConstitutiveModel::NEO_HOOKEAN) != m_obj_divider.end()) {
            auto energy_ptr = std::make_shared<NeoHookeanEnergyOp<T, dim>>(m_elem, m_vol, m_mu, m_lam, m_IB, m_obj_divider[Bow::ConstitutiveModel::NEO_HOOKEAN]);
            energy_terms.push_back(energy_ptr);
        }
        if (m_obj_divider.find(Bow::ConstitutiveModel::LINEAR_ELASTICITY) != m_obj_divider.end()) {
            auto energy_ptr = std::make_shared<LinearElasticityEnergyOp<T, dim>>(m_elem, m_vol, m_mu, m_lam, m_IB, m_obj_divider[Bow::ConstitutiveModel::LINEAR_ELASTICITY]);
            energy_terms.push_back(energy_ptr);
        }
        if constexpr (dim == 3) {
            if (m_obj_divider_codim1.find(Bow::ConstitutiveModel::NEO_HOOKEAN_MEMBRANE) != m_obj_divider_codim1.end()) {
                auto energy_ptr = std::make_shared<Shell::MembraneEnergyOp<T>>(m_elem_codim1, m_vol_codim1, m_mu_codim1, m_lam_codim1, m_IB_codim1);
                energy_terms.push_back(energy_ptr);
            }
            if (m_obj_divider_codim1.find(Bow::ConstitutiveModel::DISCRETE_HINGE_BENDING) != m_obj_divider_codim1.end()) {
                auto energy_ptr = std::make_shared<Shell::BendingEnergyOp<T>>(m_edge_stencil_codim1, m_e_codim1, m_h_codim1, m_rest_angle_codim1, m_bend_stiff_codim1);
                energy_terms.push_back(energy_ptr);
            }
        }
        TimeIntegrator update_op(BC_basis, BC_order, BC_target, BC_fixed, m_mass, m_x, m_v, m_a, m_x1, m_v1, m_x_tilde);
        additional_options(update_op, energy_terms, dt);
        for (auto energy_ptr : energy_terms) {
            update_op.m_energy_terms.push_back(energy_ptr.get());
        }
        update_op.tsMethod = tsMethod;
        update_op.dt = dt;
        update_op.tol = tol;
        update_op.max_iter = max_iter;
        update_op.line_search = this->line_search;
        update_op.project_dirichlet = false;
        update_op.return_if_backtracking_fails = true;
        update_op();
    }

    virtual void dump_output(int frame_num) override
    {
        if constexpr (dim == 2)
            IO::write_ply(this->output_directory + "/" + std::to_string(frame_num) + ".ply", m_x, m_elem);
        else {
            //TODO: output all surfaces together? how about edges and points?
            if (!m_elem.empty()) {
                IO::write_vtk(this->output_directory + "/" + std::to_string(frame_num) + ".vtk", m_x, m_elem);
            }
            if (!m_elem_codim1.empty()) {
                IO::write_ply(this->output_directory + "/" + std::to_string(frame_num) + ".ply", m_x, m_elem_codim1);
            }
        }
    }
};

} // namespace Bow::FEM