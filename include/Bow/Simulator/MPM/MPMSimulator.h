#pragma once

#include <Bow/Energy/MPM/MPMEnergies.h>
#include <Bow/ConstitutiveModel/PlasticityOp.h>
#include <Bow/Geometry/Hybrid/MPMTransfer.h>
#include <Bow/Geometry/PoissonDisk.h>
#include <Bow/Geometry/RandomSampling.h>
#include <Bow/Geometry/Hybrid/MPMGrid.h>
#include <Bow/TimeIntegrator/MPM/MPMTimeIntegrator.h>
#include <Bow/Energy/MPM/MPMEnergies.h>
#include <Bow/Simulator/BoundaryConditionManager.h>
#include <Bow/Simulator/PhysicallyBasedSimulator.h>
#include <Bow/ConstitutiveModel/PlasticityOp.h>
#include <Bow/IO/tetwild.h>
#include <Bow/IO/vtk.h>
#include <Bow/Utils/Serialization.h>
#include <Bow/IO/ply.h>
#include "BoundaryConiditionUpdateOp.h"

namespace Bow::MPM {

template <class T, int dim>
class MPMSimulator : virtual public PhysicallyBasedSimulator<T, dim> {
public:
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    Field<TV> m_X;
    Field<TV> m_V;
    Field<TM> m_C;
    Field<TV> m_x_tilde;
    std::vector<T> m_mass;

    SERIALIZATION_REGISTER(m_X)
    SERIALIZATION_REGISTER(m_V)
    SERIALIZATION_REGISTER(m_C)
    SERIALIZATION_REGISTER(m_mass)

    Field<TM> stress;
    MPMGrid<T, dim> grid;
    T dx = 0.02;
    TV gravity = TV::Zero();
    bool symplectic = false;
    bool backward_euler = true;
    bool apic = true;
    bool dump_F_for_meshing = false;
    bool verbose = true;

    std::vector<std::shared_ptr<ElasticityOp<T, dim>>> elasticity_models;
    std::vector<std::shared_ptr<PlasticityOp<T, dim>>> plasticity_models;

    BoundaryConditionManager<T, dim> BC;
    Field<Matrix<T, dim, dim>> BC_basis;
    Field<int> BC_order;

    T newton_tol = 1e-3;
    T cfl = 0.6;

    int max_PN_iter = 1000;

    MPMSimulator()
    {
        this->output_directory = "mpm_output/";
    }

    void add_particles_random(std::shared_ptr<ElasticityOp<T, dim>> model, const TV& min_corner, const TV& max_corner, const TV& velocity = TV::Zero(), T density = 1000., const unsigned count = 0)
    {
        BOW_ASSERT(count > 0);
        T vol = (max_corner - min_corner).prod() / (T)count;
        int start = m_X.size();
        Field<TV> new_samples;
        Geometry::RandomSampling<T, dim> random_sampling(min_corner, max_corner, count);
        random_sampling.sample(new_samples);
        for (auto position : new_samples) {
            m_X.push_back(position);
            m_V.push_back(velocity);
            m_C.push_back(TM::Zero());
            m_mass.push_back(density * vol);
            stress.push_back(TM::Zero());
        }
        int end = m_X.size();
        model->append(start, end, vol);
    }

    void add_particles(std::shared_ptr<ElasticityOp<T, dim>> model, const TV& min_corner, const TV& max_corner, const TV& velocity = TV::Zero(), T density = 1000., T ppc = (T)(1 << dim))
    {
        T vol = std::pow(dx, dim) / T(ppc);
        int start = m_X.size();
        Field<TV> new_samples;
        Geometry::PoissonDisk<T, dim> poisson_disk(min_corner, max_corner, dx, ppc);
        poisson_disk.sample(new_samples);
        for (auto position : new_samples) {
            m_X.push_back(position);
            m_V.push_back(velocity);
            m_C.push_back(TM::Zero());
            m_mass.push_back(density * vol);
            stress.push_back(TM::Zero());
        }
        int end = m_X.size();
        model->append(start, end, vol);
    }

    void add_particles_from_tetwild(std::shared_ptr<ElasticityOp<T, dim>> model, const std::string mesh_path, const std::string vtk_path = "tet.vtk", const TV& center = TV::Zero(), const TV& velocity = TV::Zero(), T density = 1000.)
    {
        Field<Vector<T, 3>> X;
        Field<Vector<int, 4>> cells;
        IO::read_mesh(mesh_path, X, cells);
        IO::write_vtk(vtk_path, X, cells, false);
        T total_volume = 0;
        for (size_t i = 0; i < cells.size(); i++) {
            TV p0 = X[cells[i](0)], p1 = X[cells[i](1)],
               p2 = X[cells[i](2)], p3 = X[cells[i](3)];
            Matrix<T, 4, 4> A;
            A << 1, p0(0), p0(1), p0(2), 1, p1(0), p1(1), p1(2), 1, p2(0), p2(1), p2(2), 1, p3(0), p3(1), p3(2);
            T temp = A.determinant() / (T)6;
            total_volume += (temp > 0 ? temp : (-temp));
        }
        T vol = total_volume / (T)X.size();
        T mass_per_particle = vol * density;
        int start = m_X.size();
        for (size_t i = 0; i < X.size(); i++) {
            m_X.push_back(X[i] + center);
            m_V.push_back(velocity);
            m_C.push_back(TM::Zero());
            m_mass.push_back(mass_per_particle);
            stress.push_back(TM::Zero());
        }
        int end = m_X.size();
        model->append(start, end, vol);
    }

    template <class ELASTICITY>
    std::shared_ptr<ElasticityOp<T, dim>> create_elasticity(ELASTICITY* e)
    {
        elasticity_models.push_back(std::shared_ptr<ELASTICITY>(e));
        return elasticity_models.back();
    }

    template <class PLASTICITY>
    void create_plasticity(PLASTICITY* e)
    {
        plasticity_models.push_back(std::shared_ptr<PLASTICITY>(e));
    }

    template <class BOUNDARY_CONDITION>
    std::shared_ptr<BOUNDARY_CONDITION> add_boundary_condition(BOUNDARY_CONDITION* bc)
    {
        auto bc_shared_ptr = std::shared_ptr<BOUNDARY_CONDITION>(bc);
        BC.add(bc_shared_ptr);
        return bc_shared_ptr;
    }

    template <int interpolation_degree = 2>
    void p2g(T dt)
    {
        grid.sortParticles(m_X, dx);
        if (symplectic) {
            for (auto& model : elasticity_models)
                model->compute_stress(stress);
            ParticlesToGridOp<T, dim, true, interpolation_degree> P2G{ {}, m_X, m_V, m_mass, m_C, stress, grid, dx, dt };
            P2G();
        }
        else {
            ParticlesToGridOp<T, dim, false, interpolation_degree> P2G{ {}, m_X, m_V, m_mass, m_C, stress, grid, dx, dt };
            P2G();
        }
    }

    // https://stackoverflow.com/questions/47333843/using-initializer-list-for-a-struct-with-inheritance
    template <int interpolation_degree = 2>
    void grid_update(T dt, bool direct_solver = false)
    {
        BC.update(dt);
        if (symplectic) {
            BoundaryConditionUpdateOp<T, dim> bc_update{ {}, grid, gravity, BC, dx, dt };
            bc_update();
        }
        else {
            ImplicitBoundaryConditionUpdateOp<T, dim> bc_update{ {}, grid, BC, BC_basis, BC_order, dx };
            bc_update();
            InertialEnergy<T, dim, int> inertial_energy(grid, m_x_tilde);
            GravityForceEnergy<T, dim, int> gravity_energy(grid, gravity, dx);
            ElasticityEnergy<T, dim, int, interpolation_degree> elasticity_energy(grid, m_X, elasticity_models, dx);
            TimeIntegratorUpdateOp<T, dim, int, interpolation_degree> implicit_mpm(grid, m_X, BC_basis, BC_order, m_x_tilde, dx, dt);
            implicit_mpm.m_energy_terms.push_back(&inertial_energy);
            implicit_mpm.m_energy_terms.push_back(&gravity_energy);
            implicit_mpm.m_energy_terms.push_back(&elasticity_energy);
            implicit_mpm.verbose = verbose;
            if (backward_euler) {
                implicit_mpm.tsMethod = BE;
            }
            else {
                implicit_mpm.tsMethod = NM;
            }
            implicit_mpm.direct_solver = direct_solver;
            implicit_mpm.tol = newton_tol;
            implicit_mpm.max_iter = max_PN_iter;
            implicit_mpm.gravity = gravity;
            implicit_mpm();
        }
    }

    template <int interpolation_degree = 2>
    void g2p(T dt)
    {
        Field<T> FBarMultipliers;
        GridToParticlesOp<T, dim, interpolation_degree> G2P{ {}, m_X, m_V, m_C, grid, dx, dt };
        G2P(apic);
        for (auto& model : elasticity_models)
            model->evolve_strain(G2P.m_gradXp, FBarMultipliers);
        for (auto& model : plasticity_models)
            model->project_strain();
    }

    void advance(T dt, int frame_num, T frame_dt) override
    {
        p2g(dt);
        grid_update(dt);
        g2p(dt);
    }

    virtual T calculate_dt()
    {
        T max_speed = 0;
        for (int i = 0; i < (int)m_X.size(); ++i) {
            max_speed = m_V[i].norm() > max_speed ? m_V[i].norm() : max_speed;
        }
        return cfl * dx / (max_speed + 1e-10);
    }

    void dump_output(int frame_num) override
    {
        IO::write_ply(this->output_directory + std::to_string(frame_num) + ".ply", m_X);

        if (dump_F_for_meshing) {
            Field<TM> m_Fs;
            m_Fs.resize(m_X.size(), TM::Identity());
            for (auto& model : elasticity_models)
                model->collect_strain(m_Fs);
            Bow::IO::write_meshing_data(this->output_directory + std::to_string(frame_num) + ".dat", m_X, m_Fs);
        }
    }
};

} // namespace Bow::MPM
