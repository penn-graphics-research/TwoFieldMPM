#include <Bow/Simulator/MPM/MPMSimulator.h>
#include <Bow/Utils/FileSystem.h>
#include <Bow/Utils/ResultRecorder.h>

using namespace Bow;

int main(int argc, char* argv[])
{
    if (argc != 2 && argc != 3) {
        puts("ERROR: please add parameters");
        puts("USAGE: ./mpm testcase");
        puts("       ./mpm testcase start_frame");
        exit(0);
    }

    std::string output_dir = "mpm_output/";
    Bow::FileSystem::create_path(output_dir);

    // TODO: use some parameter library
    int testcase = std::atoi(argv[1]);
    int start_frame = 0;
    if (argc == 3) {
        start_frame = std::atoi(argv[2]);
    }

    if (testcase == 1) {
        using T = double;
        static const int dim = 3;
        MPM::MPMSimulator<T, dim> mpm;
        mpm.suggested_dt = 0.001;
        mpm.dx = 0.02;
        mpm.symplectic = true;
        // Using `new` to avoid redundant copy constructor
        auto material1 = mpm.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(10000, 0.4));
        mpm.add_particles(material1, Vector<T, dim>(0, 0, 0), Vector<T, dim>(.5, .5, .5), Vector<T, dim>(1, 0, 0));
        auto material2 = mpm.create_elasticity(new MPM::EquationOfStateOp<T, dim>(10000, 7));
        mpm.add_particles(material2, Vector<T, dim>(.6, 0, 0), Vector<T, dim>(1.1, .5, .5), Vector<T, dim>(-1, 0, 0));
        mpm.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(-0.1, 0, 0), Vector<T, dim>(1, 0, 0)));
        mpm.run(start_frame);
    }

    // TODO: add 30x30x30 particles makes this extremely slow
    if (testcase == 2) {
        using T = double;
        static const int dim = 3;
        MPM::MPMSimulator<T, dim> mpm;
        mpm.suggested_dt = mpm.frame_dt;
        mpm.dx = 0.02;
        mpm.symplectic = false;
        auto material = mpm.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(10000, 0.4));
        mpm.add_particles(material, Vector<T, dim>(0, 0, 0), Vector<T, dim>(.5, .5, .5), Vector<T, dim>(-1, 0, 0));
        mpm.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(-0.05, 0, 0), Vector<T, dim>(1, 1, 0)));
        mpm.run(start_frame);
    }

    if (testcase == 3) {
        using T = double;
        static const int dim = 2;
        MPM::MPMSimulator<T, dim> mpm;
        mpm.suggested_dt = mpm.frame_dt;
        mpm.dx = 0.02;
        mpm.symplectic = false;
        mpm.newton_tol = 1e-7;
        auto material = mpm.create_elasticity(new MPM::NeoHookeanOp<T, dim>(10000, 0.4));
        mpm.add_particles(material, Vector<T, dim>(0, 0), Vector<T, dim>(.5, .5), Vector<T, dim>(-1, 0));
        mpm.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(-0.05, 0), Vector<T, dim>(1, 1)));
        mpm.run(start_frame);
    }

    if (testcase == 4) { // sand
        using T = double;
        static const int dim = 2;
        T rho = 1000;
        T Youngs = 1e6;
        T nu = 0.35;

        MPM::MPMSimulator<T, dim> mpm;
        mpm.output_directory = "outputs/sand/";
        mpm.suggested_dt = 1e-4;
        mpm.gravity = Vector<T, dim>::Unit(1) * -9.8;
        mpm.dx = 0.01;
        mpm.symplectic = true;
        mpm.newton_tol = 1e-6;
        auto material = mpm.create_elasticity(new MPM::StvkWithHenckyOp<T, dim>(Youngs, nu));
        mpm.create_plasticity(new MPM::DruckerPragerStvkHencky<T, dim>(material, 30.));
        mpm.add_particles(material, Vector<T, dim>(0, 0), Vector<T, dim>(0.1, 0.3), Vector<T, dim>(0, 0), rho);
        mpm.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, 0), Vector<T, dim>(0, 1)));
        mpm.run(start_frame);
    }

    if (testcase == 41) { // sand
        using T = double;
        static const int dim = 2;
        T rho = 1000;
        T Youngs = 1e6;
        T nu = 0.35;

        MPM::MPMSimulator<T, dim> mpm;
        mpm.output_directory = "outputs/sand_vm/";
        mpm.suggested_dt = 1e-4;
        mpm.gravity = Vector<T, dim>::Unit(1) * -10;
        mpm.dx = 0.01;
        mpm.symplectic = true;
        mpm.newton_tol = 1e-6;
        auto material = mpm.create_elasticity(new MPM::StvkWithHenckyOp<T, dim>(Youngs, nu));
        mpm.create_plasticity(new MPM::VonMisesStvkHencky<T, dim>(material, 500.));
        mpm.add_particles(material, Vector<T, dim>(0, 0), Vector<T, dim>(0.1, 0.3), Vector<T, dim>(0, 0), rho);
        mpm.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, 0), Vector<T, dim>(0, 1)));
        mpm.run(start_frame);
    }
    // dam break
    if (testcase == 5) {
        using T = double;
        static const int dim = 2;
        MPM::MPMSimulator<T, dim> mpm;
        mpm.end_frame = 1200;
        mpm.suggested_dt = 0.001;
        mpm.gravity = Vector<T, dim>::Unit(1) * -9.8;
        mpm.dx = 0.005;
        mpm.symplectic = false;
        auto material = mpm.create_elasticity(new MPM::EquationOfStateOp<T, dim>(10000, 7));
        mpm.add_particles(material, Vector<T, dim>(0., 0.), Vector<T, dim>(.3, 1.), Vector<T, dim>(0, 0));
        mpm.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(0, 0), Vector<T, dim>(1, 0)));
        mpm.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(0, 0), Vector<T, dim>(0, 1)));
        mpm.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(1, 1), Vector<T, dim>(-1, 0)));
        mpm.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(1, 1), Vector<T, dim>(0, -1)));
        mpm.run(start_frame);
    }

    // dam break
    if (testcase == 55) {
        using T = double;
        static const int dim = 2;
        MPM::MPMSimulator<T, dim> mpm;
        mpm.end_frame = 1200;
        mpm.suggested_dt = 0.001;
        mpm.frame_dt = mpm.suggested_dt;
        mpm.gravity = Vector<T, dim>::Unit(1) * -9.8;
        mpm.dx = 0.005;
        mpm.symplectic = false;
        mpm.max_PN_iter = 1;
        auto material = mpm.create_elasticity(new MPM::EquationOfStateOp<T, dim>(10000, 7));
        mpm.add_particles(material, Vector<T, dim>(0., 0.), Vector<T, dim>(.3, 1.), Vector<T, dim>(0, 0));
        mpm.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(0, 0), Vector<T, dim>(1, 0)));
        mpm.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(0, 0), Vector<T, dim>(0, 1)));
        mpm.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(1, 1), Vector<T, dim>(-1, 0)));
        mpm.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(1, 1), Vector<T, dim>(0, -1)));
        mpm.run(start_frame);
    }

    if (testcase == 6) {
        using T = double;
        static const int dim = 2;
        MPM::MPMSimulator<T, dim> mpm;
        mpm.suggested_dt = mpm.frame_dt;
        mpm.dx = 0.02;
        mpm.symplectic = false;
        auto material = mpm.create_elasticity(new MPM::StvkWithHenckyOp<T, dim>(10000, 0.4));
        mpm.add_particles(material, Vector<T, dim>(0, 0), Vector<T, dim>(.5, .5), Vector<T, dim>(0, 0));
        dynamic_cast<Bow::MPM::StvkWithHenckyOp<T, dim>*>(material.get())->m_F *= 10;
        mpm.run(start_frame);
    }

    if (testcase == 1001) {
        // for meshing
        using T = double;
        static const int dim = 3;
        MPM::MPMSimulator<T, dim> mpm;
        mpm.suggested_dt = 0.00005;
        mpm.dx = 0.02;
        mpm.symplectic = true;
        mpm.gravity = Vector<T, dim>::Unit(1) * -0.1;
        mpm.dump_F_for_meshing = true;
        auto material = mpm.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(10000, 0.3));
        mpm.add_particles_from_tetwild(material, "/home/yunuo/Desktop/data/geo/tree1_20k.mesh", "mpm_output/tet.vtk", Vector<T, dim>(0, 0, 0), Vector<T, dim>(0, 0, 0), 10000.0);
        mpm.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, 0.1, 0), Vector<T, dim>(0, 1, 0)));
        mpm.run(start_frame);
    }

    if (testcase == 1002) { // snow block falling on a mound 3D
        using T = double;
        static const int dim = 3;
        T rho = 3;
        T Youngs = 360;
        T nu = 0.3;
        const int num_particles = 8e5;
        MPM::MPMSimulator<T, dim> mpm;
        mpm.output_directory = "outputs/snow_mound_3d/";
        mpm.suggested_dt = 0.01;
        mpm.gravity = Vector<T, dim>::Unit(1) * -2.5;
        mpm.dx = 0.005;
        mpm.symplectic = false;
        mpm.newton_tol = 1e-3;
        auto material = mpm.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(Youngs, nu));
        mpm.create_plasticity(new MPM::SnowPlasticity<T, dim>(material, 10.0, 0.025, 0.005));
        mpm.add_particles_random(material, Vector<T, dim>(0.3, 0.7, 0.4), Vector<T, dim>(0.7, 0.9, 0.6), Vector<T, dim>(0, 0, 0), rho, num_particles);
        mpm.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0.4, 0.3, 0.0), Vector<T, dim>(0.6, 0.5, 1.0), Vector<T, dim + 1>(0, 0, 0.38268343, 0.92387953)));
        mpm.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, 0, 0), Vector<T, dim>(0, 1, 0)));
        mpm.run(start_frame);
    }

    return 0;
}
