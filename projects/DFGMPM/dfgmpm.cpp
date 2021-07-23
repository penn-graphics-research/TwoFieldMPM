#include "DFGMPMSimulator.h"

using namespace Bow;

int main(int argc, char *argv[])
{
    if (argc != 2 && argc != 3) {
        puts("ERROR: please add parameters");
        puts("USAGE: ./mpm testcase");
        puts("       ./mpm testcase start_frame");
        exit(0);
    }

    // TODO: use some parameter library
    int testcase = std::atoi(argv[1]);
    int start_frame = 0;
    if (argc == 3) {
        start_frame = std::atoi(argv[2]);
    }

    /*---TEST NUMBERS---*/
    // 2D tests are 200-series, 3D tests are 300 series (201, 202, 301, 302, etc.)

    //USED FOR TESTING GRID STATE SIZE
    if(testcase == 0){
        using T = float;
        static const int dim = 3;
        Bow::DFGMPM::GridState<T, dim> gs;
        std::cout << "GridState size: " << sizeof(gs) << std::endl;
        //cout << "Padding: " << sizeof(gs.padding) << std::endl;
        return 0;
        //Without Padding
        //NOTE: if we already had a power of two, need to pad to the next one up still because can't conditionally do padding = 0 B
        //Float2D: 224 B -> add 32 B -> 8 Ts
        //Float3D: 288 B -> add 224 B -> 56 Ts
        //Double2D: 448 B -> add 64 B -> 8 Ts
        //Double3D: 576 B -> add 448 B -> 56 Ts
    }
    
    /*--------------2D BEGIN (200 SERIES)---------------*/

    if (testcase == 201) {
        using T = double;
        static const int dim = 2;
        MPM::DFGMPMSimulator<T, dim> sim("output/hangingCube2D");

        //Params
        sim.dx = 0.01;
        sim.symplectic = true;
        sim.end_frame = 240;
        sim.frame_dt = (T)1. / 24;
        sim.gravity = -10;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.9;
        
        //DFG Specific Params
        sim.st = 0.0; //4.0 is close to working, but we actually don't want surfacing for this demo
        sim.useDFG = true;
        sim.fricCoeff = 0.4;
        
        //Debug mode
        sim.verbose = true;
        sim.writeGrid = true;

        //material
        T E = 1000;
        T nu = 0.4;
        T rho = 10;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        sim.samplePrecutRandomCube(material1, Vector<T, dim>(0.4, 0.4), Vector<T, dim>(0.6, 0.6), Vector<T, dim>(0, 0), rho);
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, 0.59), Vector<T, dim>(0, -1)));
        sim.run(start_frame);
    }

    if (testcase == 202) {
        using T = double;
        static const int dim = 2;
        MPM::DFGMPMSimulator<T, dim> sim("output/ballpit2D_implicitDFG_withNoContact");

        //Params
        T radius = 0.03;
        sim.dx = 0.0049002217; //from taichi
        sim.symplectic = false;
        sim.end_frame = 240;
        sim.frame_dt = (T)1. / 24;
        sim.gravity = -10;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.95;
        
        //DFG Specific Params
        sim.st = 4.2; //4.5 too high, a few in the middle
        sim.useDFG = true;
        sim.useImplicitContact = false;
        sim.fricCoeff = 0.4;
        
        //Sim Modes
        sim.verbose = false;
        sim.writeGrid = true;

        //density
        T E = 5000;
        T nu = 0.2;
        T rho = 10;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.7 * maxDt;

        if(!sim.symplectic){
            sim.suggested_dt = 1e-3;
        }

        // Using `new` to avoid redundant copy constructor
        //auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material2 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material3 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material4 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material5 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material6 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material7 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material8 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material9 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material10 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material11 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material12 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material13 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material14 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material15 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material16 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material17 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material18 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material19 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material20 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material21 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material22 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // auto material23 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        // sim.sampleRandomSphere(material1, Vector<T, dim>(0.14, 0.14), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material2, Vector<T, dim>(0.32, 0.14), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material3, Vector<T, dim>(0.5, 0.14), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material4, Vector<T, dim>(0.68, 0.14), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material5, Vector<T, dim>(0.86, 0.14), radius, Vector<T, dim>(0, 0), rho);

        // sim.sampleRandomSphere(material6, Vector<T, dim>(0.23, 0.32), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material7, Vector<T, dim>(0.41, 0.32), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material8, Vector<T, dim>(0.59, 0.32), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material9, Vector<T, dim>(0.77, 0.32), radius, Vector<T, dim>(0, 0), rho);

        // sim.sampleRandomSphere(material10, Vector<T, dim>(0.86, 0.5), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material11, Vector<T, dim>(0.14, 0.5), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material12, Vector<T, dim>(0.32, 0.5), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material13, Vector<T, dim>(0.5, 0.5), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material14, Vector<T, dim>(0.68, 0.5), radius, Vector<T, dim>(0, 0), rho);

        // sim.sampleRandomSphere(material15, Vector<T, dim>(0.23, 0.68), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material16, Vector<T, dim>(0.41, 0.68), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material17, Vector<T, dim>(0.59, 0.68), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material18, Vector<T, dim>(0.77, 0.68), radius, Vector<T, dim>(0, 0), rho);

        // sim.sampleRandomSphere(material19, Vector<T, dim>(0.68, 0.86), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material20, Vector<T, dim>(0.86, 0.86), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material21, Vector<T, dim>(0.14, 0.86), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material22, Vector<T, dim>(0.32, 0.86), radius, Vector<T, dim>(0, 0), rho);
        // sim.sampleRandomSphere(material23, Vector<T, dim>(0.5, 0.86), radius, Vector<T, dim>(0, 0), rho);

        //Sample from OBJ (obj was from Triangle in python)
        auto material = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        std::string filepath = "../projects/DFGMPM/Data/ballPit2D.obj";
        T volume = radius*radius * M_PI * 23.0; //23 discs
        sim.sampleFromObj(material, filepath, Vector<T, dim>(0, 0), volume, rho);

        //Unit square boundaries
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(0, 0.05), Vector<T, dim>(0, 1)));
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(0, 0.95), Vector<T, dim>(0, -1)));
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(0.05, 0), Vector<T, dim>(1, 0)));
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(0.95, 0), Vector<T, dim>(-1, 0)));
        sim.run(start_frame);
    }

    if (testcase == 203) {
        using T = double;
        static const int dim = 2;
        MPM::DFGMPMSimulator<T, dim> sim("output/circleExplode2D");

        //Params
        T fps = 240.0;
        sim.dx = std::sqrt(sim.ppc * 0.07 * 0.07 * M_PI / 4860.0); //from taichi demo, matches the circleOBJ file
        sim.symplectic = true;
        sim.end_frame = 240 * 3;
        sim.frame_dt = (T)1. / fps;
        sim.gravity = -10.0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.9;
        
        //DFG Specific Params
        sim.st = 4.4; //4.4 too low
        sim.useDFG = true;
        sim.fricCoeff = 0.0;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;

        //Damage Params, add these with a method
        T eta = 1e-5;
        T zeta = 1e4;
        T p = 5e-2; //5e-2 original, 6.5 too low still, 8.0 maybe too high?
        T dMin = 0.4;
        sim.addAnisoMPMDamage(eta, dMin, zeta, p);

        //Material
        T E = 1e4;
        T nu = 0.15;
        T rho = 1;

        //Compute time step for symplectic
        sim.cfl = 0.5;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        //Impulse
        Vector<T, dim> center(0.5, 0.5);
        T strength = -1e4; //-2e4 too low (too intense)
        T startTime = 0.0;
        T duration = 3.0 * fps; //3 seconds
        sim.addImpulse(center, strength, startTime, duration);

        // Using `new` to avoid redundant copy constructor
        auto material = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        
        //sim.sampleRandomSphere(material, Vector<T, dim>(0.5, 0.5), radius, Vector<T, dim>(0, 0), rho);
        
        //Sample from OBJ (obj was from Triangle in python)
        std::string filepath = "../projects/DFGMPM/Data/circle2D.obj";
        T radius = 0.07;
        T volume = radius*radius * M_PI;
        sim.sampleFromObj(material, filepath, Vector<T, dim>(0, 0), volume, rho);
        
        //Unit square boundaries
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(0, 0.05), Vector<T, dim>(0, 1)));
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(0, 0.95), Vector<T, dim>(0, -1)));
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(0.05, 0), Vector<T, dim>(1, 0)));
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(0.95, 0), Vector<T, dim>(-1, 0)));        
        
        sim.run(start_frame);
    }

    /*--------------3D BEGIN---------------*/

    if (testcase == 301) {
        using T = double;
        static const int dim = 3;
        MPM::DFGMPMSimulator<T, dim> sim("output/cubeCollide3D");

        //Params
        sim.dx = 0.02;
        sim.symplectic = true;
        sim.end_frame = 240;
        sim.frame_dt = (T)1. / 24;
        sim.suggested_dt = 1e-3;
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = true;
        
        //DFG Specific Params
        sim.st = 25.0; //25 too low?
        sim.useDFG = true;
        sim.fricCoeff = 0.1;
        
        //Debug mode
        sim.verbose = true;

        //Damage Params, add these with a method
        double eta = 0.1;
        double zeta = 1.0;
        double p = 0.1;
        double dMin = 0.25;
        sim.addAnisoMPMDamage(eta, dMin, zeta, p);

        //density
        T rho = 100;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(1000, 0.4));
        sim.sampleRandomCube(material1, Vector<T, dim>(0, 0, 0), Vector<T, dim>(.3, .3, .3), Vector<T, dim>(1, 0, 0), rho);
        auto material2 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(1000, 0.4));
        sim.sampleRandomCube(material2, Vector<T, dim>(.6, 0, 0), Vector<T, dim>(0.9, .3, .3), Vector<T, dim>(-1, 0, 0), rho);
        //sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(-0.1, 0, 0), Vector<T, dim>(1, 0, 0)));
        sim.run(start_frame);
    }

    if (testcase == 302) {
        using T = double;
        static const int dim = 3;
        MPM::DFGMPMSimulator<T, dim> sim("output/cubeFreefall3D");

        //Params
        sim.dx = 0.02;
        sim.symplectic = true;
        sim.end_frame = 240;
        sim.frame_dt = (T)1. / 24;
        sim.suggested_dt = 1e-3;
        sim.gravity = -10;

        //Interpolation Scheme
        sim.useAPIC = true;
        
        //DFG Specific Params
        sim.st = 25.0; //25 too low?
        sim.useDFG = false;
        sim.fricCoeff = 0.1;
        
        //Debug mode
        sim.verbose = true;

        //Damage Params, add these with a method
        double eta = 0.1;
        double zeta = 1.0;
        double p = 0.1;
        double dMin = 0.25;
        sim.addAnisoMPMDamage(eta, dMin, zeta, p);

        //Density
        T rho = 100;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(1000, 0.4));
        sim.sampleRandomCube(material1, Vector<T, dim>(0.45, 0.45, 0.45), Vector<T, dim>(.55, .55, .55), Vector<T, dim>(0, 0, 0), rho);
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, 0.3, 0), Vector<T, dim>(0, 1, 0)));
        sim.run(start_frame);
    }

    if (testcase == 303) {
        using T = double;
        static const int dim = 3;
        MPM::DFGMPMSimulator<T, dim> sim("output/hangingCube3D");

        //Params
        sim.dx = 0.02;
        sim.symplectic = true;
        sim.end_frame = 240;
        sim.frame_dt = (T)1. / 24;
        sim.suggested_dt = 1e-3;
        sim.gravity = -10;

        //Interpolation Scheme
        sim.useAPIC = true;
        
        //DFG Specific Params
        sim.st = 25.0; //25 too low?
        sim.useDFG = true;
        sim.fricCoeff = 0.1;
        
        //Debug mode
        sim.verbose = true;

        //density
        T rho = 100;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(10000, 0.4));
        sim.samplePrecutRandomCube(material1, Vector<T, dim>(0.35, 0.7, 0.35), Vector<T, dim>(.65, 1.0, .65), Vector<T, dim>(0, 0, 0), rho);
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, 0.97, 0), Vector<T, dim>(0, -1, 0)));
        sim.run(start_frame);
    }

    if (testcase == 304) {
        using T = double;
        static const int dim = 3;
        MPM::DFGMPMSimulator<T, dim> sim("output/explodingCube3D");

        //Params
        T fps = 240.0;
        sim.dx = 0.01;
        sim.symplectic = true;
        sim.end_frame = 240 * 3;
        sim.frame_dt = (T)1. / fps;
        sim.suggested_dt = 1e-5;
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = true;
        
        //DFG Specific Params
        sim.st = 0.0; //4.0 is close to working, but we actually don't want surfacing for this demo
        sim.useDFG = true;
        sim.fricCoeff = 0.1;
        
        //Debug mode
        sim.verbose = true;

        //Damage Params, add these with a method
        T eta = 1e-5;
        T zeta = 1e4;
        T p = 4.5e-2; //4.9e-2 too high
        T dMin = 0.25;
        sim.addAnisoMPMDamage(eta, dMin, zeta, p);

        //Impulse
        Vector<T, dim> center(0.5, 0.5, 0.5);
        T strength = -1e4; //-2e4 too low (too intense)
        T startTime = 0.0;
        T duration = 3.0 / fps;
        sim.addImpulse(center, strength, startTime, duration);

        //Material
        T E = 1e4;
        T nu = 0.15;
        T rho = 1;

        // Using `new` to avoid redundant copy constructor
        auto material = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        sim.sampleRandomCube(material, Vector<T, dim>(0.45, 0.45, 0.45), Vector<T, dim>(.55, .55, .55), Vector<T, dim>(0, 0, 0), rho);
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, 0.3, 0), Vector<T, dim>(0, 1, 0)));
        sim.run(start_frame);
    }
    return 0;
}