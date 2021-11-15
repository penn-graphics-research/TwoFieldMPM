#include "CRAMPSimulator.h"

using namespace Bow;

int main(int argc, char *argv[])
{
    if (argc < 2) {
        puts("ERROR: please add parameters");
        puts("USAGE: ./mpm testcase");
        puts("       ./mpm testcase start_frame");
        exit(0);
    }

    int testcase = std::atoi(argv[1]);
    int start_frame = 0;

    /*---TEST NUMBERS---*/
    // 2D tests are 200-series, 3D tests are 300 series (201, 202, 301, 302, etc.)

    //USED FOR TESTING GRID STATE SIZE
    if(testcase == 0){
        using T = double;
        static const int dim = 2;
        Bow::DFGMPM::GridState<T, dim> gs;
        std::cout << "GridState size: " << sizeof(gs) << std::endl;
        //std::cout << "Padding: " << sizeof(gs.padding) << std::endl;
        
        //Testing Quaternion representations
        if(false){
            Matrix<T,dim,dim> F = Matrix<T,dim,dim>::Identity();
            F(0,0) = -45;
            F(0,1) = -7;
            F(1,0) = 82;
            F(1,1) = -700;
            
            if(dim == 2){
                Matrix<T, dim, dim> U, V;
                Vector<T, dim> sigma;
                Math::svd(F, U, sigma, V);
                Matrix<T, dim, dim> Sigma = sigma.asDiagonal();
                
                std::cout << "U:\n" << U << std::endl;
                std::cout << "Sigma:\n" << Sigma << std::endl;
                std::cout << "V^T:\n" << V.transpose() << std::endl;
                std::cout << "F:\n" << F << std::endl;
                std::cout << "Freconstruct:\n" << U * Sigma * V.transpose() << std::endl;
                
                //Now convert U and V to quaternions
                Matrix<T, 3,3> Upad = Matrix<T,3,3>::Identity();
                Matrix<T, 3,3> Vpad = Matrix<T,3,3>::Identity();
                Upad.topLeftCorner(2,2) = U;
                Vpad.topLeftCorner(2,2) = V; //pad these to be 3x3 for quaternion
                Eigen::Quaternion<T> rotU(Upad);
                Eigen::Quaternion<T> rotV(Vpad);
                rotU.normalize();
                rotV.normalize(); //normalize our quaternions!

                Vector<T, 4> Uquat, Vquat; //quaternion coefficients for U and V
                Uquat = rotU.coeffs();
                Vquat = rotV.coeffs();
                Eigen::Quaternion<T> rotUreconstruct(Uquat);
                Eigen::Quaternion<T> rotVreconstruct(Vquat);

                Matrix<T,3,3> Ureconstruct = rotUreconstruct.toRotationMatrix();
                Matrix<T,3,3> Vreconstruct = rotVreconstruct.toRotationMatrix();
                U = Ureconstruct.topLeftCorner(2,2);
                V = Vreconstruct.topLeftCorner(2,2);
                std::cout << "Freconstruct after quaternions:\n" << U * Sigma * V.transpose() << std::endl;
            }
        }
        
        return 0;
        //Without Padding
        //NOTE: if we already had a power of two, need to pad to the next one up still because can't conditionally do padding = 0 B
        //Float2D: 224 B -> add 32 B -> 8 Ts
        //Float3D: 288 B -> add 224 B -> 56 Ts
        //Double2D: 448 B -> add 64 B -> 8 Ts
        //Double3D: 576 B -> add 448 B -> 56 Ts
        //Vector<T, (48 * dim) - 88> padding; //dim2 = 8 Ts, dim3 = 56 Ts --> y = 48x - 88

        //AFTER ADDING Pi1, Pi2, Fi1, Fi2 (without padding) - 8/13/21
        //Float2D: 304 B -> add 208 B -> 52 Ts
        //Float3D: 432 B -> add 80 B -> 20 Ts
        //Double2D: 608 B -> add 416 B -> 52 Ts
        //Double3D: 864 B -> add 160 B -> 20 Ts
        //Vector<T, (-32 * dim) + 116> padding; //dim2 = 52 Ts, dim3 = 20 Ts --> y = -32x + 116

        //AFTER ADDING sigma1, Uquat1, etc. and removing Fi1, Pi1, etc. ... 9/13/21
        //Float2D: 320 B -> add 192 B -> 48 Ts
        //Float3D: 384 B -> add 128 B -> 32 Ts
        //Double2D: 640 B -> add 384 B -> 48 Ts
        //Double3D: 768 B -> add 256 B -> 32 Ts
        //Vector<T, (-16 * dim) + 80> padding; //dim2 = 48 Ts, dim3 = 32 Ts --> y = -16x + 80

        //AFTER ADDING cauchy and Fi 10/5/21
        //Float2D: 384 B -> add 128 B -> 32 Ts
        //Float3D: 528 B -> add 496 B -> 124 Ts
        //Double2D: 768 B -> add 256 B -> 32 Ts
        //Double3D: 1056 B -> add 992 B -> 124 Ts
        //Vector<T, (92 * dim) - 152> padding; //dim2 = 32 Ts, dim3 = 124 Ts --> y = 92x - 152
    }
    
    /*--------------2D BEGIN (200 SERIES)---------------*/

    //SENT specimen with fibrin parameters
    if (testcase == 201) {
        
        //Fibrin Parameters from Tutwiler2020
        // fracture toughness,          Gc = 7.6 +/- 0.45 J/m^2
        // folded state stiffness,      cf = 4.4e4 N/m^2
        // unfolded state stiffness,    cu = 2.6e6 N/m^2
        // fibrinogen density,          rho = 1395 g/cm^3 = 1,395,000 kg/m^3 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3044599/

        using T = double;
        static const int dim = 2;
        MPM::CRAMPSimulator<T, dim> sim("output/SENT_1e-3_noDamp_dx0.25mm_newTensorTransfer_sigmaA_2600_actualFCR_ramp2000");

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = 0.25e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 4000;
        //sim.frame_dt = 22e-6 / sim.end_frame; //total time = 22e-6 s, want 1000 frames of this
        sim.frame_dt = 1e-3; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 4.0; //4.0 is close to working, but we actually don't want surfacing for this demo
        sim.useDFG = true;
        sim.fricCoeff = 0.4;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        //auto material1 = sim.create_elasticity(new MPM::LinearElasticityOp<T, dim>(E, nu));

        //Sample Particles
        int ppc = 9;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        sim.sampleGridAlignedBox(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho);
        //sim.samplGeridAlignedBoxWithPoissonDisk(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho);

        //Add Crack
        T crackSegmentLength = sim.dx / 5.0;
        T damageRadius = sim.dx / 2.0;
        T crackLength = 5e-3;
        T crackY = y1 + height/2.0;
        T crackX = x1 + (sim.dx / std::pow(ppc, (T)1 / dim) / 2.0);
        sim.addHorizontalCrack(Vector<T,dim>(crackX, crackY), Vector<T,dim>(crackX + crackLength, crackY), crackSegmentLength, damageRadius);

        //Add Boundary Conditions
        // bool singlePuller = false;
        // T yTop = y2 - 0.5e-3;
        // T yBottom = y1 + 0.5e-3;
        // T u2 = 0.2e-3; // pull a total displacement of 0.2 mm, so each puller will pull half this distance
        // T pullTime = (sim.frame_dt * sim.end_frame) / 2.0; //pull for half of the total time duration
        // T speed = (u2 / 2.0) / pullTime;
        // std::cout << "speed:" << speed << std::endl;
        // if(singlePuller){
        //     //fix bottom constant, pull on top the full u2
        //     sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yTop), Vector<T, dim>(0, -1), Vector<T, dim>(0, speed * 2.0), pullTime)); //top puller (pull up u2)
        //     sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yBottom), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), pullTime)); //bottom puller (constant)
        // }
        // else{
        //     //pull from top and bottom, each pulling u2/2
        //     sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yTop), Vector<T, dim>(0, -1), Vector<T, dim>(0, speed), pullTime)); //top puller (pull up u2/2)
        //     sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yBottom), Vector<T, dim>(0, 1), Vector<T, dim>(0, -speed), pullTime)); //bottom puller (pull down u2/2)
        // }
        
        T sigmaA = 2600; //1000 times smaller than E
        T rampTime = sim.frame_dt * 2000; //ramp up to full sigmaA over 500 frames
        sim.addMode1Loading(y2, y1, sigmaA, rampTime, true, width, x1, x2); //if doing nodal loading, pass y1, y2, x1, x2 as the exact min and max of the material!

        // T simpleDampFactor = 0.5;
        // T simpleDampStartTime = sim.frame_dt * 500; //start damping once we reach the full load (rampTime over)
        // T simpleDampDuration = sim.frame_dt * 500; //for 1500 frames, damp
        // sim.addSimpleDamping(simpleDampFactor, simpleDampStartTime, simpleDampDuration);

        T snapshotTime = sim.frame_dt * (sim.end_frame - 1); //1950; //take snapshot after damping, around 1600
        //snapshotTime = sim.frame_dt * 3000; //take snapshot at frame 3000
        T halfEnvelope = sim.dx;
        sim.addStressSnapshot(snapshotTime, halfEnvelope);
        sim.contourRadii.push_back(1);
        sim.contourRadii.push_back(2);
        sim.contourRadii.push_back(3);
        sim.contourRadii.push_back(4);
        sim.contourRadii.push_back(5);
        sim.contourRadii.push_back(6);
        sim.contourRadii.push_back(7);
        sim.contourRadii.push_back(8);
        sim.contourRadii.push_back(9); //contour Radii to test

        sim.contourRadii.push_back(10); //for dx = 0.25mm
        sim.contourRadii.push_back(11);
        sim.contourRadii.push_back(12);
        sim.contourRadii.push_back(13);
        sim.contourRadii.push_back(14);
        sim.contourRadii.push_back(15);
        sim.contourRadii.push_back(16);
        sim.contourRadii.push_back(17);
        sim.contourRadii.push_back(18);
        sim.contourRadii.push_back(19);

        sim.run(start_frame);
    }

    //SENT specimen from CRAMP paper with small deformation (true E = 200e9)
    if (testcase == 202) {
        
        //Simulation notes
        // Without damping, this sim comes to equilibrium within 2e-3 seconds (two frames of 1e-3 frame dt), BUT there is still no crack opening displacement
        
        using T = double;
        static const int dim = 2;
        MPM::CRAMPSimulator<T, dim> sim("output/SENT_1e-3_wDFG_woDamping_E200e9");

        //Params
        sim.dx = 0.5e-3; //0.5 mm
        sim.symplectic = true;
        sim.end_frame = 2000;
        //sim.frame_dt = 22e-6 / sim.end_frame; //total time = 22e-6 s, want 1000 frames of this
        sim.frame_dt = 1e-3; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 4.0; //4.0 is close to working, but we actually don't want surfacing for this demo
        sim.useDFG = true;
        sim.fricCoeff = 0.4;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;

        //material
        T E = 200e9; //200e9 = 200 GPa
        T nu = 0.3; //0.3
        T rho = 5000; //5.0 g/cm^3 -> 5000 kg/m^3
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        //auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        auto material1 = sim.create_elasticity(new MPM::LinearElasticityOp<T, dim>(E, nu));

        int ppc = 9;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        sim.sampleGridAlignedBox(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho);

        T crackSegmentLength = sim.dx / 5.0;
        T damageRadius = sim.dx / 2.0;
        T crackLength = 5e-3;
        T crackY = y1 + height/2.0;
        T crackX = x1 + (sim.dx / std::pow(ppc, (T)1 / dim) / 2.0);
        sim.addHorizontalCrack(Vector<T,dim>(crackX, crackY), Vector<T,dim>(crackX + crackLength, crackY), crackSegmentLength, damageRadius);

        //sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, 15e-3), Vector<T, dim>(0, -1)));

        T yTop = y2 - 0.5e-3;
        T yBottom = y1 + 0.5e-3;
        T sigmaA = 400e6; //400 MPa
        T rampTime = 0; //500e-6; //ramp up to full sigmaA over 500 microseconds
        sim.addMode1Loading(yTop, yBottom, sigmaA, rampTime, false); //particle loading

        // T simpleDampFactor = 0.5;
        // T simpleDampDuration = sim.frame_dt * 1500; //for 1500 frames, damp
        // sim.addSimpleDamping(simpleDampFactor, simpleDampDuration);

        T snapshotTime = sim.frame_dt * 7;//1950; //take snapshot after damping, around 1600
        T halfEnvelope = sim.dx;
        sim.addStressSnapshot(snapshotTime, halfEnvelope);

        sim.run(start_frame);

        
    }

    if (testcase == 203) {
        using T = double;
        static const int dim = 2;
        MPM::CRAMPSimulator<T, dim> sim("output/hangingCube2D_CRAMP");

        //Params
        sim.dx = 0.01;
        sim.ppc = 4;
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
        sim.verbose = false;
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

    //Damage propagation test, uniaxial tension
    if (testcase == 204) {

        //Fibrin Parameters from Tutwiler2020
        // fracture toughness,          Gc = 7.6 +/- 0.45 J/m^2
        // folded state stiffness,      cf = 4.4e4 N/m^2
        // unfolded state stiffness,    cu = 2.6e6 N/m^2
        // fibrinogen density,          rho = 1395 g/cm^3 = 1,395,000 kg/m^3 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3044599/

        //Setup command line options
        //argv[2] = eta
        //argv[3] = zeta
        //argv[4] = p
        //argv[5] = dMin

        // T eta = 100;
        // T zeta = 10000;
        // T p = 0.03;
        // T dMin = 0.4;

        if (argc < 6) {
            puts("ERROR: please add parameters");
            puts("TEST 204 USAGE: ./mpm testcase eta zeta p dMin");
            exit(0);
        }

        using T = double;
        static const int dim = 2;

        T eta = std::atof(argv[2]);
        T zeta = std::atof(argv[3]);
        T p = std::atof(argv[4]);
        T dMin = std::atof(argv[5]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 6; ++i){
            std::string cleanString = argv[i];
            if(i == 4 || i == 5){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/uniaxialTension_withDamage_Eta" + cleanedStrings[0] + "_Zeta" + cleanedStrings[1] + "_p" + cleanedStrings[2] + "_dMin" + cleanedStrings[3];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = 0.5e-3; //0.5 mm
        sim.symplectic = true;
        sim.end_frame = 4000;
        //sim.frame_dt = 22e-6 / sim.end_frame; //total time = 22e-6 s, want 1000 frames of this
        sim.frame_dt = 1e-3; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 4.0; //4.0 is close to working, but we actually don't want surfacing for this demo
        sim.useDFG = true;
        sim.fricCoeff = 0.4;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        //auto material1 = sim.create_elasticity(new MPM::LinearElasticityOp<T, dim>(E, nu));

        //Sample Particles
        int ppc = 9;
        T center = 0.05;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = center - width/2.0;
        T y1 = center - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        sim.sampleGridAlignedBox(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho);
        //sim.sampleGridAlignedBoxWithPoissonDisk(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho);

        //Add center slit to facilitate fracture better
        T crackSegmentLength = sim.dx / 5.0;
        T damageRadius = sim.dx / 2.0;
        T crackLength = 1e-3;
        Vector<T,dim> crackMin(center - (crackLength/2.0), center);
        Vector<T,dim> crackMax(center + (crackLength/2.0), center);
        int crackType = 1; //1 for middle crack
        sim.addHorizontalCrackWithoutPoints(crackMin, crackMax, crackSegmentLength, damageRadius, crackType);

        //Damage Params, add these with a method
        sim.addAnisoMPMDamage(eta, dMin, zeta, p);

        //Add Boundary Conditions
        bool singlePuller = false;
        T yTop = y2 - 0.5e-3;
        T yBottom = y1 + 0.5e-3;
        T u2 = 1.0e-3; // pull a total displacement of 0.2 mm, so each puller will pull half this distance
        T pullTime = (sim.frame_dt * sim.end_frame) / 2.0; //pull for half of the total time duration
        T speed = (u2 / 2.0) / pullTime;
        std::cout << "speed:" << speed << std::endl;
        // T sigmaA = 400e5;
        // T rampTime = sim.frame_dt * 500; //ramp up to full sigmaA over 500 frames
        // //rampTime = 0.0;
        // sim.addMode1Loading(yTop, yBottom, sigmaA, rampTime);
        if(singlePuller){
            //fix bottom constant, pull on top the full u2
            sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yTop), Vector<T, dim>(0, -1), Vector<T, dim>(0, speed * 2.0), pullTime)); //top puller (pull up u2)
            sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yBottom), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), pullTime)); //bottom puller (constant)
        }
        else{
            //pull from top and bottom, each pulling u2/2
            sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yTop), Vector<T, dim>(0, -1), Vector<T, dim>(0, speed), pullTime)); //top puller (pull up u2/2)
            sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yBottom), Vector<T, dim>(0, 1), Vector<T, dim>(0, -speed), pullTime)); //bottom puller (pull down u2/2)
        }

        sim.run(start_frame);
    }

    // Uniaxial Tension Test (Previously Plane Strain Tension Test)
    if (testcase == 205) {

        //Fibrin Parameters from Tutwiler2020
        // fracture toughness,          Gc = 7.6 +/- 0.45 J/m^2
        // folded state stiffness,      cf = 4.4e4 N/m^2
        // unfolded state stiffness,    cu = 2.6e6 N/m^2
        // fibrinogen density,          rho = 1395 g/cm^3 = 1,395,000 kg/m^3 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3044599/

        using T = double;
        static const int dim = 2;

        MPM::CRAMPSimulator<T, dim> sim("output/uniaxialTension_E2.6e6_nu0.25_u2_1mm_u1_toBalance");

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = 0.5e-3; //0.5 mm
        sim.symplectic = true;
        sim.end_frame = 4000;
        //sim.frame_dt = 22e-6 / sim.end_frame; //total time = 22e-6 s, want 1000 frames of this
        sim.frame_dt = 1e-3; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 4.0; //4.0 is close to working, but we actually don't want surfacing for this demo
        sim.useDFG = true;
        sim.fricCoeff = 0.4;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        //auto material1 = sim.create_elasticity(new MPM::LinearElasticityOp<T, dim>(E, nu));

        //Sample Particles
        int ppc = 9;
        T center = 0.05;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = center - width/2.0;
        T y1 = center - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        sim.sampleGridAlignedBox(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho);
        //sim.sampleGridAlignedBoxWithPoissonDisk(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho);

        //Add Boundary Conditions
        bool singlePuller = false;
        T gripMargin = 0.5e-3;
        T yTop = y2 - gripMargin;
        T yBottom = y1 + gripMargin;
        T u2 = 1.0e-3; // pull a total displacement of 0.2 mm, so each puller will pull half this distance
        T pullTime = (sim.frame_dt * sim.end_frame) / 2.0; //pull for half of the total time duration
        T speed = (u2 / 2.0) / pullTime;
        if(singlePuller){
            //fix bottom constant, pull on top the full u2
            sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yTop), Vector<T, dim>(0, -1), Vector<T, dim>(0, speed * 2.0), pullTime)); //top puller (pull up u2)
            sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yBottom), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), pullTime)); //bottom puller (constant)
        }
        else{
            //pull from top and bottom, each pulling u2/2
            sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yTop), Vector<T, dim>(0, -1), Vector<T, dim>(0, speed), pullTime)); //top puller (pull up u2/2)
            sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yBottom), Vector<T, dim>(0, 1), Vector<T, dim>(0, -speed), pullTime)); //bottom puller (pull down u2/2)
        }

        T xLeft = x1 + gripMargin;
        T xRight = x2 - gripMargin;
        T u1 = -nu * u2 * (width / height);
        T speedX = (u1 / 2.0) / pullTime;
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(xLeft, 0), Vector<T, dim>(1, 0), Vector<T, dim>(speedX, 0), pullTime)); //left puller (pull u1/2)
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(xRight, 0), Vector<T, dim>(-1, 0), Vector<T, dim>(-speedX, 0), pullTime)); //right puller (pull u1/2)

        sim.run(start_frame);
    }

    //Plate with Hole (sigma22 = 3 * sigmaApplied)
    if (testcase == 206) {

        //Fibrin Parameters from Tutwiler2020
        // fracture toughness,          Gc = 7.6 +/- 0.45 J/m^2
        // folded state stiffness,      cf = 4.4e4 N/m^2
        // unfolded state stiffness,    cu = 2.6e6 N/m^2
        // fibrinogen density,          rho = 1395 g/cm^3 = 1,395,000 kg/m^3 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3044599/

        using T = double;
        static const int dim = 2;

        MPM::CRAMPSimulator<T, dim> sim("output/plateWithHole_E2.6e6_nu0.25_sigmaA_2600_aOverb0.2_dx0.25_smoothedTensors_nodalLoading");

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = 0.25e-3; //0.5 mm
        sim.symplectic = true;
        sim.end_frame = 4000;
        //sim.frame_dt = 22e-6 / sim.end_frame; //total time = 22e-6 s, want 1000 frames of this
        sim.frame_dt = 1e-3; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 4.0; //4.0 is close to working, but we actually don't want surfacing for this demo
        sim.useDFG = true;
        sim.fricCoeff = 0.4;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        //auto material1 = sim.create_elasticity(new MPM::LinearElasticityOp<T, dim>(E, nu));

        //Sample Particles
        int ppc = 9;
        T center = 0.05;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = center - width/2.0;
        T y1 = center - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T aOverB = 0.2;
        T radius = aOverB * (width / 2.0);
        sim.sampleGridAlignedBoxWithHole(material1, minPoint, maxPoint, Vector<T,dim>(center, center), radius, Vector<T, dim>(0, 0), ppc, rho);
        //sim.sampleGridAlignedBoxWithPoissonDisk(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho);

        //Add Boundary Conditions
        // bool singlePuller = true;
        
        // T u2 = 1.0e-3; // pull a total displacement of 0.2 mm, so each puller will pull half this distance
        // T pullTime = (sim.frame_dt * sim.end_frame) / 2.0; //pull for half of the total time duration
        // T speed = (u2 / 2.0) / pullTime;
        // std::cout << "speed:" << speed << std::endl;
        T sigmaA = 2600; //1000 times smaller than E
        T rampTime = sim.frame_dt * 500; //ramp up to full sigmaA over 500 frames
        //rampTime = 0.0;

        // T yTop = y2 - 0.5e-3;
        // T yBottom = y1 + 0.5e-3;
        // sim.addMode1Loading(yTop, yBottom, sigmaA, rampTime, false); //particle loading
        sim.addMode1Loading(y2, y1, sigmaA, rampTime, true, width, x1, x2); //if doing nodal loading, pass y1, y2, x1, x2 as the exact min and max of the material!
        
        // if(singlePuller){
        //     //fix bottom constant, pull on top the full u2
        //     sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yTop), Vector<T, dim>(0, -1), Vector<T, dim>(0, speed * 2.0), pullTime)); //top puller (pull up u2)
        //     sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yBottom), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), pullTime)); //bottom puller (constant)
        // }
        // else{
        //     //pull from top and bottom, each pulling u2/2
        //     sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yTop), Vector<T, dim>(0, -1), Vector<T, dim>(0, speed), pullTime)); //top puller (pull up u2/2)
        //     sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yBottom), Vector<T, dim>(0, 1), Vector<T, dim>(0, -speed), pullTime)); //bottom puller (pull down u2/2)
        // }

        sim.run(start_frame);
    }

    return 0;
}