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

    /*---TEST INDEX---*/
    /* 2D tests are 200-series, 3D tests are 300 series (201, 202, 301, 302, etc.)
    0 ..... Sandbox (test grid state size, Eigen routines, etc)
    201 ... SENT with fibrin params, traction BCs, used for computing J Integrals with 2*dx DFG
    202 ... SENT with CRAMP params
    203 ... Hanging Cube
    204 ... [PYTHON] Old AnisoMPM Damage Propagation Test
    205 ... "Plane Strain Tension Test"
    206 ... Plate with Hole Test
    207 ... SENT with Wider Crack to compute J-integrals with single-field MPM
    208 ... Ballpit 2D (test DFG frictional contact)
    209 ... [PYTHON] FCR, SENT, Displacement BCs, "Stress Based" Damage
    210 ... [PYTHON] FCR, SENT, Displacement BCs, "Stretch Based" Damage
    211 ... [PYTHON] FCR, Shear Fracture, "Stress Based" Damage, Homel 2016 config and params
    212 ... [PYTHON] FCR, Shear Fracture, "Stretch Based" Damage
    213 ... [PYTHON] NH, SENT, Displacement BCs, "Stress Based" Damage
    214 ... [PYTHON] NH, SENT, Displacement BCs, "Stretch Based" Damage
    215 ... [PYTHON] NH, Shear Fracture, "Stress Based" Damage, Homel 2016 config and params
    216 ... [PYTHON] NH, Shear Fracture, "Stretch Based" Damage
    217 ... [PYTHON] Numerical Fracture exploration -- FCR, SENT, Displacement BCs, no Damage, variable dx
    218 ... Fluid Test
    219 ... [PYTHON] 70 Degree LARGER Shear Fracture Test (Stretch Based Damage with NH elasticity)
    220 ... Clot in Pipe Test with Reservoir
    221 ... [PYTHON] LARGER stretch SENT with Displacement BCs, using Stretch-Based Damage and NeoHookean elasticity
    222 ... [PYTHON] Horizontal Pipe w/o clot, Test for Parabolic Velocity under Viscous Fluid Model
    223 ... [PYTHON] Vertical Pipe w/o clot, Test for Parabolic Velocity under Viscous Fluid Model
    224 ... [PYTHON] Horizontal Pipe Flow with Elastic Pipe Walls -- Test for Parabolic Velocity under Viscous Fluid Model
    225 ... [PYTHON] Constant Pressure Horizontal Pipe Flow with Elastic Pipe Walls and no Gravity -- Test for Parabolic Velocity under Viscous Fluid Model
    226 ... [PYTHON] With Clot Inclusion - Constant Pressure Horizontal Pipe Flow with Elastic Pipe Walls and no Gravity -- Test for Parabolic Velocity under Viscous Fluid Model
    227 ... 2D Fluid Generator Test
    228 ... [PYTHON] Fluid Generator With Clot Inclusion - Constant Velocity Source with Elastic Pipe Walls and no Gravity
    229 ... [PYTHON] 40 Diameter Pipe With Clot Inclusion - 40 diameters long pipe with fluid Starting in pipe -- push piston from 0 to 20 diameters over some time frame
    230 ... [PYTHON] Shorter Pipe With Pressure Gradient (no clot)
    231 ... [PYTHON] Shorter Pipe With Pressure Gradient AND CLOT
    232 ... [PYTHON] Shorter Pipe With Pressure Gradient AND CLOT -- DIRICHLET PIPE WALLS
    233 ... [PYTHON] Shorter Pipe With Pressure Gradient NO CLOT -- DIRICHLET PIPE WALLS - test for finding pStart

    TGC Presentation Sims
    2001 .. SENT with Damage Region and Elasticity Degradation -> Single Field
    2002 .. SENT with Damage Region and Elasticity Degradation -> Two-Field
    2003 .. [PYTHON] SENT with Constant Width Crack & Variable Dx -> Single-Field
    2004 .. [PYTHON] SENT with Constant Width Crack & Variable Dx -> Two-Field
    2005 .. [PYTHON] Plate with Constant Radius Hole and Variable Dx -> Single Field
    2006 .. [PYTHON] Plate with Constant Radius Hole and Variable Dx -> Two-Field
    2007 .. SENT with 2*dx Wide Crack and Single Field (compare against equatorial stress results from two field 201 and single field 207)
    2008 .. SENT with Damage Region and Elasticity Degradation -> Two-Field --> Computing Dynamic J-Integral
    2009 .. SENT with Damage Region and Elasticity Degradation -> Two-Field --> Computing Dynamic J-Integral LOW-RES
    2010 .. SENT with Damage Region and Elasticity Degradation -> Single-Field --> Computing Dynamic J-Integral
    */

    //USED FOR TESTING GRID STATE SIZE
    if(testcase == 0){
        using T = float;
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

        //AFTER REMOVING sigma, Uquat, Vquat 4/7/22
        //Double2D: 608 B -> add 416 B -> 52 Ts
        //Double3D: 864 B -> add 160 B -> 20 Ts
        //Vector<T, (-32 * dim) + 116> padding; //dim2 = 52 Ts, dim3 = 20 Ts --> y = -32x + 116

        //AFTER adding displacement 4/19/22
        //Double2D: 640 B -> add 384 B -> 48 Ts
        //Double3D: 896 B -> add 160 B -> 16 Ts
        //Vector<T, (-32 * dim) + 112> padding; //dim2 = 48 Ts, dim3 = 16 Ts --> y = -32x + 112

        //adding chemicalPotential and Idx (T and int) 11/16/22
        //Double2D: 640 B -> add 384 B -> 48 Ts
        //Double3D: 928 B -> add 96 B -> 12 Ts
        //Vector<T, (-36 * dim) + 120> padding; //dim2 = 48 Ts, dim3 = 12 Ts --> y = -36x + 120
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
        MPM::CRAMPSimulator<T, dim> sim("output/201_SENT_2dxWideCrack_dx0.1mm_sigmaA_2600_FCR_ramp4s_PIC_DamageSurface");

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        bool useDisplacement = false;
        sim.dx = 0.1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 150;
        sim.frame_dt = 1e-1; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 0.0; //don't surface this, we get the outer ones from particle sampling and set the inner surface using damage
        sim.useDFG = true;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
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
        int ppc = 4;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T crackLength = 5e-3;
        T crackRadius = 0.0001; 
        T crackHeight = y1 + (height / 2.0); //- (sim.dx / 2.0); 
        //sim.sampleGridAlignedBox(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho);
        sim.sampleGridAlignedBoxWithNotch(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, true, Vector<T, dim>(0, 0), ppc, rho, true);

        //Add damage particles at the crack edges
        sim.addHorizontalCrackWithoutPoints(Vector<T,dim>(x1, crackHeight), Vector<T,dim>(x1 + crackLength, crackHeight), sim.dx/3.0, 0.000165); //damageRadius was found empirically!

        //Add Crack
        // T crackSegmentLength = sim.dx / 5.0;
        // T damageRadius = sim.dx / 2.0;
        // T crackLength = 5e-3;
        // T crackY = y1 + height/2.0 - (0.5*sim.dx);
        // T crackX = x1 + (sim.dx / std::pow(ppc, (T)1 / dim) / 2.0);
        // sim.addHorizontalCrack(Vector<T,dim>(crackX, crackY), Vector<T,dim>(crackX + crackLength, crackY), crackSegmentLength, damageRadius);
        //sim.addHorizontalCrackWithoutPoints(Vector<T,dim>(crackX, crackY), Vector<T,dim>(crackX + crackLength, crackY), crackSegmentLength, damageRadius);

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
        T rampTime = sim.frame_dt * 40; // ramp up 4 seconds
        sim.addMode1Loading(y2, y1, sigmaA, rampTime, true, width, x1, x2); //if doing nodal loading, pass y1, y2, x1, x2 as the exact min and max of the material!

        // T simpleDampFactor = 0.5;
        // T simpleDampStartTime = sim.frame_dt * 500; //start damping once we reach the full load (rampTime over)
        // T simpleDampDuration = sim.frame_dt * 500; //for 1500 frames, damp
        // sim.addSimpleDamping(simpleDampFactor, simpleDampStartTime, simpleDampDuration);

        //T snapshotTime = sim.frame_dt * (sim.end_frame - 1); //1950; //take snapshot after damping, around 1600
        //snapshotTime = sim.frame_dt * 3000; //take snapshot at frame 3000
        //T halfEnvelope = sim.dx;
        //sim.addStressSnapshot(snapshotTime, halfEnvelope);
        
        //Add Elasticity Degradation
        sim.elasticityDegradationType = 1;

        //Add Energy Tracking
        T energyDt = sim.frame_dt;
        sim.addEnergyTracking(energyDt);

        //Add Contours
        
        //DX = 0.5mm
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,15,5,15), true); //centered on crack tip
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,15,10,15), true); 
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,15,15,15), true);
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,15,20,15), true); //centered on crack tip
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,15,25,15), true); 
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,15,30,15), true);  

        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,25,5,25), true);
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,25,10,25), true); 
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,25,15,25), true);
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,25,20,25), true);
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,25,25,25), true); 
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,25,30,25), true); 

        //DX = 0.1mm
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,25,75), true, true); //second true is to mark this contour for additional tracking of data (J_I contributions)
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,50,75), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,75,75), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,100,75), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,125,75), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,145,75), true);
        //sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,150,75), true);  

        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,25,125), true, true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,50,125), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,75,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,100,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,125,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,145,125), true);  
        //sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,150,125), true); 

        //These have different L values than the other families!
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,75,25,75), true, true);    //compare to Contour A
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,75,100,75), true);         //to Contour D
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,125,25,125), true, true);  //to Contour 1
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,125,100,125), true);       //to Contour 4

        //Add contours that define the inverse intersections between each pair of contours (A and 1, B and 2, etc.) -> each pair has an upper and lower contour, each not containing the crack and should have J = 0
        Vector<T, dim> upperCenter(0.045, 0.06);
        Vector<T, dim> lowerCenter(0.045, 0.04);
        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 25, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 25, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 50, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 50, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 75, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 75, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 100, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 100, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 125, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 125, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 145, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 145, 25), false);
        
        //Add timing for contours (NOTE: without this we wont calculate anything!)
        std::vector<T> contourTimes;
        contourTimes.push_back(sim.frame_dt * 1);
        contourTimes.push_back(sim.frame_dt * 40);
        contourTimes.push_back(sim.frame_dt * 45);
        contourTimes.push_back(sim.frame_dt * 50);
        contourTimes.push_back(sim.frame_dt * 55);
        contourTimes.push_back(sim.frame_dt * 60);
        contourTimes.push_back(sim.frame_dt * 65);
        contourTimes.push_back(sim.frame_dt * 70);
        contourTimes.push_back(sim.frame_dt * 75);
        contourTimes.push_back(sim.frame_dt * 80);
        contourTimes.push_back(sim.frame_dt * 85);
        contourTimes.push_back(sim.frame_dt * 90);
        contourTimes.push_back(sim.frame_dt * 95);
        contourTimes.push_back(sim.frame_dt * 100);
        contourTimes.push_back(sim.frame_dt * 105);
        contourTimes.push_back(sim.frame_dt * 110);
        contourTimes.push_back(sim.frame_dt * 115);
        contourTimes.push_back(sim.frame_dt * 120);
        contourTimes.push_back(sim.frame_dt * 125);
        contourTimes.push_back(sim.frame_dt * 130);
        contourTimes.push_back(sim.frame_dt * 135);
        contourTimes.push_back(sim.frame_dt * 140);
        contourTimes.push_back(sim.frame_dt * 145);
        contourTimes.push_back(sim.frame_dt * 149);
        // contourTimes.push_back(sim.frame_dt * 150);
        // contourTimes.push_back(sim.frame_dt * 155);
        // contourTimes.push_back(sim.frame_dt * 160);
        // contourTimes.push_back(sim.frame_dt * 165);
        // contourTimes.push_back(sim.frame_dt * 170);
        // contourTimes.push_back(sim.frame_dt * 175);
        // contourTimes.push_back(sim.frame_dt * 180);
        // contourTimes.push_back(sim.frame_dt * 185);
        // contourTimes.push_back(sim.frame_dt * 190);
        // contourTimes.push_back(sim.frame_dt * 195);
        // contourTimes.push_back(sim.frame_dt * 199);
        sim.addJIntegralTiming(contourTimes, useDisplacement);

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
        MPM::CRAMPSimulator<T, dim> sim("output/hangingCube2D_CRAMP_closestNode");

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
        sim.samplePrecutRandomCube(material1, Vector<T, dim>(0.4, 0.4), Vector<T, dim>(0.6, 0.6), Vector<T, dim>(0, 0), rho, true);
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, 0.59), Vector<T, dim>(0, -1)));

        //Add Elasticity Degradation
        sim.elasticityDegradationType = 1;

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

    //SENT specimen but with wider crack so MPM can handle it without DFG -- SINGLE FIELD VERSION OF 201
    if (testcase == 207) {
        
        //Fibrin Parameters from Tutwiler2020
        // fracture toughness,          Gc = 7.6 +/- 0.45 J/m^2
        // folded state stiffness,      cf = 4.4e4 N/m^2
        // unfolded state stiffness,    cu = 2.6e6 N/m^2
        // fibrinogen density,          rho = 1395 g/cm^3 = 1,395,000 kg/m^3 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3044599/

        using T = double;
        static const int dim = 2;
        MPM::CRAMPSimulator<T, dim> sim("output/207_SENT_with2dxCrack_dx0.1mm_sigmaA_2600_FCR_ramp4s_singleField_damageSurface");

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = 0.1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 150; //need to simulate around 9 to 12 seconds to remove oscillations
        sim.frame_dt = 1e-1; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.useDFG = false;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));

        //Sample Particles
        int ppc = 4;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T crackLength = 5e-3;
        T crackRadius = 0.0001;
        T crackHeight = y1 + (height/2.0);
        sim.sampleGridAlignedBoxWithNotch(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, true, Vector<T, dim>(0, 0), ppc, rho);

        //Add damage particles at the crack edges
        sim.addHorizontalCrackWithoutPoints(Vector<T,dim>(x1, crackHeight), Vector<T,dim>(x1 + crackLength, crackHeight), sim.dx/3.0, 0.000165); //damageRadius was found empirically!

        //Add Traction Boundary Condition        
        T sigmaA = 2600; //1000 times smaller than E
        T rampTime = sim.frame_dt * 40; //ramp up to full sigmaA over 500 frames
        sim.addMode1Loading(y2, y1, sigmaA, rampTime, true, width, x1, x2); //if doing nodal loading, pass y1, y2, x1, x2 as the exact min and max of the material!
        
        //Add Elasticity Degradation
        sim.elasticityDegradationType = 1;

        //Add Contours
        //DX = 0.1mm
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,25,75), true, true); //second true is to mark this contour for additional tracking of data (J_I contributions)
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,50,75), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,75,75), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,100,75), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,125,75), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,145,75), true);
        //sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,150,75), true);  

        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,25,125), true, true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,50,125), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,75,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,100,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,125,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,145,125), true);  
        //sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,150,125), true); 

        //These have different L values than the other families!
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,75,25,75), true, true);    //compare to Contour A
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,75,100,75), true);         //to Contour D
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,125,25,125), true, true);  //to Contour 1
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,125,100,125), true);       //to Contour 4

        //Add contours that define the inverse intersections between each pair of contours (A and 1, B and 2, etc.) -> each pair has an upper and lower contour, each not containing the crack and should have J = 0
        Vector<T, dim> upperCenter(0.045, 0.06);
        Vector<T, dim> lowerCenter(0.045, 0.04);
        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 25, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 25, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 50, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 50, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 75, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 75, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 100, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 100, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 125, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 125, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 145, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 145, 25), false);
        
        //Add timing for contours (NOTE: without this we wont calculate anything!)
        std::vector<T> contourTimes;
        contourTimes.push_back(sim.frame_dt * 1);
        contourTimes.push_back(sim.frame_dt * 40);
        contourTimes.push_back(sim.frame_dt * 45);
        contourTimes.push_back(sim.frame_dt * 50);
        contourTimes.push_back(sim.frame_dt * 55);
        contourTimes.push_back(sim.frame_dt * 60);
        contourTimes.push_back(sim.frame_dt * 65);
        contourTimes.push_back(sim.frame_dt * 70);
        contourTimes.push_back(sim.frame_dt * 75);
        contourTimes.push_back(sim.frame_dt * 80);
        contourTimes.push_back(sim.frame_dt * 85);
        contourTimes.push_back(sim.frame_dt * 90);
        contourTimes.push_back(sim.frame_dt * 95);
        contourTimes.push_back(sim.frame_dt * 100);
        contourTimes.push_back(sim.frame_dt * 105);
        contourTimes.push_back(sim.frame_dt * 110);
        contourTimes.push_back(sim.frame_dt * 115);
        contourTimes.push_back(sim.frame_dt * 120);
        contourTimes.push_back(sim.frame_dt * 125);
        contourTimes.push_back(sim.frame_dt * 130);
        contourTimes.push_back(sim.frame_dt * 135);
        contourTimes.push_back(sim.frame_dt * 140);
        contourTimes.push_back(sim.frame_dt * 145);
        contourTimes.push_back(sim.frame_dt * 149);
        sim.addJIntegralTiming(contourTimes, false);

        sim.run(start_frame);
    }

    //Adding ballpit test to check the explicit frictional contact routines
    if(testcase == 208){
        using T = double;
        static const int dim = 2;
        MPM::CRAMPSimulator<T, dim> sim("output/ballpit2D_explicitFrictionalContact_homelFriction");

        //Params
        T radius = 0.03;
        sim.dx = 0.0049002217; //from taichi
        sim.symplectic = true;
        sim.end_frame = 240;
        sim.frame_dt = (T)1. / 24;
        sim.gravity = -10;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.95;
        
        //DFG Specific Params
        sim.st = 4.2; //4.5 too high, a few in the middle
        sim.useDFG = true;
        //sim.useImplicitContact = false;
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

        //Sample from OBJ (obj was from Triangle in python)
        auto material = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));
        std::string filepath = "../../projects/DFGMPM/Data/ballPit2D.obj";
        T volume = radius*radius * M_PI * 23.0; //23 discs
        sim.sampleFromObj(material, filepath, Vector<T, dim>(0, 0), volume, rho);

        //Unit square boundaries
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(0, 0.05), Vector<T, dim>(0, 1)));
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(0, 0.95), Vector<T, dim>(0, -1)));
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(0.05, 0), Vector<T, dim>(1, 0)));
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(0.95, 0), Vector<T, dim>(-1, 0)));
        sim.run(start_frame);
    }

    //[PYTHON] SENT with Displacement BCs, using Rankine Damage
    if(testcase == 209){
        
        using T = double;
        static const int dim = 2;
        
        //Setup command line options
        //argv[2] = Gf
        //argv[3] = sigmaC
        //argv[4] = alpha (elasticity degradation degree)
        //argv[5] = dMin
        //argv[6] = minDp

        //Good Params to Wedge Around
        // T Gf = 22.3e-3; //from Table2 Homel2016
        // T sigmaC = 2600;
        // T alpha = 1.0;
        // T dMin = 0.25;
        // T minDp = 1.0;
        
        if (argc < 7) {
            puts("ERROR: please add parameters");
            puts("TEST 209 USAGE: ./cramp testcase Gf sigmaC alpha dMin minDp");
            exit(0);
        }

        T Gf = std::atof(argv[2]);
        T sigmaC = std::atof(argv[3]);
        T alpha = std::atof(argv[4]);
        T dMin = std::atof(argv[5]);
        T minDp = std::atof(argv[6]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 7; ++i){
            std::string cleanString = argv[i];
            if(i == 2 || i == 5 || i == 6){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/currentVolumeFix_SENT_DisplacementBCs_StressBasedDamage_FCR_Gf" + cleanedStrings[0] + "_SigmaC" + cleanedStrings[1] + "_Alpha" + cleanedStrings[2] + "_dMin" + cleanedStrings[3] + "_minDp" + cleanedStrings[4];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = 0.1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 300;
        sim.frame_dt = 1e-3; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));

        //Sample Particles
        int ppc = 4;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T crackLength = 5e-3;
        T crackRadius = sim.dx;
        if(sim.useDFG == false){
            crackRadius *= 2.0;
        }
        T crackHeight = y1 + (height / 2.0); //- (sim.dx / 2.0); 
        //sim.sampleGridAlignedBox(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho);
        sim.sampleGridAlignedBoxWithNotch(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, false, Vector<T, dim>(0, 0), ppc, rho, true);

        //Add Boundary Conditions
        bool singlePuller = false;
        T heldMaterial = 2.0 * sim.dx;
        T yTop = y2 - heldMaterial;
        T yBottom = y1 + heldMaterial;
        T u2 = 1e-3; // pull a total displacement of 0.2 mm, so each puller will pull half this distance
        T pullTime = 7.5; //in seconds
        T speed = (u2 / 2.0) / pullTime;
        std::cout << "speed:" << speed << std::endl;
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

        //Add Rankine Damage Model
        T l0 = sqrt(2 * sim.dx * sim.dx);
        int degType = 1;
        sim.addRankineDamage(dMin, Gf, l0, degType, -1.0, sigmaC); //-1 is for p which we dont want to use here
        
        //Set degradation alpha
        sim.degAlpha = alpha;

        //set minDp
        sim.minDp = minDp;

        sim.run(start_frame);
    }

    //[PYTHON] SENT with Displacement BCs, using Hyperbolic Tangent Damage
    if(testcase == 210){
        
        using T = double;
        static const int dim = 2;
        
        //Setup command line options
        //argv[2] = lamC
        //argv[3] = tanhWidth
        //argv[4] = alpha (elasticity degradation degree)
        //argv[5] = dMin
        //argv[6] = minDp

        //Good Params to Wedge Around
        // T lamC = ???; //from Table2 Homel2016
        // T tanhWidth = ???;
        // T alpha = 1.0;
        // T dMin = 0.25;
        // T minDp = 1.0;
        
        if (argc < 7) {
            puts("ERROR: please add parameters");
            puts("TEST 210 USAGE: ./cramp testcase lamC tanhWidth alpha dMin minDp");
            exit(0);
        }

        T lamC = std::atof(argv[2]);
        T tanhWidth = std::atof(argv[3]);
        T alpha = std::atof(argv[4]);
        T dMin = std::atof(argv[5]);
        T minDp = std::atof(argv[6]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 7; ++i){
            std::string cleanString = argv[i];
            if(i == 3 || i == 5 || i == 6){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/currentVolumeFix_SENT_DisplacementBCs_StretchBasedDamage_FCR_lamC" + cleanedStrings[0] + "_tanhWidth" + cleanedStrings[1] + "_Alpha" + cleanedStrings[2] + "_dMin" + cleanedStrings[3] + "_minDp" + cleanedStrings[4];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = 0.1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 100;
        sim.frame_dt = 1e-3; //500 frames at 1e-3 is 0.5s
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));

        //Sample Particles
        int ppc = 4;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T crackLength = 5e-3;
        T crackRadius = sim.dx;
        if(sim.useDFG == false){
            crackRadius *= 2.0;
        }
        T crackHeight = y1 + (height / 2.0); //- (sim.dx / 2.0); 
        //sim.sampleGridAlignedBox(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho);
        sim.sampleGridAlignedBoxWithNotch(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, false, Vector<T, dim>(0, 0), ppc, rho, true);

        //Add Boundary Conditions
        bool singlePuller = false;
        T heldMaterial = 2.0 * sim.dx;
        T yTop = y2 - heldMaterial;
        T yBottom = y1 + heldMaterial;
        T u2 = height*0.5; // pull a total displacement of 0.2 mm, so each puller will pull half this distance
        T pullTime = 0.3; //in seconds
        T speed = (u2 / 2.0) / pullTime;
        std::cout << "speed:" << speed << std::endl;
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

        //Add Tanh Damage Model
        int degType = 1;
        sim.addHyperbolicTangentDamage(lamC, tanhWidth, dMin, degType);
        
        //Set degradation alpha
        sim.degAlpha = alpha;

        //set minDp
        sim.minDp = minDp;

        sim.run(start_frame);
    }

    //[PYTHON] 70 Degree Shear Fracture Test (Homel 2016 configuration, Stress based damage with FCR elasticity)
    if(testcase == 211){
        
        using T = double;
        static const int dim = 2;
        
        //Setup command line options
        //argv[2] = Gf
        //argv[3] = sigmaC
        //argv[4] = alpha (elasticity degradation degree)
        //argv[5] = dMin
        //argv[6] = minDp

        //Good Params to Wedge Around
        // T Gf = 22.3e-3; //from Table2 Homel2016
        // T sigmaC = 2600;
        // T alpha = 1.0;
        // T dMin = 0.25;
        // T minDp = 1.0;
        
        if (argc < 7) {
            puts("ERROR: please add parameters");
            puts("TEST 211 USAGE: ./cramp testcase Gf sigmaC alpha dMin minDp");
            exit(0);
        }

        T Gf = std::atof(argv[2]);
        T sigmaC = std::atof(argv[3]);
        T alpha = std::atof(argv[4]);
        T dMin = std::atof(argv[5]);
        T minDp = std::atof(argv[6]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 7; ++i){
            std::string cleanString = argv[i];
            if(i == 5 || i == 6){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/currentVolumeFix_HomelShearFractureTest_StressBasedDamage_FCR_Gf" + cleanedStrings[0] + "_SigmaC" + cleanedStrings[1] + "_Alpha" + cleanedStrings[2] + "_dMin" + cleanedStrings[3] + "_minDp" + cleanedStrings[4];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //material (from Homel)
        T E = 1.9e11;
        T nu = 0.2647;
        T rho = 8000;

        //Params
        sim.dx = 1e-3; //1 mm
        sim.symplectic = true;
        sim.end_frame = 300;
        sim.frame_dt = 1e-6; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));

        //Sample Target's Particles
        int ppc = 4;
        T height = 100e-3; //32mm
        T width = 100e-3; //20mm
        T x1 = 0.5 - width/2.0;
        T y1 = 0.5 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T crackLength = 0.05; //50mm
        T crackRadius = sim.dx;
        if(sim.useDFG == false){
            crackRadius *= 2.0;
        }
        T crackHeight = y1 + (height / 4.0); //- (sim.dx / 2.0); 
        sim.sampleGridAlignedBoxWithNotch(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, false, Vector<T, dim>(0, 0), ppc, rho, true);

        //Sample Impactor Particles
        T distFromTarget = sim.dx*2.0;
        x1 -= (0.1 + distFromTarget);
        x2 -= (width + distFromTarget);
        T y2New = y2 - 0.075 - crackRadius;
        Vector<T,dim> minPoint2(x1, y1);
        Vector<T,dim> maxPoint2(x2, y2New);
        T impactorSpeed = 33.0;
        sim.sampleGridAlignedBox(material1, minPoint2, maxPoint2, Vector<T,dim>(impactorSpeed, 0.0), ppc, rho, false);

        //Add Boundary Conditions
        T heldMaterial = 2.0 * sim.dx;
        T yTop = y2 - heldMaterial;
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yTop), Vector<T, dim>(0, -1), Vector<T, dim>(0, 0))); //hold the top

        //Add Rankine Damage Model
        T l0 = sqrt(2 * sim.dx * sim.dx);
        int degType = 1;
        sim.addRankineDamage(dMin, Gf, l0, degType, -1.0, sigmaC); //-1 is for p which we dont want to use here
        
        //Set degradation alpha
        sim.degAlpha = alpha;

        //set minDp
        sim.minDp = minDp;

        sim.run(start_frame);
    }

    //[PYTHON] 70 Degree Shear Fracture Test (Stretch Based Damage with FCR elasticity)
    if(testcase == 212){
        
        using T = double;
        static const int dim = 2;
        
        //Setup command line options
        //argv[2] = lamC
        //argv[3] = tanhWidth
        //argv[4] = alpha (elasticity degradation degree)
        //argv[5] = dMin
        //argv[6] = minDp

        //Good Params to Wedge Around
        // T lamC = 1.5; //from Table2 Homel2016
        // T tanhWidth = 0.2;
        // T alpha = 1.0;
        // T dMin = 0.25;
        // T minDp = 1.0;
        
        if (argc < 7) {
            puts("ERROR: please add parameters");
            puts("TEST 212 USAGE: ./cramp testcase lamC tanhWidth alpha dMin minDp");
            exit(0);
        }

        T lamC = std::atof(argv[2]);
        T tanhWidth = std::atof(argv[3]);
        T alpha = std::atof(argv[4]);
        T dMin = std::atof(argv[5]);
        T minDp = std::atof(argv[6]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 7; ++i){
            std::string cleanString = argv[i];
            if(i == 3 || i == 5 || i == 6){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/currentVolumeFix_HomelShearFractureTest_StretchBasedDamage_FCR_lamC" + cleanedStrings[0] + "_tanhWidth" + cleanedStrings[1] + "_Alpha" + cleanedStrings[2] + "_dMin" + cleanedStrings[3] + "_minDp" + cleanedStrings[4];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //material (from Homel)
        T E = 1.9e11;
        T nu = 0.2647;
        T rho = 8000;

        //Params
        sim.dx = 1e-3; //1 mm
        sim.symplectic = true;
        sim.end_frame = 300;
        sim.frame_dt = 1e-6; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));

        //Sample Target's Particles
        int ppc = 4;
        T height = 100e-3; //32mm
        T width = 100e-3; //20mm
        T x1 = 0.5 - width/2.0;
        T y1 = 0.5 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T crackLength = 0.05; //50mm
        T crackRadius = sim.dx;
        if(sim.useDFG == false){
            crackRadius *= 2.0;
        }
        T crackHeight = y1 + (height / 4.0); //- (sim.dx / 2.0); 
        sim.sampleGridAlignedBoxWithNotch(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, false, Vector<T, dim>(0, 0), ppc, rho, true);

        //Sample Impactor Particles
        T distFromTarget = sim.dx*2.0;
        x1 -= (0.1 + distFromTarget);
        x2 -= (width + distFromTarget);
        T y2New = y2 - 0.075 - crackRadius;
        Vector<T,dim> minPoint2(x1, y1);
        Vector<T,dim> maxPoint2(x2, y2New);
        T impactorSpeed = 33.0;
        sim.sampleGridAlignedBox(material1, minPoint2, maxPoint2, Vector<T,dim>(impactorSpeed, 0.0), ppc, rho, false);

        //Add Boundary Conditions
        T heldMaterial = 2.0 * sim.dx;
        T yTop = y2 - heldMaterial;
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yTop), Vector<T, dim>(0, -1), Vector<T, dim>(0, 0))); //hold the top

        //Add Tanh Damage Model
        int degType = 1;
        sim.addHyperbolicTangentDamage(lamC, tanhWidth, dMin, degType);
        
        //Set degradation alpha
        sim.degAlpha = alpha;

        //set minDp
        sim.minDp = minDp;

        sim.run(start_frame);
    }

    //[PYTHON] SENT with Displacement BCs, using Stress-Based Damage and NeoHookean elasticity
    if(testcase == 213){
        
        using T = double;
        static const int dim = 2;
        
        //Setup command line options
        //argv[2] = Gf
        //argv[3] = sigmaC
        //argv[4] = alpha (elasticity degradation degree)
        //argv[5] = dMin
        //argv[6] = minDp

        //Good Params to Wedge Around
        // T Gf = 22.3e-3; //from Table2 Homel2016
        // T sigmaC = 2600;
        // T alpha = 1.0;
        // T dMin = 0.25;
        // T minDp = 1.0;
        
        if (argc < 7) {
            puts("ERROR: please add parameters");
            puts("TEST 213 USAGE: ./cramp testcase Gf sigmaC alpha dMin minDp");
            exit(0);
        }

        T Gf = std::atof(argv[2]);
        T sigmaC = std::atof(argv[3]);
        T alpha = std::atof(argv[4]);
        T dMin = std::atof(argv[5]);
        T minDp = std::atof(argv[6]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 7; ++i){
            std::string cleanString = argv[i];
            if(i == 2 || i == 5 || i == 6){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/currentVolumeFix_SENT_DisplacementBCs_StressBasedDamage_NH_Gf" + cleanedStrings[0] + "_SigmaC" + cleanedStrings[1] + "_Alpha" + cleanedStrings[2] + "_dMin" + cleanedStrings[3] + "_minDp" + cleanedStrings[4];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = 0.1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 300;
        sim.frame_dt = 1e-3; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));

        //Sample Particles
        int ppc = 4;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T crackLength = 5e-3;
        T crackRadius = sim.dx;
        if(sim.useDFG == false){
            crackRadius *= 2.0;
        }
        T crackHeight = y1 + (height / 2.0); //- (sim.dx / 2.0); 
        //sim.sampleGridAlignedBox(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho);
        sim.sampleGridAlignedBoxWithNotch(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, false, Vector<T, dim>(0, 0), ppc, rho, true);

        //Add Boundary Conditions
        bool singlePuller = false;
        T heldMaterial = 2.0 * sim.dx;
        T yTop = y2 - heldMaterial;
        T yBottom = y1 + heldMaterial;
        T u2 = 1e-3; // pull a total displacement of 0.2 mm, so each puller will pull half this distance
        T pullTime = 7.5; //in seconds
        T speed = (u2 / 2.0) / pullTime;
        std::cout << "speed:" << speed << std::endl;
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

        //Add Rankine Damage Model
        T l0 = sqrt(2 * sim.dx * sim.dx);
        int degType = 1;
        sim.addRankineDamage(dMin, Gf, l0, degType, -1.0, sigmaC); //-1 is for p which we dont want to use here
        
        //Set degradation alpha
        sim.degAlpha = alpha;

        //set minDp
        sim.minDp = minDp;

        sim.run(start_frame);
    }

    //[PYTHON] SENT with Displacement BCs, using Stretch-Based Damage and NeoHookean elasticity
    if(testcase == 214){
        
        using T = double;
        static const int dim = 2;
        
        //Setup command line options
        //argv[2] = lamC
        //argv[3] = tanhWidth
        //argv[4] = alpha (elasticity degradation degree)
        //argv[5] = dMin
        //argv[6] = minDp

        //Good Params to Wedge Around
        // T lamC = ???; //from Table2 Homel2016
        // T tanhWidth = ???;
        // T alpha = 1.0;
        // T dMin = 0.25;
        // T minDp = 1.0;
        
        if (argc < 7) {
            puts("ERROR: please add parameters");
            puts("TEST 214 USAGE: ./cramp testcase lamC tanhWidth alpha dMin minDp");
            exit(0);
        }

        T lamC = std::atof(argv[2]);
        T tanhWidth = std::atof(argv[3]);
        T alpha = std::atof(argv[4]);
        T dMin = std::atof(argv[5]);
        T minDp = std::atof(argv[6]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 7; ++i){
            std::string cleanString = argv[i];
            if(i == 3 || i == 5 || i == 6){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/currentVolumeFix_SENT_DisplacementBCs_StretchBasedDamage_NH_lamC" + cleanedStrings[0] + "_tanhWidth" + cleanedStrings[1] + "_Alpha" + cleanedStrings[2] + "_dMin" + cleanedStrings[3] + "_minDp" + cleanedStrings[4];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = 0.1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 100;
        sim.frame_dt = 1e-3; //500 frames at 1e-3 is 0.5s
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));

        //Sample Particles
        int ppc = 4;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T crackLength = 5e-3;
        T crackRadius = sim.dx;
        if(sim.useDFG == false){
            crackRadius *= 2.0;
        }
        T crackHeight = y1 + (height / 2.0); //- (sim.dx / 2.0); 
        //sim.sampleGridAlignedBox(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho);
        sim.sampleGridAlignedBoxWithNotch(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, false, Vector<T, dim>(0, 0), ppc, rho, true);

        //Add Boundary Conditions
        bool singlePuller = false;
        T heldMaterial = 2.0 * sim.dx;
        T yTop = y2 - heldMaterial;
        T yBottom = y1 + heldMaterial;
        T u2 = height*0.5; // pull a total displacement of 0.2 mm, so each puller will pull half this distance
        T pullTime = 0.3; //in seconds
        T speed = (u2 / 2.0) / pullTime;
        std::cout << "speed:" << speed << std::endl;
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

        //Add Tanh Damage Model
        int degType = 1;
        sim.addHyperbolicTangentDamage(lamC, tanhWidth, dMin, degType);
        
        //Set degradation alpha
        sim.degAlpha = alpha;

        //set minDp
        sim.minDp = minDp;

        sim.run(start_frame);
    }

    //[PYTHON] 70 Degree Shear Fracture Test (Homel 2016 configuration, Stress based damage with NH elasticity)
    if(testcase == 215){
        
        using T = double;
        static const int dim = 2;
        
        //Setup command line options
        //argv[2] = Gf
        //argv[3] = sigmaC
        //argv[4] = alpha (elasticity degradation degree)
        //argv[5] = dMin
        //argv[6] = minDp

        //Good Params to Wedge Around
        // T Gf = 22.3e-3; //from Table2 Homel2016
        // T sigmaC = 2600;
        // T alpha = 1.0;
        // T dMin = 0.25;
        // T minDp = 1.0;
        
        if (argc < 7) {
            puts("ERROR: please add parameters");
            puts("TEST 215 USAGE: ./cramp testcase Gf sigmaC alpha dMin minDp");
            exit(0);
        }

        T Gf = std::atof(argv[2]);
        T sigmaC = std::atof(argv[3]);
        T alpha = std::atof(argv[4]);
        T dMin = std::atof(argv[5]);
        T minDp = std::atof(argv[6]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 7; ++i){
            std::string cleanString = argv[i];
            if(i == 5 || i == 6){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/currentVolumeFix_HomelShearFractureTest_StressBasedDamage_NH_Gf" + cleanedStrings[0] + "_SigmaC" + cleanedStrings[1] + "_Alpha" + cleanedStrings[2] + "_dMin" + cleanedStrings[3] + "_minDp" + cleanedStrings[4];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //material (from Homel)
        T E = 1.9e11;
        T nu = 0.2647;
        T rho = 8000;

        //Params
        sim.dx = 1e-3; //1 mm
        sim.symplectic = true;
        sim.end_frame = 300;
        sim.frame_dt = 1e-6; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));

        //Sample Target's Particles
        int ppc = 4;
        T height = 100e-3; //32mm
        T width = 100e-3; //20mm
        T x1 = 0.5 - width/2.0;
        T y1 = 0.5 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T crackLength = 0.05; //50mm
        T crackRadius = sim.dx;
        if(sim.useDFG == false){
            crackRadius *= 2.0;
        }
        T crackHeight = y1 + (height / 4.0); //- (sim.dx / 2.0); 
        sim.sampleGridAlignedBoxWithNotch(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, false, Vector<T, dim>(0, 0), ppc, rho, true);

        //Sample Impactor Particles
        T distFromTarget = sim.dx*2.0;
        x1 -= (0.1 + distFromTarget);
        x2 -= (width + distFromTarget);
        T y2New = y2 - 0.075 - crackRadius;
        Vector<T,dim> minPoint2(x1, y1);
        Vector<T,dim> maxPoint2(x2, y2New);
        T impactorSpeed = 33.0;
        sim.sampleGridAlignedBox(material1, minPoint2, maxPoint2, Vector<T,dim>(impactorSpeed, 0.0), ppc, rho, false);

        //Add Boundary Conditions
        T heldMaterial = 2.0 * sim.dx;
        T yTop = y2 - heldMaterial;
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yTop), Vector<T, dim>(0, -1), Vector<T, dim>(0, 0))); //hold the top

        //Add Rankine Damage Model
        T l0 = sqrt(2 * sim.dx * sim.dx);
        int degType = 1;
        sim.addRankineDamage(dMin, Gf, l0, degType, -1.0, sigmaC); //-1 is for p which we dont want to use here
        
        //Set degradation alpha
        sim.degAlpha = alpha;

        //set minDp
        sim.minDp = minDp;

        sim.run(start_frame);
    }

    //[PYTHON] 70 Degree Shear Fracture Test (Stretch Based Damage with NH elasticity)
    if(testcase == 216){
        
        using T = double;
        static const int dim = 2;
        
        //Setup command line options
        //argv[2] = lamC
        //argv[3] = tanhWidth
        //argv[4] = alpha (elasticity degradation degree)
        //argv[5] = dMin
        //argv[6] = minDp

        //Good Params to Wedge Around
        // T lamC = 1.5; //from Table2 Homel2016
        // T tanhWidth = 0.2;
        // T alpha = 1.0;
        // T dMin = 0.25;
        // T minDp = 1.0;
        
        if (argc < 7) {
            puts("ERROR: please add parameters");
            puts("TEST 216 USAGE: ./cramp testcase lamC tanhWidth alpha dMin minDp");
            exit(0);
        }

        T lamC = std::atof(argv[2]);
        T tanhWidth = std::atof(argv[3]);
        T alpha = std::atof(argv[4]);
        T dMin = std::atof(argv[5]);
        T minDp = std::atof(argv[6]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 7; ++i){
            std::string cleanString = argv[i];
            if(i == 3 || i == 5 || i == 6){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/currentVolumeFix_HomelShearFractureTest_StretchBasedDamage_NH_lamC" + cleanedStrings[0] + "_tanhWidth" + cleanedStrings[1] + "_Alpha" + cleanedStrings[2] + "_dMin" + cleanedStrings[3] + "_minDp" + cleanedStrings[4];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //material (from Homel)
        T E = 1.9e11;
        T nu = 0.2647;
        T rho = 8000;

        //Params
        sim.dx = 1e-3; //1 mm
        sim.symplectic = true;
        sim.end_frame = 300;
        sim.frame_dt = 1e-6; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));

        //Sample Target's Particles
        int ppc = 4;
        T height = 100e-3; //32mm
        T width = 100e-3; //20mm
        T x1 = 0.5 - width/2.0;
        T y1 = 0.5 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T crackLength = 0.05; //50mm
        T crackRadius = sim.dx;
        if(sim.useDFG == false){
            crackRadius *= 2.0;
        }
        T crackHeight = y1 + (height / 4.0); //- (sim.dx / 2.0); 
        sim.sampleGridAlignedBoxWithNotch(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, false, Vector<T, dim>(0, 0), ppc, rho, true);

        //Sample Impactor Particles
        T distFromTarget = sim.dx*2.0;
        x1 -= (0.1 + distFromTarget);
        x2 -= (width + distFromTarget);
        T y2New = y2 - 0.075 - crackRadius;
        Vector<T,dim> minPoint2(x1, y1);
        Vector<T,dim> maxPoint2(x2, y2New);
        T impactorSpeed = 33.0;
        sim.sampleGridAlignedBox(material1, minPoint2, maxPoint2, Vector<T,dim>(impactorSpeed, 0.0), ppc, rho, false);

        //Add Boundary Conditions
        T heldMaterial = 2.0 * sim.dx;
        T yTop = y2 - heldMaterial;
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yTop), Vector<T, dim>(0, -1), Vector<T, dim>(0, 0))); //hold the top

        //Add Tanh Damage Model
        int degType = 1;
        sim.addHyperbolicTangentDamage(lamC, tanhWidth, dMin, degType);
        
        //Set degradation alpha
        sim.degAlpha = alpha;

        //set minDp
        sim.minDp = minDp;

        sim.run(start_frame);
    }

    //[PYTHON] Numerical Fracture exploration -- FCR, SENT, Displacement BCs, no Damage, variable dx
    if(testcase == 217){
        
        using T = double;
        static const int dim = 2;
        
        //Setup command line options
        //argv[2] = dx
        //argv[3] = ppc
        
        if (argc < 4) {
            puts("ERROR: please add parameters");
            puts("TEST 217 USAGE: ./cramp testcase dx ppc");
            exit(0);
        }

        T resolution = std::atof(argv[2]);
        int particlesPerCell = std::atoi(argv[3]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 4; ++i){
            std::string cleanString = argv[i];
            if(i == 2){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/SENT_DisplacementBCs_noDamage_FCR_dx" + cleanedStrings[0] + "_ppc" + cleanedStrings[1];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = resolution; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 300;
        sim.frame_dt = 1e-3; //500 frames at 1e-3 is 0.5s
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));

        //Sample Particles
        int ppc = particlesPerCell;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T crackLength = 5e-3;
        T crackRadius = sim.dx;
        if(sim.useDFG == false){
            crackRadius *= 2.0;
        }
        T crackHeight = y1 + (height / 2.0); //- (sim.dx / 2.0); 
        //sim.sampleGridAlignedBox(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho);
        sim.sampleGridAlignedBoxWithNotch(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, false, Vector<T, dim>(0, 0), ppc, rho, true);

        //Add Boundary Conditions
        bool singlePuller = false;
        T heldMaterial = 2.0 * sim.dx;
        T yTop = y2 - heldMaterial;
        T yBottom = y1 + heldMaterial;
        T u2 = height*0.5; // pull a total displacement of 0.2 mm, so each puller will pull half this distance
        T pullTime = 0.3; //in seconds
        T speed = (u2 / 2.0) / pullTime;
        std::cout << "speed:" << speed << std::endl;
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

    //Fluid-Solid Coupling Test
    if(testcase == 218){
        
        using T = double;
        static const int dim = 2;
        std::string path = "output/SolidFluidCouplingDemo_AddSolidCube_withViscosity4e-1";
        MPM::CRAMPSimulator<T, dim> sim(path);

        //water material
        T bulk = 1e4;
        T gamma = 7;
        T rho = 1000; //density of water

        //solid material
        T E = 2.6e6;
        T nu = 0.25;
        T rho2 = 900; //make it float

        //Params
        sim.dx = 1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 240;
        sim.frame_dt = 1.0/60.0; //500 frames at 1e-3 is 0.5s
        sim.gravity = -9.81;

        //Interpolation Scheme
        sim.useAPIC = true;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = false;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        // sim.cfl = 0.4;
        // T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 1e-5;

        // Using `new` to avoid redundant copy constructor
        auto material = sim.create_elasticity(new MPM::ViscousEquationOfStateOp<T, dim>(bulk, gamma, 4e-1)); //K = 1e7 from glacier, gamma = 7 always for water
        auto material2 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));
        
        //Sample Fluid Particles
        int ppc = 4;
        T height = 30e-3; //32mm
        T width = 30e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2); 
        //sim.sampleRandomCube(material, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho, false);
        sim.sampleGridAlignedBoxWithPoissonDisk(material, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho, false, 4); //marker = 4 for fluids, helps with analysis under the hood

        //Sample solid cube
        T h = 10e-3;
        T w = 10e-3;
        x1 = 0.05 - w/2.0;
        y1 = 0.075 - h/2.0;
        x2 = x1 + w;
        y2 = y1 + h;
        Vector<T,dim> minPoint2(x1, y1);
        Vector<T,dim> maxPoint2(x2, y2);
        sim.sampleGridAlignedBox(material2, minPoint2, maxPoint2, Vector<T, dim>(0, 0), ppc, rho2, false);

        //Add Boundary Conditions
        T boundLength = width*2.0;
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0.05 - (boundLength/2.0), 0), Vector<T, dim>(1, 0), Vector<T, dim>(0, 0), 0)); //left
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0.05 + (boundLength/2.0), 0), Vector<T, dim>(-1, 0), Vector<T, dim>(0, 0), 0)); //right
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, 0.05 + (boundLength/2.0) + 0.02), Vector<T, dim>(0, -1), Vector<T, dim>(0, 0), 0)); //top
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, 0.05 - (boundLength/2.0)), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //bottom

        sim.run(start_frame);
    }

    //[PYTHON] 70 Degree LARGER Shear Fracture Test (Stretch Based Damage with NH elasticity)
    if(testcase == 219){
        
        using T = double;
        static const int dim = 2;
        
        //Setup command line options
        //argv[2] = lamC
        //argv[3] = tanhWidth
        //argv[4] = alpha (elasticity degradation degree)
        //argv[5] = dMin
        //argv[6] = minDp

        //Good Params to Wedge Around
        // T lamC = 1.5; //from Table2 Homel2016
        // T tanhWidth = 0.2;
        // T alpha = 1.0;
        // T dMin = 0.25;
        // T minDp = 1.0;
        
        if (argc < 7) {
            puts("ERROR: please add parameters");
            puts("TEST 219 USAGE: ./cramp testcase lamC tanhWidth alpha dMin minDp");
            exit(0);
        }

        T lamC = std::atof(argv[2]);
        T tanhWidth = std::atof(argv[3]);
        T alpha = std::atof(argv[4]);
        T dMin = std::atof(argv[5]);
        T minDp = std::atof(argv[6]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 7; ++i){
            std::string cleanString = argv[i];
            if(i == 3 || i == 5 || i == 6){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/currentVolumeFix_HomelLARGERShearFractureTest_StretchBasedDamage_NH_lamC" + cleanedStrings[0] + "_tanhWidth" + cleanedStrings[1] + "_Alpha" + cleanedStrings[2] + "_dMin" + cleanedStrings[3] + "_minDp" + cleanedStrings[4];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //material (from Homel)
        T E = 1.9e11;
        T nu = 0.2647;
        T rho = 8000;

        //Params
        sim.dx = 1e-3; //1 mm
        sim.symplectic = true;
        sim.end_frame = 300;
        sim.frame_dt = 1e-6; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));

        //Sample Target's Particles
        int ppc = 4;
        T height = 100e-3; //32mm
        T width = 100e-3; //20mm
        T x1 = 0.5 - width/2.0;
        T y1 = 0.5 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T crackLength = 0.05; //50mm
        T crackRadius = sim.dx;
        if(sim.useDFG == false){
            crackRadius *= 2.0;
        }
        T crackHeight = y1 + (height / 4.0); //- (sim.dx / 2.0); 
        sim.sampleGridAlignedBoxWithNotch(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, false, Vector<T, dim>(0, 0), ppc, rho, true);

        //Sample Impactor Particles
        // T distFromTarget = sim.dx*2.0;
        // x1 -= (0.1 + distFromTarget);
        // x2 -= (width + distFromTarget);
        // T y2New = y2 - 0.075 - crackRadius;
        // Vector<T,dim> minPoint2(x1, y1);
        // Vector<T,dim> maxPoint2(x2, y2New);
        // T impactorSpeed = 33.0;
        // T multiplier = 10.0;
        // sim.sampleGridAlignedBox(material1, minPoint2, maxPoint2, Vector<T,dim>(impactorSpeed*multiplier, 0.0), ppc, rho, false);

        //Add Boundary Conditions
        T heldMaterial = 2.0 * sim.dx;
        T yTop = y2 - heldMaterial;
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yTop), Vector<T, dim>(0, -1), Vector<T, dim>(0, 0))); //hold the top
        
        //Add bottom mover (move to the right)
        T yBottom = y1 + heldMaterial;
        T u1 = width*0.5;
        T duration = sim.frame_dt * sim.end_frame;
        T speed = u1 / duration;
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, yBottom), Vector<T, dim>(0, 1), Vector<T, dim>(speed, 0), duration));

        //Add Tanh Damage Model
        int degType = 1;
        sim.addHyperbolicTangentDamage(lamC, tanhWidth, dMin, degType);
        
        //Set degradation alpha
        sim.degAlpha = alpha;

        //set minDp
        sim.minDp = minDp;

        sim.run(start_frame);
    }

    //Fluid Tank with Clot in Pipe
    if(testcase == 220){
        
        using T = double;
        static const int dim = 2;
        std::string path = "output/FluidTankWithClotInPipe_CFLExplore";
        MPM::CRAMPSimulator<T, dim> sim(path);

        //water material
        T bulk = 1e4;
        T gamma = 7;
        T rho = 1000; //density of water

        //solid material
        T E = 2.6e6;
        T nu = 0.25;
        T rho2 = 900; //make it float

        //Params
        sim.dx = 1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 240;
        sim.frame_dt = 1.0/60.0; //500 frames at 1e-3 is 0.5s
        sim.gravity = -9.81;

        //Interpolation Scheme
        sim.useAPIC = true;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = false;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        sim.suggested_dt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl); //Solid CFL condition, will be overridden when particle velocity gets too big though!
        //sim.suggested_dt = 1e-2;

        // Using `new` to avoid redundant copy constructor
        auto material = sim.create_elasticity(new MPM::EquationOfStateOp<T, dim>(bulk, gamma)); //K = 1e7 from glacier, gamma = 7 always for water
        auto material2 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));
        
        //Sample Fluid Particles
        int ppc = 4;
        T minX = 0.05;
        T minY = 0.05;
        T height_f = 120e-3; //32mm
        T width_f = 40e-3; //20mm
        T x1 = minX;
        T y1 = minY;
        T x2 = x1 + width_f;
        T y2 = y1 + height_f;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2); 
        //sim.sampleRandomCube(material, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho, false);
        sim.sampleGridAlignedBoxWithPoissonDisk(material, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho, false, 4); //marker = 4 for fluids, helps with analysis under the hood

        //Sample solid clot hemisphere
        T radius = 5e-3;
        T x_s = 30e-3; //dist into pipe
        Vector<T,dim> center(minX + width_f + x_s, minY);
        sim.sampleHemispherePoissonDisk(material2, center, radius, Vector<T, dim>(0, 0), ppc, rho2, false);

        //Add Boundary Conditions
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(minX, 0), Vector<T, dim>(1, 0), Vector<T, dim>(0, 0), 0)); //left wall, SEP
        T pipeLength = 150e-3;
        T height_floor = height_f * 1.1;
        T width_floor = width_f + pipeLength;
        sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(minX - 5e-3, minY - (1.1*height_floor)), Vector<T, dim>(minX + width_floor, minY), Vector<T, 4>(0, 0, 0, 1.0))); //FLOOR, SEP
        
        //RESERVOIR
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(0, minY - height_floor), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //reservoir floor, SEP
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(minX + width_floor + width_f, 0), Vector<T, dim>(-1, 0), Vector<T, dim>(0, 0), 0)); //right wall, SEP

        //Add box boundaries now
        T pipeHeight = 10e-3;
        sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(minX + width_f, minY + pipeHeight), Vector<T, dim>(minX + width_f + pipeLength + (width_f*1.1), minY + (1.2*height_f)), Vector<T, 4>(0, 0, 0, 1.0))); //box enforcing pipe, SLIP

        //Add STICKY box to hold clot in place
        T height_holder = 1e-3;
        sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(center[0] - radius, center[1] - (height_holder/2.0)), Vector<T, dim>(center[0] + radius, center[1] + (height_holder/2.0)), Vector<T, 4>(0, 0, 0, 1.0)));
        //sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY - belowPipeAmt), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //floor, SLIP

        sim.run(start_frame);
    }

    //[PYTHON] LARGER stretch SENT with Displacement BCs, using Stretch-Based Damage and NeoHookean elasticity
    if(testcase == 221){
        
        using T = double;
        static const int dim = 2;
        
        //Setup command line options
        //argv[2] = lamC
        //argv[3] = tanhWidth
        //argv[4] = alpha (elasticity degradation degree)
        //argv[5] = dMin
        //argv[6] = minDp

        //Good Params to Wedge Around
        // T lamC = ???; //from Table2 Homel2016
        // T tanhWidth = ???;
        // T alpha = 1.0;
        // T dMin = 0.25;
        // T minDp = 1.0;
        
        if (argc < 7) {
            puts("ERROR: please add parameters");
            puts("TEST 221 USAGE: ./cramp testcase lamC tanhWidth alpha dMin minDp");
            exit(0);
        }

        T lamC = std::atof(argv[2]);
        T tanhWidth = std::atof(argv[3]);
        T alpha = std::atof(argv[4]);
        T dMin = std::atof(argv[5]);
        T minDp = std::atof(argv[6]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 7; ++i){
            std::string cleanString = argv[i];
            if(i == 3 || i == 5 || i == 6){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/currentVolumeFix_LARGERStretch_SENT_DisplacementBCs_StretchBasedDamage_NH_lamC" + cleanedStrings[0] + "_tanhWidth" + cleanedStrings[1] + "_Alpha" + cleanedStrings[2] + "_dMin" + cleanedStrings[3] + "_minDp" + cleanedStrings[4];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = 0.5e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 300;
        sim.frame_dt = 1e-3; //500 frames at 1e-3 is 0.5s
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 35; //5.5 good for dx = 0.5e-3 and ppc = 4, 38 for ppc = 25 (both for uniform grid distributed particles)  
        sim.useDFG = true;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));

        //Sample Particles
        int ppc = 25;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T crackLength = 5e-3;
        T crackRadius = sim.dx;
        T crackHeight = y1 + (height / 2.0); //- (sim.dx / 2.0); 
        //sim.sampleGridAlignedBoxWithNotch(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, false, Vector<T, dim>(0, 0), ppc, rho, false);
        sim.sampleGridAlignedBoxWithTriangularNotchWithPoissonDisk(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, Vector<T, dim>(0, 0), ppc, rho, true);
        //sim.sampleGridAlignedBoxWithNotchWithPoissonDisk(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, false, Vector<T, dim>(0, 0), ppc, rho, false);

        //Add Boundary Conditions
        bool singlePuller = false;
        T heldMaterial = 2.0 * sim.dx;
        T yTop = y2 - heldMaterial;
        T yBottom = y1 + heldMaterial;
        T u2 = height*0.5; // pull a total displacement of 0.2 mm, so each puller will pull half this distance
        T pullTime = 0.3; //in seconds
        T speed = (u2 / 2.0) / pullTime;
        std::cout << "speed:" << speed << std::endl;
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

        //Add Tanh Damage Model
        int degType = 1;
        sim.addHyperbolicTangentDamage(lamC, tanhWidth, dMin, degType);
        
        //Set degradation alpha
        sim.degAlpha = alpha;

        //set minDp
        sim.minDp = minDp;

        sim.run(start_frame);
    }

    //[PYTHON] Pipe w/o clot, Test for Parabolic Velocity under Viscous Fluid Model
    if(testcase == 222){
        
        using T = double;
        static const int dim = 2;

        if (argc < 5) {
            puts("ERROR: please add parameters");
            puts("TEST 222 USAGE: ./cramp testcase bulk gamma viscosity");
            exit(0);
        }

        T bulk = std::atof(argv[2]);
        T gamma = std::atof(argv[3]);
        T viscosity = std::atof(argv[4]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 5; ++i){
            std::string cleanString = argv[i];
            if(i == 4){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/HorizontalPipeFlowTest_ViscousFluid_BulkMod" + cleanedStrings[0] + "_Gamma" + cleanedStrings[1] + "_Viscosity" + cleanedStrings[2];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //water material
        // T bulk = 1e4;
        // T gamma = 7;
        T rho = 1000; //density of water

        //Params
        sim.dx = 1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 240;
        sim.frame_dt = 1.0/60.0; //500 frames at 1e-3 is 0.5s
        sim.gravity = -9.81;

        //Interpolation Scheme
        sim.useAPIC = true;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = false;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        sim.suggested_dt = 1e-5; //Solid CFL condition, will be overridden when particle velocity gets too big though!

        // Using `new` to avoid redundant copy constructor
        // T viscosity = 4e-3;
        auto material = sim.create_elasticity(new MPM::ViscousEquationOfStateOp<T, dim>(bulk, gamma, viscosity)); //K = 1e7 from glacier, gamma = 7 always for water, viscosity = ?
        
        //Sample Fluid Particles
        int ppc = 4;
        T minX = 0.05;
        T minY = 0.05;
        T height_f = 200e-3; //32mm
        T width_f = 40e-3; //20mm
        T x1 = minX;
        T y1 = minY;
        T x2 = x1 + width_f;
        T y2 = y1 + height_f;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2); 
        //sim.sampleRandomCube(material, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho, false);
        sim.sampleGridAlignedBoxWithPoissonDisk(material, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho, false, 4); //marker = 4 for fluids, helps with analysis under the hood

        //Add Boundary Conditions
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(minX, 0), Vector<T, dim>(1, 0), Vector<T, dim>(0, 0), 0)); //left wall, SEP
        T pipeLength = 150e-3;
        T height_floor = height_f * 1.1;
        T width_floor = width_f + pipeLength;
        sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(minX - 5e-3, minY - (1.1*height_floor)), Vector<T, dim>(minX + width_floor, minY), Vector<T, 4>(0, 0, 0, 1.0))); //FLOOR, SEP
        
        //RESERVOIR
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(0, minY - height_floor), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //reservoir floor, SEP
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(minX + width_floor + width_f, 0), Vector<T, dim>(-1, 0), Vector<T, dim>(0, 0), 0)); //right wall, SEP

        //Add box boundaries now
        T pipeHeight = 5e-3;
        sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(minX + width_f, minY + pipeHeight), Vector<T, dim>(minX + width_f + pipeLength + (width_f*1.1), minY + (1.2*height_f)), Vector<T, 4>(0, 0, 0, 1.0))); //box enforcing pipe, SLIP

        sim.run(start_frame);
    }

    //[PYTHON] Vertical Pipe w/o clot, Test for Parabolic Velocity under Viscous Fluid Model
    if(testcase == 223){
        
        using T = double;
        static const int dim = 2;

        if (argc < 5) {
            puts("ERROR: please add parameters");
            puts("TEST 223 USAGE: ./cramp testcase bulk gamma viscosity");
            exit(0);
        }

        T bulk = std::atof(argv[2]);
        T gamma = std::atof(argv[3]);
        T viscosity = std::atof(argv[4]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 5; ++i){
            std::string cleanString = argv[i];
            if(i == 4){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/VerticalPipeFlowTest_ViscousFluid_BulkMod" + cleanedStrings[0] + "_Gamma" + cleanedStrings[1] + "_Viscosity" + cleanedStrings[2];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //water material
        // T bulk = 1e4;
        // T gamma = 7;
        T rho = 1000; //density of water

        //Params
        sim.dx = 1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 240;
        sim.frame_dt = 1.0/60.0; //500 frames at 1e-3 is 0.5s
        sim.gravity = -9.81;

        //Interpolation Scheme
        sim.useAPIC = true;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = false;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        sim.suggested_dt = 1e-5; //Solid CFL condition, will be overridden when particle velocity gets too big though!

        // Using `new` to avoid redundant copy constructor
        // T viscosity = 4e-3;
        auto material = sim.create_elasticity(new MPM::ViscousEquationOfStateOp<T, dim>(bulk, gamma, viscosity)); //K = 1e7 from glacier, gamma = 7 always for water, viscosity = ?
        
        //Sample Fluid Particles
        int ppc = 4;
        T minX = 0.05;
        T minY = 0.05;
        T height_f = 200e-3; //32mm
        T width_f = 40e-3; //20mm
        T x1 = minX;
        T y1 = minY;
        T x2 = x1 + width_f;
        T y2 = y1 + height_f;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2); 
        //sim.sampleRandomCube(material, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho, false);
        sim.sampleGridAlignedBoxWithPoissonDisk(material, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho, false, 4); //marker = 4 for fluids, helps with analysis under the hood

        //Add Boundary Conditions
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(minX, 0), Vector<T, dim>(1, 0), Vector<T, dim>(0, 0), 0)); //left wall, SEP
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(minX + width_f, 0), Vector<T, dim>(-1, 0), Vector<T, dim>(0, 0), 0)); //right wall, SEP

        //Add boxes for left andd right side of pipe
        T pipeLength = 150e-3;
        T pipeWidth = 5e-3;
        sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(minX - 5e-3, minY - pipeLength), Vector<T, dim>(minX + (0.5*width_f) - (pipeWidth*0.5), minY), Vector<T, 4>(0, 0, 0, 1.0))); //PIPE WALL LEFT, SEP
        sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(minX + (0.5*width_f) + (pipeWidth*0.5), minY - pipeLength), Vector<T, dim>(minX + width_f + 5e-3, minY), Vector<T, 4>(0, 0, 0, 1.0))); //PIPE WALL RIGHT, SEP
        
        //Cap for the pipe
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SEPARATE, Vector<T, dim>(0, minY - pipeLength - height_f), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //bottom wall, SEP

        sim.run(start_frame);
    }

    //[PYTHON] Horizontal Pipe Flow with Elastic Pipe Walls -- Test for Parabolic Velocity under Viscous Fluid Model
    if(testcase == 224){
        
        using T = double;
        static const int dim = 2;

        if (argc < 5) {
            puts("ERROR: please add parameters");
            puts("TEST 224 USAGE: ./cramp testcase bulk gamma viscosity");
            exit(0);
        }

        T bulk = std::atof(argv[2]);
        T gamma = std::atof(argv[3]);
        T viscosity = std::atof(argv[4]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 5; ++i){
            std::string cleanString = argv[i];
            if(i == 4){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/HorizontalPipeFlow_DeformablePipeWalls_ViscousFluid_wDFG_BulkMod" + cleanedStrings[0] + "_Gamma" + cleanedStrings[1] + "_Viscosity" + cleanedStrings[2];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //water material
        // T bulk = 1e4;
        // T gamma = 7;
        T rhoFluid = 1000; //density of water

        //Params
        sim.dx = 1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 360;
        sim.frame_dt = 1.0/60.0; //500 frames at 1e-3 is 0.5s
        sim.gravity = -9.81;

        //Interpolation Scheme
        sim.useAPIC = true;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 0; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0.3; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Solid Material properties (soft artery walls)
        T E = 1e6;
        T nu = 0.2;
        T rhoSolid = 1300;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        sim.suggested_dt = sim.suggestedDt(E, nu, rhoSolid, sim.dx, sim.cfl); //Solid CFL condition, will be overridden when particle velocity gets too big though!        

        auto material = sim.create_elasticity(new MPM::ViscousEquationOfStateOp<T, dim>(bulk, gamma, viscosity)); //K = 1e7 from glacier, gamma = 7 always for water, viscosity = ?
        auto material2 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));
        
        //Sample Fluid Particles
        int ppc = 4;
        T minX = 0.05;
        T minY = 0.05;
        T height_f = 200e-3; //32mm
        T width_f = 40e-3; //20mm
        T x1 = minX;
        T y1 = minY + sim.dx;
        T x2 = x1 + width_f;
        T y2 = y1 + height_f;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2); 
        sim.sampleGridAlignedBoxWithPoissonDisk(material, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rhoFluid, false, 4); //marker = 4 for fluids, helps with analysis under the hood

        //Add solid arterial walls
        T wallWidth = sim.dx * 2;
        T heldMaterial = wallWidth;
        T pipeLength = 150e-3;
        T pipeWidth = sim.dx * 10;
        sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX - 3e-3, minY - wallWidth - heldMaterial), Vector<T,dim>(minX + width_f + pipeLength, minY), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Bottom Arterial Wall
        sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX + width_f + sim.dx, minY + pipeWidth), Vector<T,dim>(minX + width_f + pipeLength + width_f + 3e-3, minY + pipeWidth + wallWidth + heldMaterial), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Top Arterial Wall

        //Add Boundary Conditions
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX, 0), Vector<T, dim>(1, 0), Vector<T, dim>(0, 0), 0)); //left wall, SEP
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + pipeLength + width_f, 0), Vector<T, dim>(-1, 0), Vector<T, dim>(0, 0), 0)); //right wall, SEP
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY - wallWidth - height_f), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //bottom wall, SEP

        //Add boxes to hold the free ends of the arterial walls
        T boxWidth = wallWidth;
        sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T,dim>(minX + width_f, minY + pipeWidth), Vector<T, dim>(minX + width_f + boxWidth, minY + pipeWidth + height_f + 5e-3), Vector<T, 4>(0, 0, 0, 1.0))); //TOP BOX - HOLDING TOP ARTERY WALL
        sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + pipeLength - boxWidth, minY - pipeWidth - height_f - 5e-3), Vector<T,dim>(minX + width_f + pipeLength, minY), Vector<T, 4>(0, 0, 0, 1.0))); //BOTTOM BOX, HOLDING BTTOM ARTERY WALL   

        //Add boxes to hold the arterial walls in place
        sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T,dim>(minX + width_f + (1.1*boxWidth), minY + pipeWidth + wallWidth), Vector<T,dim>(minX + width_f + pipeLength + width_f + 5e-3, minY + pipeWidth + wallWidth + heldMaterial), Vector<T, 4>(0, 0, 0, 1.0))); //TOP BOX - HOLDING TOP ARTERY WALL
        sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T,dim>(minX - 3e-3, minY - wallWidth - heldMaterial), Vector<T,dim>(minX + width_f + pipeLength - (1.1*boxWidth), minY - wallWidth), Vector<T, 4>(0, 0, 0, 1.0))); //BOTTOM BOX, HOLDING BTTOM ARTERY WALL       

        sim.run(start_frame);
    }

    //[PYTHON] Constant Pressure Horizontal Pipe Flow with Elastic Pipe Walls and no Gravity -- Test for Parabolic Velocity under Viscous Fluid Model
    if(testcase == 225){
        
        using T = double;
        static const int dim = 2;

        if (argc < 6) {
            puts("ERROR: please add parameters");
            puts("TEST 225 USAGE: ./cramp testcase bulk gamma viscosity massRatio");
            exit(0);
        }

        T bulk = std::atof(argv[2]);
        T gamma = std::atof(argv[3]);
        T viscosity = std::atof(argv[4]);
        T massRatio = std::atof(argv[5]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 6; ++i){
            std::string cleanString = argv[i];
            if(i == 4){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/225_DFGfriction0.9_pressureDrop_Duration4s_ConstantPressureHorizontalPipeFlow_DeformablePipeWalls_ViscousFluid_BulkMod" + cleanedStrings[0] + "_Gamma" + cleanedStrings[1] + "_Viscosity" + cleanedStrings[2] + "_massRatio" + cleanedStrings[3];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //water material
        // T bulk = 1e4;
        // T gamma = 7;
        T rhoFluid = 1000; //density of water

        //Params
        sim.dx = 1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 360;
        sim.frame_dt = 1.0/60.0; //500 frames at 1e-3 is 0.5s
        sim.gravity = 0.0;

        //Interpolation Scheme
        sim.useAPIC = true;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 0; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0.9;
        sim.useExplicitContact = true;
        sim.massRatio = massRatio; //set massRatio using user param
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Solid Material properties (soft artery walls)
        T E = 1e6;
        T nu = 0.2;
        T rhoSolid = 1300;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        sim.suggested_dt = sim.suggestedDt(E, nu, rhoSolid, sim.dx, sim.cfl); //Solid CFL condition, will be overridden when particle velocity gets too big though!        

        auto material = sim.create_elasticity(new MPM::ViscousEquationOfStateOp<T, dim>(bulk, gamma, viscosity)); //K = 1e7 from glacier, gamma = 7 always for water, viscosity = ?
        auto material2 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));
        
        //-----PARTICLE SAMPLING-----

        //Sample Fluid Particles
        int ppc = 4;
        T minX = 0.05;
        T minY = 0.05;
        T height_f = 90e-3; //32mm
        T width_f = 90e-3; //20mm
        T x1 = minX + sim.dx;
        T y1 = minY + sim.dx;
        T x2 = x1 + width_f;
        T y2 = y1 + height_f;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2); 
        sim.sampleGridAlignedBoxWithPoissonDisk(material, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rhoFluid, false, 4); //marker = 4 for fluids, helps with analysis under the hood

        //Add solid arterial walls
        T wallWidth = sim.dx * 4.0;
        T heldMaterial = sim.dx * 2.0;
        T pipeLength = 300e-3;
        T pipeWidth = sim.dx * 10;
        sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX + width_f + (2.0*sim.dx), minY + (0.5*height_f) + (0.5*pipeWidth)), Vector<T,dim>(minX + width_f + (2.0*sim.dx) + pipeLength, minY + (0.5*height_f) + (0.5*pipeWidth) + wallWidth), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Bottom Arterial Wall
        sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX + width_f + (2.0*sim.dx), minY + (0.5*height_f) - (0.5*pipeWidth) - wallWidth), Vector<T,dim>(minX + width_f + (2.0*sim.dx) + pipeLength, minY + (0.5*height_f) - (0.5*pipeWidth)), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Top Arterial Wall

        //-----BOUNDARY CONDITIONS-----

        //Add Static Half Spaces
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY + height_f + (2.0*sim.dx)), Vector<T, dim>(0, -1), Vector<T, dim>(0, 0), 0)); //top wall
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + pipeLength + width_f + (4.0*sim.dx), 0), Vector<T, dim>(-1, 0), Vector<T, dim>(0, 0), 0)); //right wall
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //bottom wall
        
        //Piston Wall
        T dist = width_f + (2.*sim.dx); //distance to compress in one second
        T duration = 4;
        T speed = dist / duration;
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX, 0), Vector<T, dim>(1, 0), Vector<T, dim>(speed, 0), duration)); //left side piston wall

        //Add boxes to hold the free ends of the arterial walls
        T boxHeight = height_f * 1.5; //just make sure this overshoots so we fully contain the fluid
        sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T,dim>(minX + width_f + (2.0*sim.dx) - sim.dx, minY + (0.5*height_f) + (0.5*pipeWidth) + heldMaterial), Vector<T, dim>(minX + width_f + (2.0*sim.dx) + pipeLength + sim.dx,minY + (0.5*height_f) + (0.5*pipeWidth) + heldMaterial + boxHeight), Vector<T, 4>(0, 0, 0, 1.0))); //TOP BOX - HOLDING TOP ARTERY WALL
        sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + (2.0*sim.dx) - sim.dx, minY + (0.5*height_f) - (0.5*pipeWidth) - heldMaterial - boxHeight), Vector<T,dim>(minX + width_f + (2.0*sim.dx) + pipeLength + sim.dx, minY + (0.5*height_f) - (0.5*pipeWidth) - heldMaterial), Vector<T, 4>(0, 0, 0, 1.0))); //BOTTOM BOX, HOLDING BTTOM ARTERY WALL         

        sim.run(start_frame);
    }

    //[PYTHON] With Clot Inclusion - Constant Pressure Horizontal Pipe Flow with Elastic Pipe Walls and no Gravity -- Test for Parabolic Velocity under Viscous Fluid Model
    if(testcase == 226){
        
        using T = double;
        static const int dim = 2;

        if (argc < 10) {
            puts("ERROR: please add parameters");
            puts("TEST 226 USAGE: ./cramp testcase bulk gamma viscosity lamC tanhWidth alpha dMin minDp");
            exit(0);
        }

        T bulk = std::atof(argv[2]);
        T gamma = std::atof(argv[3]);
        T viscosity = std::atof(argv[4]);
        T lamC = std::atof(argv[5]);
        T tanhWidth = std::atof(argv[6]);
        T alpha = std::atof(argv[7]);
        T dMin = std::atof(argv[8]);
        T minDp = std::atof(argv[9]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 10; ++i){
            std::string cleanString = argv[i];
            if(i == 4 || i == 6 || i == 8 || i == 9){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/226_with2dxCrack_andTanhDamage_Duration_4s_ConstantPressureFlowWithClot_wDFG_BulkMod" + cleanedStrings[0] + "_Gamma" + cleanedStrings[1] + "_Viscosity" + cleanedStrings[2] + "_lamC" + cleanedStrings[3] + "_tanhWidth" + cleanedStrings[4] + "_Alpha" + cleanedStrings[5] + "_dMin" + cleanedStrings[6] + "_minDp" + cleanedStrings[7];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //water material
        // T bulk = 1e4;
        // T gamma = 7;
        T rhoFluid = 1000; //density of water

        //Params
        sim.dx = 0.5e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 360;
        sim.frame_dt = 1.0/60.0; //500 frames at 1e-3 is 0.5s
        sim.gravity = 0.0;

        //Interpolation Scheme
        sim.useAPIC = true;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 4.7; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0.3; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Solid Material properties (soft artery walls)
        T E = 1e6;
        T nu = 0.2;
        T rhoSolid = 1300;
        
        //solid material (fibrin clot)
        T E2 = 2.6e6;
        T nu2 = 0.25;
        T rhoSolid2 = 1200; //make it float

        //Compute time step for symplectic
        sim.cfl = 0.4;
        T t1 = sim.suggestedDt(E, nu, rhoSolid, sim.dx, sim.cfl);
        T t2 = sim.suggestedDt(E2, nu2, rhoSolid2, sim.dx, sim.cfl);
        sim.suggested_dt = std::min(t1, t2); //Solid CFL condition, will be overridden when particle velocity gets too big though!     

        auto material = sim.create_elasticity(new MPM::ViscousEquationOfStateOp<T, dim>(bulk, gamma, viscosity)); //K = 1e7 from glacier, gamma = 7 always for water, viscosity = ?
        auto material2 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));
        auto material3 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E2, nu2));
        
        //-----PARTICLE SAMPLING-----

        //Sample Fluid Particles
        int ppc = 4;
        T minX = 0.05;
        T minY = 0.05;
        T height_f = 90e-3; //32mm
        T width_f = 90e-3; //20mm
        T x1 = minX + sim.dx;
        T y1 = minY + sim.dx;
        T x2 = x1 + width_f;
        T y2 = y1 + height_f;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2); 
        sim.sampleGridAlignedBoxWithPoissonDisk(material, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rhoFluid, false, 4); //marker = 4 for fluids, helps with analysis under the hood

        //Add solid arterial walls
        T wallWidth = sim.dx * 4.0;
        T heldMaterial = sim.dx * 2.0;
        T pipeLength = 300e-3;
        T pipeWidth = 0.01; //keep this part constant despite dx
        sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX + width_f + (2.0*sim.dx), minY + (0.5*height_f) + (0.5*pipeWidth)), Vector<T,dim>(minX + width_f + (2.0*sim.dx) + pipeLength, minY + (0.5*height_f) + (0.5*pipeWidth) + wallWidth), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Bottom Arterial Wall
        sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX + width_f + (2.0*sim.dx), minY + (0.5*height_f) - (0.5*pipeWidth) - wallWidth), Vector<T,dim>(minX + width_f + (2.0*sim.dx) + pipeLength, minY + (0.5*height_f) - (0.5*pipeWidth)), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Top Arterial Wall

        //Add fibrin clot
        T radius = pipeWidth * 0.5;
        T x_s = pipeWidth * 10.0; //dist into pipe
        Vector<T,dim> center(minX + width_f + (sim.dx*2.0) + x_s, minY + (height_f * 0.5) - (pipeWidth*0.5));
        Vector<T, dim> notchMin(center[0] - radius, center[1] + sim.dx);
        Vector<T, dim> notchMax(center[0] - radius*0.5, center[1] + sim.dx*3.0);
        bool damageRegion = false; //toggle this to switch between damage region and material discontinuity
        sim.sampleHemispherePoissonDisk_WithNotch(material3, center, radius, notchMin, notchMax, damageRegion, Vector<T, dim>(0, 0), ppc, rhoSolid2, true, 0);

        //Add elastodamage coupling
        sim.elasticityDegradationType = 1;
        sim.computeLamMaxFlag = true;

        //-----BOUNDARY CONDITIONS-----

        //Add Static Half Spaces
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY + height_f + (2.0*sim.dx)), Vector<T, dim>(0, -1), Vector<T, dim>(0, 0), 0)); //top wall
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + pipeLength + width_f + (4.0*sim.dx), 0), Vector<T, dim>(-1, 0), Vector<T, dim>(0, 0), 0)); //right wall
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //bottom wall
        
        //Piston Wall
        T dist = width_f + (2.*sim.dx); //distance to compress in one second
        T duration = 4;
        T speed = dist / duration;
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX, 0), Vector<T, dim>(1, 0), Vector<T, dim>(speed, 0), duration)); //left side piston wall

        //Add boxes to hold the free ends of the arterial walls
        T boxHeight = height_f * 1.5; //just make sure this overshoots so we fully contain the fluid
        sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T,dim>(minX + width_f + (2.0*sim.dx) - sim.dx, minY + (0.5*height_f) + (0.5*pipeWidth) + heldMaterial), Vector<T, dim>(minX + width_f + (2.0*sim.dx) + pipeLength + sim.dx,minY + (0.5*height_f) + (0.5*pipeWidth) + heldMaterial + boxHeight), Vector<T, 4>(0, 0, 0, 1.0))); //TOP BOX - HOLDING TOP ARTERY WALL
        sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + (2.0*sim.dx) - sim.dx, minY + (0.5*height_f) - (0.5*pipeWidth) - heldMaterial - boxHeight), Vector<T,dim>(minX + width_f + (2.0*sim.dx) + pipeLength + sim.dx, minY + (0.5*height_f) - (0.5*pipeWidth) - heldMaterial), Vector<T, 4>(0, 0, 0, 1.0))); //BOTTOM BOX, HOLDING BTTOM ARTERY WALL         

        //Add Tanh Damage Model
        int degType = 1;
        sim.addHyperbolicTangentDamage(lamC, tanhWidth, dMin, degType);
        
        //Set degradation alpha
        sim.degAlpha = alpha;

        //set minDp
        sim.minDp = minDp;

        sim.run(start_frame);
    }
    
    // [PYTHON] 2D Fluid Generator Test
    if(testcase == 227){
        
        using T = double;
        static const int dim = 2;

        if (argc < 5) {
            puts("ERROR: please add parameters");
            puts("TEST 227 USAGE: ./cramp testcase bulk gamma viscosity");
            exit(0);
        }

        T bulk = std::atof(argv[2]);
        T gamma = std::atof(argv[3]);
        T viscosity = std::atof(argv[4]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 5; ++i){
            std::string cleanString = argv[i];
            if(i == 4){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/227_withDFG_FluidGeneratorTest" + cleanedStrings[0] + "_Gamma" + cleanedStrings[1] + "_Viscosity" + cleanedStrings[2];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //water material
        T rhoFluid = 1000; //density of water

        //solid mats
        T E = 1e4;
        T nu = 0.25;
        T rhoSolid = 700;

        //Params
        sim.dx = 0.5e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 180;
        sim.frame_dt = 1.0/60.0; //500 frames at 1e-3 is 0.5s
        sim.gravity = -9.8;

        //Interpolation Scheme
        sim.useAPIC = true;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 0; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0.3; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;

        //time step for symplectic
        sim.cfl = 0.4;
        T t1 = sim.suggestedDt(E, nu, rhoSolid, sim.dx, sim.cfl);
        T tFluid = 1e-5; //works well
        sim.suggested_dt = std::min(t1, tFluid); 

        auto material = sim.create_elasticity(new MPM::ViscousEquationOfStateOp<T, dim>(bulk, gamma, viscosity)); //K = 1e7 from glacier, gamma = 7 always for water, viscosity = ?
        auto materialSolid = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));

        //Box Dimensions
        T minX = 0.05;
        T minY = 0.05;
        T length = 100 * sim.dx;
        T height = 100 * sim.dx;
        
        //Fluid Generator
        Vector<T,dim> center(minX + (length/2.0), minY + (height/2.0));
        T radius = sim.dx * 10;
        Vector<T, dim> velocity(0.3, 0.0);
        int ppc = 9;
        T source_dt = 1/60.0;
        bool parabolic = false;
        sim.addFluidSource(material, center, radius, velocity, rhoFluid, ppc, source_dt, parabolic);

        Vector<T,2> timing(0.0, 0.3); //run fluid source from t = 0 to t = 2s
        std::vector<Vector<T,2>> timings;
        timings.push_back(timing);
        sim.addFluidSourceTiming(timings);

        //Solid Box
        T solidLength = 10 * sim.dx;
        T solidHeight = solidLength;
        T centerX = center[0];
        T centerY = center[1] - 25*sim.dx;
        Vector<T, dim> minCorner(centerX - (solidLength/2.0), centerY - (solidHeight/2.0));
        Vector<T, dim> maxCorner(centerX + (solidLength/2.0), centerY + (solidHeight/2.0));
        sim.sampleGridAlignedBox(materialSolid, minCorner, maxCorner, Vector<T, dim>(0, 0), ppc, rhoSolid, true, 0);

        //Add elastodamage coupling
        sim.elasticityDegradationType = 1;
        sim.computeLamMaxFlag = true;

        //-----BOUNDARY CONDITIONS-----

        //Add Static Half Spaces
        
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY + height), Vector<T, dim>(0, -1), Vector<T, dim>(0, 0), 0)); //top wall
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + length, 0), Vector<T, dim>(-1, 0), Vector<T, dim>(0, 0), 0)); //right wall
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //bottom wall
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX, 0), Vector<T, dim>(1, 0), Vector<T, dim>(0, 0), 0)); //left wall

        sim.run(start_frame);
    }

    //[PYTHON] Fluid Generator With Clot Inclusion - Constant Velocity Source with Elastic Pipe Walls and no Gravity
    if(testcase == 228){
        
        using T = double;
        static const int dim = 2;

        if (argc < 10) {
            puts("ERROR: please add parameters");
            puts("TEST 228 USAGE: ./cramp testcase bulk gamma viscosity lamC tanhWidth alpha dMin minDp");
            exit(0);
        }

        T bulk = std::atof(argv[2]);
        T gamma = std::atof(argv[3]);
        T viscosity = std::atof(argv[4]);
        //T lamC = std::atof(argv[5]);
        //T tanhWidth = std::atof(argv[6]);
        T alpha = std::atof(argv[7]);
        //T dMin = std::atof(argv[8]);
        T minDp = std::atof(argv[9]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 10; ++i){
            std::string cleanString = argv[i];
            if(i == 4 || i == 6 || i == 8 || i == 9){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/228_FluidGeneratorPipeWithClot_SolidClot_noDamage_Duration_4s_BulkMod" + cleanedStrings[0] + "_Gamma" + cleanedStrings[1] + "_Viscosity" + cleanedStrings[2] + "_lamC" + cleanedStrings[3] + "_tanhWidth" + cleanedStrings[4] + "_Alpha" + cleanedStrings[5] + "_dMin" + cleanedStrings[6] + "_minDp" + cleanedStrings[7];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //water material
        T rhoFluid = 1000; //density of water

        //Params
        sim.dx = 1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 360;
        sim.frame_dt = 1.0/60.0; //500 frames at 1e-3 is 0.5s
        sim.gravity = 0.0;

        //Interpolation Scheme
        sim.useAPIC = true;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 4.7; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0.3; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Solid Material properties (soft artery walls)
        T E = 1e6;
        T nu = 0.2;
        T rhoSolid = 1300;
        
        //solid material (fibrin clot)
        T E2 = 2.6e6;
        T nu2 = 0.25;
        T rhoSolid2 = 1200; //make it float

        //Compute time step for symplectic
        sim.cfl = 0.4;
        T t1 = sim.suggestedDt(E, nu, rhoSolid, sim.dx, sim.cfl);
        T t2 = sim.suggestedDt(E2, nu2, rhoSolid2, sim.dx, sim.cfl);
        sim.suggested_dt = std::min(t1, t2); //Solid CFL condition, will be overridden when particle velocity gets too big though!     

        auto material = sim.create_elasticity(new MPM::ViscousEquationOfStateOp<T, dim>(bulk, gamma, viscosity)); //K = 1e7 from glacier, gamma = 7 always for water, viscosity = ?
        auto material2 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));
        auto material3 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E2, nu2));
        
        //-----PARTICLE SAMPLING-----

        //Sample Fluid Particles
        int ppc = 4;
        T minX = 0.05;
        T minY = 0.05;
        T height_f = 90e-3; //32mm
        T width_f = 90e-3; //20mm
        // T x1 = minX + sim.dx;
        // T y1 = minY + sim.dx;
        // T x2 = x1 + width_f;
        // T y2 = y1 + height_f;
        // Vector<T,dim> minPoint(x1, y1);
        // Vector<T,dim> maxPoint(x2, y2); 
        // sim.sampleGridAlignedBoxWithPoissonDisk(material, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rhoFluid, false, 4); //marker = 4 for fluids, helps with analysis under the hood               

        //Add solid arterial walls
        T wallWidth = sim.dx * 4.0;
        T heldMaterial = sim.dx * 2.0;
        T pipeLength = 300e-3;
        T pipeWidth = 0.01; //keep this part constant despite dx
        sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX + width_f + (2.0*sim.dx), minY + (0.5*height_f) + (0.5*pipeWidth)), Vector<T,dim>(minX + width_f + (2.0*sim.dx) + pipeLength, minY + (0.5*height_f) + (0.5*pipeWidth) + wallWidth), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Bottom Arterial Wall
        sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX + width_f + (2.0*sim.dx), minY + (0.5*height_f) - (0.5*pipeWidth) - wallWidth), Vector<T,dim>(minX + width_f + (2.0*sim.dx) + pipeLength, minY + (0.5*height_f) - (0.5*pipeWidth)), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Top Arterial Wall

        //Blood Generator
        T sourceRadius = (pipeWidth / 2.0) - sim.dx;
        Vector<T,dim> sourceCenter(minX + width_f - sourceRadius, minY + (0.5 * height_f));
        Vector<T, dim> velocity(0.24, 0.0);
        T source_dt = sim.frame_dt;
        bool parabolic = false;
        sim.addFluidSource(material, sourceCenter, sourceRadius, velocity, rhoFluid, ppc, source_dt, parabolic);

        Vector<T,2> timing(0.0, 6.0); //run fluid source from t = 0 to t = 2s
        std::vector<Vector<T,2>> timings;
        timings.push_back(timing);
        sim.addFluidSourceTiming(timings); 

        //Add fibrin clot
        T radius = pipeWidth * 0.5;
        T x_s = pipeWidth * 10.0; //dist into pipe
        Vector<T,dim> center(minX + width_f + (sim.dx*2.0) + x_s, minY + (height_f * 0.5) - (pipeWidth*0.5));
        Vector<T, dim> notchMin(center[0] - radius, center[1] + sim.dx);
        Vector<T, dim> notchMax(center[0] - radius*0.5, center[1] + sim.dx*3.0);
        //bool damageRegion = false; //toggle this to switch between damage region and material discontinuity
        //sim.sampleHemispherePoissonDisk_WithNotch(material3, center, radius, notchMin, notchMax, damageRegion, Vector<T, dim>(0, 0), ppc, rhoSolid2, true, 0);
        sim.sampleHemispherePoissonDisk(material3, center, radius, Vector<T, dim>(0, 0), ppc, rhoSolid2, true, 0);

        //Add elastodamage coupling
        sim.elasticityDegradationType = 1;
        sim.computeLamMaxFlag = true;

        //-----BOUNDARY CONDITIONS-----

        //Add Static Half Spaces
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY + height_f + (2.0*sim.dx)), Vector<T, dim>(0, -1), Vector<T, dim>(0, 0), 0)); //top wall
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + pipeLength + width_f + (4.0*sim.dx), 0), Vector<T, dim>(-1, 0), Vector<T, dim>(0, 0), 0)); //right wall
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //bottom wall
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX, 0), Vector<T, dim>(1, 0), Vector<T, dim>(0, 0), 0)); //left side wall

        //Add boxes to hold the free ends of the arterial walls
        T boxHeight = height_f * 1.5; //just make sure this overshoots so we fully contain the fluid
        sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T,dim>(minX + width_f + (2.0*sim.dx) - sim.dx, minY + (0.5*height_f) + (0.5*pipeWidth) + heldMaterial), Vector<T, dim>(minX + width_f + (2.0*sim.dx) + pipeLength + sim.dx,minY + (0.5*height_f) + (0.5*pipeWidth) + heldMaterial + boxHeight), Vector<T, 4>(0, 0, 0, 1.0))); //TOP BOX - HOLDING TOP ARTERY WALL
        sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + (2.0*sim.dx) - sim.dx, minY + (0.5*height_f) - (0.5*pipeWidth) - heldMaterial - boxHeight), Vector<T,dim>(minX + width_f + (2.0*sim.dx) + pipeLength + sim.dx, minY + (0.5*height_f) - (0.5*pipeWidth) - heldMaterial), Vector<T, 4>(0, 0, 0, 1.0))); //BOTTOM BOX, HOLDING BTTOM ARTERY WALL         

        //Add Tanh Damage Model
        //int degType = 1;
        //sim.addHyperbolicTangentDamage(lamC, tanhWidth, dMin, degType);
        
        //Set degradation alpha
        sim.degAlpha = alpha;

        //set minDp
        sim.minDp = minDp;

        sim.run(start_frame);
    }

    //[PYTHON] With Clot Inclusion - 40 diameters long pipe with fluid Starting in pipe -- push piston from 0 to 20 diameters over some time frame
    if(testcase == 229){
        
        using T = double;
        static const int dim = 2;

        if (argc < 10) {
            puts("ERROR: please add parameters");
            puts("TEST 229 USAGE: ./cramp testcase bulk gamma viscosity lamC tanhWidth alpha dMin minDp");
            exit(0);
        }

        T bulk = std::atof(argv[2]);
        T gamma = std::atof(argv[3]);
        T viscosity = std::atof(argv[4]);
        //T lamC = std::atof(argv[5]);
        //T tanhWidth = std::atof(argv[6]);
        T alpha = std::atof(argv[7]);
        //T dMin = std::atof(argv[8]);
        T minDp = std::atof(argv[9]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 10; ++i){
            std::string cleanString = argv[i];
            if(i == 4 || i == 6 || i == 8 || i == 9){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/229_1e-6_duration2s_0.95Friction_55DiameterPipe_wClot_d1cm_BulkMod" + cleanedStrings[0] + "_Gamma" + cleanedStrings[1] + "_Viscosity" + cleanedStrings[2] + "_lamC" + cleanedStrings[3] + "_tanhWidth" + cleanedStrings[4] + "_Alpha" + cleanedStrings[5] + "_dMin" + cleanedStrings[6] + "_minDp" + cleanedStrings[7];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //water material
        T rhoFluid = 1000; //density of water

        //Params
        sim.dx = 1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 125;
        sim.frame_dt = 1.0/60.0; //500 frames at 1e-3 is 0.5s
        sim.gravity = 0.0;

        //Interpolation Scheme
        sim.useAPIC = true;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 4.7; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0.95; //for no slip condition between solid and fluid
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Solid Material properties (soft artery walls)
        T E = 1e6;
        T nu = 0.2;
        T rhoSolid = 1300;
        
        //solid material (fibrin clot)
        T E2 = 2.6e6;
        T nu2 = 0.25;
        T rhoSolid2 = 1200; //make it float

        //Compute time step for symplectic
        sim.cfl = 0.4;
        T t1 = sim.suggestedDt(E, nu, rhoSolid, sim.dx, sim.cfl);
        T t2 = sim.suggestedDt(E2, nu2, rhoSolid2, sim.dx, sim.cfl);
        sim.suggested_dt = std::min(t1, t2); //Solid CFL condition, will be overridden when particle velocity gets too big though!     
        sim.suggested_dt = 1e-6;

        auto material = sim.create_elasticity(new MPM::ViscousEquationOfStateOp<T, dim>(bulk, gamma, viscosity)); //K = 1e7 from glacier, gamma = 7 always for water, viscosity = ?
        auto material2 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));
        auto material3 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E2, nu2));
        
        //-----PARTICLE SAMPLING-----

        //Sampling Constants
        int ppc = 4;
        T minX = 0.05;
        T minY = 0.05;

        //Add solid arterial walls
        T wallWidth = sim.dx * 4.0;
        T heldMaterial = sim.dx * 2.0;
        T pipeWidth = sim.dx * 10; //d = 0.01m for dx = 1e-3
        T pipeLength = pipeWidth * 55.0;
        sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX, minY), Vector<T,dim>(minX + pipeLength, minY + wallWidth), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Bottom Arterial Wall
        sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX, minY + pipeWidth + wallWidth), Vector<T,dim>(minX + pipeLength, minY + pipeWidth + (2.0*wallWidth)), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Top Arterial Wall

        //Add fibrin clot
        T radius = pipeWidth * 0.5;
        T x_s = pipeWidth * 40.0; //dist into pipe
        Vector<T,dim> center(minX + x_s, minY + wallWidth);
        //Vector<T, dim> notchMin(center[0] - radius, center[1] + sim.dx);
        //Vector<T, dim> notchMax(center[0] - radius*0.5, center[1] + sim.dx*3.0);
        //bool damageRegion = false; //toggle this to switch between damage region and material discontinuity
        sim.sampleHemispherePoissonDisk(material3, center, radius, Vector<T, dim>(0, 0), ppc, rhoSolid2, true, 0);

        //Add blood particles inside artery
        T height_f = pipeWidth - (sim.dx * 2.0); //32mm
        T width_f = pipeLength;
        T x1 = minX;
        T y1 = minY + wallWidth + sim.dx;
        T x2 = x1 + width_f;
        T y2 = y1 + height_f;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2); 
        T maxVelocity = 0.0;
        bool parabolicVelocity = false;
        sim.sampleGridAlignedBoxWithPoissonDisk_ClotCutOut(material, minPoint, maxPoint, center, radius, Vector<T, dim>(maxVelocity, 0), ppc, rhoFluid, false, 4, false, parabolicVelocity); //marker = 4 for fluids, helps with analysis under the hood

        //Add elastodamage coupling
        sim.elasticityDegradationType = 1;
        sim.computeLamMaxFlag = true;

        //-----BOUNDARY CONDITIONS-----

        //Add Static Half Spaces
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY + wallWidth + pipeWidth + heldMaterial), Vector<T, dim>(0, -1), Vector<T, dim>(0, 0), 0)); //top wall -- holds artery in place as well
        //sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + pipeLength + width_f + (4.0*sim.dx), 0), Vector<T, dim>(-1, 0), Vector<T, dim>(0, 0), 0)); //right wall
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY + heldMaterial), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //bottom wall - hold artery in place
        
        //Piston Box
        T dist = pipeWidth * 20.0; //total distance to push
        T duration = 2;
        T speed = dist / duration;
        T boxRadius = height_f / 2.0;
        Vector<T,dim> boxMin(minX - (2.0 * boxRadius), minY + wallWidth);
        Vector<T,dim> boxMax(minX, minY + wallWidth + pipeWidth);
        Vector<T,4> rot(0.0, 0.0, 0.0, 1.0);
        Vector<T,dim> vel(speed, 0.0);
        sim.add_boundary_condition(new Geometry::MovingBoxLevelSet<T, dim>(Geometry::STICKY, boxMin, boxMax, rot, vel, duration)); //left side piston wall -- THIS NEEDS TO BE STICKY!! All moving BCs should be implemented as STICKY

        //Add boxes to hold the free ends of the arterial walls
        //T boxHeight = height_f * 1.5; //just make sure this overshoots so we fully contain the fluid
        //sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T,dim>(minX + width_f + (2.0*sim.dx) - sim.dx, minY + (0.5*height_f) + (0.5*pipeWidth) + heldMaterial), Vector<T, dim>(minX + width_f + (2.0*sim.dx) + pipeLength + sim.dx,minY + (0.5*height_f) + (0.5*pipeWidth) + heldMaterial + boxHeight), Vector<T, 4>(0, 0, 0, 1.0))); //TOP BOX - HOLDING TOP ARTERY WALL
        //sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + (2.0*sim.dx) - sim.dx, minY + (0.5*height_f) - (0.5*pipeWidth) - heldMaterial - boxHeight), Vector<T,dim>(minX + width_f + (2.0*sim.dx) + pipeLength + sim.dx, minY + (0.5*height_f) - (0.5*pipeWidth) - heldMaterial), Vector<T, 4>(0, 0, 0, 1.0))); //BOTTOM BOX, HOLDING BTTOM ARTERY WALL         

        //Add Tanh Damage Model
        //int degType = 1;
        //sim.addHyperbolicTangentDamage(lamC, tanhWidth, dMin, degType);
        
        //Set degradation alpha
        sim.degAlpha = alpha;

        //set minDp
        sim.minDp = minDp;

        sim.run(start_frame);
    }

    //[PYTHON] Shorter Pipe With Pressure Gradient (no clot)
    if(testcase == 230){
        
        using T = double;
        static const int dim = 2;

        if (argc < 8) {
            puts("ERROR: please add parameters");
            puts("TEST 230 USAGE: ./cramp testcase bulk gamma viscosity pStart pGrad");
            exit(0);
        }

        T bulk = std::atof(argv[2]);
        T gamma = std::atof(argv[3]);
        T viscosity = std::atof(argv[4]);
        T pStart = std::atof(argv[5]);
        T pGrad = std::atof(argv[6]);
        T couplingFriction = std::atof(argv[7]);
        // T alpha = std::atof(argv[7]);
        // T dMin = std::atof(argv[8]);
        // T minDp = std::atof(argv[9]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 8; ++i){
            std::string cleanString = argv[i];
            if(i == 4 || i == 7){// || i == 6 || i == 8 || i == 9){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/230_ToroidalPressureGradient_d1cm_BulkMod" + cleanedStrings[0] + "_Gamma" + cleanedStrings[1] + "_Viscosity" + cleanedStrings[2] + "_pStart" + cleanedStrings[3] + "_pGrad" + cleanedStrings[4] + "_couplingFriction" + cleanedStrings[5];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //water material
        T rhoFluid = 1060; //density of blood

        //Params
        sim.dx = 1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 240;
        sim.frame_dt = 1.0/60.0; //500 frames at 1e-3 is 0.5s
        sim.gravity = 0.0;

        //Interpolation Scheme
        sim.useAPIC = true;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 4.7; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = couplingFriction; //for no slip condition between solid and fluid
        sim.useExplicitContact = true;
        //sim.massRatio = 15.0;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Solid Material properties (soft artery walls)
        T E = 1e6;
        T nu = 0.2;
        T rhoSolid = 1300;
        
        //solid material (fibrin clot)
        // T E2 = 2.6e6;
        // T nu2 = 0.25;
        // T rhoSolid2 = 1200; //make it float

        //Compute time step for symplectic
        sim.cfl = 0.4;
        T t1 = sim.suggestedDt(E, nu, rhoSolid, sim.dx, sim.cfl);
        // T t2 = sim.suggestedDt(E2, nu2, rhoSolid2, sim.dx, sim.cfl);
        sim.suggested_dt = std::min(t1, t1); //Solid CFL condition, will be overridden when particle velocity gets too big though!     
        //sim.suggested_dt = 1e-6;

        auto material = sim.create_elasticity(new MPM::ViscousEquationOfStateOp<T, dim>(bulk, gamma, viscosity)); //K = 1e7 from glacier, gamma = 7 always for water, viscosity = ?
        auto material2 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));
        //auto material3 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E2, nu2));
        
        //-----PARTICLE SAMPLING-----

        //Sampling Constants
        int ppc = 4;
        T minX = 0.05;
        T minY = 0.05;

        //Add solid arterial walls
        T wallWidth = sim.dx * 4.0;
        T heldMaterial = sim.dx * 2.0;
        T pipeWidth = sim.dx * 10; //d = 0.01m for dx = 1e-3
        T pipeLength = pipeWidth * 30.0;
        sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX, minY), Vector<T,dim>(minX + pipeLength, minY + wallWidth), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Bottom Arterial Wall
        sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX, minY + pipeWidth + wallWidth), Vector<T,dim>(minX + pipeLength, minY + pipeWidth + (2.0*wallWidth)), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Top Arterial Wall

        //Add fibrin clot
        // T radius = pipeWidth * 0.5;
        // T x_s = pipeWidth * 40.0; //dist into pipe
        // Vector<T,dim> center(minX + x_s, minY + wallWidth);
        // //Vector<T, dim> notchMin(center[0] - radius, center[1] + sim.dx);
        // //Vector<T, dim> notchMax(center[0] - radius*0.5, center[1] + sim.dx*3.0);
        // //bool damageRegion = false; //toggle this to switch between damage region and material discontinuity
        //sim.sampleHemispherePoissonDisk(material3, center, radius, Vector<T, dim>(0, 0), ppc, rhoSolid2, true, 0);

        //Add blood particles inside artery
        T height_f = pipeWidth - (sim.dx * 2.0); //32mm
        T width_f = pipeLength;
        T x1 = minX;
        T y1 = minY + wallWidth + sim.dx;
        T x2 = x1 + width_f;
        T y2 = y1 + height_f;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2); 
        //T maxVelocity = 0.0;
        //bool parabolicVelocity = false;
        //sim.sampleGridAlignedBoxWithPoissonDisk_ClotCutOut(material, minPoint, maxPoint, center, radius, Vector<T, dim>(maxVelocity, 0), ppc, rhoFluid, false, 4, false, parabolicVelocity); //marker = 4 for fluids, helps with analysis under the hood
        sim.sampleGridAlignedBoxWithPoissonDisk(material, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rhoFluid, false, 4, false);

        //Add elastodamage coupling
        sim.elasticityDegradationType = 1;
        sim.computeLamMaxFlag = true;

        //-----BOUNDARY CONDITIONS-----

        //Add Static Half Spaces
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY + wallWidth + pipeWidth + heldMaterial), Vector<T, dim>(0, -1), Vector<T, dim>(0, 0), 0)); //top wall -- holds artery in place as well
        //sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + pipeLength + width_f + (4.0*sim.dx), 0), Vector<T, dim>(-1, 0), Vector<T, dim>(0, 0), 0)); //right wall
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY + heldMaterial), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //bottom wall - hold artery in place
        
        //---EXTERNAL LOADING---

        Vector<T,dim> pgMin = minPoint;
        Vector<T,dim> pgMax = maxPoint;
        Vector<T,dim> toroidMin = minPoint;
        Vector<T,dim> toroidMax = maxPoint;
        pgMin[1] -= (sim.dx + heldMaterial);
        pgMax[1] += (sim.dx + heldMaterial);
        toroidMin[1] = pgMin[1];
        toroidMax[1] = pgMax[1];
        bool toroidal = true;
        //pgMin[0] -= (sim.dx * 10); //add some additional padding in front
        pgMax[0] += (sim.dx * 10); //make sure fluid will keep moving!
        //dp/dx = -u_max * (2mu / b^2)
        //want u_max = 0.15 -> dp/dx = -48 (b = 0.005, mu = 0.004)
        sim.addPressureGradient(pgMin, pgMax, pStart, pGrad, toroidMin, toroidMax, toroidal);

        //Add boxes to hold the free ends of the arterial walls
        //T boxHeight = height_f * 1.5; //just make sure this overshoots so we fully contain the fluid
        //sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T,dim>(minX + width_f + (2.0*sim.dx) - sim.dx, minY + (0.5*height_f) + (0.5*pipeWidth) + heldMaterial), Vector<T, dim>(minX + width_f + (2.0*sim.dx) + pipeLength + sim.dx,minY + (0.5*height_f) + (0.5*pipeWidth) + heldMaterial + boxHeight), Vector<T, 4>(0, 0, 0, 1.0))); //TOP BOX - HOLDING TOP ARTERY WALL
        //sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + (2.0*sim.dx) - sim.dx, minY + (0.5*height_f) - (0.5*pipeWidth) - heldMaterial - boxHeight), Vector<T,dim>(minX + width_f + (2.0*sim.dx) + pipeLength + sim.dx, minY + (0.5*height_f) - (0.5*pipeWidth) - heldMaterial), Vector<T, 4>(0, 0, 0, 1.0))); //BOTTOM BOX, HOLDING BTTOM ARTERY WALL         

        //Add Tanh Damage Model
        //int degType = 1;
        //sim.addHyperbolicTangentDamage(lamC, tanhWidth, dMin, degType);
        
        //Set degradation alpha
        //sim.degAlpha = alpha;

        //set minDp
        //sim.minDp = minDp;

        sim.run(start_frame);
    }

    //[PYTHON] Shorter Pipe With Pressure Gradient AND CLOT
    if(testcase == 231){
        
        using T = double;
        static const int dim = 2;

        if (argc < 8) {
            puts("ERROR: please add parameters");
            puts("TEST 231 USAGE: ./cramp testcase bulk gamma viscosity pStart pGrad couplingFriction ");
            exit(0);
        }

        T bulk = std::atof(argv[2]);
        T gamma = std::atof(argv[3]);
        T viscosity = std::atof(argv[4]);
        T pStart = std::atof(argv[5]);
        T pGrad = std::atof(argv[6]);
        T couplingFriction = std::atof(argv[7]);
        // T alpha = std::atof(argv[7]);
        // T dMin = std::atof(argv[8]);
        // T minDp = std::atof(argv[9]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 8; ++i){
            std::string cleanString = argv[i];
            if(i == 4 || i == 7){// || i == 6 || i == 8 || i == 9){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/231_ToroidalPressureGradientWithPoroelasticClot_c1_300k_c2_3.0_d1cm_BulkMod" + cleanedStrings[0] + "_Gamma" + cleanedStrings[1] + "_Viscosity" + cleanedStrings[2] + "_pStart" + cleanedStrings[3] + "_pGrad" + cleanedStrings[4] + "_couplingFriction" + cleanedStrings[5];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //water material
        T rhoFluid = 1060; //density of blood

        //Params
        sim.dx = 1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 240;
        sim.frame_dt = 1.0/60.0; //500 frames at 1e-3 is 0.5s
        sim.gravity = 0.0;

        //Interpolation Scheme
        sim.useAPIC = true;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 4.7; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = couplingFriction; //for no slip condition between solid and fluid
        sim.useExplicitContact = true;
        //sim.massRatio = 15.0;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Solid Material properties (soft artery walls)
        T E = 1e6;
        T nu = 0.2;
        T rhoSolid = 1300;
        
        //solid material (fibrin clot)
        T c1 = 300000;
        T c2 = 3.0;
        T phi_s0 = 0.01;
        T pi_0 = 1000.0;
        T beta_1 = 1.02;
        T rhoSolid2 = 1200;

        //Compute time step for symplectic
        sim.cfl = 0.4;
        sim.suggested_dt = sim.suggestedDt(E, nu, rhoSolid, sim.dx, sim.cfl); //Solid CFL condition, will be overridden when particle velocity gets too big though!     
        //sim.suggested_dt = 1e-6;

        auto material = sim.create_elasticity(new MPM::ViscousEquationOfStateOp<T, dim>(bulk, gamma, viscosity)); //K = 1e7 from glacier, gamma = 7 always for water, viscosity = ?
        auto material2 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));
        //auto material3 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));
        auto material3 = sim.create_elasticity(new MPM::FibrinPoroelasticityOp<T, dim>(c1, c2, phi_s0, pi_0, beta_1));
        
        //-----PARTICLE SAMPLING-----

        //Sampling Constants
        int ppc = 4;
        T minX = 0.05;
        T minY = 0.05;

        //Add solid arterial walls
        T wallWidth = sim.dx * 4.0;
        T heldMaterial = sim.dx * 2.0;
        T pipeWidth = sim.dx * 10; //d = 0.01m for dx = 1e-3
        T pipeLength = pipeWidth * 30.0;
        sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX, minY), Vector<T,dim>(minX + pipeLength, minY + wallWidth), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Bottom Arterial Wall
        sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX, minY + pipeWidth + wallWidth), Vector<T,dim>(minX + pipeLength, minY + pipeWidth + (2.0*wallWidth)), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Top Arterial Wall

        //Add fibrin clot
        T radius = pipeWidth * 0.5;
        T x_s = pipeWidth * 15.0; //dist into pipe
        Vector<T,dim> center(minX + x_s, minY + wallWidth);
        //Vector<T, dim> notchMin(center[0] - radius, center[1] + sim.dx);
        //Vector<T, dim> notchMax(center[0] - radius*0.5, center[1] + sim.dx*3.0);
        //bool damageRegion = false; //toggle this to switch between damage region and material discontinuity
        sim.sampleHemispherePoissonDisk(material3, center, radius, Vector<T, dim>(0, 0), ppc, rhoSolid2, true, 0);

        //Add blood particles inside artery
        T height_f = pipeWidth - (sim.dx * 2.0); //32mm
        T width_f = pipeLength;
        T x1 = minX;
        T y1 = minY + wallWidth + sim.dx;
        T x2 = x1 + width_f;
        T y2 = y1 + height_f;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2); 
        //T maxVelocity = 0.0;
        //bool parabolicVelocity = false;
        sim.sampleGridAlignedBoxWithPoissonDisk_ClotCutOut(material, minPoint, maxPoint, center, radius, Vector<T, dim>(0, 0), ppc, rhoFluid, false, 4, false); //marker = 4 for fluids, helps with analysis under the hood
        //sim.sampleGridAlignedBoxWithPoissonDisk(material, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rhoFluid, false, 4, false);

        //Add elastodamage coupling
        sim.elasticityDegradationType = 1;
        sim.computeLamMaxFlag = true;

        //-----BOUNDARY CONDITIONS-----

        //Add Static Half Spaces
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY + wallWidth + pipeWidth + heldMaterial), Vector<T, dim>(0, -1), Vector<T, dim>(0, 0), 0)); //top wall -- holds artery in place as well
        //sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + pipeLength + width_f + (4.0*sim.dx), 0), Vector<T, dim>(-1, 0), Vector<T, dim>(0, 0), 0)); //right wall
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY + heldMaterial), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //bottom wall - hold artery in place
        
        //---EXTERNAL LOADING---

        Vector<T,dim> pgMin = minPoint;
        Vector<T,dim> pgMax = maxPoint;
        Vector<T,dim> toroidMin = minPoint;
        Vector<T,dim> toroidMax = maxPoint;
        pgMin[1] -= (sim.dx + heldMaterial);
        pgMax[1] += (sim.dx + heldMaterial);
        toroidMin[1] = pgMin[1];
        toroidMax[1] = pgMax[1];
        bool toroidal = true;
        //pgMin[0] -= (sim.dx * 10); //add some additional padding in front
        pgMax[0] += (sim.dx * 10); //make sure fluid will keep moving!
        //dp/dx = -u_max * (2mu / b^2)
        //want u_max = 0.15 -> dp/dx = -48 (b = 0.005, mu = 0.004)
        sim.addPressureGradient(pgMin, pgMax, pStart, pGrad, toroidMin, toroidMax, toroidal);

        //Add boxes to hold the free ends of the arterial walls
        //T boxHeight = height_f * 1.5; //just make sure this overshoots so we fully contain the fluid
        //sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T,dim>(minX + width_f + (2.0*sim.dx) - sim.dx, minY + (0.5*height_f) + (0.5*pipeWidth) + heldMaterial), Vector<T, dim>(minX + width_f + (2.0*sim.dx) + pipeLength + sim.dx,minY + (0.5*height_f) + (0.5*pipeWidth) + heldMaterial + boxHeight), Vector<T, 4>(0, 0, 0, 1.0))); //TOP BOX - HOLDING TOP ARTERY WALL
        //sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + (2.0*sim.dx) - sim.dx, minY + (0.5*height_f) - (0.5*pipeWidth) - heldMaterial - boxHeight), Vector<T,dim>(minX + width_f + (2.0*sim.dx) + pipeLength + sim.dx, minY + (0.5*height_f) - (0.5*pipeWidth) - heldMaterial), Vector<T, 4>(0, 0, 0, 1.0))); //BOTTOM BOX, HOLDING BTTOM ARTERY WALL         

        //Add Tanh Damage Model
        //int degType = 1;
        //sim.addHyperbolicTangentDamage(lamC, tanhWidth, dMin, degType);
        
        //Set degradation alpha
        //sim.degAlpha = alpha;

        //set minDp
        //sim.minDp = minDp;

        sim.run(start_frame);
    }

    //[PYTHON] Shorter Pipe With Pressure Gradient AND CLOT -- DIRICHLET PIPE WALLS
    if(testcase == 232){
        
        using T = double;
        static const int dim = 2;

        if (argc < 8) {
            puts("ERROR: please add parameters");
            puts("TEST 232 USAGE: ./cramp testcase bulk gamma viscosity pStart pGrad couplingFriction ");
            exit(0);
        }

        T bulk = std::atof(argv[2]);
        T gamma = std::atof(argv[3]);
        T viscosity = std::atof(argv[4]);
        T pStart = std::atof(argv[5]);
        T pGrad = std::atof(argv[6]);
        T couplingFriction = std::atof(argv[7]);
        // T alpha = std::atof(argv[7]);
        // T dMin = std::atof(argv[8]);
        // T minDp = std::atof(argv[9]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 8; ++i){
            std::string cleanString = argv[i];
            if(i == 4 || i == 7){// || i == 6 || i == 8 || i == 9){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/232_ToroidalPressureGradientWPoroelasticClotAndChemPotentialEvolution_DirichletPipeWallsSTICKY_c1_300k_c2_2.0_d1cm_BulkMod" + cleanedStrings[0] + "_Gamma" + cleanedStrings[1] + "_Viscosity" + cleanedStrings[2] + "_pStart" + cleanedStrings[3] + "_pGrad" + cleanedStrings[4] + "_couplingFriction" + cleanedStrings[5];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //water material
        T rhoFluid = 1060; //density of blood

        //Params
        sim.dx = 1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 240;
        sim.frame_dt = 1.0/60.0; //500 frames at 1e-3 is 0.5s
        sim.gravity = 0.0;

        //Interpolation Scheme
        sim.useAPIC = true;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 4.7; //5.5 good for dx = 0.2, 
        sim.useDFG = false;
        sim.fricCoeff = couplingFriction; //for no slip condition between solid and fluid
        sim.useExplicitContact = true;
        //sim.massRatio = 15.0;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Solid Material properties (soft artery walls)
        T E = 1e6;
        T nu = 0.2;
        T rhoSolid = 1300;
        
        //solid material (fibrin clot)
        T c1 = 300000;
        T c2 = 2.0;
        T phi_s0 = 0.01;
        T pi_0 = 1000.0;
        T beta_1 = 1.02;
        T rhoSolid2 = 1200;

        //Compute time step for symplectic
        sim.cfl = 0.4;
        sim.suggested_dt = sim.suggestedDt(E, nu, rhoSolid, sim.dx, sim.cfl); //Solid CFL condition, will be overridden when particle velocity gets too big though!     
        sim.suggested_dt = 1e-6;

        auto material = sim.create_elasticity(new MPM::ViscousEquationOfStateOp<T, dim>(bulk, gamma, viscosity)); //K = 1e7 from glacier, gamma = 7 always for water, viscosity = ?
        //auto material2 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));
        //auto material3 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));
        auto material3 = sim.create_elasticity(new MPM::FibrinPoroelasticityOp<T, dim>(c1, c2, phi_s0, pi_0, beta_1));
        
        //-----PARTICLE SAMPLING-----

        //Sampling Constants
        int ppc = 4;
        T minX = 0.05;
        T minY = 0.05;

        //Add solid arterial walls
        T wallWidth = sim.dx * 4.0;
        T heldMaterial = sim.dx * 2.0;
        T pipeWidth = sim.dx * 10; //d = 0.01m for dx = 1e-3
        T pipeLength = pipeWidth * 30.0;
        //sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX, minY), Vector<T,dim>(minX + pipeLength, minY + wallWidth), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Bottom Arterial Wall
        //sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX, minY + pipeWidth + wallWidth), Vector<T,dim>(minX + pipeLength, minY + pipeWidth + (2.0*wallWidth)), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Top Arterial Wall

        //Add fibrin clot
        T radius = pipeWidth * 0.5;
        T x_s = pipeWidth * 15.0; //dist into pipe
        Vector<T,dim> center(minX + x_s, minY + wallWidth);
        //Vector<T, dim> notchMin(center[0] - radius, center[1] + sim.dx);
        //Vector<T, dim> notchMax(center[0] - radius*0.5, center[1] + sim.dx*3.0);
        //bool damageRegion = false; //toggle this to switch between damage region and material discontinuity
        sim.sampleHemispherePoissonDisk(material3, center, radius, Vector<T, dim>(0, 0), ppc, rhoSolid2, true, 5); //add marker = 5 for poroelastic clot

        //Add blood particles inside artery
        //T height_f = pipeWidth - (sim.dx * 2.0); //32mm
        T width_f = pipeLength;
        T x1 = minX;
        T y1 = minY + wallWidth;
        T x2 = x1 + width_f;
        T y2 = y1 + pipeWidth;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2); 
        //T maxVelocity = 0.0;
        //bool parabolicVelocity = false;
        sim.sampleGridAlignedBoxWithPoissonDisk_ClotCutOut(material, minPoint, maxPoint, center, radius, Vector<T, dim>(0, 0), ppc, rhoFluid, false, 4, false); //marker = 4 for fluids, helps with analysis under the hood
        //sim.sampleGridAlignedBoxWithPoissonDisk(material, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rhoFluid, false, 4, false);

        //Add elastodamage coupling
        sim.elasticityDegradationType = 1;
        sim.computeLamMaxFlag = true;

        //-----BOUNDARY CONDITIONS-----

        //Add Static Half Spaces
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY + wallWidth + pipeWidth), Vector<T, dim>(0, -1), Vector<T, dim>(0, 0), 0)); //top wall -- holds artery in place as well
        //sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + pipeLength + width_f + (4.0*sim.dx), 0), Vector<T, dim>(-1, 0), Vector<T, dim>(0, 0), 0)); //right wall
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY + wallWidth), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //bottom wall - hold artery in place

        //left and right bounds
        //T wallBuffer = sim.dx * 7.0;
        //sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX - wallBuffer, 0), Vector<T, dim>(1, 0), Vector<T, dim>(0, 0), 0)); //left wall - hold artery in place
        //sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + wallBuffer, 0), Vector<T, dim>(-1, 0), Vector<T, dim>(0, 0), 0)); //right wall - hold artery in place
        
        //---EXTERNAL LOADING---

        Vector<T,dim> pgMin = minPoint;
        Vector<T,dim> pgMax = maxPoint;
        Vector<T,dim> toroidMin = minPoint;
        Vector<T,dim> toroidMax = maxPoint;
        pgMin[1] -= (sim.dx + heldMaterial);
        pgMax[1] += (sim.dx + heldMaterial);
        toroidMin[1] = pgMin[1];
        toroidMax[1] = pgMax[1];
        bool toroidal = true;
        //pgMin[0] -= (sim.dx * 10); //add some additional padding in front
        pgMax[0] += (sim.dx * 10); //make sure fluid will keep moving!
        //dp/dx = -u_max * (2mu / b^2)
        //want u_max = 0.15 -> dp/dx = -48 (b = 0.005, mu = 0.004)
        sim.addPressureGradient(pgMin, pgMax, pStart, pGrad, toroidMin, toroidMax, toroidal);

        //Add boxes to hold the free ends of the arterial walls
        //T boxHeight = height_f * 1.5; //just make sure this overshoots so we fully contain the fluid
        //sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T,dim>(minX + width_f + (2.0*sim.dx) - sim.dx, minY + (0.5*height_f) + (0.5*pipeWidth) + heldMaterial), Vector<T, dim>(minX + width_f + (2.0*sim.dx) + pipeLength + sim.dx,minY + (0.5*height_f) + (0.5*pipeWidth) + heldMaterial + boxHeight), Vector<T, 4>(0, 0, 0, 1.0))); //TOP BOX - HOLDING TOP ARTERY WALL
        //sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + (2.0*sim.dx) - sim.dx, minY + (0.5*height_f) - (0.5*pipeWidth) - heldMaterial - boxHeight), Vector<T,dim>(minX + width_f + (2.0*sim.dx) + pipeLength + sim.dx, minY + (0.5*height_f) - (0.5*pipeWidth) - heldMaterial), Vector<T, 4>(0, 0, 0, 1.0))); //BOTTOM BOX, HOLDING BTTOM ARTERY WALL         

        //Add Tanh Damage Model
        //int degType = 1;
        //sim.addHyperbolicTangentDamage(lamC, tanhWidth, dMin, degType);
        
        //Set degradation alpha
        //sim.degAlpha = alpha;

        //set minDp
        //sim.minDp = minDp;

        sim.run(start_frame);
    }

    //[PYTHON] Shorter Pipe With Pressure Gradient NO CLOT -- DIRICHLET PIPE WALLS - test for finding pStart
    if(testcase == 233){
        
        using T = double;
        static const int dim = 2;

        if (argc < 8) {
            puts("ERROR: please add parameters");
            puts("TEST 233 USAGE: ./cramp testcase bulk gamma viscosity pStart pGrad couplingFriction ");
            exit(0);
        }

        T bulk = std::atof(argv[2]);
        T gamma = std::atof(argv[3]);
        T viscosity = std::atof(argv[4]);
        T pStart = std::atof(argv[5]);
        T pGrad = std::atof(argv[6]);
        T couplingFriction = std::atof(argv[7]);
        // T alpha = std::atof(argv[7]);
        // T dMin = std::atof(argv[8]);
        // T minDp = std::atof(argv[9]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 8; ++i){
            std::string cleanString = argv[i];
            if(i == 4 || i == 7){// || i == 6 || i == 8 || i == 9){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/233_ToroidalPressureGradient_noClot_DirichletPipeWallsSTICKY_c1_300k_c2_2.0_d1cm_BulkMod" + cleanedStrings[0] + "_Gamma" + cleanedStrings[1] + "_Viscosity" + cleanedStrings[2] + "_pStart" + cleanedStrings[3] + "_pGrad" + cleanedStrings[4] + "_couplingFriction" + cleanedStrings[5];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //water material
        T rhoFluid = 1060; //density of blood

        //Params
        sim.dx = 1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 240;
        sim.frame_dt = 1.0/60.0; //500 frames at 1e-3 is 0.5s
        sim.gravity = 0.0;

        //Interpolation Scheme
        sim.useAPIC = true;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 4.7; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = couplingFriction; //for no slip condition between solid and fluid
        sim.useExplicitContact = true;
        //sim.massRatio = 15.0;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Solid Material properties (soft artery walls)
        T E = 1e6;
        T nu = 0.2;
        T rhoSolid = 1300;
        
        //solid material (fibrin clot)
        // T c1 = 300000;
        // T c2 = 2.0;
        // T phi_s0 = 0.01;
        // T pi_0 = 1000.0;
        // T beta_1 = 1.02;
        // T rhoSolid2 = 1200;

        //Compute time step for symplectic
        sim.cfl = 0.4;
        sim.suggested_dt = sim.suggestedDt(E, nu, rhoSolid, sim.dx, sim.cfl); //Solid CFL condition, will be overridden when particle velocity gets too big though!     
        //sim.suggested_dt = 1e-6;

        auto material = sim.create_elasticity(new MPM::ViscousEquationOfStateOp<T, dim>(bulk, gamma, viscosity)); //K = 1e7 from glacier, gamma = 7 always for water, viscosity = ?
        //auto material2 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));
        //auto material3 = sim.create_elasticity(new MPM::NeoHookeanOp<T, dim>(E, nu));
        //auto material3 = sim.create_elasticity(new MPM::FibrinPoroelasticityOp<T, dim>(c1, c2, phi_s0, pi_0, beta_1));
        
        //-----PARTICLE SAMPLING-----

        //Sampling Constants
        int ppc = 4;
        T minX = 0.05;
        T minY = 0.05;

        //Add solid arterial walls
        T wallWidth = sim.dx * 4.0;
        T heldMaterial = sim.dx * 2.0;
        T pipeWidth = sim.dx * 10; //d = 0.01m for dx = 1e-3
        T pipeLength = pipeWidth * 30.0;
        //sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX, minY), Vector<T,dim>(minX + pipeLength, minY + wallWidth), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Bottom Arterial Wall
        //sim.sampleGridAlignedBoxWithPoissonDisk(material2, Vector<T,dim>(minX, minY + pipeWidth + wallWidth), Vector<T,dim>(minX + pipeLength, minY + pipeWidth + (2.0*wallWidth)), Vector<T, dim>(0, 0), ppc, rhoSolid, false, 0); //Top Arterial Wall

        //Add fibrin clot
        //T radius = pipeWidth * 0.5;
        //T x_s = pipeWidth * 15.0; //dist into pipe
        //Vector<T,dim> center(minX + x_s, minY + wallWidth);
        //Vector<T, dim> notchMin(center[0] - radius, center[1] + sim.dx);
        //Vector<T, dim> notchMax(center[0] - radius*0.5, center[1] + sim.dx*3.0);
        //bool damageRegion = false; //toggle this to switch between damage region and material discontinuity
        //sim.sampleHemispherePoissonDisk(material3, center, radius, Vector<T, dim>(0, 0), ppc, rhoSolid2, true, 0);

        //Add blood particles inside artery
        //T height_f = pipeWidth - (sim.dx * 2.0); //32mm
        T width_f = pipeLength;
        T x1 = minX;
        T y1 = minY + wallWidth;
        T x2 = x1 + width_f;
        T y2 = y1 + pipeWidth;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2); 
        //T maxVelocity = 0.0;
        //bool parabolicVelocity = false;
        //sim.sampleGridAlignedBoxWithPoissonDisk_ClotCutOut(material, minPoint, maxPoint, center, radius, Vector<T, dim>(0, 0), ppc, rhoFluid, false, 4, false); //marker = 4 for fluids, helps with analysis under the hood
        sim.sampleGridAlignedBoxWithPoissonDisk(material, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rhoFluid, false, 4, false);

        //Add elastodamage coupling
        sim.elasticityDegradationType = 1;
        sim.computeLamMaxFlag = true;

        //-----BOUNDARY CONDITIONS-----

        //Add Static Half Spaces
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY + wallWidth + pipeWidth), Vector<T, dim>(0, -1), Vector<T, dim>(0, 0), 0)); //top wall -- holds artery in place as well
        //sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + pipeLength + width_f + (4.0*sim.dx), 0), Vector<T, dim>(-1, 0), Vector<T, dim>(0, 0), 0)); //right wall
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY + wallWidth), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //bottom wall - hold artery in place

        //left and right bounds
        //T wallBuffer = sim.dx * 7.0;
        //sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX - wallBuffer, 0), Vector<T, dim>(1, 0), Vector<T, dim>(0, 0), 0)); //left wall - hold artery in place
        //sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + wallBuffer, 0), Vector<T, dim>(-1, 0), Vector<T, dim>(0, 0), 0)); //right wall - hold artery in place
        
        //---EXTERNAL LOADING---

        Vector<T,dim> pgMin = minPoint;
        Vector<T,dim> pgMax = maxPoint;
        Vector<T,dim> toroidMin = minPoint;
        Vector<T,dim> toroidMax = maxPoint;
        pgMin[1] -= (sim.dx + heldMaterial);
        pgMax[1] += (sim.dx + heldMaterial);
        toroidMin[1] = pgMin[1];
        toroidMax[1] = pgMax[1];
        bool toroidal = true;
        //pgMin[0] -= (sim.dx * 10); //add some additional padding in front
        pgMax[0] += (sim.dx * 10); //make sure fluid will keep moving!
        //dp/dx = -u_max * (2mu / b^2)
        //want u_max = 0.15 -> dp/dx = -48 (b = 0.005, mu = 0.004)
        sim.addPressureGradient(pgMin, pgMax, pStart, pGrad, toroidMin, toroidMax, toroidal);

        //Add boxes to hold the free ends of the arterial walls
        //T boxHeight = height_f * 1.5; //just make sure this overshoots so we fully contain the fluid
        //sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T,dim>(minX + width_f + (2.0*sim.dx) - sim.dx, minY + (0.5*height_f) + (0.5*pipeWidth) + heldMaterial), Vector<T, dim>(minX + width_f + (2.0*sim.dx) + pipeLength + sim.dx,minY + (0.5*height_f) + (0.5*pipeWidth) + heldMaterial + boxHeight), Vector<T, 4>(0, 0, 0, 1.0))); //TOP BOX - HOLDING TOP ARTERY WALL
        //sim.add_boundary_condition(new Geometry::BoxLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(minX + width_f + (2.0*sim.dx) - sim.dx, minY + (0.5*height_f) - (0.5*pipeWidth) - heldMaterial - boxHeight), Vector<T,dim>(minX + width_f + (2.0*sim.dx) + pipeLength + sim.dx, minY + (0.5*height_f) - (0.5*pipeWidth) - heldMaterial), Vector<T, 4>(0, 0, 0, 1.0))); //BOTTOM BOX, HOLDING BTTOM ARTERY WALL         

        //Add Tanh Damage Model
        //int degType = 1;
        //sim.addHyperbolicTangentDamage(lamC, tanhWidth, dMin, degType);
        
        //Set degradation alpha
        //sim.degAlpha = alpha;

        //set minDp
        //sim.minDp = minDp;

        sim.run(start_frame);
    }

    //Chem Potential Solve Simple Mode 1 Tension Test
    if(testcase == 234){
        
        using T = double;
        static const int dim = 2;

        // if (argc < 8) {
        //     puts("ERROR: please add parameters");
        //     puts("TEST 234 USAGE: ./cramp testcase bulk gamma viscosity pStart pGrad couplingFriction ");
        //     exit(0);
        // }

        // T bulk = std::atof(argv[2]);
        // T gamma = std::atof(argv[3]);
        // T viscosity = std::atof(argv[4]);
        // T pStart = std::atof(argv[5]);
        // T pGrad = std::atof(argv[6]);
        // T couplingFriction = std::atof(argv[7]);
        // // T alpha = std::atof(argv[7]);
        // // T dMin = std::atof(argv[8]);
        // // T minDp = std::atof(argv[9]);
        // std::vector<std::string> cleanedStrings;
        // for(int i = 2; i < 8; ++i){
        //     std::string cleanString = argv[i];
        //     if(i == 4 || i == 7){// || i == 6 || i == 8 || i == 9){
        //         cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
        //     }
        //     cleanedStrings.push_back(cleanString);
        // }
        std::string path = "output/234_ChemPotentialSolveTest_Mode1Tension";
        MPM::CRAMPSimulator<T, dim> sim(path);

        //Params
        sim.dx = 5e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 240;
        sim.frame_dt = 1.0/60.0; //500 frames at 1e-3 is 0.5s
        sim.gravity = 0.0;

        //Interpolation Scheme
        sim.useAPIC = true;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 4.7; //5.5 good for dx = 0.2, 
        sim.useDFG = false;
        //sim.fricCoeff = couplingFriction; //for no slip condition between solid and fluid
        sim.useExplicitContact = true;
        //sim.massRatio = 15.0;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //solid material (fibrin clot)
        T c1 = 300000;
        T c2 = 2.0;
        T phi_s0 = 0.01;
        T pi_0 = 1000.0;
        T beta_1 = 1.02;
        T rhoSolid2 = 1200;

        //Compute time step for symplectic
        sim.suggested_dt = 1e-4;

        auto material3 = sim.create_elasticity(new MPM::FibrinPoroelasticityOp<T, dim>(c1, c2, phi_s0, pi_0, beta_1));
        
        //-----PARTICLE SAMPLING-----

        //Sampling Constants
        int ppc = 4;
        T minX = 0.06; //tuning this enabled making the BoxSampling actually a square lol must be rounding error :O
        T minY = 0.05;

        //Add fibrin block
        T width = sim.dx * 2;
        T height = width;
        sim.sampleGridAlignedBox(material3, Vector<T,dim>(minX, minY), Vector<T, dim>(minX + width, minY + height), Vector<T, dim>(0,0), ppc, rhoSolid2, false, 5);

        //-----BOUNDARY CONDITIONS-----

        //Add Static Half Spaces
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(minX, 0), Vector<T, dim>(1, 0), Vector<T, dim>(0, 0), 0)); //left wall
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::SLIP, Vector<T, dim>(minX + width, 0), Vector<T, dim>(-1, 0), Vector<T, dim>(0, 0), 0)); //right wall
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY + 0.5*sim.dx), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //bottom wall - hold artery in place

        //Displacement Half Space
        T moveTime = 3.0;
        T displacement = sim.dx * 0.5;
        T speed = displacement / moveTime;
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, minY + height - 0.5*sim.dx), Vector<T, dim>(0, -1), Vector<T, dim>(0, speed), moveTime)); //bottom wall - hold artery in place
    
        sim.run(start_frame);
    }

    //---------------------------------------------------------------------
    //---------------------------------------------------------------------
    //----------------------  FINAL TESTING SIMS  -------------------------
    //---------------------------------------------------------------------
    //---------------------------------------------------------------------


    //SENT with Damage Region and Elasticity Degradation -> Single Field MPM
    if (testcase == 2001) {
        
        //Fibrin Parameters from Tutwiler2020
        // fracture toughness,          Gc = 7.6 +/- 0.45 J/m^2
        // folded state stiffness,      cf = 4.4e4 N/m^2
        // unfolded state stiffness,    cu = 2.6e6 N/m^2
        // fibrinogen density,          rho = 1395 g/cm^3 = 1,395,000 kg/m^3 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3044599/

        using T = double;
        static const int dim = 2;
        MPM::CRAMPSimulator<T, dim> sim("output/2001_SENT_withDamageRegion_andElasticityDegradation_SingleFieldMPM_dx0.5mm_PIC_BETWEEN");

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = 0.5e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 100;
        //sim.frame_dt = 22e-6 / sim.end_frame; //total time = 22e-6 s, want 1000 frames of this
        sim.frame_dt = 1e-1; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = false;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));

        //Sample Particles
        int ppc = 4;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        sim.sampleGridAlignedBox(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho, false, 0);

        //Add Crack
        T crackSegmentLength = sim.dx / 5.0;
        T damageRadius = sim.dx / 2.0;
        T crackLength = 5e-3;
        T crackY = y1 + height/2.0 - (0.5*sim.dx);
        T crackX = x1 + (sim.dx / std::pow(ppc, (T)1 / dim) / 2.0);
        sim.addHorizontalCrackWithoutPoints(Vector<T,dim>(crackX, crackY), Vector<T,dim>(crackX + crackLength, crackY), crackSegmentLength, damageRadius);
        
        T sigmaA = 2600; //1000 times smaller than E
        T rampTime = sim.frame_dt * 40; // ramp up 4 seconds
        sim.addMode1Loading(y2, y1, sigmaA, rampTime, true, width, x1, x2); //if doing nodal loading, pass y1, y2, x1, x2 as the exact min and max of the material!
        
        //Add Elasticity Degradation
        sim.elasticityDegradationType = 1;

        T energyDt = sim.frame_dt;
        sim.addEnergyTracking(energyDt);

        sim.run(start_frame);
    }

    //SENT with Damage Region and Elasticity Degradation -> Two-Field MPM
    if (testcase == 2002) {
        
        //Fibrin Parameters from Tutwiler2020
        // fracture toughness,          Gc = 7.6 +/- 0.45 J/m^2
        // folded state stiffness,      cf = 4.4e4 N/m^2
        // unfolded state stiffness,    cu = 2.6e6 N/m^2
        // fibrinogen density,          rho = 1395 g/cm^3 = 1,395,000 kg/m^3 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3044599/

        using T = double;
        static const int dim = 2;
        MPM::CRAMPSimulator<T, dim> sim("output/2002_SENT_withDamageRegion_andElasticityDegradation_TwoFieldMPM_dx0.5mm_PIC_dMin0.25_rpDefault_CENTERED");

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = 0.5e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 100;
        //sim.frame_dt = 22e-6 / sim.end_frame; //total time = 22e-6 s, want 1000 frames of this
        sim.frame_dt = 1e-1; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 0; //automatically markign surface for sampleGridAlignedBox
        sim.useDFG = true;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        //sim.rpFactor = 1.8; //TODO: explore if this helps results or not? default value is sqrt(2) ~= 1.414
        sim.dMin = 0.25;

        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));

        //Sample Particles
        int ppc = 4;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        sim.sampleGridAlignedBox(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho, true, 0);

        //Add Crack
        T crackSegmentLength = sim.dx / 5.0;
        T damageRadius = sim.dx / 2.0;
        T crackLength = 5e-3;
        T crackY = y1 + height/2.0;// - (0.5*sim.dx);
        T crackX = x1 + (sim.dx / std::pow(ppc, (T)1 / dim) / 2.0);
        sim.addHorizontalCrackWithoutPoints(Vector<T,dim>(crackX, crackY), Vector<T,dim>(crackX + crackLength, crackY), crackSegmentLength, damageRadius);
        
        T sigmaA = 2600; //1000 times smaller than E
        T rampTime = sim.frame_dt * 40; // ramp up 4 seconds
        sim.addMode1Loading(y2, y1, sigmaA, rampTime, true, width, x1, x2); //if doing nodal loading, pass y1, y2, x1, x2 as the exact min and max of the material!
        
        //Add Elasticity Degradation
        sim.elasticityDegradationType = 1;

        sim.run(start_frame);
    }

    //[PYTHON] SENT with Constant Dx & Variable Crack Thickness -> Single-Field
    if (testcase == 2003) {

        using T = double;
        static const int dim = 2;

        if (argc < 4) {
            puts("ERROR: please add parameters");
            puts("TEST 2003 USAGE: ./cramp testcase crackThickness damageRadius");
            exit(0);
        }

        T crackThickness = std::atof(argv[2]);
        T userDamageRadius = std::atof(argv[3]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 4; ++i){
            std::string cleanString = argv[i];
            if(i == 2){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/2003_withHolder_SENT_dx0.1mm_SingleFieldMPM_crackThickness" + cleanedStrings[0] + "_damageRadius" + cleanedStrings[1];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = 0.1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 80; //need to simulate around 9 to 12 seconds to remove oscillations
        sim.frame_dt = 1e-1; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.useDFG = false;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));

        //Sample Particles
        int ppc = 4;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T crackLength = 5e-3;
        T crackRadius = crackThickness / 2.0; //0.6e-3 crack width = 3.75% of specimen height 1.2/32
        T crackHeight = y1 + (height/2.0);
        sim.sampleGridAlignedBoxWithNotch(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, true, Vector<T, dim>(0, 0), ppc, rho);

        //Add Tracton Boundary Condition        
        T sigmaA = 2600; //1000 times smaller than E
        T rampTime = sim.frame_dt * 40; //ramp up to full sigmaA over 500 frames
        sim.addMode1Loading(y2, y1, sigmaA, rampTime, true, width, x1, x2); //if doing nodal loading, pass y1, y2, x1, x2 as the exact min and max of the material!
        
        //Add damage particles at the crack edges
        sim.addHorizontalCrackWithoutPoints(Vector<T,dim>(x1, crackHeight), Vector<T,dim>(x1 + crackLength, crackHeight), sim.dx/3.0, userDamageRadius);

        //Add Holder at the bottom (variable dx makes nodal loading imbalanced, so hold the bottom of the specimen)
        T heldMaterial = sim.dx;
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, y1 + heldMaterial), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //bottom holder

        //Add Contours
        
        //DX = 0.1mm
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,25,75), true, true); //second true is to mark this contour for additional tracking of data (J_I contributions)
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,50,75), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,75,75), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,100,75), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,125,75), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,145,75), true);
        //sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,150,75), true);  

        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,25,125), true, true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,50,125), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,75,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,100,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,125,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,145,125), true);  
        //sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,150,125), true); 

        //These have different L values than the other families!
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,75,25,75), true, true);    //compare to Contour A
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,75,100,75), true);         //to Contour D
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,125,25,125), true, true);  //to Contour 1
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,125,100,125), true);       //to Contour 4

        //Add contours that define the inverse intersections between each pair of contours (A and 1, B and 2, etc.) -> each pair has an upper and lower contour, each not containing the crack and should have J = 0
        Vector<T, dim> upperCenter(0.045, 0.06);
        Vector<T, dim> lowerCenter(0.045, 0.04);
        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 25, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 25, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 50, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 50, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 75, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 75, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 100, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 100, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 125, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 125, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 145, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 145, 25), false);
        
        //Add timing for contours (NOTE: without this we wont calculate anything!)
        std::vector<T> contourTimes;
        contourTimes.push_back(sim.frame_dt * 40);
        contourTimes.push_back(sim.frame_dt * 45);
        contourTimes.push_back(sim.frame_dt * 50);
        contourTimes.push_back(sim.frame_dt * 55);
        contourTimes.push_back(sim.frame_dt * 60);
        contourTimes.push_back(sim.frame_dt * 65);
        contourTimes.push_back(sim.frame_dt * 70);
        contourTimes.push_back(sim.frame_dt * 75);
        contourTimes.push_back(sim.frame_dt * 79);
        sim.addJIntegralTiming(contourTimes, false);

        sim.run(start_frame);
    }

    //[PYTHON] SENT with Constant Dx & Variable Crack Thickness -> Two-Field
    if (testcase == 2004) {

        using T = double;
        static const int dim = 2;

        if (argc < 4) {
            puts("ERROR: please add parameters");
            puts("TEST 2004 USAGE: ./cramp testcase crackThickness damageRadius");
            exit(0);
        }

        T crackThickness = std::atof(argv[2]);
        T userDamageRadius = std::atof(argv[3]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 4; ++i){
            std::string cleanString = argv[i];
            if(i == 2){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/2004_withHolder_SENT_dx0.1mm_TwoFieldMPM_crackThickness" + cleanedStrings[0] + "_damageRadius" + cleanedStrings[1];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = 0.1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 80; //need to simulate around 9 to 12 seconds to remove oscillations
        sim.frame_dt = 1e-1; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 0.0;
        sim.useDFG = true;
        sim.fricCoeff = 0.15;
        sim.useExplicitContact = true;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));

        //Sample Particles
        int ppc = 4;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T crackLength = 5e-3;
        T crackRadius = crackThickness / 2.0; //0.6e-3 crack width = 3.75% of specimen height 1.2/32
        T crackHeight = y1 + (height/2.0);
        sim.sampleGridAlignedBoxWithNotch(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, true, Vector<T, dim>(0, 0), ppc, rho, true);

        //Add damage particles at the crack edges
        sim.addHorizontalCrackWithoutPoints(Vector<T,dim>(x1, crackHeight), Vector<T,dim>(x1 + crackLength, crackHeight), sim.dx/3.0, userDamageRadius);

        //Add Tracton Boundary Condition        
        T sigmaA = 2600; //1000 times smaller than E
        T rampTime = sim.frame_dt * 40; //ramp up to full sigmaA over 500 frames
        sim.addMode1Loading(y2, y1, sigmaA, rampTime, true, width, x1, x2); //if doing nodal loading, pass y1, y2, x1, x2 as the exact min and max of the material!
        
        //Add Elasticity Degradation
        sim.elasticityDegradationType = 1;

        //Add Holder at the bottom (variable dx makes nodal loading imbalanced, so hold the bottom of the specimen)
        T heldMaterial = sim.dx;
        sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, y1 + heldMaterial), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0)); //bottom holder

        //Add Contours
        
        //DX = 0.1mm
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,25,75), true, true); //second true is to mark this contour for additional tracking of data (J_I contributions)
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,50,75), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,75,75), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,100,75), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,125,75), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,145,75), true);
        //sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,150,75), true);  

        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,25,125), true, true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,50,125), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,75,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,100,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,125,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,145,125), true);  
        //sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,150,125), true); 

        //These have different L values than the other families!
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,75,25,75), true, true);    //compare to Contour A
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,75,100,75), true);         //to Contour D
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,125,25,125), true, true);  //to Contour 1
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,125,100,125), true);       //to Contour 4

        //Add contours that define the inverse intersections between each pair of contours (A and 1, B and 2, etc.) -> each pair has an upper and lower contour, each not containing the crack and should have J = 0
        Vector<T, dim> upperCenter(0.045, 0.06);
        Vector<T, dim> lowerCenter(0.045, 0.04);
        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 25, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 25, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 50, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 50, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 75, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 75, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 100, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 100, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 125, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 125, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 145, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 145, 25), false);
        
        //Add timing for contours (NOTE: without this we wont calculate anything!)
        std::vector<T> contourTimes;
        contourTimes.push_back(sim.frame_dt * 40);
        contourTimes.push_back(sim.frame_dt * 45);
        contourTimes.push_back(sim.frame_dt * 50);
        contourTimes.push_back(sim.frame_dt * 55);
        contourTimes.push_back(sim.frame_dt * 60);
        contourTimes.push_back(sim.frame_dt * 65);
        contourTimes.push_back(sim.frame_dt * 70);
        contourTimes.push_back(sim.frame_dt * 75);
        contourTimes.push_back(sim.frame_dt * 79);
        sim.addJIntegralTiming(contourTimes, false);

        sim.run(start_frame);
    }

    //[PYTHON] Plate with Constant Radius Hole and Variable Dx -> Single Field
    if (testcase == 2005) {


        using T = double;
        static const int dim = 2;

        if (argc < 3) {
            puts("ERROR: please add parameters");
            puts("TEST 2005 USAGE: ./cramp testcase dx");
            exit(0);
        }

        T userDx = std::atof(argv[2]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 3; ++i){
            std::string cleanString = argv[i];
            if(i == 2){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/2005_HoleinPlate_PoissonDisk_ppc4_with_SingleFieldMPM_dx" + cleanedStrings[0];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = userDx; //0.5 mm
        sim.symplectic = true;
        sim.end_frame = 40;
        //sim.frame_dt = 22e-6 / sim.end_frame; //total time = 22e-6 s, want 1000 frames of this
        sim.frame_dt = 1e-1; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 4.0; //4.0 is close to working, but we actually don't want surfacing for this demo
        sim.useDFG = false;
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
        int ppc = 4;
        T center = 0.05;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = center - width/2.0;
        T y1 = center - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T aOverB = 0.1;
        T radius = aOverB * (width / 2.0);
        //sim.sampleGridAlignedBoxWithHole(material1, minPoint, maxPoint, Vector<T,dim>(center, center), radius, Vector<T, dim>(0, 0), ppc, rho, false);
        sim.sampleGridAlignedBoxWithHole_PoissonDisk(material1, minPoint, maxPoint, Vector<T,dim>(center, center), radius, Vector<T, dim>(0, 0), ppc, rho, false);

        T sigmaA = 2600; //1000 times smaller than E
        T rampTime = sim.frame_dt * 5; //ramp up to full sigmaA over 500 frames

        sim.addMode1Loading(y2, y1, sigmaA, rampTime, true, width, x1, x2); //if doing nodal loading, pass y1, y2, x1, x2 as the exact min and max of the material!
    
        sim.run(start_frame);
    }

    //[PYTHON] Plate with Constant Radius Hole and Variable Dx -> Two-Field
    if (testcase == 2006) {

        using T = double;
        static const int dim = 2;

        if (argc < 4) {
            puts("ERROR: please add parameters");
            puts("TEST 2006 USAGE: ./cramp testcase dx st");
            exit(0);
        }

        T userDx = std::atof(argv[2]);
        T userSt = std::atof(argv[3]);
        std::vector<std::string> cleanedStrings;
        for(int i = 2; i < 4; ++i){
            std::string cleanString = argv[i];
            if(i == 2){
                cleanString.erase(cleanString.find_last_not_of('0') + 1, std::string::npos);
            }
            cleanedStrings.push_back(cleanString);
        }
        std::string path = "output/2006_HoleinPlate_UniformSampling_w4PPC_with_TwoFieldMPM_dx" + cleanedStrings[0] + "_st" + cleanedStrings[1];
        MPM::CRAMPSimulator<T, dim> sim(path);

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = userDx; //0.5 mm
        sim.symplectic = true;
        sim.end_frame = 40;
        //sim.frame_dt = 22e-6 / sim.end_frame; //total time = 22e-6 s, want 1000 frames of this
        sim.frame_dt = 1e-1; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = userSt; //4.0 is close to working, but we actually don't want surfacing for this demo
        sim.useDFG = true;
        sim.fricCoeff = 0.15;
        
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
        int ppc = 4;
        T center = 0.05;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = center - width/2.0;
        T y1 = center - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T aOverB = 0.1;
        T radius = aOverB * (width / 2.0);
        sim.sampleGridAlignedBoxWithHole(material1, minPoint, maxPoint, Vector<T,dim>(center, center), radius, Vector<T, dim>(0, 0), ppc, rho, true);

        T sigmaA = 2600; //1000 times smaller than E
        T rampTime = sim.frame_dt * 5; //ramp up to full sigmaA over 500 frames

        //Add Elasticity Degradation
        sim.elasticityDegradationType = 1;

        sim.addMode1Loading(y2, y1, sigmaA, rampTime, true, width, x1, x2); //if doing nodal loading, pass y1, y2, x1, x2 as the exact min and max of the material!

        //T heldMaterial = sim.dx * 1.5;
        //sim.add_boundary_condition(new Geometry::HalfSpaceLevelSet<T, dim>(Geometry::STICKY, Vector<T, dim>(0, y1 + heldMaterial), Vector<T, dim>(0, 1), Vector<T, dim>(0, 0), 0.0)); //bottom puller (constant)
    
        sim.run(start_frame);
    }

    // SENT with 2*dx Wide Crack and Single Field (compare against equatorial stress results from two field 201 and single field 207)
    if (testcase == 2007) {

        using T = double;
        static const int dim = 2;
        MPM::CRAMPSimulator<T, dim> sim("output/2007_SENT_2dxCrack_singleField_dx0.1mm_sigmaA_2600_FCR_ramp4s");

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        sim.dx = 0.1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 150; //need to simulate around 9 to 12 seconds to remove oscillations
        sim.frame_dt = 1e-1; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.useDFG = false;
        
        //Debug mode
        sim.verbose = false;
        sim.writeGrid = true;
        
        //Compute time step for symplectic
        sim.cfl = 0.4;
        T maxDt = sim.suggestedDt(E, nu, rho, sim.dx, sim.cfl);
        sim.suggested_dt = 0.9 * maxDt;

        // Using `new` to avoid redundant copy constructor
        auto material1 = sim.create_elasticity(new MPM::FixedCorotatedOp<T, dim>(E, nu));

        //Sample Particles
        int ppc = 4;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        T crackLength = 5e-3;
        T crackRadius = sim.dx;
        T crackHeight = y1 + (height/2.0);
        sim.sampleGridAlignedBoxWithNotch(material1, minPoint, maxPoint, crackLength, crackRadius, crackHeight, false, Vector<T, dim>(0, 0), ppc, rho);

        //Add Tracton Boundary Condition        
        T sigmaA = 2600; //1000 times smaller than E
        T rampTime = sim.frame_dt * 40; //ramp up to full sigmaA over 500 frames
        sim.addMode1Loading(y2, y1, sigmaA, rampTime, true, width, x1, x2); //if doing nodal loading, pass y1, y2, x1, x2 as the exact min and max of the material!
        
        //Add Contours
        
        //contain crack tip
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int, 4>(25, 75, 25, 75), true); //LEFT, DOWN, RIGHT, UP
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int, 4>(25, 75, 50, 75), true);
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int, 4>(25, 75, 75, 75), true);
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int, 4>(25, 75, 100, 75), true);
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int, 4>(25, 75, 125, 75), true);
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int, 4>(25, 75, 150, 75), true);
        
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int, 4>(25, 125, 25, 125), true);
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int, 4>(25, 125, 50, 125), true);
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int, 4>(25, 125, 75, 125), true);
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int, 4>(25, 125, 100, 125), true);
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int, 4>(25, 125, 125, 125), true);
        // sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int, 4>(25, 125, 150, 125), true);

        // //sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(10,54,50,54)); //centered on crack tip
        // //sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,49,45,49)); //centered on crack tip. inset contour to check against

        // //do not contain crack tip
        // //sim.addJIntegralContour(Vector<T,dim>(0.05175, 0.05), Vector<int,4>(15,54,20,54)); //centered ahead of crack tip and contains NO SINGULARITY
        // //sim.addJIntegralContour(Vector<T,dim>(0.05175, 0.05), Vector<int,4>(10,5,15,49)); //same center, but inset and upper half
        // //sim.addJIntegralContour(Vector<T,dim>(0.05175, 0.05), Vector<int,4>(10,49,15,5)); //same center, but inset and lower half
        
        // //Add timing for contours (NOTE: without this we wont calculate anything!)
        // std::vector<T> contourTimes;
        // // contourTimes.push_back(sim.frame_dt * 0.2);
        // // contourTimes.push_back(sim.frame_dt * 0.7);
        // // contourTimes.push_back(sim.frame_dt * 200);
        // // contourTimes.push_back(sim.frame_dt * 2500);
        // // contourTimes.push_back(sim.frame_dt * 300);
        // // contourTimes.push_back(sim.frame_dt * 350);

        // contourTimes.push_back(sim.frame_dt * 400);
        // contourTimes.push_back(sim.frame_dt * 450);
        // contourTimes.push_back(sim.frame_dt * 500);
        // contourTimes.push_back(sim.frame_dt * 550);
        // contourTimes.push_back(sim.frame_dt * 600);
        // contourTimes.push_back(sim.frame_dt * 650);
        // contourTimes.push_back(sim.frame_dt * 700);
        // contourTimes.push_back(sim.frame_dt * 750);
        // contourTimes.push_back(sim.frame_dt * 800);
        // contourTimes.push_back(sim.frame_dt * 850);
        // contourTimes.push_back(sim.frame_dt * 900);
        // contourTimes.push_back(sim.frame_dt * 950);
        // contourTimes.push_back(sim.frame_dt * 1000);
        // contourTimes.push_back(sim.frame_dt * 1050);
        // contourTimes.push_back(sim.frame_dt * 1100);
        // contourTimes.push_back(sim.frame_dt * 1150);
        // contourTimes.push_back(sim.frame_dt * 1200);
        // contourTimes.push_back(sim.frame_dt * 1250);
        // contourTimes.push_back(sim.frame_dt * 1300);
        // contourTimes.push_back(sim.frame_dt * 1350);
        // contourTimes.push_back(sim.frame_dt * 1400);
        // contourTimes.push_back(sim.frame_dt * 1450);
        // contourTimes.push_back(sim.frame_dt * 1499);
        // sim.addJIntegralTiming(contourTimes);

        sim.run(start_frame);
    }

    //SENT with damage region and elasticity degradation -- computing the Dynamic J-Integral with a sharp crack
    if (testcase == 2008) {
        
        //Fibrin Parameters from Tutwiler2020
        // fracture toughness,          Gc = 7.6 +/- 0.45 J/m^2
        // folded state stiffness,      cf = 4.4e4 N/m^2
        // unfolded state stiffness,    cu = 2.6e6 N/m^2
        // fibrinogen density,          rho = 1395 g/cm^3 = 1,395,000 kg/m^3 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3044599/

        using T = double;
        static const int dim = 2;
        MPM::CRAMPSimulator<T, dim> sim("output/2008_SENT_damageRegionWithElastDeg_dx0.1mm_sigmaA_2600_FCR_ramp4s_PIC_tensorTransfer_CENTERED");

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        bool useDisplacement = false;
        sim.dx = 0.1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 100;
        //sim.frame_dt = 22e-6 / sim.end_frame; //total time = 22e-6 s, want 1000 frames of this
        sim.frame_dt = 1e-1; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
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
        int ppc = 4;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        sim.sampleGridAlignedBox(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho, true);

        //Add Crack
        T damageRadius = sim.dx / 2.0;
        T crackLength = 5e-3;
        T crackY = y1 + height/2.0;// - (sim.dx/2.0);
        T crackX = x1;
        Vector<T, dim> damageRegionMin(crackX, crackY - damageRadius);
        Vector<T, dim> damageRegionMax(crackX + crackLength, crackY + damageRadius);
        sim.addRectangularDamageRegion(damageRegionMin, damageRegionMax);

        //ADd crack segments for sharp J-integral
        T crackSegmentLength = sim.dx / 5.0;
        sim.addHorizontalCrack(Vector<T,dim>(crackX, crackY), Vector<T,dim>(crackX + crackLength, crackY), crackSegmentLength, 0.0);

        //Add Boundary Condition
        T sigmaA = 2600; //1000 times smaller than E
        T rampTime = sim.frame_dt * 40; // ramp up 4 seconds
        sim.addMode1Loading(y2, y1, sigmaA, rampTime, true, width, x1, x2); //if doing nodal loading, pass y1, y2, x1, x2 as the exact min and max of the material!

        // T simpleDampFactor = 0.5;
        // T simpleDampStartTime = sim.frame_dt * 500; //start damping once we reach the full load (rampTime over)
        // T simpleDampDuration = sim.frame_dt * 500; //for 1500 frames, damp
        // sim.addSimpleDamping(simpleDampFactor, simpleDampStartTime, simpleDampDuration);
        
        //Add Elasticity Degradation
        sim.elasticityDegradationType = 1;

        //Add Energy Tracking
        T energyDt = sim.frame_dt;
        sim.addEnergyTracking(energyDt);

        //Add Contours

        //DX = 0.1mm
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,25,75), true, true); //second true is to mark this contour for additional tracking of data (J_I contributions)
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,50,75), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,75,75), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,100,75), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,125,75), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,145,75), true);
        //sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,150,75), true);  

        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,25,125), true, true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,50,125), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,75,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,100,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,125,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,145,125), true);  
        //sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,150,125), true); 

        //These have different L values than the other families!
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,75,25,75), true, true);    //compare to Contour A
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,75,100,75), true);         //to Contour D
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,125,25,125), true, true);  //to Contour 1
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,125,100,125), true);       //to Contour 4

        //Add contours that define the inverse intersections between each pair of contours (A and 1, B and 2, etc.) -> each pair has an upper and lower contour, each not containing the crack and should have J = 0
        Vector<T, dim> upperCenter(0.045, 0.06);
        Vector<T, dim> lowerCenter(0.045, 0.04);
        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 25, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 25, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 50, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 50, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 75, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 75, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 100, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 100, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 125, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 125, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 145, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 145, 25), false);
        
        //Add timing for contours (NOTE: without this we wont calculate anything!)
        std::vector<T> contourTimes;
        contourTimes.push_back(sim.frame_dt * 1);
        contourTimes.push_back(sim.frame_dt * 40);
        contourTimes.push_back(sim.frame_dt * 45);
        contourTimes.push_back(sim.frame_dt * 50);
        contourTimes.push_back(sim.frame_dt * 55);
        contourTimes.push_back(sim.frame_dt * 60);
        contourTimes.push_back(sim.frame_dt * 65);
        contourTimes.push_back(sim.frame_dt * 70);
        contourTimes.push_back(sim.frame_dt * 75);
        contourTimes.push_back(sim.frame_dt * 80);
        contourTimes.push_back(sim.frame_dt * 85);
        contourTimes.push_back(sim.frame_dt * 90);
        contourTimes.push_back(sim.frame_dt * 95);
        contourTimes.push_back(sim.frame_dt * 99);
        sim.addJIntegralTiming(contourTimes, useDisplacement);

        sim.run(start_frame);
    }

    //SENT with damage region and elasticity degradation -- computing the Dynamic J-Integral with a sharp crack -- LOW RES 0.5mm dx
    if (testcase == 2009) {
        
        //Fibrin Parameters from Tutwiler2020
        // fracture toughness,          Gc = 7.6 +/- 0.45 J/m^2
        // folded state stiffness,      cf = 4.4e4 N/m^2
        // unfolded state stiffness,    cu = 2.6e6 N/m^2
        // fibrinogen density,          rho = 1395 g/cm^3 = 1,395,000 kg/m^3 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3044599/

        using T = double;
        static const int dim = 2;
        MPM::CRAMPSimulator<T, dim> sim("output/2009_SENT_damageRegionWithElastDeg_dx0.5mm_sigmaA_2600_FCR_ramp4s_APIC_FullDynamicJIntegral_usingTensorTransfer_halfDxDown_computeJ");

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        bool useDisplacement = false;
        sim.dx = 0.5e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 150;
        //sim.frame_dt = 22e-6 / sim.end_frame; //total time = 22e-6 s, want 1000 frames of this
        sim.frame_dt = 1e-1; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = true;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = true;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
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
        int ppc = 4;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        sim.sampleGridAlignedBox(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho, true);

        //Add Crack
        T damageRadius = sim.dx / 2.0;
        T crackLength = 5e-3;
        T crackY = y1 + height/2.0 - (sim.dx/2.0);
        T crackX = x1;
        Vector<T, dim> damageRegionMin(crackX, crackY - damageRadius);
        Vector<T, dim> damageRegionMax(crackX + crackLength, crackY + damageRadius);
        sim.addRectangularDamageRegion(damageRegionMin, damageRegionMax);

        //ADd crack segments for sharp J-integral
        T crackSegmentLength = sim.dx / 5.0;
        sim.addHorizontalCrack(Vector<T,dim>(crackX, crackY), Vector<T,dim>(crackX + crackLength, crackY), crackSegmentLength, 0.0);

        //Add Boundary Condition
        T sigmaA = 2600; //1000 times smaller than E
        T rampTime = sim.frame_dt * 40; // ramp up 4 seconds
        sim.addMode1Loading(y2, y1, sigmaA, rampTime, true, width, x1, x2); //if doing nodal loading, pass y1, y2, x1, x2 as the exact min and max of the material!

        // T simpleDampFactor = 0.5;
        // T simpleDampStartTime = sim.frame_dt * 500; //start damping once we reach the full load (rampTime over)
        // T simpleDampDuration = sim.frame_dt * 500; //for 1500 frames, damp
        // sim.addSimpleDamping(simpleDampFactor, simpleDampStartTime, simpleDampDuration);
        
        //Add Elasticity Degradation
        sim.elasticityDegradationType = 1;

        //Add Energy Tracking
        T energyDt = sim.frame_dt;
        sim.addEnergyTracking(energyDt);

        //Add Contours
        
        //DX = 0.5mm
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,15,5,15), true); //centered on crack tip
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,15,10,15), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,15,15,15), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,15,20,15), true); //centered on crack tip
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,15,25,15), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,15,30,15), true);  

        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,25,5,25), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,25,10,25), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,25,15,25), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,25,20,25), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,25,25,25), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(5,25,30,25), true); 
        
        //Add timing for contours (NOTE: without this we wont calculate anything!)
        std::vector<T> contourTimes;
        contourTimes.push_back(sim.frame_dt * 1);
        contourTimes.push_back(sim.frame_dt * 40);
        contourTimes.push_back(sim.frame_dt * 45);
        contourTimes.push_back(sim.frame_dt * 50);
        contourTimes.push_back(sim.frame_dt * 55);
        contourTimes.push_back(sim.frame_dt * 60);
        contourTimes.push_back(sim.frame_dt * 65);
        contourTimes.push_back(sim.frame_dt * 70);
        contourTimes.push_back(sim.frame_dt * 75);
        contourTimes.push_back(sim.frame_dt * 80);
        contourTimes.push_back(sim.frame_dt * 85);
        contourTimes.push_back(sim.frame_dt * 90);
        contourTimes.push_back(sim.frame_dt * 95);
        contourTimes.push_back(sim.frame_dt * 100);
        contourTimes.push_back(sim.frame_dt * 105);
        contourTimes.push_back(sim.frame_dt * 110);
        contourTimes.push_back(sim.frame_dt * 115);
        contourTimes.push_back(sim.frame_dt * 120);
        contourTimes.push_back(sim.frame_dt * 125);
        contourTimes.push_back(sim.frame_dt * 130);
        contourTimes.push_back(sim.frame_dt * 135);
        contourTimes.push_back(sim.frame_dt * 140);
        contourTimes.push_back(sim.frame_dt * 145);
        contourTimes.push_back(sim.frame_dt * 149);
        // contourTimes.push_back(sim.frame_dt * 150);
        // contourTimes.push_back(sim.frame_dt * 155);
        // contourTimes.push_back(sim.frame_dt * 160);
        // contourTimes.push_back(sim.frame_dt * 165);
        // contourTimes.push_back(sim.frame_dt * 170);
        // contourTimes.push_back(sim.frame_dt * 175);
        // contourTimes.push_back(sim.frame_dt * 180);
        // contourTimes.push_back(sim.frame_dt * 185);
        // contourTimes.push_back(sim.frame_dt * 190);
        // contourTimes.push_back(sim.frame_dt * 195);
        // contourTimes.push_back(sim.frame_dt * 199);
        sim.addJIntegralTiming(contourTimes, useDisplacement);

        sim.run(start_frame);
    }

    //SENT with damage region and elasticity degradation -- SINGLE FIELD -- computing the Dynamic J-Integral with a sharp crack
    if (testcase == 2010) {
        
        //Fibrin Parameters from Tutwiler2020
        // fracture toughness,          Gc = 7.6 +/- 0.45 J/m^2
        // folded state stiffness,      cf = 4.4e4 N/m^2
        // unfolded state stiffness,    cu = 2.6e6 N/m^2
        // fibrinogen density,          rho = 1395 g/cm^3 = 1,395,000 kg/m^3 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3044599/

        using T = double;
        static const int dim = 2;
        MPM::CRAMPSimulator<T, dim> sim("output/2010_SENT_damageRegionWithElastDeg_singleField_dx0.1mm_sigmaA_2600_FCR_ramp4s_PIC_tensorTransfer_CENTERED");

        //material
        T E = 2.6e6;
        T nu = 0.25;
        T rho = 1395000;

        //Params
        bool useDisplacement = false;
        sim.dx = 0.1e-3; //0.5 mm --> make sure this evenly fits into the width and height
        sim.symplectic = true;
        sim.end_frame = 100;
        //sim.frame_dt = 22e-6 / sim.end_frame; //total time = 22e-6 s, want 1000 frames of this
        sim.frame_dt = 1e-1; //1e-6 -> 1000 micro seconds total duration, 1e-3 -> 1 second duration
        sim.gravity = 0;

        //Interpolation Scheme
        sim.useAPIC = false;
        sim.flipPicRatio = 0.0; //0 -> want full PIC for analyzing static configurations (this is our damping)
        
        //DFG Specific Params
        sim.st = 5.5; //5.5 good for dx = 0.2, 
        sim.useDFG = false;
        sim.fricCoeff = 0; //try making this friction coefficient 0 to prevent any friction forces, only normal contact forces
        sim.useExplicitContact = true;
        
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
        int ppc = 4;
        T height = 32e-3; //32mm
        T width = 20e-3; //20mm
        T x1 = 0.05 - width/2.0;
        T y1 = 0.05 - height/2.0;
        T x2 = x1 + width;
        T y2 = y1 + height;
        Vector<T,dim> minPoint(x1, y1);
        Vector<T,dim> maxPoint(x2, y2);
        sim.sampleGridAlignedBox(material1, minPoint, maxPoint, Vector<T, dim>(0, 0), ppc, rho, true);

        //Add Crack
        T damageRadius = sim.dx / 2.0;
        T crackLength = 5e-3;
        T crackY = y1 + height/2.0;// - (sim.dx/2.0);
        T crackX = x1;
        Vector<T, dim> damageRegionMin(crackX, crackY - damageRadius);
        Vector<T, dim> damageRegionMax(crackX + crackLength, crackY + damageRadius);
        sim.addRectangularDamageRegion(damageRegionMin, damageRegionMax);

        //ADd crack segments for sharp J-integral
        //T crackSegmentLength = sim.dx / 5.0;
        //sim.addHorizontalCrack(Vector<T,dim>(crackX, crackY), Vector<T,dim>(crackX + crackLength, crackY), crackSegmentLength, 0.0);

        //Add Boundary Condition
        T sigmaA = 2600; //1000 times smaller than E
        T rampTime = sim.frame_dt * 40; // ramp up 4 seconds
        sim.addMode1Loading(y2, y1, sigmaA, rampTime, true, width, x1, x2); //if doing nodal loading, pass y1, y2, x1, x2 as the exact min and max of the material!

        // T simpleDampFactor = 0.5;
        // T simpleDampStartTime = sim.frame_dt * 500; //start damping once we reach the full load (rampTime over)
        // T simpleDampDuration = sim.frame_dt * 500; //for 1500 frames, damp
        // sim.addSimpleDamping(simpleDampFactor, simpleDampStartTime, simpleDampDuration);
        
        //Add Elasticity Degradation
        sim.elasticityDegradationType = 1;

        //Add Energy Tracking
        T energyDt = sim.frame_dt;
        sim.addEnergyTracking(energyDt);

        //Add Contours

        //DX = 0.1mm
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,25,75), true, true); //second true is to mark this contour for additional tracking of data (J_I contributions)
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,50,75), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,75,75), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,100,75), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,125,75), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,145,75), true);
        //sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,75,150,75), true);  

        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,25,125), true, true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,50,125), true); 
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,75,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,100,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,125,125), true);
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,145,125), true);  
        //sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(25,125,150,125), true); 

        //These have different L values than the other families!
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,75,25,75), true, true);    //compare to Contour A
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,75,100,75), true);         //to Contour D
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,125,25,125), true, true);  //to Contour 1
        sim.addJIntegralContour(Vector<T,dim>(0.045, 0.05), Vector<int,4>(30,125,100,125), true);       //to Contour 4

        //Add contours that define the inverse intersections between each pair of contours (A and 1, B and 2, etc.) -> each pair has an upper and lower contour, each not containing the crack and should have J = 0
        Vector<T, dim> upperCenter(0.045, 0.06);
        Vector<T, dim> lowerCenter(0.045, 0.04);
        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 25, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 25, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 50, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 50, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 75, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 75, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 100, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 100, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 125, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 125, 25), false);

        sim.addJIntegralContour(upperCenter, Vector<int,4>(25, 25, 145, 25), false);
        sim.addJIntegralContour(lowerCenter, Vector<int,4>(25, 25, 145, 25), false);
        
        //Add timing for contours (NOTE: without this we wont calculate anything!)
        std::vector<T> contourTimes;
        contourTimes.push_back(sim.frame_dt * 1);
        contourTimes.push_back(sim.frame_dt * 40);
        contourTimes.push_back(sim.frame_dt * 45);
        contourTimes.push_back(sim.frame_dt * 50);
        contourTimes.push_back(sim.frame_dt * 55);
        contourTimes.push_back(sim.frame_dt * 60);
        contourTimes.push_back(sim.frame_dt * 65);
        contourTimes.push_back(sim.frame_dt * 70);
        contourTimes.push_back(sim.frame_dt * 75);
        contourTimes.push_back(sim.frame_dt * 80);
        contourTimes.push_back(sim.frame_dt * 85);
        contourTimes.push_back(sim.frame_dt * 90);
        contourTimes.push_back(sim.frame_dt * 95);
        contourTimes.push_back(sim.frame_dt * 99);
        sim.addJIntegralTiming(contourTimes, useDisplacement);

        sim.run(start_frame);
    }

    return 0;
}