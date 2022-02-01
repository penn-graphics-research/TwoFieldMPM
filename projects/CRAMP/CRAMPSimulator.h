#pragma once

#include <Bow/Simulator/MPM/MPMSimulator.h>
#include <Bow/Energy/MPM/ElasticityOp.h>
#include <Bow/Energy/MPM/MPMEnergies.h>
#include <Bow/Simulator/BoundaryConditionManager.h>
#include <Bow/Simulator/PhysicallyBasedSimulator.h>
#include <Bow/Utils/Serialization.h>
#include <Bow/Utils/Logging.h>
#include <Bow/Utils/FileSystem.h>
#include "CRAMPOp.h"
#include "../DFGMPM/DFGMPMOp.h"
#include "../DFGMPM/DFGMPMGrid.h"
//#include "../DFGMPM/TwoFieldHodgepodgeEnergy.h"
//#include "../DFGMPM/TwoFieldBackwardEuler.h"
#include <Bow/IO/ply.h>

namespace Bow::MPM {

template <class T, int dim>
class CRAMPSimulator : public MPMSimulator<T, dim> {
public:
    using Base = MPM::MPMSimulator<T,dim>;
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;

    T ppc = (T)(1 << dim);

    /* We have these from inheriting from MPMSimulator, use them prefixed with Base::
    Field<TV> m_X;
    Field<TV> m_V;
    Field<TM> m_C;
    std::vector<T> m_mass;
    Field<TM> stress;
    MPMGrid<T, dim> grid;
    T dx = 0.02;
    TV gravity = TV::Zero();
    bool symplectic = true;
    std::vector<std::shared_ptr<ElasticityOp<T, dim>>> elasticity_models;
    BoundaryConditionManager<T, dim> BC; */

    Field<int> m_marker; //0 = solid material particle, 1 = crack particle, 2 = top plane, 3 = bottom plane, 4 = fluid
    int crackPlane_startIdx = 0;
    int topPlane_startIdx = 0;
    int bottomPlane_startIdx = 0;
    int bottomPlane_endIdx = 0; //only need this one end idx, other end ideces defined by the next start idx
    int crackType;

    //Mode I Loading Params - pull up on nodes above y1, pull down below y2, all with total stress sigmaA
    T y1, y2;
    T sigmaA;
    T rampTime;
    T width;
    T x1, x2;

    //Additional Particle Data
    Field<T> m_vol;
    Field<T> m_mu, m_la;
    Field<TM> m_F; //def grad --> for j integral
    Field<TM> m_FSmoothed;

    //Sim Data
    std::string outputPath;
    T flipPicRatio = 0.95;
    T gravity = 0.0;
    T elapsedTime = 0.0;
    int currSubstep = 0;
    int scale = 30; //TODO: don't need this later

    //Material Data
    T st; //surfaceThreshold

    //Sim Flags
    bool verbose = false; //true = write every substep, false = write only frames
    bool writeGrid = false;
    bool useAPIC = false;
    bool useDFG = false;
    bool useExplicitContact = true;
    bool useImplicitContact = false;
    bool useImpulse = false;
    bool crackInitialized = false;
    bool loading = false;
    bool nodalLoading = false;
    bool particlesMarkedForLoading = false;
    int damageType = 0; //0 = none, 1 = Rankine, 2 = AnisoMPM, 3 = tanh
    int elasticityDegradationType = 0; //0 = none, 1 = simpleLinearTension
    
    //Particle Data
    Field<TM> m_cauchy; //for anisoMPM
    Field<TM> m_cauchySmoothed; //smoothed particle Cauchy stress
    std::vector<T> Dp; //particle damage
    std::vector<T> damageLaplacians; //particle damage
    std::vector<T> dTildeH; //particle damage
    std::vector<T> sigmaC; //particle damage threshold
    std::vector<int> sp; //surface particle or not
    Field<TV> particleDG; //particle damage gradients
    Field<std::vector<int>> particleAF; //store which activefield each particle belongs to for the 3^d grid nodes it maps to
    Field<std::vector<int>> p_cached_idx; //store which DOF index each particle maps to for each of the 3^dim nodes

    //Data for Stress Snapshot and J Integral
    bool takeStressSnapshot = false;
    bool computeJIntegral = false;
    Field<T> m_sigmaYY; //used for stress ahead of crack tip
    Field<T> m_r;
    Field<T> m_posX;
    Field<int> m_idx;
    T stressSnapshotTime = 0;
    T halfEnvelope = 0;
    std::vector<Vector<int,4>> contourRadii; //holds contours defined by 4 integers: L, M, N, O (L left of center, D down from center, R right of center, U up from center)
    std::vector<Vector<T,dim>> contourCenters; //center points of the contours
    std::vector<bool> contourTypes; //contains true if contour contains the crack tip, false if not
    std::vector<T> contourTimes; //hold the times to take the contours
    int contourIdx = 0;
    std::vector<std::vector<T>> contourData; //holds a vector of vectors, each vector is the set of computed contour values for a given time

    //Data for Simple Damping
    bool useSimpleDamping = true;
    T simpleDampingFactor = 0.5;
    T simpleDampingStartTime = 0.0;
    T simpleDampingEndTime = 0.0;

    SERIALIZATION_REGISTER(m_cauchy)
    SERIALIZATION_REGISTER(Dp)
    SERIALIZATION_REGISTER(damageLaplacians)
    SERIALIZATION_REGISTER(dTildeH)
    SERIALIZATION_REGISTER(sigmaC)
    SERIALIZATION_REGISTER(sp)
    SERIALIZATION_REGISTER(particleDG)

    //Regular Grid Data
    Bow::DFGMPM::DFGMPMGrid<T, dim> grid;

    //Spatial Hash Data (use same grid for spatial hash and for sim)
    T rp;
    Bow::Field<std::vector<int>> particleNeighbors;

    //DFGMPM Params
    T minDp = 1.0;
    T dMin = 0.25;
    T fricCoeff = 0.2; //friction coefficient for contact forces
    Field<bool> m_useDamage;
    Field<T> m_lamMax;

    //Rankine Damage Params
    T Gf = 0.0;
    std::vector<T> Hs;
    Field<TM> m_scaledCauchy; //scaled cauchy stresses

    //Elasticity Degradation Params
    T degAlpha = 1.0; //exponent for simple linear tension

    //AnisoMPM Params
    T eta = 0.01;
    T sigmaCRef = -1.0;
    T percentStretch = -1.0;
    T zeta = 1.0;
    T l0;

    //Tanh Damage Params
    T lamC = 0.0;
    T tanhWidth = 0.0;

    //Impulse Data
    TV impulseCenter = TV::Zero();
    T impulseStrength = 0;
    T impulseStartTime = 0;
    T impulseDuration = 0;

    //Grid Data to Save and Vis
    Field<TV> activeNodesX;
    Field<TM> activeNodesCauchy1;
    Field<TM> activeNodesCauchy2;
    Field<TM> activeNodesFi1;
    Field<TM> activeNodesFi2;
    Field<TV> activeNodesDG;
    Field<TV> activeNodesV1;
    Field<TV> activeNodesV2;
    Field<TV> activeNodesFct1;
    Field<TV> activeNodesFct2;
    Field<TV> activeNodesN1;
    Field<TV> activeNodesN2;
    std::vector<T> activeNodesM1;
    std::vector<T> activeNodesM2;
    std::vector<T> activeNodesSeparability1;
    std::vector<T> activeNodesSeparability2;
    std::vector<int> activeNodesSeparable;

    CRAMPSimulator(std::string _outputPath){
        //This ensures we capture everything in the log file
        outputPath = _outputPath;
        Bow::FileSystem::create_path(outputPath);
        Bow::Logging::new_logger(outputPath + "/log.txt", Bow::Logging::Info, true);
    }

    /* Initialization Routines */
    void initialize()
    {
        //This routine gets called once by the base class to initialize sim!

        Bow::Logging::info("Simulation starts with ", std::is_same<T, double>::value ? "double" : "float", " ", dim);

        //Collect mu and lambda
        for (auto& model : Base::elasticity_models){
            m_mu = model->m_mu;
            m_la = model->m_lambda;
        }

        if(damageType == 2){
            l0 = 0.5 * Base::dx;
        }
        
        if(useDFG){
            if constexpr (dim == 2) {
                rp = sqrt(2.0 * Base::dx * Base::dx);
            }
            else if constexpr (dim == 3) {
                rp = sqrt(3.0 * Base::dx * Base::dx);
            }

            //Initialize particle neighbor lists
            for (size_t i = 0; i < Base::m_X.size(); ++i) {
                std::vector<int> placeholder, placeholder2, placeholder3;
                particleNeighbors.push_back(placeholder);
                particleAF.push_back(placeholder2);
                p_cached_idx.push_back(placeholder3);
            }     
        }   

        //Initialize sigmaC if we are using damage
        if(damageType == 1 || damageType == 2){ //for Rankine and AnisoMPM damage

            if(percentStretch > 0){
                for (auto& model : Base::elasticity_models)
                    model->compute_criticalStress(percentStretch, Base::stress); //use stress as a dummy structure here to hold stretchedCauchy
                tbb::parallel_for(size_t(0), sigmaC.size(), [&](size_t i) {
                    Eigen::EigenSolver<TM> es(Base::stress[i]);
                    T maxVal = 0;
                    for(int i = 0; i < dim; ++i){
                        maxVal = std::max(maxVal, es.eigenvalues()(i).real());
                    }
                    sigmaC[i] = maxVal;
                });
                Bow::Logging::info("[Damage] Stretched SigmaC: ", sigmaC[0]);
            }
            else{
                //set sigmaC directly from sigmaCRef
                tbb::parallel_for(size_t(0), sigmaC.size(), [&](size_t i) {
                    sigmaC[i] = sigmaCRef;
                });
                Bow::Logging::info("[Damage] Directly Set SigmaC: ", sigmaC[0]);
            }
        }

        if(damageType == 1){ //for Rankine damage only
            for(int i = 0; i < (int)Base::m_mass.size(); ++i){
                T mu = m_mu[i];
                T la = m_la[i];
                T E = (mu*(3*la + 2*mu)) / (la + mu); //recompute E for this particle
                T HsBar = (sigmaC[i] * sigmaC[i]) / (2 * E * Gf);
                T HsRegularized = (HsBar * l0) / (1 - (HsBar*l0));
                Hs.push_back(HsRegularized);
            }
            Bow::Logging::info("[Rankine Damage] Computed Hs: ", Hs[0]);
        }
    }

    //DFG specific routines (partitioning)
    void partitioningRoutines(){
        //First sort particles into a grid with dx = rp
        grid.sortParticles(Base::m_X, rp);

        Bow::DFGMPM::BackGridSortOp<T, dim> backGrid_sort{ {}, Base::m_X, grid, rp };
        backGrid_sort(); //Sort particles into spatial hash grid
        
        Bow::DFGMPM::NeighborSortOp<T, dim> neighbor_sort{ {}, Base::m_X, particleNeighbors, grid, rp };
        neighbor_sort(); //Create neighbor list for each particle

        //Now, with particle neighbor lists in hand, we need to resort into a grid with dx = dx
        grid.sortParticles(Base::m_X, Base::dx);

        //Surface Detection -> only on first substep
        if (elapsedTime == 0.0) {
            Bow::DFGMPM::SurfaceDetectionOp<T, dim> surface_detection{ {}, Base::m_X, particleNeighbors, rp, st, sp, grid };
            surface_detection(); //Detect surface particles on first substep
        }
        
        //Damage Routines (Rankine or Tanh)
        if (damageType == 1) { //Rankine
            //NOTE: we can put rankine damage here because, unlike AnisoMPM damage, we update this BEFORE computing DGs!
            Bow::CRAMP::UpdateRankineDamageOp<T, dim> update_rankine{ {}, m_cauchy, Dp, grid, sigmaC, Hs, m_useDamage};
            update_rankine();
        }
        else if(damageType == 3){ //tanh damage
            Bow::CRAMP::UpdateTanhDamageOp<T,dim> update_tanh{ {}, m_F, Dp, grid, lamC, tanhWidth, m_useDamage, m_lamMax };
            update_tanh();
        }

        Bow::DFGMPM::ComputeDamageGradientsOp<T, dim> compute_DGs{ {}, Base::m_X, particleNeighbors, rp, Base::dx, particleDG, Dp, sp, grid };
        compute_DGs(); //Compute particle damage gradients

        Bow::DFGMPM::PartitioningOp<T, dim> partition{ {}, Base::m_X, Base::m_mass, particleDG, particleAF, Dp, sp, Base::dx, minDp, dMin, grid };
        partition(); //Partition particles into their fields, transfer mass to those fields, and compute node separability
    }

    //AnisoMPM Routines
    void anisoMPMDamage(T dt){
        Bow::DFGMPM::ComputeDamageLaplaciansOp<T, dim> compute_damageLaplacians{ {}, Base::m_X, Dp, damageLaplacians, particleAF, Base::dx, grid };
        compute_damageLaplacians();

        Bow::DFGMPM::UpdateAnisoMPMDamageOp<T, dim> update_anisoMPM_damage{ {}, Dp, damageLaplacians, dTildeH, sigmaC, m_cauchy, dt, eta, zeta, l0, grid };
        update_anisoMPM_damage();
    }

    //Compute forces, P2G, and grid update --> all depends on symplectic or not
    void p2g(T dt){
        if(!useDFG){
            //if single field MPM, sort particles before P2G (for two field we already did this)
            Bow::Logging::info("Starting sorting particles for single-field MPM...");
            grid.sortParticles(Base::m_X, Base::dx);
            Bow::Logging::info("Finished sorting particles for single-field MPM...");
        }
        if(Base::symplectic){
            //Now compute forces for P2G and grid update (ONLY FOR SYMPLECTIC)
            for (auto& model : Base::elasticity_models){
                model->compute_stress(Base::stress);
                m_vol = model->m_vol;
            }
        }
        //Notice that this P2G is from CRAMPOp.h
        Bow::CRAMP::ParticlesToGridOp<T, dim> P2G{ {}, Base::m_X, Base::m_V, Base::m_mass, Base::m_C, Base::stress, gravity, particleAF, grid, Base::dx, dt, Base::symplectic, useDFG, useAPIC, useImplicitContact, elasticityDegradationType, m_vol, m_scaledCauchy };
        P2G();
    }

    //Symplectic: Collision Updates; Implicit: Collisions + Implicit Update
    void gridUpdate(T dt){
        if(Base::symplectic){
            //Timing Updates to Boundary Conditions
            Base::BC.update(dt);
            Base::BC.timingUpdates(elapsedTime);

            //Boundary Collisions
            Bow::DFGMPM::BoundaryConditionUpdateOp<T, dim> bc_update{ {}, grid, Base::BC, Base::dx, dt };
            bc_update();
        }
        // else{
        //     Bow::DFGMPM::ImplicitBoundaryConditionUpdateOp<T, dim> bc_update{ {}, grid, Base::BC, Base::BC_basis, Base::BC_order, Base::dx, useDFG };
        //     bc_update();
        //     Bow::DFGMPM::TwoFieldHodgepodgeEnergy<T, dim> energy(grid, gravity, Base::dx, dt, Base::m_X, Base::elasticity_models, particleAF, p_cached_idx, useDFG, useImplicitContact);
        //     Bow::DFGMPM::TwoFieldBackwardEulerUpdateOp<T, dim, int> implicit_mpm(grid, Base::BC_basis, Base::BC_order, Base::dx, dt, useDFG);
        //     implicit_mpm.m_energy_terms.push_back(&energy);
        //     if (Base::backward_euler) {
        //         energy.tsMethod = BE;
        //         implicit_mpm.tsMethod = BE;
        //     }
        //     else {
        //         energy.tsMethod = NM;
        //         implicit_mpm.tsMethod = NM;
        //     }
        //     implicit_mpm.tol = Base::newton_tol;
        //     implicit_mpm.gravity = TV::Unit(1) * gravity;

        //     //implicit_mpm.diff_test_with_matrix();
        //     implicit_mpm();
        // }
    }

    void g2p(T dt){
        Bow::DFGMPM::GridToParticlesOp<T, dim> G2P{ {}, Base::m_X, Base::m_V, Base::m_C, particleAF, grid, Base::dx, dt, flipPicRatio, useDFG };
        G2P(useAPIC); //P2G

        //Now evolve strain (updateF)
        for (auto& model : Base::elasticity_models){
            model->evolve_strain(G2P.m_gradXp);
        }

        //Now project strain (plasticity)
        for (auto& model : Base::plasticity_models)
            model->project_strain();
    }

    /* Write our own advance function to override*/
    void advance(T dt) override
    {

        std::cout << "Begin advance with dt= " << dt << std::endl;

        //Always collect cauchy and F for each particle for analysis
        for (auto& model : Base::elasticity_models){
            model->compute_cauchy(m_cauchy); //we also use this for anisoMPM damage --> do not take out unless replace it in AnisoMPM damage
            m_F = model->m_F;
            m_FSmoothed = model->m_F; //dummy values to set up the right size
            m_cauchySmoothed = model->m_F; //dummy vals
            m_scaledCauchy = model->m_F; //dummy vals
        }

        if((int)m_F.size() > 0){
            //Now let's compute the maximum stretch for each particle
            Bow::CRAMP::ComputeLamMaxOp<T,dim>computeLamMax{ {}, grid, m_F, m_lamMax };
            computeLamMax();
        }

        std::cout << "Finished collecting F, Cauchy, and lamMax..." << std::endl;

        if(useDFG) {
            //DFG specific routines (partitioning)
            partitioningRoutines();
            
            std::cout << "Partitioned..." << std::endl;

            //AnisoMPM Routines
            if(damageType == 2) {
                anisoMPMDamage(dt); //note that we simply update and track damage, there is no elasticity deg
                std::cout << "AnisoMPM Damage Updated..." << std::endl;
            }
        }
        //if no DFG but we still want to use damage
        else{ 
            //Damage Routines (Rankine or Tanh)
            if (damageType == 1) { //Rankine
                Bow::CRAMP::UpdateRankineDamageOp<T, dim> update_rankine{ {}, m_cauchy, Dp, grid, sigmaC, Hs, m_useDamage};
                update_rankine();
            }
            else if(damageType == 3){ //tanh damage
                Bow::CRAMP::UpdateTanhDamageOp<T,dim> update_tanh{ {}, m_F, Dp, grid, lamC, tanhWidth, m_useDamage, m_lamMax };
                update_tanh();
            }
        }

        //Compute Scaled Stress from Elasticity Degradation
        if(elasticityDegradationType == 1){
            Bow::CRAMP::SimpleLinearTensionElasticityDegOp<T,dim>linearTensionDegradation{ {}, m_cauchy, m_scaledCauchy, Dp, degAlpha, grid };
            linearTensionDegradation();
        }

        p2g(dt); //compute forces, p2g transfer

        std::cout << "P2G Done..." << std::endl;

        //Now transfer cauchy and F to the grid (requires grid masses, so, after P2G)
        if(elasticityDegradationType == 1 && ((int)m_F.size() > 0)){
            Bow::CRAMP::TensorP2GOp<T,dim>tensorP2G{ {}, Base::m_X, Base::m_mass, m_scaledCauchy, m_F, particleAF, grid, Base::dx, useDFG, m_marker }; //use scaledCauchy if we are using RankineDamage
            tensorP2G();
            std::cout << "Tensor P2G Done (Scaled)..." << std::endl;
        }
        else if((int)m_F.size() > 0){
            Bow::CRAMP::TensorP2GOp<T,dim>tensorP2G{ {}, Base::m_X, Base::m_mass, m_cauchy, m_F, particleAF, grid, Base::dx, useDFG, m_marker }; //use regular Cauchy stress otherwise
            tensorP2G();
            std::cout << "Tensor P2G Done (Unscaled)..." << std::endl;
        }
        
        if((int)m_F.size() > 0){
            Bow::CRAMP::TensorG2POp<T,dim>tensorG2P{ {}, Base::m_X, m_cauchySmoothed, m_FSmoothed, particleAF, grid, Base::dx, useDFG, m_marker };
            tensorG2P();
            std::cout << "Tensor G2P Done..." << std::endl;
        }
        
        //Now take our stress snapshot (if we have one, and it's the right time)
        if(takeStressSnapshot && elapsedTime >= stressSnapshotTime){

            takeStressSnapshot = false; //for now only take one snapshot
            Vector<T,dim> crackTip = Base::m_X[topPlane_startIdx - 1]; //crack tip should be last massless particle before topPlane particles
            Bow::CRAMP::StressSnapshotOp<T,dim>stressSnapshot{ {}, Base::m_X, crackTip, m_cauchy, grid, Base::dx, m_sigmaYY, m_r, m_posX, m_idx, halfEnvelope };
            stressSnapshot();
            writeStressSnapshot(elapsedTime);

            std::cout << "Stress Snapshot Computed..." << std::endl;

        }

        //Now compute all J-integral contours for the next time stamp (if it's time) (e.g. t = 0.5s, 0.7s, ...)
        if(computeJIntegral && elapsedTime >= contourTimes[contourIdx]){

            //Mark Jintegral computation as finished if necessary
            contourIdx++;
            if(contourIdx >= (int)contourTimes.size()){
                computeJIntegral = false;
            }

            std::vector<T> contourValues; //empty vector to hold a value for each contour at this time

            //For this time, we will compute the J integral using however many contour radii the user asks for
            std::string jIntFilePath = outputPath + "/JIntegralData" + std::to_string(elapsedTime) + ".txt";
            std::ofstream jIntFile(jIntFilePath);
            Bow::CRAMP::ComputeJIntegralOp<T,dim>computeJIntegralOp{ {}, Base::m_X, topPlane_startIdx, bottomPlane_startIdx, m_cauchy, grid, Base::dx, dt, m_mu[0], m_la[0], useDFG };
            for(int i = 0; i < (int)contourRadii.size(); ++i){
                T J_I = 0;
                J_I = computeJIntegralOp(contourCenters[i], contourRadii[i], contourTypes[i], jIntFile);
                contourValues.push_back(J_I);
            }
            jIntFile.close();
            contourData.push_back(contourValues);

            std::cout << "J Integral Computed At t = " << elapsedTime << std::endl;

            //Now write a data file with all contour values across all times if we are done computing them
            if(!computeJIntegral){
                std::string jIntFilePath = outputPath + "/JIntegralData_COMPLETE.txt";
                std::ofstream file(jIntFilePath);
                file << "=====Complete Computed J-Integral Data=====\n";
                for(int i = 0; i < (int)contourTimes.size(); ++i){
                    file << "Time = " << contourTimes[i] << " s: ";
                    for(int j = 0; j < (int)contourValues.size(); ++j){
                        file << contourData[i][j] << ", ";
                    }
                    file << "\n";
                }
                file.close();
            }
        }

        //If Loading this specimen:
        if(loading){
            //If we've not yet marked particles for loading, do so!
            if(!particlesMarkedForLoading){
                Bow::CRAMP::MarkParticlesForLoadingOp<T, dim> markParticles{ {}, Base::m_X, m_marker, y1, y2, grid };
                markParticles();
                particlesMarkedForLoading = true;
            }
            
            //Pass the right portion of sigmaA to the loading (based on the user defined rampTime)
            T scaledSigmaA = sigmaA;
            if(elapsedTime < rampTime && rampTime > 0.0){
                scaledSigmaA *= (elapsedTime / rampTime);
            }
            Bow::CRAMP::ApplyMode1LoadingOp<T, dim> mode1Loading{ {}, Base::m_X, m_marker, scaledSigmaA, nodalLoading, width, y1, y2, x1, x2, Base::dx, dt, grid, m_vol, ppc };
            mode1Loading();

            std::cout << "Mode 1 Loading Applied..." << std::endl;
        }

        //Apply Impulse (if user added one) -> apply directly for symplectic, save forces for later if implicit
        if(useImpulse){
            if((elapsedTime >= impulseStartTime) && (elapsedTime < impulseStartTime + impulseDuration)){
                Bow::DFGMPM::ApplyImpulseOp<T, dim> apply_impulse{ {}, impulseCenter, impulseStrength, grid, Base::dx, dt, Base::symplectic, useImpulse };
                apply_impulse();
                std::cout << "Impulse Applied..." << std::endl;
            }
        }

        //Frictional Contact -> apply directly for symplectic, for implicit we compute normals here (but only if we want implicit contact)
        if (useDFG && ((Base::symplectic && useExplicitContact) || (!Base::symplectic && useImplicitContact))) {
            Bow::DFGMPM::ContactForcesOp<T, dim> frictional_contact{ {}, dt, fricCoeff, Base::symplectic, useImplicitContact, grid };
            frictional_contact();
            std::cout << "Frictional Contact Applied..." << std::endl;
        }

        gridUpdate(dt); //collisions + implicit grid update

        std::cout << "Grid Updated..." << std::endl;

        //Explicit Frictional Contact -> ONLY for implicit two field with EXPLICIT frictional contact
        if (useDFG && (!Base::symplectic && !useImplicitContact)) {
            Bow::DFGMPM::ContactForcesOp<T, dim> frictional_contact{ {}, dt, fricCoeff, Base::symplectic, useImplicitContact, grid };
            frictional_contact();
            std::cout << "Applied Explicit Frictional Contact..." << std::endl;
        }

        g2p(dt); //transfer, updateF, and plastic projection

        std::cout << "G2P done..." << std::endl;

        //Now damp particle velocities if we want damping
        if(useSimpleDamping && (elapsedTime < simpleDampingEndTime) && (elapsedTime > simpleDampingStartTime)){
            //Use simple damping
            Bow::CRAMP::SimpleDampingOp<T, dim> applySimpleDamping{ {}, Base::m_V, simpleDampingFactor, grid };
            applySimpleDamping();
            std::cout << "Simple Damping Applied..." << std::endl;

        }

        if(crackInitialized){
            Bow::CRAMP::EvolveCrackPlanesOp<T, dim> evolve_cracks{ {}, Base::m_X, Base::m_V, topPlane_startIdx, bottomPlane_startIdx, grid, Base::dx, dt, flipPicRatio, useAPIC, crackType };
            evolve_cracks();
            std::cout << "Evolved cracks..." << std::endl;
        }

        //Helpful timers and counters
        elapsedTime += dt;
        currSubstep++;

        //Now dump substep data if verbose is active (verbose = write every substep)
        if(verbose){
            BOW_TIMER_FLAG("writeSubstep");
            
            IO::writeTwoField_particles_ply(outputPath + "/p" + std::to_string(currSubstep) + ".ply", Base::m_X, Base::m_V, particleDG, Base::m_mass, Dp, sp, m_marker, m_cauchySmoothed, m_FSmoothed, m_lamMax);

            std::cout << "Substep Written..." << std::endl;

            //Write Grid
            if(writeGrid){
                Bow::DFGMPM::CollectGridDataOp<T, dim> collect_gridData{ {}, grid, Base::dx, activeNodesX, activeNodesCauchy1, activeNodesCauchy2, activeNodesFi1, activeNodesFi2, activeNodesDG, activeNodesV1, activeNodesV2, activeNodesFct1, activeNodesFct2, activeNodesM1, activeNodesM2, activeNodesSeparability1, activeNodesSeparability2, activeNodesSeparable, activeNodesN1, activeNodesN2 };
                collect_gridData();
                IO::writeTwoField_nodes_ply(outputPath + "/i" + std::to_string(currSubstep) + ".ply", activeNodesX, activeNodesCauchy1, activeNodesCauchy2, activeNodesFi1, activeNodesFi2, activeNodesDG, activeNodesV1, activeNodesV2, activeNodesFct1, activeNodesFct2, activeNodesM1, activeNodesM2, activeNodesSeparability1, activeNodesSeparability2, activeNodesSeparable, activeNodesN1, activeNodesN2);
            }
        }
    }

    // virtual T calculate_dt()
    // {
    //     T max_speed = 0;
    //     for (int i = 0; i < (int)Base::m_X.size(); ++i) {
    //         max_speed = Base::m_V[i].norm() > max_speed ? Base::m_V[i].norm() : max_speed;
    //     }
    //     return Base::cfl * Base::dx / (max_speed + 1e-10);
    // }

    //------------ADDING TO SIM--------------

    //Setup sim for AnisoMPM Damage -- NOTE: if you want to set sigmaC directly, pass p < 0 and your sigmaC
    void addAnisoMPMDamage(T _eta, T _dMin, T _zeta, T _p = -1.0, T _sigmaC = -1.0)
    {
        assert(_p > 0 ^ _sigmaC > 0); //assert that exactly one of these is set
        assert(damageType == 0); //if we've already added a damage model we can't add another!
        Bow::Logging::info("[AnisoMPM Damage] Simulating with AnisoMPM Damage");
        damageType = 2;
        eta = _eta;
        dMin = _dMin;
        sigmaCRef = _sigmaC;
        percentStretch = _p;
        zeta = _zeta;
        if (_p > 0) {
            Bow::Logging::info("[AnisoMPM Damage] Percent Stretch: ", percentStretch);
        }
        else {
            Bow::Logging::info("[AnisoMPM Damage] SigmaC: ", sigmaCRef);
        }
        Bow::Logging::info("[AnisoMPM Damage] Eta: ", eta);
        Bow::Logging::info("[AnisoMPM Damage] Zeta: ", zeta);
        Bow::Logging::info("[AnisoMPM Damage] dMin: ", dMin);
    }

    //Setup sim for Rankine damage (from Homel 2016)
    void addRankineDamage(T _dMin, T _Gf, T _l0, int _degType, T _p = -1.0, T _sigmaC = -1.0){
        assert(_p > 0 ^ _sigmaC > 0); //assert that exactly one of these is set
        assert(damageType == 0); //if we've already added a damage model we can't add another!
        Bow::Logging::info("[Rankine Damage] Simulating with Rankine Damage");
        damageType = 1;
        dMin = _dMin;
        Gf = _Gf;
        l0 = _l0;
        sigmaCRef = _sigmaC;
        percentStretch = _p;
        elasticityDegradationType = _degType;
        if (_p > 0) {
            Bow::Logging::info("[Rankine Damage] Percent Stretch: ", percentStretch);
        }
        else {
            Bow::Logging::info("[Rankine Damage] SigmaC: ", sigmaCRef);
        }
        Bow::Logging::info("[Rankine Damage] Gf: ", Gf);
        Bow::Logging::info("[Rankine Damage] dMin: ", dMin);
    }

    //Setup sim for Hyperbolic Tangent Damage
    void addHyperbolicTangentDamage(T _lamC, T _tanhWidth, T _dMin, int _degType){
        assert(damageType == 0); //if we've already added a damage model we can't add another!
        Bow::Logging::info("[Hyperbolic Tangent Damage] Simulating with Hyperbolic Tangent Damage");
        damageType = 3;
        dMin = _dMin;
        lamC = _lamC;
        tanhWidth = _tanhWidth;
        elasticityDegradationType = _degType;
    }

    //Setup sim for an impulse of user defined strength and duration
    void addImpulse(TV _center, T _strength, T _startTime, T _duration)
    {
        BOW_ASSERT(0); //NOTE: disabling this for now just in case
        useImpulse = true;
        impulseCenter = _center;
        impulseStrength = _strength;
        impulseStartTime = _startTime;
        impulseDuration = _duration;
    }

    //Setup a mode I constant loading for the configuration, we pull up on nodes above y1, and pull down on nodes below y2, each with total stress sigmaA.
    void addMode1Loading(T _y1, T _y2, T _sigmaA, T _rampTime, bool _nodalLoading, T _width = 0, T _x1 = 0, T _x2 = 0)
    {
        if(_nodalLoading){
            BOW_ASSERT_INFO(_width > 0, "ERROR: Nodal loading requires passing the specimen width");
        }
        y1 = _y1;
        y2 = _y2;
        sigmaA = _sigmaA;
        rampTime = _rampTime;
        loading = true;
        nodalLoading = _nodalLoading;
        width = _width;
        x1 = _x1;
        x2 = _x2;
    }

    //Setup taking a snapshot of stress at a given time
    void addStressSnapshot(T _time, T _envelope){
        stressSnapshotTime = _time;
        halfEnvelope = _envelope;
        takeStressSnapshot = true;
    }

    //Add a contour for us to take the J-integral over
    //NOTE: for ALL times we will calculate ALL contours (for more elegant design)
    void addJIntegralContour(Vector<T,dim> _center, Vector<int,4> _contour, bool _containsCrackTip){
        contourRadii.push_back(_contour);
        contourCenters.push_back(_center);
        contourTypes.push_back(_containsCrackTip);
    }
    //Add times for these contours to be integrated over, ONLY CALL THIS ONCE WITH FULL LIST OF TIMES!
    void addJIntegralTiming(std::vector<T>& _times){
        BOW_ASSERT_INFO(!computeJIntegral, "ERROR: Only call addJIntegralTimes once");
        computeJIntegral = true;
        contourTimes = _times;
        contourIdx = 0;
    }

    //------------TIME STEP--------------

    //Allows a user to set dt based on symplectic CFL limit
    T suggestedDt(T _E, T _nu, T _rho, T _dx, T _cfl){
        T elasticity = std::sqrt(_E * (1 - _nu) / ((1 + _nu) * (1 - 2 * _nu) * _rho));
        return _cfl * _dx / elasticity;
    }

    //Ensure that our dt never allows a particle to move faster than one grid cell per step
    virtual T calculate_dt()
    {
        T max_speed = 0;
        for (int i = 0; i < (int)Base::m_X.size(); ++i) {
            max_speed = Base::m_V[i].norm() > max_speed ? Base::m_V[i].norm() : max_speed;
        }
        return Base::cfl * Base::dx / (max_speed + 1e-10);
    }

    //------------OUTPUT--------------

    //FRAME OUTPUT
    void dump_output(int frame_num) override
    {
        if(!frame_num || !verbose){
            BOW_TIMER_FLAG("writeFrame");
            IO::writeTwoField_particles_ply(outputPath + "/p" + std::to_string(frame_num) + ".ply", Base::m_X, Base::m_V, particleDG, Base::m_mass, Dp, sp, m_marker, m_cauchySmoothed, m_FSmoothed, m_lamMax);

            std::cout << "Frame Written (p)..." << std::endl;

            //Write Grid
            if(writeGrid){
                Bow::DFGMPM::CollectGridDataOp<T, dim> collect_gridData{ {}, grid, Base::dx, activeNodesX, activeNodesCauchy1, activeNodesCauchy2, activeNodesFi1, activeNodesFi2, activeNodesDG, activeNodesV1, activeNodesV2, activeNodesFct1, activeNodesFct2, activeNodesM1, activeNodesM2, activeNodesSeparability1, activeNodesSeparability2, activeNodesSeparable, activeNodesN1, activeNodesN2 };
                collect_gridData();
                IO::writeTwoField_nodes_ply(outputPath + "/i" + std::to_string(frame_num) + ".ply", activeNodesX, activeNodesCauchy1, activeNodesCauchy2, activeNodesFi1, activeNodesFi2, activeNodesDG, activeNodesV1, activeNodesV2, activeNodesFct1, activeNodesFct2, activeNodesM1, activeNodesM2, activeNodesSeparability1, activeNodesSeparability2, activeNodesSeparable, activeNodesN1, activeNodesN2);
                std::cout << "Frame Written (i)..." << std::endl;
            }
        }
    }

    //Stress Snapshot Output -> to CSV
    void writeStressSnapshot(T elapsedTime){
        std::string filepath = outputPath + "/stressSnapshot" + std::to_string(elapsedTime) + ".csv";
        std::ofstream file(filepath);
        file << "Radius,SigmaYY,X Dist From Crack Tip,Idx\n";
        for(unsigned int i = 0; i < m_r.size(); i++){
            file << std::to_string(m_r[i]);
            file << ",";
            file << std::to_string(m_sigmaYY[i]);
            file << ",";
            file << std::to_string(m_posX[i]);
            file << ",";
            file << std::to_string(m_idx[i]);
            file << "\n";
        }
        file.close();
    }

    //Add simple damping routine (multiply particle velocities after G2P by dampFactor)
    void addSimpleDamping(T _factor, T _startTime, T _duration){
        useSimpleDamping = true;
        simpleDampingFactor = _factor;
        simpleDampingStartTime = _startTime;
        simpleDampingEndTime = _startTime + _duration;
    }

    //------------PARTICLE SAMPLING--------------

    void addParticle(TV X, TV V, T mass, T damage, int surface, int marker, bool useDamage){
        Base::m_X.push_back(X);
        Base::m_V.push_back(V);
        Base::m_C.push_back(TM::Zero());
        Base::m_mass.push_back(mass);
        Base::stress.push_back(TM::Zero());
        m_cauchy.push_back(TM::Zero());
        Dp.push_back(damage);
        damageLaplacians.push_back(0.0);
        sp.push_back(surface);
        particleDG.push_back(TV::Zero());
        dTildeH.push_back(0.0);
        sigmaC.push_back(0.0);
        m_marker.push_back(marker);
        m_useDamage.push_back(useDamage);
        m_lamMax.push_back(0.0);
    }

    void sampleRandomCube(std::shared_ptr<ElasticityOp<T, dim>> model, const TV& min_corner, const TV& max_corner, const TV& velocity = TV::Zero(), int _ppc = 4, T density = 1000., bool useDamage = false, int marker = 0)
    {
        // sample particles
        ppc = (T)_ppc; //set sim ppc
        T vol = std::pow(Base::dx, dim) / T(_ppc);
        T interval = Base::dx / std::pow(_ppc, (T)1 / dim);
        Vector<int, dim> region = ((max_corner - min_corner) / interval).template cast<int>();
        printf("%d %d %d\n", region(0), region(1), region(2));
        int start = Base::m_X.size();
        iterateRegion(region, [&](const Vector<int, dim>& offset) {
            TV position = min_corner + offset.template cast<T>() * interval;
            position += TV::Ones() * 0.5 * interval + TV::Random() * 0.5 * interval;
            addParticle(position, velocity, density*vol, 0.0, 0, marker, useDamage);
        });
        int end = Base::m_X.size();
        model->append(start, end, vol);
    }

    void samplePrecutRandomCube(std::shared_ptr<ElasticityOp<T, dim>> model, const TV& min_corner, const TV& max_corner, const TV& velocity = TV::Zero(), T density = 1000., bool useDamage = false, int marker = 0)
    {
        // sample particles
        //T vol = dim == 2 ? Base::dx * Base::dx / 4 : Base::dx * Base::dx * Base::dx / 8;
        T vol = std::pow(Base::dx, dim) / (T)ppc;
        T interval = Base::dx / std::pow((T)ppc, (T)1 / dim);
        Vector<int, dim> region = ((max_corner - min_corner) / interval).template cast<int>();
        printf("%d %d\n", region(0), region(1));
        int start = Base::m_X.size();
        iterateRegion(region, [&](const Vector<int, dim>& offset) {
            TV position = min_corner + offset.template cast<T>() * interval;
            position += TV::Ones() * 0.5 * interval + TV::Random() * 0.5 * interval;
            
            //give full damage for the cut particles
            T buffer = 0.01;
            T diffY = max_corner[1] - min_corner[1];
            T diffX = max_corner[0] - min_corner[0];
            T y1 = max_corner[1] - (diffY/3.0);
            T y2 = max_corner[1] - ((2*diffY)/3.0);
            T x1 = min_corner[0] + (0.3 * diffX);
            T x2 = max_corner[0] - (0.3 * diffX);

            T damage = 0;
            if((position[1] <= y1 + buffer) && (position[1] >= y1 - buffer) && (position[0] >= x1)){
                damage = 1.0;
            } else if ((position[1] <= y2 + buffer) && (position[1] >= y2 - buffer) && (position[0] <= x2)){
                damage = 1.0;
            } else{
                damage = 0.0;
            }

            addParticle(position, velocity, density*vol, damage, 0, marker, useDamage);
        });
        int end = Base::m_X.size();
        model->append(start, end, vol);
    }

    void sampleRandomSphere(std::shared_ptr<ElasticityOp<T, dim>> model, const TV& center, const T radius, const TV& velocity = TV::Zero(), T density = 1000., bool useDamage = false, int marker = 0)
    {
        // sample particles
        T vol = dim == 2 ? Base::dx * Base::dx / 4 : Base::dx * Base::dx * Base::dx / 8;
        T interval = Base::dx / std::pow((T)ppc, (T)1 / dim);
        TV min_corner = center;
        TV max_corner = center;
        for(int i = 0; i < dim; ++i){
            min_corner[i] -= radius;
            max_corner[i] += radius;
        }
        Vector<int, dim> region = ((max_corner - min_corner) / interval).template cast<int>();
        printf("%d %d %d\n", region(0), region(1), region(2));
        int start = Base::m_X.size();
        iterateRegion(region, [&](const Vector<int, dim>& offset) {
            TV position = min_corner + offset.template cast<T>() * interval;
            position += TV::Ones() * 0.5 * interval + TV::Random() * 0.5 * interval;
            if((position - center).norm() < radius){
                addParticle(position, velocity, density*vol, 0.0, 0, marker, useDamage);
            }
        });
        int end = Base::m_X.size();
        model->append(start, end, vol);
    }

    void sampleFromObj(std::shared_ptr<ElasticityOp<T, dim>> model, const std::string filepath, const TV& velocity, T volume, T density, bool useDamage = false, int marker = 0)
    {
        // sample particles from OBJ file
        int start = Base::m_X.size();
        
        std::ifstream fs;
        fs.open(filepath);
        BOW_ASSERT_INFO(fs, "[OBJ Sampling] Failed to open file");

        //Know vertex is always preceded by 'v' and these are the only lines we care about
        std::string line;
        TV position;
        while(std::getline(fs, line)){
            std::stringstream ss(line); 
            if(line[0] == 'v'){
                ss.ignore(128, ' '); //skip the v and get to the position
                for (size_t i = 0; i < dim; i++)
                    ss >> position[i];
                //std::cout << "Position:" << position << std::endl;
                addParticle(position, velocity, 0.0, 0.0, 0, marker, useDamage); //dummy mass passed in
            }
        }

        //now that we know total num particles, calculate particle masses
        T vol = volume / (T)Base::m_X.size();
        for(size_t i = 0; i < Base::m_X.size(); ++i){
            Base::m_mass[i] = density * vol; //actually set mass here
        }
        
        int end = Base::m_X.size();
        model->append(start, end, vol);
        fs.close();
    }

    //NOTE: This routine works best if the dimensions of the box are even multiples of the grid resolution (width = c1 * dx, height = c2 * dx)
    void sampleGridAlignedBox(std::shared_ptr<ElasticityOp<T, dim>> model, const TV& min_corner, const TV& max_corner, const TV& velocity = TV::Zero(), int _ppc = 4, T density = 1000., bool useDamage = false, int marker = 0)
    {
        BOW_ASSERT_INFO(min_corner != max_corner, "min_corner == max_corner in sampleGridAlignedBox");
        // T width = max_corner[0] - min_corner[0];
        // T height = max_corner[1] - min_corner[1];
        // T widthMod = fmod(width, Base::dx);
        // T heightMod = fmod(height, Base::dx);
        // BOW_ASSERT_INFO(widthMod == Base::dx || widthMod == 0, "width not divisible by dx in sampleGridAlignedBox");
        // BOW_ASSERT_INFO(heightMod == Base::dx || heightMod == 0, "height not divisible by dx in sampleGridAlignedBox");

        // sample particles
        ppc = (T)_ppc; //set sim ppc
        T vol = std::pow(Base::dx, dim) / T(_ppc);
        T interval = Base::dx / std::pow(_ppc, (T)1 / dim);
        Vector<int, dim> region = ((max_corner - min_corner) / interval).template cast<int>();
        region(0)++;
        // region(1)++;
        // if(dim == 3){
        //     region(2)++;
        // }
        printf("%d %d\n", region(0), region(1));
        int start = Base::m_X.size();
        T translation = interval / 2.0;
        iterateRegion(region, [&](const Vector<int, dim>& offset) {
            TV position = min_corner + offset.template cast<T>() * interval;
            position(0) += translation;
            position(1) += translation;
            if(dim == 3){
                position(2) += translation;
            } 
            
            int surface = 0;
            if(offset[0] == 0 || offset[1] == 0 || offset[0] == region[0] - 1 || offset[1] == region[1] - 1){
                surface = 1;
            }
            else{
                surface = 0;
            }
            addParticle(position, velocity, density*vol, 0.0, surface, marker, useDamage);
        });
        int end = Base::m_X.size();
        model->append(start, end, vol);
    }

    //NOTE: This routine works best if the dimensions of the box are even multiples of the grid resolution (width = c1 * dx, height = c2 * dx)
    void sampleGridAlignedBoxWithHole(std::shared_ptr<ElasticityOp<T, dim>> model, const TV& min_corner, const TV& max_corner, const TV& center, const T radius, const TV& velocity = TV::Zero(), int _ppc = 4, T density = 1000., bool useDamage = false, int marker = 0)
    {
        BOW_ASSERT_INFO(min_corner != max_corner, "min_corner == max_corner in sampleGridAlignedBox");
        // T width = max_corner[0] - min_corner[0];
        // T height = max_corner[1] - min_corner[1];
        // T widthMod = fmod(width, Base::dx);
        // T heightMod = fmod(height, Base::dx);
        // BOW_ASSERT_INFO(widthMod == Base::dx || widthMod == 0, "width not divisible by dx in sampleGridAlignedBox");
        // BOW_ASSERT_INFO(heightMod == Base::dx || heightMod == 0, "height not divisible by dx in sampleGridAlignedBox");

        // sample particles
        ppc = (T)_ppc;
        T vol = std::pow(Base::dx, dim) / T(_ppc);
        T interval = Base::dx / std::pow(_ppc, (T)1 / dim);
        Vector<int, dim> region = ((max_corner - min_corner) / interval).template cast<int>();
        region(0)++;
        // region(1)++;
        // if(dim == 3){
        //     region(2)++;
        // }
        printf("%d %d\n", region(0), region(1));
        int start = Base::m_X.size();
        T translation = interval / 2.0;
        iterateRegion(region, [&](const Vector<int, dim>& offset) {
            TV position = min_corner + offset.template cast<T>() * interval;
            position(0) += translation;
            position(1) += translation;
            if(dim == 3){
                position(2) += translation;
            }
            //Now check to make sure this is outside the desired hole
            T dist = (position - center).norm();
            if(dist > radius){ //outside hole
                int surface = 0;
                if(offset[0] == 0 || offset[1] == 0 || offset[0] == region[0] - 1 || offset[1] == region[1] - 1){
                    surface = 1;
                }
                else{
                    surface = 0;
                }
                addParticle(position, velocity, density*vol, 0.0, surface, marker, useDamage);
            }
        });
        int end = Base::m_X.size();
        model->append(start, end, vol);
    }

    //NOTE: This routine works best if the dimensions of the box are even multiples of the grid resolution (width = c1 * dx, height = c2 * dx)
    void sampleGridAlignedBoxWithNotch(std::shared_ptr<ElasticityOp<T, dim>> model, const TV& min_corner, const TV& max_corner, const T length, const T radius, const T crackHeight, const bool useCircularNotch = false, const TV& velocity = TV::Zero(), int _ppc = 4, T density = 1000., bool useDamage = false, int marker = 0)
    {
        BOW_ASSERT_INFO(min_corner != max_corner, "min_corner == max_corner in sampleGridAlignedBox");
        // T width = max_corner[0] - min_corner[0];
        T height = max_corner[1] - min_corner[1];
        // T widthMod = fmod(width, Base::dx);
        // T heightMod = fmod(height, Base::dx);
        // BOW_ASSERT_INFO(widthMod == Base::dx || widthMod == 0, "width not divisible by dx in sampleGridAlignedBox");
        // BOW_ASSERT_INFO(heightMod == Base::dx || heightMod == 0, "height not divisible by dx in sampleGridAlignedBox");

        // sample particles
        ppc = (T)_ppc;
        T vol = std::pow(Base::dx, dim) / T(_ppc);
        T interval = Base::dx / std::pow(_ppc, (T)1 / dim);
        Vector<int, dim> region = ((max_corner - min_corner) / interval).template cast<int>();
        region(0)++;
        // region(1)++;
        // if(dim == 3){
        //     region(2)++;
        // }
        printf("%d %d\n", region(0), region(1));
        int start = Base::m_X.size();
        T translation = interval / 2.0;
        iterateRegion(region, [&](const Vector<int, dim>& offset) {
            TV position = min_corner + offset.template cast<T>() * interval;
            position(0) += translation;
            position(1) += translation;
            if(dim == 3){
                position(2) += translation;
            }
            //Now check to make sure this is outside the desired notch
            bool pointIncluded = true;
            TV center = min_corner;
            center(0) += length;
            center(1) += height/2.0;
            T dist = (position - center).norm();
            T yMin = crackHeight - radius;
            T yMax = crackHeight + radius;
            //T xMin = min_corner(0);
            T xMax = min_corner(0) + length;
            if(position(1) > yMin && position(1) < yMax && position(0) < xMax){ //excluded rectangle of size length * (2*radius)
                pointIncluded = false;
            }
            if(dist < radius){ //exclude points in a semi circle around the end point of the crack rectangle
                if(useCircularNotch){
                    pointIncluded = false;
                }
            }
            if(pointIncluded){ //outside hole
                int surface = 0;
                if(offset[0] == 0 || offset[1] == 0 || offset[0] == region[0] - 1 || offset[1] == region[1] - 1){
                    surface = 1;
                }
                else{
                    surface = 0;
                }
                addParticle(position, velocity, density*vol, 0.0, surface, marker, useDamage);
            }
            
        });
        int end = Base::m_X.size();
        model->append(start, end, vol);
    }

    //NOTE: This routine works best if the dimensions of the box are even multiples of the grid resolution (width = c1 * dx, height = c2 * dx)
    void sampleGridAlignedBoxWithTriangularNotch(std::shared_ptr<ElasticityOp<T, dim>> model, const TV& min_corner, const TV& max_corner, const T length, const T radius, const T crackHeight, const TV& velocity = TV::Zero(), int _ppc = 4, T density = 1000., bool useDamage = false, int marker = 0)
    {
        BOW_ASSERT_INFO(min_corner != max_corner, "min_corner == max_corner in sampleGridAlignedBox");
        // sample particles
        ppc = (T)_ppc;
        T vol = std::pow(Base::dx, dim) / T(_ppc);
        T interval = Base::dx / std::pow(_ppc, (T)1 / dim);
        Vector<int, dim> region = ((max_corner - min_corner) / interval).template cast<int>();
        region(0)++;
        // region(1)++;
        // if(dim == 3){
        //     region(2)++;
        // }
        printf("%d %d\n", region(0), region(1));
        int start = Base::m_X.size();
        T translation = interval / 2.0;
        iterateRegion(region, [&](const Vector<int, dim>& offset) {
            TV position = min_corner + offset.template cast<T>() * interval;
            position(0) += translation;
            position(1) += translation;
            if(dim == 3){
                position(2) += translation;
            }
            //Now check to make sure this is outside the desired triangular notch
            T yMin = crackHeight - radius;
            T yMax = crackHeight + radius;
            TV A(min_corner(0), yMax);
            TV B(min_corner(0) + length, crackHeight);
            TV C(min_corner(0), yMin);
            T cross1 = ((B(0)-A(0))*(position(1)-A(1)) - (B(1)-A(1))*(position(0)-A(0)));
            T cross2 = ((B(0)-C(0))*(position(1)-C(1)) - (B(1)-C(1))*(position(0)-C(0)));
            bool pointIncluded = true;
            if(cross1 < 0 && cross2 > 0 && position(0) < B(0)){ //excluded rectangle of size length * (2*radius)
                pointIncluded = false;
            }
            if(pointIncluded){
                addParticle(position, velocity, density*vol, 0.0, 0, marker, useDamage);
            }
        });
        int end = Base::m_X.size();
        model->append(start, end, vol);
    }

    void sampleGridAlignedBoxWithTriangularNotchWithPoissonDisk(std::shared_ptr<ElasticityOp<T, dim>> model, const TV& min_corner, const TV& max_corner, const T length, const T radius, const T crackHeight, const TV& velocity = TV::Zero(), int _ppc = 4, T density = 1000., bool useDamage = false, int marker = 0)
    {
        // sample particles
        ppc = (T)_ppc;
        T vol = std::pow(Base::dx, dim) / T(_ppc);
        int start = Base::m_X.size();
        Field<TV> new_samples;
        Geometry::PoissonDisk<T, dim> poisson_disk(min_corner, max_corner, Base::dx, T(_ppc));
        poisson_disk.sample(new_samples);
        for(auto position : new_samples) {
            //Now check to make sure this is outside the desired triangular notch
            T yMin = crackHeight - radius;
            T yMax = crackHeight + radius;
            TV A(min_corner(0), yMax);
            TV B(min_corner(0) + length, crackHeight);
            TV C(min_corner(0), yMin);
            T cross1 = ((B(0)-A(0))*(position(1)-A(1)) - (B(1)-A(1))*(position(0)-A(0)));
            T cross2 = ((B(0)-C(0))*(position(1)-C(1)) - (B(1)-C(1))*(position(0)-C(0)));
            bool pointIncluded = true;
            if(cross1 < 0 && cross2 > 0 && position(0) < B(0)){ //excluded rectangle of size length * (2*radius)
                pointIncluded = false;
            }
            if(pointIncluded){
                addParticle(position, velocity, density*vol, 0.0, 0, marker, useDamage);
            }
        }
        int end = Base::m_X.size();
        model->append(start, end, vol);
    }

    void sampleGridAlignedBoxWithPoissonDisk(std::shared_ptr<ElasticityOp<T, dim>> model, const TV& min_corner, const TV& max_corner, const TV& velocity = TV::Zero(), int _ppc = 4, T density = 1000., bool useDamage = false, int marker = 0){
        // sample particles
        ppc = (T)_ppc;
        T vol = std::pow(Base::dx, dim) / T(_ppc);
        int start = Base::m_X.size();
        Field<TV> new_samples;
        Geometry::PoissonDisk<T, dim> poisson_disk(min_corner, max_corner, Base::dx, T(_ppc));
        poisson_disk.sample(new_samples);
        for(auto position : new_samples){
            addParticle(position, velocity, density*vol, 0.0, 0, marker, useDamage);
        }
        int end = Base::m_X.size();
        model->append(start, end, vol);
    }

    void sampleGridAlignedBoxWithNotchWithPoissonDisk(std::shared_ptr<ElasticityOp<T, dim>> model, const TV& min_corner, const TV& max_corner, const T length, const T radius, const T crackHeight, const bool useCircularNotch = false, const TV& velocity = TV::Zero(), int _ppc = 4, T density = 1000., bool useDamage = false, int marker = 0){
        // sample particles
        T height = max_corner[1] - min_corner[1];
        ppc = (T)_ppc;
        T vol = std::pow(Base::dx, dim) / T(_ppc);
        int start = Base::m_X.size();
        Field<TV> new_samples;
        Geometry::PoissonDisk<T, dim> poisson_disk(min_corner, max_corner, Base::dx, T(_ppc));
        poisson_disk.sample(new_samples);
        for(auto position : new_samples){
            //Now check to make sure this is outside the desired notch
            bool pointIncluded = true;
            TV center = min_corner;
            center(0) += length;
            center(1) += height/2.0;
            T dist = (position - center).norm();
            T yMin = crackHeight - radius;
            T yMax = crackHeight + radius;
            //T xMin = min_corner(0);
            T xMax = min_corner(0) + length;
            if(position(1) > yMin && position(1) < yMax && position(0) < xMax){ //excluded rectangle of size length * (2*radius)
                pointIncluded = false;
            }
            if(dist < radius){ //exclude points in a semi circle around the end point of the crack rectangle
                if(useCircularNotch){
                    pointIncluded = false;
                }
            }
            if(pointIncluded){ //outside hole
                addParticle(position, velocity, density*vol, 0.0, 0, marker, useDamage);
            }
        }
        int end = Base::m_X.size();
        model->append(start, end, vol);
    }

    void sampleHemispherePoissonDisk(std::shared_ptr<ElasticityOp<T, dim>> model, const TV& center, T radius, const TV& velocity = TV::Zero(), int _ppc = 4, T density = 1000., bool useDamage = false, int marker = 0){
        // sample particles
        ppc = (T)_ppc;
        T vol = std::pow(Base::dx, dim) / T(_ppc);
        int start = Base::m_X.size();
        Field<TV> new_samples;
        TV min_corner, max_corner;
        for(int d = 0; d < dim; ++d){
            min_corner[d] = center[d] - radius;
            max_corner[d] = center[d] + radius;
        }
        Geometry::PoissonDisk<T, dim> poisson_disk(min_corner, max_corner, Base::dx, T(_ppc));
        poisson_disk.sample(new_samples);
        for(auto position : new_samples){
            T dist = (position - center).norm();
            if(dist < radius){
                addParticle(position, velocity, density*vol, 0.0, 0, marker, useDamage);
            }
        }
        int end = Base::m_X.size();
        model->append(start, end, vol);
    }

    //------------CRACK DEFINITION--------------

    void addHorizontalCrack(const TV& minPoint, const TV& maxPoint, T increment, T radius, int _crackType = 0)
    {
        //Crack type: 0 = left side, 1 = middle, 2 = right side
        crackType = _crackType;
        BOW_ASSERT(dim == 2);
        T region = (maxPoint(0) - minPoint(0)) / increment;
        crackPlane_startIdx = Base::m_X.size(); //start idx for crack path particles
        for(int i = 0; i < region; i++){
            TV position = minPoint;
            position(0) += i * increment;

            //Add crack particles directly to the particle list, BUT ZERO MASS
            Base::m_X.push_back(position);
            Base::m_V.push_back(TV::Zero());
            m_marker.push_back(1);

            Base::m_C.push_back(TM::Zero());
            Base::m_mass.push_back(0.0); //zero mass
            Base::stress.push_back(TM::Zero());
            m_cauchy.push_back(TM::Zero());
            Dp.push_back(0.0);
            damageLaplacians.push_back(0.0);
            sp.push_back(0);
            particleDG.push_back(TV::Zero());
            //dTildeH.push_back(0.0);
            //sigmaC.push_back(0.0);
        }
        topPlane_startIdx = Base::m_X.size(); //now add duplicate particles for the top plane
        for(int i = 0; i < region; i++){
            TV position = Base::m_X[crackPlane_startIdx + i];
            Base::m_X.push_back(position);
            Base::m_V.push_back(TV::Zero());
            m_marker.push_back(2);

            Base::m_C.push_back(TM::Zero());
            Base::m_mass.push_back(0.0); //zero mass
            Base::stress.push_back(TM::Zero());
            m_cauchy.push_back(TM::Zero());
            Dp.push_back(0.0);
            damageLaplacians.push_back(0.0);
            sp.push_back(0);
            particleDG.push_back(TV::Zero());
            //dTildeH.push_back(0.0);
            //sigmaC.push_back(0.0);
        }
        bottomPlane_startIdx = Base::m_X.size(); //now add duplicate particles for the bottom plane
        for(int i = 0; i < region; i++){
            TV position = Base::m_X[topPlane_startIdx + i];
            Base::m_X.push_back(position);
            Base::m_V.push_back(TV::Zero());
            m_marker.push_back(3);

            Base::m_C.push_back(TM::Zero());
            Base::m_mass.push_back(0.0); //zero mass
            Base::stress.push_back(TM::Zero());
            m_cauchy.push_back(TM::Zero());
            Dp.push_back(0.0);
            damageLaplacians.push_back(0.0);
            sp.push_back(0);
            particleDG.push_back(TV::Zero());
            //dTildeH.push_back(0.0);
            //sigmaC.push_back(0.0);
        }
        bottomPlane_endIdx = Base::m_X.size() - 1;

        crackInitialized = true;

        grid.crackParticlesStartIdx = crackPlane_startIdx; //we'll use this in DFGMPMOp to make sure we don't access these particles for other operations!
        grid.crackInitialized = true;
        grid.horizontalCrack = true;

        //int crackTipIdx = topPlane_startIdx - 1;

        //Now we need to mark particles near the crack as fully damaged
        for(int i = 0; i < crackPlane_startIdx; i++){ //iter normal material particles
            TV p = Base::m_X[i];
            //TV crackTip = Base::m_X[crackTipIdx];
            for(int j = crackPlane_startIdx; j < topPlane_startIdx; j++){ //check material particles against every crack plane particle
                TV c = Base::m_X[j];
                T dX = p(0) - c(0);
                T dY = p(1) - c(1);
                T dist = std::sqrt(dX*dX + dY*dY);
                // if(dist < radius && p(0) < crackTip(0)){
                //     Dp[i] = 1.0;
                // }
                if(dist < radius){
                    Dp[i] = 1.0;
                }
            }
        }
    }

    void addHorizontalCrackWithoutPoints(const TV& minPoint, const TV& maxPoint, T increment, T radius, int _crackType = 0)
    {
        //Crack type: 0 = left side, 1 = middle, 2 = right side
        crackType = _crackType;
        BOW_ASSERT(dim == 2);
        T region = (maxPoint(0) - minPoint(0)) / increment;
        std::vector<TV> crackPoints;
        for(int i = 0; i < region; i++){
            TV position = minPoint;
            position(0) += i * increment;
            crackPoints.push_back(position);
        }

        //Now we need to mark particles near the crack as fully damaged
        for(int i = 0; i < (int)Base::m_X.size(); i++){ //iter normal material particles
            TV p = Base::m_X[i];
            for(int j = 0; j < (int)crackPoints.size(); j++){ //check material particles against every crack plane particle
                TV c = crackPoints[j];
                T dX = p(0) - c(0);
                T dY = p(1) - c(1);
                T dist = std::sqrt(dX*dX + dY*dY);
                // if(dist < radius && p(0) < crackTip(0)){
                //     Dp[i] = 1.0;
                // }
                if(dist < radius){
                    //Dp[i] = 1.0;
                    sp[i] = 1;
                }
            }
        }
    }


};

}