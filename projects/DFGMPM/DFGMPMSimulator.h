#pragma once

#include <Bow/Simulator/MPM/MPMSimulator.h>
#include <Bow/Energy/MPM/ElasticityOp.h>
#include <Bow/Simulator/PhysicallyBasedSimulator.h>
#include <Bow/Utils/Serialization.h>
#include <Bow/Utils/Logging.h>
#include <Bow/Utils/FileSystem.h>
#include "DFGMPMOp.h"
#include "DFGMPMGrid.h"
// #include "TwoFieldHodgepodgeEnergy.h"
// #include "TwoFieldBackwardEuler.h"
#include <Bow/IO/ply.h>

namespace Bow::MPM {

template <class T, int dim>
class DFGMPMSimulator : public MPMSimulator<T, dim> {
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

    Field<int> m_marker; //0 = material particle, 1 = crack particle, 2 = top plane, 3 = bottom plane

    //Additional Particle Data
    Field<T> m_vol;
    Field<T> m_mu;
    Field<T> m_la;

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
    bool useDamage = false;
    bool useImplicitContact = false;
    bool useRankineDamage = false;
    bool useAnisoMPMDamage = false;
    bool initialized = false;
    bool useImpulse = false;

    //Particle Data
    Field<TM> cauchy; //for anisoMPM
    std::vector<T> Dp; //particle damage
    std::vector<T> damageLaplacians; //particle damage
    std::vector<T> dTildeH; //particle damage
    std::vector<T> sigmaC; //particle damage
    std::vector<int> sp; //surface particle or not
    Field<TV> particleDG; //particle damage gradients
    Field<std::vector<int>> particleAF; //store which activefield each particle belongs to for the 3^d grid nodes it maps to
    Field<std::vector<int>> p_cached_idx; //store which DOF index each particle maps to for each of the 3^dim nodes

    SERIALIZATION_REGISTER(cauchy)
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

    //AnisoMPM Params
    T eta = 0.01;
    T sigmaCRef = -1.0;
    T percentStretch = -1.0;
    T zeta = 1.0;
    T l0;

    //Impulse Data
    TV impulseCenter = TV::Zero();
    T impulseStrength = 0;
    T impulseStartTime = 0;
    T impulseDuration = 0;

    //Grid Data to Save and Vis
    Field<TV> activeNodesX;
    Field<TV> activeNodesDG;
    Field<TV> activeNodesV1;
    Field<TV> activeNodesV2;
    Field<TV> activeNodesFct1;
    Field<TV> activeNodesFct2;
    std::vector<T> activeNodesM1;
    std::vector<T> activeNodesM2;
    std::vector<T> activeNodesSeparability1;
    std::vector<T> activeNodesSeparability2;
    std::vector<int> activeNodesSeparable;

    DFGMPMSimulator(std::string _outputPath){
        //This ensures we capture everything in the log file
        outputPath = _outputPath;
        Bow::FileSystem::create_path(outputPath);
        Bow::Logging::new_logger(outputPath + "/log.txt", Bow::Logging::Info, true);
    }

    /* Initialization Routines */
    void initialize()
    {
        
        Bow::Logging::info("Simulation starts with ", std::is_same<T, double>::value ? "double" : "float", " ", dim);

        l0 = 0.5 * Base::dx;
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

        //Initialize sigmaC if we are using damage
        if(useDamage){

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
                Bow::Logging::info("[AnisoMPM Damage] Stretched SigmaC: ", sigmaC[0]);
            }
            else{
                //set sigmaC directly from sigmaCRef
                tbb::parallel_for(size_t(0), sigmaC.size(), [&](size_t i) {
                    sigmaC[i] = sigmaCRef;
                });
                Bow::Logging::info("[AnisoMPM Damage] Directly Set SigmaC: ", sigmaC[0]);
            }
        }

        initialized = true;
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
        
        //Rankine Damage Routines
        if (useRankineDamage) {
            //TODO: updateRankineDamage
            //NOTE: we can put rankine damage here because, unlike AnisoMPM damage, we can update this BEFORE computing DGs!
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

        for (auto& model : Base::elasticity_models) //compute cauchy stress to pass to next method
            model->compute_cauchy(cauchy);

        Bow::DFGMPM::UpdateAnisoMPMDamageOp<T, dim> update_anisoMPM_damage{ {}, Dp, damageLaplacians, dTildeH, sigmaC, cauchy, dt, eta, zeta, l0, grid };
        update_anisoMPM_damage();
    }

    //Compute forces, P2G, and grid update --> all depends on symplectic or not
    void p2g(T dt){
        if(!useDFG){
            //if single field MPM, sort particles before P2G (for two field we already did this)
            grid.sortParticles(Base::m_X, Base::dx);
        }
        if(Base::symplectic){
            //Now compute forces for P2G and grid update (ONLY FOR SYMPLECTIC)
            for (auto& model : Base::elasticity_models)
                model->compute_stress(Base::stress);
        }
        else if(useImplicitContact){ //grab mu, la, and vol for barrier eval -- TAG=BARRIER
            for (auto& model : Base::elasticity_models){
                //std::shared_ptr<Bow::MPM::FBasedElastiticityOp> abcd = std::dynamic_pointer_cast<Bow::MPM::FBasedElastiticityOp>(model);
                m_mu = model->m_mu;
                m_la = model->m_lambda;
                m_vol = model->m_vol;
            }
        }
        Bow::DFGMPM::ParticlesToGridOp<T, dim> P2G{ {}, Base::m_X, Base::m_V, Base::m_mass, Base::m_C, Base::stress, gravity, particleAF, grid, Base::dx, dt, Base::symplectic, useDFG, useAPIC, useImplicitContact, m_mu, m_la, m_vol };
        P2G();
    }

    //Symplectic: Collision Updates; Implicit: Collisions + Implicit Update
    void gridUpdate(T dt){
        if(Base::symplectic){
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
        for (auto& model : Base::elasticity_models)
            model->evolve_strain(G2P.m_gradXp);

        //Now project strain (plasticity)
        for (auto& model : Base::plasticity_models)
            model->project_strain();
    }

    /* Write our own advance function to override*/
    void advance(T dt) override
    {
        if(!initialized){
            initialize();
        }

        std::cout << "Initialized..." << std::endl;    
        
        if(useDFG) {
            //DFG specific routines (partitioning)
            partitioningRoutines();
            
            //AnisoMPM Routines
            if(useAnisoMPMDamage) {
                anisoMPMDamage(dt); //note that we simply update and track damage, there is no elasticity deg
            }
        }

        std::cout << "Partitioned..." << std::endl;   

        p2g(dt); //compute forces, p2g transfer

        std::cout << "P2G done..." << std::endl;

        //Apply Impulse (if user added one) -> apply directly for symplectic, save forces for later if implicit
        if(useImpulse){
            if((elapsedTime >= impulseStartTime) && (elapsedTime < impulseStartTime + impulseDuration)){
                Bow::DFGMPM::ApplyImpulseOp<T, dim> apply_impulse{ {}, impulseCenter, impulseStrength, grid, Base::dx, dt, Base::symplectic, useImpulse };
                apply_impulse();
            }
        }

        //Frictional Contact -> apply directly for symplectic, for implicit we compute normals here (but only if we want implicit contact)
        if (useDFG && (Base::symplectic || (!Base::symplectic && useImplicitContact))) {
            Bow::DFGMPM::ContactForcesOp<T, dim> frictional_contact{ {}, dt, fricCoeff, Base::symplectic, useImplicitContact, grid };
            frictional_contact();
        }

        gridUpdate(dt); //collisions + implicit grid update

        //Explicit Frictional Contact -> ONLY for implicit two field with EXPLICIT frictional contact
        if (useDFG && (!Base::symplectic && !useImplicitContact)) {
            Bow::DFGMPM::ContactForcesOp<T, dim> frictional_contact{ {}, dt, fricCoeff, Base::symplectic, useImplicitContact, grid };
            frictional_contact();
        }

        g2p(dt); //transfer, updateF, and plastic projection

        std::cout << "G2P done..." << std::endl;

        //Helpful timers and counters
        elapsedTime += dt;
        currSubstep++;

        //Now dump substep data if verbose is active (verbose = write every substep)
        if(verbose){
            BOW_TIMER_FLAG("writeSubstep");
            IO::writeTwoField_particles_ply(outputPath + "/p" + std::to_string(currSubstep) + ".bgeo", Base::m_X, Base::m_V, particleDG, Base::m_mass, Dp, sp, m_marker);

            //Write Grid
            if(writeGrid){
                Bow::DFGMPM::CollectGridDataOp<T, dim> collect_gridData{ {}, grid, Base::dx, activeNodesX, activeNodesDG, activeNodesV1, activeNodesV2, activeNodesFct1, activeNodesFct2, activeNodesM1, activeNodesM2, activeNodesSeparability1, activeNodesSeparability2, activeNodesSeparable };
                collect_gridData();
                IO::writeTwoField_nodes_ply(outputPath + "/i" + std::to_string(currSubstep) + ".bgeo", activeNodesX, activeNodesDG, activeNodesV1, activeNodesV2, activeNodesFct1, activeNodesFct2, activeNodesM1, activeNodesM2, activeNodesSeparability1, activeNodesSeparability2, activeNodesSeparable);
            }
        }
    }

    //------------ADDING TO SIM--------------

    //Setup sim for AnisoMPM Damage -- NOTE: if you want to set sigmaC directly, pass p < 0 and your sigmaC
    void addAnisoMPMDamage(T _eta, T _dMin, T _zeta, T _p = -1.0, T _sigmaC = -1.0)
    {
        assert(_p > 0 ^ _sigmaC > 0); //assert that exactly one of these is set
        assert(!useDamage); //if we've already added a damage model we can't add another!
        Bow::Logging::info("[AnisoMPM Damage] Simulating with AnisoMPM Damage");
        useAnisoMPMDamage = true;
        useDamage = true;
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

    //Setup sim for an impulse of user defined strength and duration
    void addImpulse(TV _center, T _strength, T _startTime, T _duration)
    {
        useImpulse = true;
        impulseCenter = _center;
        impulseStrength = _strength;
        impulseStartTime = _startTime;
        impulseDuration = _duration;
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

    void dump_output(int frame_num) override
    {
        if(!frame_num || !verbose){
            BOW_TIMER_FLAG("writeFrame");
            IO::writeTwoField_particles_ply(outputPath + "/p" + std::to_string(frame_num) + ".bgeo", Base::m_X, Base::m_V, particleDG, Base::m_mass, Dp, sp, m_marker);

            //Write Grid
            if(writeGrid){
                Bow::DFGMPM::CollectGridDataOp<T, dim> collect_gridData{ {}, grid, Base::dx, activeNodesX, activeNodesDG, activeNodesV1, activeNodesV2, activeNodesFct1, activeNodesFct2, activeNodesM1, activeNodesM2, activeNodesSeparability1, activeNodesSeparability2, activeNodesSeparable };
                collect_gridData();
                IO::writeTwoField_nodes_ply(outputPath + "/i" + std::to_string(frame_num) + ".bgeo", activeNodesX, activeNodesDG, activeNodesV1, activeNodesV2, activeNodesFct1, activeNodesFct2, activeNodesM1, activeNodesM2, activeNodesSeparability1, activeNodesSeparability2, activeNodesSeparable);
            }
        }
    }

    //------------PARTICLE SAMPLING--------------

    void sampleRandomCube(std::shared_ptr<ElasticityOp<T, dim>> model, const TV& min_corner, const TV& max_corner, const TV& velocity = TV::Zero(), T density = 1000.)
    {
        // sample particles
        T vol = dim == 2 ? Base::dx * Base::dx / 4 : Base::dx * Base::dx * Base::dx / 8;
        T interval = Base::dx / std::pow((T)ppc, (T)1 / dim);
        Vector<int, dim> region = ((max_corner - min_corner) / interval).template cast<int>();
        printf("%d %d %d\n", region(0), region(1), region(2));
        int start = Base::m_X.size();
        iterateRegion(region, [&](const Vector<int, dim>& offset) {
            TV position = min_corner + offset.template cast<T>() * interval;
            position += TV::Ones() * 0.5 * interval + TV::Random() * 0.5 * interval;
            Base::m_X.push_back(position);
            Base::m_V.push_back(velocity);
            Base::m_C.push_back(TM::Zero());
            Base::m_mass.push_back(density * vol);
            Base::stress.push_back(TM::Zero());
            cauchy.push_back(TM::Zero());
            Dp.push_back(0.0);
            damageLaplacians.push_back(0.0);
            sp.push_back(0);
            particleDG.push_back(TV::Zero());
            dTildeH.push_back(0.0);
            sigmaC.push_back(10.0);
            m_marker.push_back(0);
        });
        int end = Base::m_X.size();
        model->append(start, end, vol);
    }

    void samplePrecutRandomCube(std::shared_ptr<ElasticityOp<T, dim>> model, const TV& min_corner, const TV& max_corner, const TV& velocity = TV::Zero(), T density = 1000.)
    {
        // sample particles
        T vol = dim == 2 ? Base::dx * Base::dx / 4 : Base::dx * Base::dx * Base::dx / 8;
        T interval = Base::dx / std::pow((T)ppc, (T)1 / dim);
        Vector<int, dim> region = ((max_corner - min_corner) / interval).template cast<int>();
        printf("%d %d %d\n", region(0), region(1), region(2));
        int start = Base::m_X.size();
        iterateRegion(region, [&](const Vector<int, dim>& offset) {
            TV position = min_corner + offset.template cast<T>() * interval;
            position += TV::Ones() * 0.5 * interval + TV::Random() * 0.5 * interval;
            Base::m_X.push_back(position);
            Base::m_V.push_back(velocity);
            Base::m_C.push_back(TM::Zero());
            Base::m_mass.push_back(density * vol);
            Base::stress.push_back(TM::Zero());
            cauchy.push_back(TM::Zero());
            
            //give full damage for the cut particles
            T buffer = 0.01;
            T diffY = max_corner[1] - min_corner[1];
            T diffX = max_corner[0] - min_corner[0];
            T y1 = max_corner[1] - (diffY/3.0);
            T y2 = max_corner[1] - ((2*diffY)/3.0);
            T x1 = min_corner[0] + (0.3 * diffX);
            T x2 = max_corner[0] - (0.3 * diffX);

            if((position[1] <= y1 + buffer) && (position[1] >= y1 - buffer) && (position[0] >= x1)){
                Dp.push_back(1.0);
            } else if ((position[1] <= y2 + buffer) && (position[1] >= y2 - buffer) && (position[0] <= x2)){
                Dp.push_back(1.0);
            } else{
                Dp.push_back(0.0);
            }

            damageLaplacians.push_back(0.0);
            sp.push_back(0);
            particleDG.push_back(TV::Zero());
            dTildeH.push_back(0.0);
            sigmaC.push_back(10.0);
            m_marker.push_back(0);
        });
        int end = Base::m_X.size();
        model->append(start, end, vol);
    }

    void sampleRandomSphere(std::shared_ptr<ElasticityOp<T, dim>> model, const TV& center, const T radius, const TV& velocity = TV::Zero(), T density = 1000.)
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
                Base::m_X.push_back(position);
                Base::m_V.push_back(velocity);
                Base::m_C.push_back(TM::Zero());
                Base::m_mass.push_back(density * vol);
                Base::stress.push_back(TM::Zero());
                cauchy.push_back(TM::Zero());
                Dp.push_back(0.0);
                damageLaplacians.push_back(0.0);
                sp.push_back(0);
                particleDG.push_back(TV::Zero());
                dTildeH.push_back(0.0);
                sigmaC.push_back(10.0);
                m_marker.push_back(0);
            }
        });
        int end = Base::m_X.size();
        model->append(start, end, vol);
    }

    void sampleFromObj(std::shared_ptr<ElasticityOp<T, dim>> model, const std::string filepath, const TV& velocity, T volume, T density)
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
                Base::m_X.push_back(position);
                Base::m_V.push_back(velocity);
                Base::m_C.push_back(TM::Zero());
                Base::stress.push_back(TM::Zero());
                cauchy.push_back(TM::Zero());
                Dp.push_back(0.0);
                damageLaplacians.push_back(0.0);
                sp.push_back(0);
                particleDG.push_back(TV::Zero());
                dTildeH.push_back(0.0);
                sigmaC.push_back(10.0);
                m_marker.push_back(0);
            }
        }

        //now that we know total num particles, calculate particle masses
        T vol = volume / (T)Base::m_X.size();
        for(size_t i = 0; i < Base::m_X.size(); ++i){
            Base::m_mass.push_back(density * vol);
        }
        
        int end = Base::m_X.size();
        model->append(start, end, vol);
        fs.close();
    }

};

}