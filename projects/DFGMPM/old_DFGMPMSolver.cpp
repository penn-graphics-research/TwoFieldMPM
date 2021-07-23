//#include <Bow/Simulation/MPM/DFGMPMOp.h>
#include "DFGMPMOp.h"
#include <Bow/Simulation/MPM/ElasticityOp.h>
#include <Bow/Simulation/MPM/MPMImplicit.h>
#include <Bow/Types.h>
#include <Bow/Utils/Timer.h>
#include <Bow/Utils/Logging.h>
#include <Bow/Utils/FileSystem.h>
#include <cstdlib>
#include <Bow/IO/partio.h>
// #include <tbb/tbb.h>
// #include <tbb/task_scheduler_init.h>

using T = double;
const int dim = 3;
class DFGMPMSolver {
public:
    using TV = Bow::Vector<T, dim>;

    //Sim Data
    std::string outputPath;
    T dx;
    T dt;
    T maxDt;
    int endFrame;
    int fps;
    int numSubsteps;
    int numParticles;
    T frameDt;
    T elapsedTime;
    T flipPicRatio;
    T gravity;
    int scale = 30; //TODO: don't need this later

    //Material Data
    T st; //surfaceThreshold

    //Sim Flags
    bool symplectic;
    bool verbose;
    bool useAPIC;
    bool useDFG;
    bool useDamage;
    bool useImplicitContactForce;
    bool useRankineDamage;
    bool useAnisoMPMDamage;

    //Particle Data
    Bow::Field<Bow::Vector<T, dim>> x;
    Bow::Field<Bow::Vector<T, dim>> v;
    Bow::Field<Bow::Matrix<T, dim, dim>> C;
    std::vector<T> mp;
    Bow::Field<Bow::Matrix<T, dim, dim>> F;
    Bow::Field<Bow::Matrix<T, dim, dim>> cauchy; //for anisoMPM
    std::vector<T> Dp; //particle damage
    std::vector<T> damageLaplacians; //particle damage
    std::vector<T> dTildeH; //particle damage
    std::vector<T> sigmaC; //particle damage
    std::vector<int> sp; //surface particle or not
    Bow::Field<Bow::Vector<T, dim>> particleDG; //particle damage gradients
    Bow::Field<std::vector<int>> particleAF; //store which activefield each particle belongs to for the 3^d grid nodes it maps to

    //Regular Grid Data
    Bow::DFGMPM::DFGMPMGrid<T, dim> grid;

    //Spatial Hash Data (use same grid for spatial hash and for sim)
    T rp;
    Bow::Field<std::vector<int>> particleNeighbors;

    //DFGMPM Params
    T minDp;
    T dMin;
    T fricCoeff; //friction coefficient for contact forces

    //AnisoMPM Params
    T eta;
    T sigmaCRef;
    T percentStretch;
    T zeta;
    T l0;

    //Grid Data to Save and Vis
    Bow::Field<Bow::Vector<T, dim>> activeNodesX;
    Bow::Field<Bow::Vector<T, dim>> activeNodesDG;
    Bow::Field<Bow::Vector<T, dim>> activeNodesV1;
    Bow::Field<Bow::Vector<T, dim>> activeNodesV2;
    Bow::Field<Bow::Vector<T, dim>> activeNodesFct1;
    Bow::Field<Bow::Vector<T, dim>> activeNodesFct2;
    std::vector<T> activeNodesM1;
    std::vector<T> activeNodesM2;
    std::vector<int> activeNodesSeparable;

    DFGMPMSolver(T _dx, T _dt, int _endFrame, int _fps, T _gravity, T _st, bool _symplectic, bool _verbose, bool _useDFG, T _fricCoeff, bool _useAPIC, T _flipPicRatio = -1.0)
    {
        if (!_useAPIC) {
            assert(_flipPicRatio >= 0); //need this ratio if we arent using APIC
            flipPicRatio = _flipPicRatio;
        }
        else if (_useAPIC) {
            flipPicRatio = 0.0; //set this to be 0 if we are using APIC, this helps for computing updated velocity in G2P
        }

        dx = _dx;
        l0 = 0.5 * _dx;
        dt = _dt;
        maxDt = _dt;
        endFrame = _endFrame;
        fps = _fps;
        frameDt = 1.0 / _fps;
        elapsedTime = 0.0;
        numSubsteps = static_cast<int>(frameDt / dt);
        gravity = _gravity;
        st = _st;
        symplectic = _symplectic;
        verbose = _verbose;
        useAPIC = _useAPIC;
        useDFG = _useDFG;
        useDamage = false; //init to false, only change with a member method
        fricCoeff = _fricCoeff;

        useImplicitContactForce = false; //TODO: hard coded for now

        //TODO: set these in separate methods later (i.e. addAnisoMPMDamage)
        useRankineDamage = false;
        minDp = 1.0;

        //Spatial Hash
        if (dim == 2) {
            rp = sqrt(2.0 * _dx * _dx);
        }
        else if (dim == 3) {
            rp = sqrt(3.0 * _dx * _dx);
        }
    }

    // void collidingCubes2D()
    // {
    //     for (int i = 0; i < scale; ++i)
    //         for (int j = 0; j < scale; ++j) {
    //             T a = (T(std::rand()) / T(RAND_MAX) + i) * dx * 0.5;
    //             T b = (T(std::rand()) / T(RAND_MAX) + j) * dx * 0.5;
    //             x.push_back(Bow::Vector<T, dim>(a, b));
    //             v.push_back(Bow::Vector<T, dim>(1, 0));
    //             C.push_back(Bow::Matrix<T, dim, dim>::Zero());
    //             mp.push_back(1000 * dx * dx / 4);
    //             F.push_back(Bow::Matrix<T, dim, dim>::Zero());
    //             Dp.push_back(0.0);
    //damageLaplacians.push_back(0.0);
    //             sp.push_back(0);
    //             particleDG.push_back(Bow::Vector<T, dim>::Zero());
    //         }
    //     for (int i = 0; i < scale; ++i)
    //         for (int j = 0; j < scale; ++j) {
    //             T a = (T(std::rand()) / T(RAND_MAX) + i) * dx * 0.5 + scale * dx;
    //             T b = (T(std::rand()) / T(RAND_MAX) + j) * dx * 0.5;
    //             x.push_back(Bow::Vector<T, dim>(a, b));
    //             v.push_back(Bow::Vector<T, dim>(-1, 0));
    //             C.push_back(Bow::Matrix<T, dim, dim>::Zero());
    //             mp.push_back(1000 * dx * dx / 4);
    //             F.push_back(Bow::Matrix<T, dim, dim>::Zero());
    //             Dp.push_back(0.0);
    //damageLaplacians.push_back(0.0);
    //             sp.push_back(0);
    //             particleDG.push_back(Bow::Vector<T, dim>::Zero());
    //         }
    //     return;
    // }

    // void sampleStaticCube2D()
    // {
    //     T w;
    //     w = 0.5;
    //     for (int i = 0; i < scale; ++i)
    //         for (int j = 0; j < scale; ++j) {
    //             T a = (T(std::rand()) / T(RAND_MAX) + i) * dx * w;
    //             T b = (T(std::rand()) / T(RAND_MAX) + j) * dx * w;
    //             x.push_back(Bow::Vector<T, dim>(a, b));
    //             v.push_back(Bow::Vector<T, dim>(0, 0));
    //             C.push_back(Bow::Matrix<T, dim, dim>::Zero());
    //             mp.push_back(1000 * dx * dx / 4);
    //             F.push_back(Bow::Matrix<T, dim, dim>::Zero());
    //             Dp.push_back(0.0);
    //damageLaplacians.push_back(0.0);
    //             sp.push_back(0);
    //             particleDG.push_back(Bow::Vector<T, dim>::Zero());
    //         }
    //     return;
    // }

    void collidingCubes3D()
    {
        for (int i = 0; i < scale; ++i)
            for (int j = 0; j < scale; ++j)
                for (int k = 0; k < scale; ++k) {
                    T a = (T(std::rand()) / T(RAND_MAX) + i) * dx * 0.5;
                    T b = (T(std::rand()) / T(RAND_MAX) + j) * dx * 0.5;
                    T c = (T(std::rand()) / T(RAND_MAX) + k) * dx * 0.5;
                    x.push_back(Bow::Vector<T, dim>(a, b, c));
                    v.push_back(Bow::Vector<T, dim>(1, 0, 0));
                    C.push_back(Bow::Matrix<T, dim, dim>::Zero());
                    mp.push_back(1000 * dx * dx * dx / 8);
                    F.push_back(Bow::Matrix<T, dim, dim>::Zero());
                    cauchy.push_back(Bow::Matrix<T, dim, dim>::Zero());
                    Dp.push_back(0.0);
                    damageLaplacians.push_back(0.0);
                    sp.push_back(0);
                    particleDG.push_back(Bow::Vector<T, dim>::Zero());
                    dTildeH.push_back(0.0);
                    sigmaC.push_back(10.0);
                }
        for (int i = 0; i < scale; ++i)
            for (int j = 0; j < scale; ++j)
                for (int k = 0; k < scale; ++k) {
                    T a = (T(std::rand()) / T(RAND_MAX) + i) * dx * 0.5 + scale * dx;
                    T b = (T(std::rand()) / T(RAND_MAX) + j) * dx * 0.5;
                    T c = (T(std::rand()) / T(RAND_MAX) + k) * dx * 0.5;
                    x.push_back(Bow::Vector<T, dim>(a, b, c));
                    v.push_back(Bow::Vector<T, dim>(-1, 0, 0));
                    C.push_back(Bow::Matrix<T, dim, dim>::Zero());
                    mp.push_back(1000 * dx * dx * dx / 8);
                    F.push_back(Bow::Matrix<T, dim, dim>::Zero());
                    cauchy.push_back(Bow::Matrix<T, dim, dim>::Zero());
                    Dp.push_back(0.0);
                    damageLaplacians.push_back(0.0);
                    sp.push_back(0);
                    particleDG.push_back(Bow::Vector<T, dim>::Zero());
                    dTildeH.push_back(0.0);
                    sigmaC.push_back(10.0);
                }
        numParticles = x.size();
        return;
    }

    // void sampleStaticCube3D()
    // {
    //     T w;
    //     w = 0.5;
    //     for (int i = 0; i < scale; ++i)
    //         for (int j = 0; j < scale; ++j)
    //             for (int k = 0; k < scale; ++k) {
    //                 T a = (T(std::rand()) / T(RAND_MAX) + i) * dx * w;
    //                 T b = (T(std::rand()) / T(RAND_MAX) + j) * dx * w;
    //                 T c = (T(std::rand()) / T(RAND_MAX) + k) * dx * w;
    //                 x.push_back(Bow::Vector<T, dim>(a, b, c));
    //                 v.push_back(Bow::Vector<T, dim>(0, 1, 0));
    //                 C.push_back(Bow::Matrix<T, dim, dim>::Zero());
    //                 mp.push_back(1000 * dx * dx * dx / 8);
    //                 F.push_back(Bow::Matrix<T, dim, dim>::Zero());
    //                 Dp.push_back(0.0);
    //                 sp.push_back(0);
    //                 particleDG.push_back(Bow::Vector<T, dim>::Zero());
    //             }
    //     numParticles = x.size();
    //     return;
    // }

    //Setup sim for AnisoMPM Damage -- NOTE: if you want to set sigmaC directly, pass p < 0 and your sigmaC
    void addAnisoMPMDamage(T _eta, T _dMin, T _zeta, T _p = -1.0, T _sigmaC = -1.0)
    {
        assert(_p > 0 ^ _sigmaC > 0); //assert that exactly one of these is set
        assert(!useDamage); //if we've already added a damage model we can't add another!
        std::cout << "[AnisoMPM Damage] Simulating with AnisoMPM Damage " << std::endl;
        useAnisoMPMDamage = true;
        useDamage = true;
        eta = _eta;
        dMin = _dMin;
        sigmaCRef = _sigmaC;
        percentStretch = _p;
        zeta = _zeta;
        if (_p > 0) {
            std::cout << "[AnisoMPM Damage] Percent Stretch: " << percentStretch << std::endl;
        }
        else {
            std::cout << "[AnisoMPM Damage] SigmaC: " << sigmaCRef << std::endl;
        }
        std::cout << "[AnisoMPM Damage] Eta: " << eta << std::endl;
        std::cout << "[AnisoMPM Damage] Zeta: " << zeta << std::endl;
        std::cout << "[AnisoMPM Damage] dMin: " << dMin << std::endl;
    }

    void initializeStructures()
    {
        //Initialize particle neighbor lists
        for (size_t i = 0; i < x.size(); ++i) {
            std::vector<int> placeholder, placeholder2;
            particleNeighbors.push_back(placeholder);
            particleAF.push_back(placeholder2);
        }
    }

    void simulate()
    {
        Bow::FileSystem::create_path(outputPath);
        Bow::Logging::new_logger(outputPath + "/log.txt", Bow::Logging::Info, true);
        Bow::Logging::info("Simulation starts with ", std::is_same<T, double>::value ? "double" : "float", " ", dim);

        collidingCubes3D();
        //sampleStaticCube3D();
        //collidingCubes2D();

        initializeStructures();

        //Define substep routines
        Bow::DFGMPM::BackGridSortOp<T, dim> backGrid_sort{ {}, x, grid, rp };
        Bow::DFGMPM::NeighborSortOp<T, dim> neighbor_sort{ {}, x, particleNeighbors, grid, rp };
        Bow::DFGMPM::SurfaceDetectionOp<T, dim> surface_detection{ {}, x, particleNeighbors, rp, st, sp };
        Bow::DFGMPM::ComputeDamageGradientsOp<T, dim> compute_DGs{ {}, x, particleNeighbors, rp, dx, particleDG, Dp, sp, grid };
        Bow::DFGMPM::PartitioningOp<T, dim> partition{ {}, x, mp, particleDG, particleAF, Dp, sp, dx, minDp, dMin, grid };
        Bow::MPM::FixedCorotatedOp<T, dim> compute_fcr{ {}, F };
        compute_fcr.add(1000, 0.4, dx * dx * dx / 8, 0, (int)x.size());
        Bow::DFGMPM::ComputeDamageLaplaciansOp<T, dim> compute_damageLaplacians{ {}, x, Dp, damageLaplacians, particleAF, dx, grid };
        Bow::DFGMPM::UpdateAnisoMPMDamageOp<T, dim> update_anisoMPM_damage{ {}, Dp, damageLaplacians, dTildeH, sigmaC, cauchy, dt, eta, zeta, l0, grid };
        Bow::DFGMPM::ParticlesToGridOp<T, dim> p2g{ {}, x, v, mp, C, F, gravity, particleAF, grid, dx, dt, symplectic, useDFG, useAPIC };
        Bow::DFGMPM::GridVelocityExplicitUpdateOp<T, dim> grid_update{ {}, grid };
        Bow::DFGMPM::ContactForcesOp<T, dim> frictional_contact{ {}, dt, fricCoeff, symplectic, useImplicitContactForce, grid };
        Bow::DFGMPM::GridToParticlesOp<T, dim> g2p{ {}, x, v, C, particleAF, grid, dx, dt, flipPicRatio, useDFG };
        Bow::DFGMPM::CollectGridDataOp<T, dim> collect_gridData{ {}, grid, dx, activeNodesX, activeNodesDG, activeNodesV1, activeNodesV2, activeNodesFct1, activeNodesFct2, activeNodesM1, activeNodesM2, activeNodesSeparable };

        //Frame Loop
        for (int frame = 0; frame < endFrame; ++frame) {
            bool frameFinished = false;
            T frameTime = 0.0;
            int currSubstep = 0;

            //Write frame 0 before any substeps
            if (frame == 0) {
                BOW_TIMER_FLAG("writeFrame");
                Bow::writePositionVectorToPartio(outputPath + "/p" + std::to_string(frame) + ".bgeo", x);
                Bow::Timer::flush();
            }

            //Substep Loop
            while (!frameFinished) {
                if (symplectic) {
                    dt = maxDt; //reset dt (even after making it smaller to finish the substep)
                }

                //Make sure we don't go beyond frameDt
                if (frameDt - frameTime < dt * 1.001) {
                    frameFinished = true;
                    dt = frameDt - frameTime;
                }
                else if (frameDt - frameTime < 2 * dt) {
                    dt = (frameDt - frameTime) / 2.0;
                }

                Bow::Timer::progress("Step " + std::to_string(currSubstep) + " of Frame " + std::to_string(frame), currSubstep, numSubsteps);

                //Substep routines
                if (symplectic) {
                    grid.sortParticles(x, dx); //Sort particles into grid

                    //DFG specific routines (partitioning)
                    if (useDFG) {
                        backGrid_sort(); //Sort particles into spatial hash grid
                        neighbor_sort(); //Create neighbor list for each particle
                        if (elapsedTime == 0.0) {
                            surface_detection(); //Detect surface particles on first substep
                        }
                        if (useRankineDamage) {
                            //TODO: updateRankineDamage
                        }
                        compute_DGs(); //Compute particle damage gradients
                        partition(); //Partition particles into their fields, transfer mass to those fields, and compute node separability
                        if (useAnisoMPMDamage) {
                            compute_damageLaplacians();
                            compute_fcr.cauchy(cauchy); //compute cauchy stress to pass to next method
                            update_anisoMPM_damage();
                        }
                    }
                    compute_fcr(); //Compute forces for P2G
                    p2g(); //P2G
                    grid_update(); //Update Grid
                    if (useDFG) {
                        frictional_contact();
                    }
                    g2p(); //P2G
                    compute_fcr.evolve_strain(g2p.m_gradVp, dt); //Update F
                }
                else {
                    //TODO: Implicit Substep
                }

                //Increment progress trackers
                currSubstep++;
                frameTime += dt;
                elapsedTime += dt;

                //Only write substep when frame is finished
                if (frameFinished && !verbose) {
                    BOW_TIMER_FLAG("writeFrame");
                    Bow::writeDFGMPMToPartio(outputPath + "/p" + std::to_string(frame + 1) + ".bgeo", x, v, particleDG, mp, Dp, sp);
                }
                else if (verbose) {
                    BOW_TIMER_FLAG("writeSubstep");
                    Bow::writeDFGMPMToPartio(outputPath + "/p" + std::to_string(frame * numSubsteps + currSubstep) + ".bgeo", x, v, particleDG, mp, Dp, sp);

                    //Write Grid
                    collect_gridData();
                    Bow::writeDFGMPMNodesToPartio(outputPath + "/i" + std::to_string(frame * numSubsteps + currSubstep) + ".bgeo", activeNodesX, activeNodesDG, activeNodesV1, activeNodesV2, activeNodesFct1, activeNodesFct2, activeNodesM1, activeNodesM2, activeNodesSeparable);
                }

                Bow::Timer::flush();
            }
        }
        return;
    }
};

int main()
{
    // Bow::DFGMPM::GridState<T, dim> gs;
    // cout << sizeof(gs) << std::endl;
    // return 0;

    std::string outputPath = "output/DFGMPM";
    double dx = 0.02;
    double dt = 0.001;
    int endFrame = 240;
    int fps = 24;
    T gravity = 0;
    double st = 25.0; //25 too low?
    bool symplectic = true;
    bool useAPIC = true;
    bool verbose = true;
    bool useDFG = true;
    T fricCoeff = 0.1;

    T flipPicRatio = 0.95;

    //Damage Params
    double eta = 0.1;
    double zeta = 1.0;
    double p = 0.1;
    double dMin = 0.25;

    DFGMPMSolver solver(dx, dt, endFrame, fps, gravity, st, symplectic, verbose, useDFG, fricCoeff, useAPIC, flipPicRatio);
    solver.outputPath = outputPath;
    solver.addAnisoMPMDamage(eta, dMin, zeta, p);
    solver.simulate();
    return 0;
}
