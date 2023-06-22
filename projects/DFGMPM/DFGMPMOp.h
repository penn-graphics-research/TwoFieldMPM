#pragma once

//#include <Bow/Simulation/MPM/DFGMPMGrid.h>
#include "DFGMPMGrid.h"
#include <Bow/IO/ply.h>
#include <Bow/Geometry/Hybrid/ParticlesLevelSet.h>
//#include <Bow/Math/LinearSolver/ConjugateGradient.h>
#include <Bow/Utils/Timer.h>
#include <Bow/Utils/Logging.h>
#include <tbb/tbb.h>
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>

using namespace SPGrid;

namespace Bow {
namespace DFGMPM {

class AbstractOp {
};

/* Sort particles into backGrid */
template <class T, int dim>
class BackGridSortOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;

    DFGMPMGrid<T, dim>& grid;
    T rp; //kernel radius for DFG
    Field<int>& m_marker;
    Field<bool> m_useDamage;

    void operator()()
    {
        BOW_TIMER_FLAG("backGridSort");

        //First, clear mappedParticles for all grid nodes
        grid.iterateGrid([&](const Vector<int, dim>& node, DFGMPM::GridState<T, dim>& g) {
            g.mappedParticles.clear();
        });
        
        //Now we can do backGrid sorting!
        grid.serial_for([&](int i) {   
            if((!grid.crackInitialized || i < grid.crackParticlesStartIdx) && (m_marker[i] == 0 || m_marker[i] == 5)){ //skip crack particles, fluid particles, or non-damaging solid particles     
                const Vector<T, dim> pos = m_X[i];

                //convert particle position to gridIdx
                Vector<int, dim> gridIdx;
                Vector<T, dim> temp;
                gridIdx = Vector<int, dim>::Zero();
                temp = (pos / rp);
                for (int d = 0; d < dim; ++d) {
                    gridIdx[d] = (int)temp[d];
                }
                //add this particle index to the grid node's mappedParticle list
                grid[gridIdx].mappedParticles.push_back(i);
            }
        });
    }
};

/* Sort into particle neighbor lists */
template <class T, int dim>
class NeighborSortOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;

    Bow::Field<std::vector<int>>& particleNeighbors; //neighbor list

    DFGMPMGrid<T, dim>& grid;
    T rp; //kernel radius for DFG
    Field<int>& m_marker;
    Field<bool> m_useDamage;

    void operator()()
    {
        BOW_TIMER_FLAG("particleNeighborSorting");
        grid.parallel_for([&](int p_i) {
            if((!grid.crackInitialized || p_i < grid.crackParticlesStartIdx) && (m_marker[p_i] == 0 || m_marker[p_i] == 5)){ //skip crack particles, fluid particles, or non-damaging solid particles    
                const Vector<T, dim> pos_i = m_X[p_i];
                BSplineWeights<T, dim> spline(pos_i, rp);
                particleNeighbors[p_i].clear(); //empty neighbor list for this particle before we fill it up
                grid.iterateNeighbors_ClosestNode(spline, 1, [&](const Vector<int, dim>& node, GridState<T, dim>& g) { //note we pass radius = 1 for this operation
                    //Iterate through all particles mapped to this node to see if they neighbor the current particle
                    for (size_t j = 0; j < g.mappedParticles.size(); ++j) {
                        int p_j = g.mappedParticles[j];
                        const Vector<T, dim> pos_j = m_X[p_j];
                        T dist = (pos_i - pos_j).norm();
                        if ((p_i != p_j) && (dist < rp)) {
                            //if particle is not the same particle AND distance is less than kernel radius, add to neighbor list
                            particleNeighbors[p_i].push_back(p_j);
                        }
                    }
                });
            }
        });
    }
};

/* Surface Detection (set sp values before DGs!!!)
   NOTE: Set whether this particle is a surface particle or not by comparing the kernel sum, S(x), against a user threshold (st) */
template <class T, int dim>
class SurfaceDetectionOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;

    Bow::Field<std::vector<int>>& particleNeighbors; //neighbor list

    T rp; //kernel radius for DFG
    T st; //surface threshold

    std::vector<int>& sp; //contains whether particle is surface or not

    DFGMPMGrid<T, dim>& grid;
    Field<int>& m_marker;
    
    void operator()()
    {
        BOW_TIMER_FLAG("surfaceDetection");

        //parallel loop over particles
        tbb::parallel_for(tbb::blocked_range<int>(0, particleNeighbors.size()), [&](tbb::blocked_range<int> b) {
            for (int p_i = b.begin(); p_i < b.end(); ++p_i) {
                if((!grid.crackInitialized || p_i < grid.crackParticlesStartIdx) && (m_marker[p_i] == 0 || m_marker[p_i] == 5)){ //skip crack particles or fluid particles if we have them   
                    const Vector<T, dim> pos_i = m_X[p_i];
                    T S = 0.0;
                    for (size_t j = 0; j < particleNeighbors[p_i].size(); ++j) { //iter particle neighbors
                        int p_j = particleNeighbors[p_i][j];
                        const Vector<T, dim> pos_j = m_X[p_j];
                        T rBar = (pos_i - pos_j).norm() / rp;
                        T omega = 1 - (3 * rBar * rBar) + (2 * rBar * rBar * rBar);
                        if (rBar < 0.0 || rBar > 1.0) {
                            omega = 0.0;
                        }
                        S += omega;
                    }
                    if (S <= st && st != 0) {
                        sp[p_i] = 1;
                    }
                    else if (sp[p_i] != 1) {
                        sp[p_i] = 0;
                    }
                }
            }
        });
    }
};

/* Compute DG for all particles and for all nodes, because of colored looper we can do this more easily! */
template <class T, int dim>
class ComputeDamageGradientsOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    Bow::Field<std::vector<int>>& particleNeighbors; //neighbor list
    T rp; //kernel radius for DFG
    T dx;
    Bow::Field<Bow::Vector<T, dim>>& particleDG;
    std::vector<T>& Dp;
    std::vector<int>& sp;

    DFGMPMGrid<T, dim>& grid;
    Field<int>& m_marker;
    Field<bool> m_useDamage;

    void operator()()
    {

        BOW_TIMER_FLAG("computeDGs");
        grid.colored_for([&](int p_i) {
            if((!grid.crackInitialized || p_i < grid.crackParticlesStartIdx) && (m_marker[p_i] == 0 || m_marker[p_i] == 5) && m_useDamage[p_i]){ //skip crack particles, fluid particles, or non-damaging solid particles 
                const Vector<T, dim> pos_i = m_X[p_i];
                T D = 0.0;
                T S = 0.0;
                Bow::Vector<T, dim> nablaD, nablaS;
                nablaD = Bow::Vector<T, dim>::Zero();
                nablaS = Bow::Vector<T, dim>::Zero();

                //Compute kernel contributions from neighbor particles
                for (size_t j = 0; j < particleNeighbors[p_i].size(); ++j) { //iter particle neighbors
                    int p_j = particleNeighbors[p_i][j];
                    const Vector<T, dim> pos_j = m_X[p_j];
                    T dist = (pos_i - pos_j).norm();
                    T rBar = dist / rp;
                    Vector<T, dim> rBarGrad = (pos_i - pos_j) * (1.0 / (rp * dist));

                    T omega = 1 - (3 * rBar * rBar) + (2 * rBar * rBar * rBar);
                    T omegaPrime = 6 * ((rBar * rBar) - rBar);
                    if (rBar < 0.0 || rBar > 1.0) {
                        omega = 0.0;
                        omegaPrime = 0.0;
                    }

                    T maxD = std::max(Dp[p_j], (T)sp[p_j]);

                    D += maxD * omega;
                    S += omega;
                    nablaD += (maxD * omegaPrime * rBarGrad);
                    nablaS += (omegaPrime * rBarGrad);
                }

                //Now, with contributions summed up, compute this particle's DG
                Bow::Vector<T, dim> nablaDBar;
                if (S == 0) {
                    nablaDBar = Bow::Vector<T, dim>::Zero();
                }
                else {
                    nablaDBar = (nablaD * S - D * nablaS) / (S * S);
                }
                particleDG[p_i] = nablaDBar;

                //Now, we still have to iterate over all grid nodes that this particle maps to so we can compute the gridDG as the maximum norm particle DG mapping to it!
                //NOTE: we use colored looper in this op to prevent race conditions
                BSplineWeights<T, dim> spline(pos_i, dx);
                grid.iterateKernelWithoutWeights(spline, [&](const Vector<int, dim>& node, GridState<T, dim>& g) {
                    if (g.gridMaxNorm < nablaDBar.norm()) {
                        g.gridMaxNorm = nablaDBar.norm();
                        g.gridDG = nablaDBar;
                    }
                });
            }
        });
    }
};

/* Partition particles into field 1 or field 2 while transferring mass to the proper field 
NOTE: mass transfer occurs here for DFG (but velocity transfer is later after computing separability) */
template <class T, int dim>
class PartitioningOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    std::vector<T>& m_mass;
    Bow::Field<Bow::Vector<T, dim>>& particleDG;
    Bow::Field<std::vector<int>>& particleAF;
    std::vector<T>& Dp;
    std::vector<int>& sp;

    T dx;
    T minDp;
    T dMin;

    DFGMPMGrid<T, dim>& grid;
    Field<int>& m_marker;

    T massRatio;

    void operator()()
    {   
        //SEPARABLE VALUE KEY
        //0 = single field
        //1 = two field with contact
        //2 = two field, contact treats like single field with v_cm
        //3 = solid and fluid detected
        //4 = only fluid detected so far
        //5 = only solid detected so far
        
        BOW_TIMER_FLAG("partitioning");
        grid.colored_for([&](int i) {
            if(!grid.crackInitialized || i < grid.crackParticlesStartIdx){ //skip crack particles if we have them
                const Vector<T, dim> pos = m_X[i];
                const T mass = m_mass[i];
                particleAF[i].clear(); //clear out this particle's activeField list because we are about to fill it for this substep

                T maxD = std::max(Dp[i], (T)sp[i]); //use max of damage and surface particle markers so we detect green case correctly

                //Set Active Fields for each grid node!
                BSplineWeights<T, dim> spline(pos, dx);
                grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, const Vector<T, dim>& dw, GridState<T, dim>& g) {
                    if (particleDG[i].dot(g.gridDG) >= 0) {
                        
                        //field 1
                        g.m1 += mass * w; //add mass to active field for this particle
                        g.gridSeparability[0] += w * maxD * mass; //numerator, field 1
                        g.gridSeparability[2] += w * mass; //denom, field 1
                        particleAF[i].push_back(0); //set this particle's AF to 0 for this grid node
                        if (g.gridMaxDamage[0] < maxD) {
                            g.gridMaxDamage[0] = maxD; //compute the max damage seen in this field at this grid node
                        }

                        //Now mark separable = 4 if this was fluid, 5 if it was solid, and mark it 3 if we've now been hit by both solid and fluid!
                        if(g.separable == 0){ //if previously never mapped to
                            if(m_marker[i] == 0 || m_marker[i] == 5){
                                g.separable = 5;
                            }
                            else if(m_marker[i] == 4){
                                g.separable = 4;
                            }
                        }
                        else if(g.separable == 4){ //if previously hit fluid
                            if(m_marker[i] == 5){
                                g.separable = 3; //already hit by fluid, now hit by solid
                            }
                        }
                        else if(g.separable == 5){ //if previously hit solid
                            if(m_marker[i] == 4){
                                g.separable = 3; //already hit by solid, now hit by fluid
                            }
                        }
                    }
                    else {
                        //field 2
                        g.m2 += mass * w;
                        g.gridSeparability[1] += w * maxD * mass; //numerator, field 2
                        g.gridSeparability[3] += w * mass; //denom, field 2
                        particleAF[i].push_back(1); //set this particle's AF to 1 for this grid node
                        if (g.gridMaxDamage[1] < maxD) {
                            g.gridMaxDamage[1] = maxD; //compute the max damage seen in this field at this grid node
                        }
                    }
                });
            }
        });

        //TODO: SPECIAL CASE when m2 > 0 AND separable = 3 (partitioned by damage AND coupling)

        //Now iterate grid nodes to compute each one's separability
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            //Compute seperability for field 1 and store as idx 0
            if (g.gridSeparability[2] > 0) {
                g.gridSeparability[0] /= g.gridSeparability[2]; //divide numerator by denominator
            }
            else {
                g.gridSeparability[0] = 0.0;
            }

            //Compute seperability for field 2 and store as idx 1
            if (g.gridSeparability[3] > 0) {
                g.gridSeparability[1] /= g.gridSeparability[3]; //divide numerator by denominator
            }
            else {
                g.gridSeparability[1] = 0.0;
            }

            //NOTE: separable = 0 indicates single field node, separable = 1 indicates two field node
            if (g.m1 > 0 && g.m2 > 0) {
                T minSep, maxMax;
                minSep = std::min(g.gridSeparability[0], g.gridSeparability[1]);
                maxMax = std::max(g.gridMaxDamage[0], g.gridMaxDamage[1]);
                if (maxMax >= minDp && minSep > dMin) {
                    g.separable = 1;
                }
                else {
                    //in this case, still treat as two fields, but will get different contact
                    g.separable = 2;
                    //g.m1 += g.m2;
                    //g.m2 = 0.0;
                }
            }
            else if(g.separable != 3){
                g.separable = 0; //if nothing in field 2 and we didn't detect solid-fluid coupling reset separable = 0
            }

            if(g.separable == 3){
                g.m1 = 0;
                g.m2 = 0; //if solid-fluid coupling, set these 0 for later transfer in P2G
            }
        });

        
        //if massRatio is set, we accumulate solid and fluid masses for this sep3 node, then check the mass ratio to see whether it should be sep3 or sep6
        if(massRatio > 0){

            //Now accumulate solid and fluid mass for separable = 3 case to detect whether massRatio is low enough
            grid.colored_for([&](int i) {
                if(!grid.crackInitialized || i < grid.crackParticlesStartIdx){ //skip crack particles if we have them
                    const Vector<T, dim> pos = m_X[i];
                    const T mass = m_mass[i];
                    BSplineWeights<T, dim> spline(pos, dx);
                    grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, const Vector<T, dim>& dw, GridState<T, dim>& g) {
                        if(g.separable == 3){ //coupling case, always transfer solid to field 1 and fluid to field 2
                            int materialIdx = m_marker[i];
                            if(materialIdx == 0 || materialIdx == 5){
                                g.m1 += mass * w; //have to do this here since we couldn't earlier without interfering with DFG partitioning
                            }
                            else if(materialIdx == 4){ //transfer fluid particles to field 2
                                g.m2 += mass * w; //have to do this here since we couldn't earlier without interfering with DFG partitioning
                            }
                        }
                    });
                }
            });

            //check massRatio (if it's set), if good stay sep = 3, otherwise switch to sep = 0
            grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
                //Now check massRatio to determine whether it should model solid fluid coupling or not
                T maxMass, minMass, particleMassRatio;
                if(g.separable == 3){
                    maxMass = std::max(g.m1, g.m2);
                    minMass = std::min(g.m1, g.m2);
                    particleMassRatio = maxMass / minMass;
                    if(particleMassRatio > massRatio){ //TODO: massRatio can be user defined!!
                        g.separable = 6; //like sep = 2 case but for solid fluid coupling
                        // g.separable = 0;
                        // g.m1 += g.m2;
                        // g.m2 = 0.0;
                    }
                }
            });
        }

        grid.countSeparableNodes();
    }
};

/* Transfer damage to the grid, then compute for each particle the damage Laplacian */
template <class T, int dim>
class ComputeDamageLaplaciansOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    std::vector<T>& m_Dp; //particle damage
    std::vector<T>& m_damageLaplacians; //particle damage Laplacians
    Bow::Field<std::vector<int>>& particleAF;
    T dx;

    DFGMPMGrid<T, dim>& grid;

    void operator()()
    {
        BOW_TIMER_FLAG("computeDamageLaplacians");

        //P2G STYLE COLORED PARALLEL LOOP: here we transfer numerator and denominator for grid damage
        grid.colored_for([&](int i) {
            if(!grid.crackInitialized || i < grid.crackParticlesStartIdx){ //skip crack particles if we have them
                const Vector<T, dim> pos = m_X[i];
                const T Dp = m_Dp[i];
                BSplineWeights<T, dim> spline(pos, dx);
                grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, const Vector<T, dim>& dw, GridState<T, dim>& g) {
                    if (g.separable == 0) {
                        //single field treatment
                        g.d1[0] += w * Dp;
                        g.d1[1] += w;
                    }
                    else {
                        //two-field treatment -- must transfer to the correct field in the separable grid nodes
                        int fieldIdx = particleAF[i][oidx]; //grab the field that this particle belongs in for this grid node (oidx)
                        if (fieldIdx == 0) {
                            g.d1[0] += w * Dp;
                            g.d1[1] += w;
                        }
                        else {
                            g.d2[0] += w * Dp;
                            g.d2[1] += w;
                        }
                    }
                });
            }
        });

        //PARALLEL GRID ITERATION: now we divide out the denominator to get final d_i
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            //Compute d_i for field 1 and store as idx 0
            if (g.d1[1] > 0) {
                g.d1[0] /= g.d1[1]; //divide numerator by denominator
            }
            else {
                g.d1[0] = 0.0;
            }
            if (g.separable != 0) {
                //Compute d_i for field 2 and store as idx 0
                if (g.d2[1] > 0) {
                    g.d2[0] /= g.d2[1]; //divide numerator by denominator
                }
                else {
                    g.d2[0] = 0.0;
                }
            }
        });

        //G2P STYLE PARALLEL PARTICLE LOOP: now we compute each particle's damage laplacian
        grid.parallel_for([&](int i) {
            if(!grid.crackInitialized || i < grid.crackParticlesStartIdx){ //skip crack particles if we have them
                Vector<T, dim>& pos = m_X[i];
                BSplineWeightsWithSecondOrder<T, dim> spline(pos, dx);
                grid.iterateKernelWithLaplacian(spline, [&](const Vector<int, dim>& node, int oidx, T w, Vector<T, dim> dw, T laplacian, GridState<T, dim>& g) {
                    if (g.separable == 0) {
                        //single field treatment
                        m_damageLaplacians[i] += g.d1[0] * laplacian;
                    }
                    else {
                        //two-field treatment -- must transfer to the correct field in the separable grid nodes
                        int fieldIdx = particleAF[i][oidx]; //grab the field that this particle belongs in for this grid node (oidx)
                        if (fieldIdx == 0) {
                            m_damageLaplacians[i] += g.d1[0] * laplacian;
                        }
                        else {
                            m_damageLaplacians[i] += g.d2[0] * laplacian;
                        }
                    }
                });
            }
        });
    }
};

/* Update damage using explicit AnisoMPM style damage */
template <class T, int dim>
class UpdateAnisoMPMDamageOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPMGrid<T, dim>::SparseMask;
    std::vector<T>& m_Dp; //particle damage
    std::vector<T>& m_damageLaplacians; //particle damage Laplacians
    std::vector<T>& m_dTildeH; //particle dTilde history (max seen)
    std::vector<T>& m_sigmaC; //particle sigmaC
    Field<Matrix<T, dim, dim>>& cauchy; //this holds cauchy stress
    T dt;
    T eta;
    T zeta;
    T l0;

    DFGMPMGrid<T, dim>& grid;

    void operator()()
    {
        BOW_TIMER_FLAG("updateAnisoMPMDamage");

        //PARALLEL PARTICLE LOOP: compute updated particle damage for each particle
        grid.parallel_for([&](int i) {
            if(!grid.crackInitialized || i < grid.crackParticlesStartIdx){ //skip crack particles if we have them
                const T Dp = m_Dp[i];
                const T Dc = Dp - (l0 * l0 * m_damageLaplacians[i]);
                const T sigmaC = m_sigmaC[i];

                //Compute sigmaPlus (tensile portion of cauchy stress)
                Matrix<T, dim, dim> sigmaPlus = Matrix<T, dim, dim>::Zero();
                Vector<T, dim> eigenVecs;
                Eigen::EigenSolver<Matrix<T, dim, dim>> es(cauchy[i]);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++) {
                        eigenVecs(j) = es.eigenvectors().col(i)(j).real(); //get the real parts of each eigenvector
                    }
                    T lambda = es.eigenvalues()(i).real();
                    sigmaPlus += ((lambda + std::abs(lambda)) / 2.0) * (eigenVecs * eigenVecs.transpose());
                }

                //Structural Tensor (dictates anisotropy)
                Matrix<T, dim, dim> A = Matrix<T, dim, dim>::Identity(); //TODO: insert real structural tensor eventually, but for now this will be hardcoded to be isotropic

                //Compute Phi
                T contraction = 0;
                Matrix<T, dim, dim> Asig = A * sigmaPlus;
                Matrix<T, dim, dim> sigA = sigmaPlus * A;
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++) {
                        contraction += (Asig(i, j) * sigA(i, j));
                    }
                }
                T phi = ((T)1 / (sigmaC * sigmaC)) * contraction;

                //Compute dTilde
                T dTilde = std::max(m_dTildeH[i], zeta * (((phi - 1) + std::abs(phi - 1)) / (T)2.0)); //remember macaulay function is y(x) = (x+|x|)/2
                m_dTildeH[i] = dTilde; //update max history

                //Update damage if necessary
                T diff = ((1 - Dp) * dTilde) - Dc;
                T newD = Dp + ((dt / eta) * ((diff + std::abs(diff)) / 2.0)); //macaulay ensures that we only update when we should (if diff <= 0 expression returns 0)
                m_Dp[i] = std::min(newD, (T)1); //update damage
            }
        });
    }
};

/* Transfer particle quantities to the grid as well as explicit update for velocity*/
template <class T, int dim>
class ParticlesToGridOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    Field<Vector<T, dim>>& m_V;
    std::vector<T>& m_mass;
    Field<Matrix<T, dim, dim>>& m_C;
    Field<Matrix<T, dim, dim>>& stress;

    T gravity;

    Bow::Field<std::vector<int>>& particleAF;

    DFGMPMGrid<T, dim>& grid;
    T dx;
    T dt;

    bool symplectic;
    bool useDFG;
    bool useAPIC;
    bool useImplicitContact;

    Field<T>& m_mu;
    Field<T>& m_la;
    Field<T>& m_vol;

    void operator()()
    {
        BOW_TIMER_FLAG("P2G");
        grid.colored_for([&](int i) {
            if(!grid.crackInitialized || i < grid.crackParticlesStartIdx){ //skip crack particles if we have them
                const Vector<T, dim> pos = m_X[i];
                const Vector<T, dim> v = m_V[i];
                const T mass = m_mass[i];
                const Matrix<T, dim, dim> C = m_C[i] * mass; //C * m_p
                const Vector<T, dim> momentum = mass * v; //m_p * v_p
                const Matrix<T, dim, dim> delta_t_tmp_force = -dt * stress[i]; //stress holds Vp^0 * PF^T
                BSplineWeights<T, dim> spline(pos, dx);

                //Computations for implicit barrier contact
                T mu = 0;
                T la = 0;
                T E = 0;
                T vol = 0;
                if(!symplectic && useImplicitContact){
                    mu = m_mu[i];
                    la = m_la[i];
                    E = (mu*(3*la + 2*mu)) / (la + mu);
                    vol = m_vol[i];
                }
                
                grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, const Vector<T, dim>& dw, GridState<T, dim>& g) {
                    Vector<T, dim> xi_minus_xp = node.template cast<T>() * dx - pos;
                    Vector<T, dim> velocity_term_APIC = Vector<T, dim>::Zero();
                    velocity_term_APIC = momentum + C * xi_minus_xp; //mv + (C @ dpos)
                    Vector<T, dim> stress_term_dw = Vector<T, dim>::Zero();
                    if (symplectic) {
                        stress_term_dw = delta_t_tmp_force * dw; //only add forces if symplectic, else added in residual
                    }
                    Vector<T, dim> delta_APIC = w * velocity_term_APIC + stress_term_dw;
                    Vector<T, dim> delta_FLIP = w * momentum + stress_term_dw;
                    Vector<T, dim> delta_vn = w * momentum; //we'll use this to compute v1^n and v2^n for FLIP

                    //Notice we treat single-field and two-field nodes differently
                    //NOTE: remember we are also including explicit force here if symplectic!
                    if (g.separable == 0 || !useDFG) {
                        //Single-field treatment if separable = 0 OR if we are using single field MPM
                        if (useAPIC) {
                            g.v1 += delta_APIC;
                        }
                        else {
                            g.v1 += delta_FLIP;
                            g.vn1 += delta_vn; //transfer momentum to compute v^n
                        }

                        //Now transfer mass if we aren't using DFG (this means we skipped massP2G earlier)
                        if (!useDFG) {
                            g.m1 += mass * w;
                        }
                    }
                    else if (g.separable != 0 && useDFG) {
                        //Treat node as having two fields
                        int fieldIdx = particleAF[i][oidx]; //grab the field that this particle belongs in for this grid node (oidx)
                        if (fieldIdx == 0) {
                            if (useAPIC) {
                                g.v1 += delta_APIC;
                            }
                            else {
                                g.v1 += delta_FLIP;
                                g.vn1 += delta_vn; //transfer momentum to compute v^n
                            }
                            //Compute normal for field 1 particles
                            g.n1 += mass * dw; //remember to normalize this later!
                            
                            //Compute Barrier ViYi value (if implicit)
                            if(!symplectic && useImplicitContact){
                                g.gridViYi1 += vol * E * w;
                            }

                        }
                        else if (fieldIdx == 1) {
                            if (useAPIC) {
                                g.v2 += delta_APIC;
                            }
                            else {
                                g.v2 += delta_FLIP;
                                g.vn2 += delta_vn; //transfer momentum to compute v^n
                            }
                            //Compute normal for field 2 particles
                            g.n2 += mass * dw; //remember to normalize this later!

                            //Compute Barrier ViYi value (if implicit)
                            if(!symplectic && useImplicitContact){
                                g.gridViYi2 += vol * E * w;
                            }
                        }
                    }
                });
            }
        });

        grid.countNumNodes();

        /* Iterate grid to divide out the grid masses from momentum and then add gravity */
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            Vector<T, dim> gravity_term = Vector<T, dim>::Zero();
            if (symplectic) {
                gravity_term[1] = gravity * dt; //only nonzero if symplectic, else we add gravity in residual of implicit solve
            }
            T mass1 = g.m1;
            Vector<T, dim> alpha1;
            alpha1 = Vector<T, dim>::Ones() * ((T)1 / mass1);
            g.v1 = g.v1.cwiseProduct(alpha1);
            g.v1 += gravity_term;
            g.vn1 = g.vn1.cwiseProduct(alpha1); // this is how we get v1^n
            g.x1 = node.template cast<T>() * dx; //put nodal position in x1 regardless of separability
            if (g.separable != 0) {
                T mass2 = g.m2;
                Vector<T, dim> alpha2;
                alpha2 = Vector<T, dim>::Ones() * ((T)1 / mass2);
                g.v2 = g.v2.cwiseProduct(alpha2);
                g.v2 += gravity_term;
                g.vn2 = g.vn2.cwiseProduct(alpha2); //this is how we get v2^n
            }
        });
    }
};

/* Iterate grid to calculate impulse and either apply it (symplectic) or save for later (implicit) */
template <class T, int dim>
class ApplyImpulseOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPMGrid<T, dim>::SparseMask;
    Vector<T, dim> center;
    T strength;

    DFGMPMGrid<T, dim>& grid;
    T dx;
    T dt;

    bool symplectic;
    bool useImpulse;

    void operator()()
    {
        BOW_TIMER_FLAG("applyImpulse");

        /* Iterate grid to calculate impulse and either apply it (symplectic) or save for later (implicit) */
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            
            Vector<T, dim> dist = center - (node.template cast<T>() * dx);
            Vector<T, dim> dv = dist / (0.01 + dist.norm()) * strength * dt;

            if(symplectic){
                //apply directly
                g.v1 += dv;
                if(g.separable != 0){
                    g.v2 += dv;
                }
            } 
            else {
                //save for later
                g.fi1 += dv;
                if(g.separable != 0){
                    g.fi2 += dv;
                }
            }
        });
    }
};

/* For each separable node, compute the frictional contact forces for each field
NOTE: We will directly apply these, but it happens at different stages in the pipeline for explicit/implicit */
template <class T, int dim>
class ContactForcesOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPMGrid<T, dim>::SparseMask;

    T dt;
    T fricCoeff;
    bool symplectic;
    bool useImplicitContact;

    DFGMPMGrid<T, dim>& grid;

    void operator()()
    {
        BOW_TIMER_FLAG("frictionalContact");

        //Iterate grid nodes to compute contact forces
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            //For separable nodes, compute the frictional contact forces for each field
            if (g.separable != 0) {
                //Grab momentum
                Vector<T, dim> q1 = g.v1 * g.m1;
                Vector<T, dim> q2 = g.v2 * g.m2;
                Vector<T, dim> q_cm = q1 + q2;

                //Grab mass
                T m_1 = g.m1;
                T m_2 = g.m2;
                T m_cm = m_1 + m_2;

                //Grab velocity
                Vector<T, dim> v_1 = g.v1;
                Vector<T, dim> v_2 = g.v2;
                Vector<T, dim> v_cm = q_cm / m_cm; //THIS IS CRUCIAL we need to compute v_cm using momentum

                //Grab normals and normalize them
                Vector<T, dim> n_1 = g.n1;
                Vector<T, dim> n_2 = g.n2;
                n_1.normalize(); //in place normalize (NOTE: also makes sure no div by zero)
                n_2.normalize();

                //Compute COM normals and save them in original space
                Vector<T, dim> n_cm1 = (n_1 - n_2).normalized();
                Vector<T, dim> n_cm2 = -n_cm1;
                g.n1 = n_cm1;
                g.n2 = n_cm2;

                //NOTE: this is where we stop for implicitMPM with implicitContact

                if (symplectic || !useImplicitContact) { //only compute these forces for explicitMPM or if we want to use explicitContact in implicitMPM
                    T fNormal1 = (m_1 / dt) * (v_cm - v_1).dot(n_cm1);
                    T fNormal2 = (m_2 / dt) * (v_cm - v_2).dot(n_cm2);
                    Vector<T, dim> fTanComp1 = Bow::Vector<T, dim>::Zero();
                    Vector<T, dim> fTanComp2 = Bow::Vector<T, dim>::Zero();
                    if constexpr (dim == 2) { //compute tangent component for 2D
                        Vector<T, dim> s_cm1 = Bow::Vector<T, dim>(-1 * n_cm1[1], n_cm1[0]);
                        Vector<T, dim> s_cm2 = s_cm1;
                        T fTan1 = (m_1 / dt) * (v_cm - v_1).dot(s_cm1);
                        T fTan2 = (m_2 / dt) * (v_cm - v_2).dot(s_cm2);
                        fTanComp1 = fTan1 * s_cm1;
                        fTanComp2 = fTan2 * s_cm2;
                    }
                    else { //compute tangent component for 3D
                        T v1x = n_cm1[0];
                        T v1y = n_cm1[1];
                        T v1z = n_cm1[2];
                        //Compute 3D orthonormal basis with n_cm1 = v1
                        //NOTE: we have 6 cases here depending on the zeros in n_cm1
                        Vector<T, dim> s_cm1A = Bow::Vector<T, dim>::Zero();
                        if (v1x == 0 && v1y == 0 && v1z != 0) {
                            // (0 0 x)
                            s_cm1A = Bow::Vector<T, dim>(1, 0, 0);
                        }
                        else if (v1x == 0 && v1y != 0 && v1z == 0) {
                            // (0 x 0)
                            s_cm1A = Bow::Vector<T, dim>(1, 0, 0);
                        }
                        else if (v1x != 0 && v1y == 0 && v1z == 0) {
                            // (x 0 0)
                            s_cm1A = Bow::Vector<T, dim>(0, 1, 0);
                        }
                        else if (v1x == 0 && v1y == 0 && v1z == 0) {
                            // (0 0 0)
                            s_cm1A = Bow::Vector<T, dim>::Zero();
                        }
                        else if (v1x != 0 && v1y != 0 && v1z == 0) {
                            // (x x 0)
                            s_cm1A = Bow::Vector<T, dim>(-1 * v1y, v1x, 0).normalized();
                        }
                        else {
                            // (x x x), (0 x x), (x 0 x)
                            //Let v2x = v1x, v2y = 2*v1y to prevent linear dependence on v1. Then set v3 based on solving v1 dot v2 = 0
                            T v2x = 1 * v1x;
                            T v2y = 2 * v1y;
                            T v2z = -1 * ((v1x * v1x) + (2 * (v1y * v1y))) / v1z;
                            s_cm1A = Bow::Vector<T, dim>(v2x, v2y, v2z).normalized();
                        }

                        //third vector is always cross product of other two
                        Vector<T, dim> s_cm1B = n_cm1.cross(s_cm1A).normalized();

                        //set same as for field 1
                        Vector<T, dim> s_cm2A = s_cm1A;
                        Vector<T, dim> s_cm2B = s_cm1B;

                        //Compute tangent components
                        T fTan1A = (m_1 / dt) * (v_cm - v_1).dot(s_cm1A);
                        T fTan1B = (m_1 / dt) * (v_cm - v_1).dot(s_cm1B);
                        T fTan2A = (m_2 / dt) * (v_cm - v_2).dot(s_cm2A);
                        T fTan2B = (m_2 / dt) * (v_cm - v_2).dot(s_cm2B);

                        //Compute tangent component for each field
                        fTanComp1 = (fTan1A * s_cm1A) + (fTan1B * s_cm1B);
                        fTanComp2 = (fTan2A * s_cm2A) + (fTan2B * s_cm2B);
                    }

                    Vector<T, dim> f_c1 = Bow::Vector<T, dim>::Zero();
                    Vector<T, dim> f_c2 = Bow::Vector<T, dim>::Zero();
                    if(g.separable == 1 || g.separable == 3){ //separable two field nodes
                        //Compute magnitude of tangent components for each field
                        T fTanMag1 = fTanComp1.norm();
                        T fTanMag2 = fTanComp2.norm();

                        //Sign of tangent component
                        T fTanSign1 = (fTanMag1 > 0) ? 1.0 : 0.0; //NOTE: L2 norm is always >= 0
                        T fTanSign2 = (fTanMag2 > 0) ? 1.0 : 0.0;

                        //Tangent directions, all 0 if sign of mag was 0
                        Vector<T, dim> tanDirection1 = (fTanSign1 == 0) ? Bow::Vector<T, dim>::Zero() : fTanComp1.normalized();
                        Vector<T, dim> tanDirection2 = (fTanSign2 == 0) ? Bow::Vector<T, dim>::Zero() : fTanComp2.normalized();

                        T tanMin1 = (fricCoeff * std::abs(fNormal1) < std::abs(fTanMag1)) ? (fricCoeff * std::abs(fNormal1)) : std::abs(fTanMag1);
                        T tanMin2 = (fricCoeff * std::abs(fNormal2) < std::abs(fTanMag2)) ? (fricCoeff * std::abs(fNormal2)) : std::abs(fTanMag2);

                        //Finally compute contact forces!
                        //NOTE: we ONLY compute this if we detect interpenetration b/w the fields
                        // if ((v_cm - v_1).dot(n_cm1) + (v_cm - v_2).dot(n_cm2) < 0) {
                        //     //interpenetration detected
                        //     f_c1 = (fNormal1 * n_cm1) + (tanMin1 * fTanSign1 * tanDirection1);
                        //     f_c2 = (fNormal2 * n_cm2) + (tanMin2 * fTanSign2 * tanDirection2);
                        // }
                        if ((v_cm - v_1).dot(n_cm1) < 0) { //NOTE: homel2016 has this sign flipped, and upon experimenting this is DEFINITELY wrong. Keep it < 0!
                            //interpenetration detected, field 1
                            f_c1 = (fNormal1 * n_cm1) + (tanMin1 * fTanSign1 * tanDirection1);
                        }
                        if ((v_cm - v_2).dot(n_cm2) < 0) {
                            //interpenetration detected, field 2
                            f_c2 = (fNormal2 * n_cm2) + (tanMin2 * fTanSign2 * tanDirection2);
                        }
                    } 
                    else if(g.separable == 2 || g.separable == 6){ //non separable two field nodes (sep6 is the non separable (based on mass ratio) solid fluid two field case)
                        //treat two-field non-separable as "single field" --> correct each to v_cm I think
                        f_c1 = (fNormal1 * n_cm1) + fTanComp1;
                        f_c2 = (fNormal2 * n_cm2) + fTanComp2;
                    }

                    //Now let's save these contact forces and update velocity
                    g.fct1 = f_c1;
                    g.fct2 = f_c2;
                    g.v1 += (f_c1 / m_1) * dt;
                    g.v2 += (f_c2 / m_2) * dt;
                }
            }
        });
    }
};

template <class T, int dim>
class BoundaryConditionUpdateOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPMGrid<T, dim>::SparseMask;
    DFGMPMGrid<T, dim>& grid;
    BoundaryConditionManager<T, dim>& BC;
    T dx;
    T dt;

    void operator()()
    {
        BOW_TIMER_FLAG("boundaryCollisions");
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            //always process first field velocity
            Vector<T, dim> new_v1 = g.v1;
            Vector<T, dim> new_vn1 = g.vn1; //for FLIP
            BC.mpm_explicit_update(node.template cast<T>() * dx, new_v1);
            BC.mpm_explicit_update(node.template cast<T>() * dx, new_vn1);
            g.v1 = new_v1;
            g.vn1 = new_vn1; //for FLIP
            g.x1 = node.template cast<T>() * dx + dt * new_v1;

            //only process second field if node is separable
            if (g.separable != 0) {
                Vector<T, dim> new_v2 = g.v2;
                Vector<T, dim> new_vn2 = g.vn2;
                BC.mpm_explicit_update(node.template cast<T>() * dx, new_v2);
                BC.mpm_explicit_update(node.template cast<T>() * dx, new_vn2);
                g.v2 = new_v2;
                g.vn2 = new_vn2;
                g.x2 = node.template cast<T>() * dx + dt * new_v2; //advect second field nodal position separately... this feels weird
            }            
        });
    }
};

template <class T, int dim>
class ImplicitBoundaryConditionUpdateOp : public AbstractOp {
public:
    using TM = Matrix<T, dim, dim>;
    DFGMPMGrid<T, dim>& grid;
    BoundaryConditionManager<T, dim>& BC;
    Field<Matrix<T, dim, dim>>& BC_basis;
    Field<int>& BC_order;
    T dx;

    bool useDFG;

    void operator()()
    {   
        int ndof = grid.num_nodes;
        int sdof = useDFG ? grid.separable_nodes : 0;
        BC_basis.assign(ndof + sdof, TM::Identity());
        BC_order.assign(ndof + sdof, 0);
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            BC.mpm_implicit_update(node.template cast<T>() * dx, BC_basis[g.idx], BC_order[g.idx]);
        });
        if(sdof > 0){
            grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
                BC.mpm_implicit_update(node.template cast<T>() * dx, BC_basis[ndof + g.sep_idx], BC_order[ndof + g.sep_idx]);
            });
        }
    }
};

template <class T, int dim>
class GridToParticlesOp : public AbstractOp {
public:
    Field<Vector<T, dim>>& m_X;
    Field<Vector<T, dim>>& m_V;
    Field<Matrix<T, dim, dim>>& m_C;

    Bow::Field<std::vector<int>>& particleAF;

    DFGMPMGrid<T, dim>& grid;
    T dx;
    T dt;

    T flipPicRatio;
    bool useDFG;
    Field<int>& m_marker;

    Field<Matrix<T, dim, dim>> m_gradXp = Field<Matrix<T, dim, dim>>();
    Field<Matrix<T, dim, dim>> m_deformationRates = Field<Matrix<T, dim, dim>>();

    template <bool useAPIC>
    void gridToParticles()
    {
        BOW_TIMER_FLAG("G2P");
        T D_inverse = (T)4 / (dx * dx);
        m_gradXp.assign(m_X.size(), Matrix<T, dim, dim>());
        m_deformationRates.assign(m_X.size(), Matrix<T, dim, dim>());
        grid.parallel_for([&](int i) {
            if(!grid.crackInitialized || i < grid.crackParticlesStartIdx){ //skip crack particles if we have them
                Vector<T, dim>& Xp = m_X[i];
                Vector<T, dim> picV = Vector<T, dim>::Zero();
                Vector<T, dim> oldV = Vector<T, dim>::Zero();
                BSplineWeights<T, dim> spline(Xp, dx);

                Matrix<T, dim, dim> gradXp = Matrix<T, dim, dim>::Identity();
                Matrix<T, dim, dim> gradVp = Matrix<T, dim, dim>::Identity();
                Vector<T, dim> picX = Vector<T, dim>::Zero();
                grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, Vector<T, dim> dw, GridState<T, dim>& g) {
                    if (g.idx < 0) return;
                    
                    Vector<T, dim> xn = node.template cast<T>() * dx; //same regardless of separability
                    //these steps depend on which field the particle is in
                    if (g.separable == 0 || !useDFG) {
                        //treat as single field node
                        picV += w * g.v1;
                        oldV += w * g.vn1; 
                        picX += w * g.x1;
                        //std::cout << "oidx: " << oidx << ", g.x1: " << g.x1 << std::endl;
                        gradXp.noalias() += (g.x1 - xn) * dw.transpose();
                        gradVp.noalias() += g.v1 * dw.transpose();
                    }
                    else if (g.separable == 1 || g.separable == 2) {
                        //treat as two-field node
                        int fieldIdx = particleAF[i][oidx]; //grab the field that this particle belongs in for this grid node (oidx)
                        if (fieldIdx == 0) {
                            picV += w * g.v1;
                            oldV += w * g.vn1;
                            picX += w * g.x1;
                            gradXp.noalias() += (g.x1 - xn) * dw.transpose(); 
                            gradVp.noalias() += g.v1 * dw.transpose(); 
                        }
                        else if (fieldIdx == 1) {
                            picV += w * g.v2;
                            oldV += w * g.vn2;
                            picX += w * g.x2;
                            gradXp.noalias() += (g.x2 - xn) * dw.transpose();
                            gradVp.noalias() += g.v2 * dw.transpose();
                        }
                    }
                    else if (g.separable == 3 || g.separable == 6) { //solid-fluid coupling case
                        //treat as two-field node
                        int materialIdx = m_marker[i]; //solid in field 1, fluid in field 2
                        if (materialIdx == 0 || materialIdx == 5) {
                            picV += w * g.v1;
                            oldV += w * g.vn1;
                            picX += w * g.x1;
                            gradXp.noalias() += (g.x1 - xn) * dw.transpose();  
                            gradVp.noalias() += g.v1 * dw.transpose();
                        }
                        else if (materialIdx == 4) {
                            picV += w * g.v2;
                            oldV += w * g.vn2;
                            picX += w * g.x2;
                            gradXp.noalias() += (g.x2 - xn) * dw.transpose();
                            gradVp.noalias() += g.v2 * dw.transpose();
                        }
                    }
                });
                if constexpr (useAPIC){
                    Matrix<T, dim, dim> Bp = Matrix<T, dim, dim>::Zero();
                    grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, Vector<T, dim> dw, GridState<T, dim>& g) {
                        if (g.idx < 0) return;
                        Vector<T, dim> xn = dx * node.template cast<T>();
                        Vector<T, dim> g_v_new = g.v1;
                        Vector<T, dim> g_v2_new = g.v2;
                        if (g.separable == 0 || !useDFG) {
                            Bp += 0.5 * w * (g_v_new * (xn - m_X[i] + g.x1 - picX).transpose() + (xn - m_X[i] - g.x1 + picX) * g_v_new.transpose());
                        }
                        else if (g.separable == 1 || g.separable == 2){
                            int fieldIdx = particleAF[i][oidx]; //grab field
                            if (fieldIdx == 0) {
                                Bp += 0.5 * w * (g_v_new * (xn - m_X[i] + g.x1 - picX).transpose() + (xn - m_X[i] - g.x1 + picX) * g_v_new.transpose());
                            }
                            else if (fieldIdx == 1) {
                                Bp += 0.5 * w * (g_v2_new * (xn - m_X[i] + g.x2 - picX).transpose() + (xn - m_X[i] - g.x2 + picX) * g_v2_new.transpose());
                            }
                        }
                        else if (g.separable == 3 || g.separable == 6){ //solid-fluid coupling cases
                            int materialIdx = m_marker[i]; //grab materialIdx, solid in field 1, fluid in field 2
                            if (materialIdx == 0 || materialIdx == 5) {
                                Bp += 0.5 * w * (g_v_new * (xn - m_X[i] + g.x1 - picX).transpose() + (xn - m_X[i] - g.x1 + picX) * g_v_new.transpose());
                            }
                            else if (materialIdx == 4) {
                                Bp += 0.5 * w * (g_v2_new * (xn - m_X[i] + g.x2 - picX).transpose() + (xn - m_X[i] - g.x2 + picX) * g_v2_new.transpose());
                            }
                        }
                    });
                    m_C[i] = Bp * D_inverse;
                    m_V[i] = picV;
                }
                else{
                    m_C[i].setZero();
                    //Finish computing FLIP velocity: v_p^n+1 = v_p^n + dt (v_i^n+1 - v_i^n) * wip
                    m_V[i] *= flipPicRatio;
                    m_V[i] += picV - flipPicRatio * oldV; //simplified form
                }
                
                m_X[i] = picX; //use PIC for advection
                m_gradXp[i] = gradXp; //used for updating strain
                m_deformationRates[i] = 0.5 * (gradVp + gradVp.transpose());
            }
        });
    }

    void operator()(bool useAPIC = true)
    {
        if (useAPIC)
            gridToParticles<true>();
        else
            gridToParticles<false>();
    }
};

template <class T, int dim, bool symplectic = true>
class CollectGridDataOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPMGrid<T, dim>::SparseMask;
    DFGMPMGrid<T, dim>& grid;
    T dx;

    Bow::Field<Bow::Vector<T, dim>>& activeNodesX;
    Bow::Field<Bow::Matrix<T, dim, dim>>& activeNodesCauchy1;
    Bow::Field<Bow::Matrix<T, dim, dim>>& activeNodesCauchy2;
    Bow::Field<Bow::Matrix<T, dim, dim>>& activeNodesFi1;
    Bow::Field<Bow::Matrix<T, dim, dim>>& activeNodesFi2;
    Bow::Field<Bow::Vector<T, dim>>& activeNodesDG;
    Bow::Field<Bow::Vector<T, dim>>& activeNodesV1;
    Bow::Field<Bow::Vector<T, dim>>& activeNodesV2;
    Bow::Field<Bow::Vector<T, dim>>& activeNodesFct1;
    Bow::Field<Bow::Vector<T, dim>>& activeNodesFct2;
    std::vector<T>& activeNodesM1;
    std::vector<T>& activeNodesM2;
    std::vector<T>& activeNodesSeparability1;
    std::vector<T>& activeNodesSeparability2;
    std::vector<int>& activeNodesSeparable;
    Bow::Field<Bow::Vector<T, dim>>& activeNodesN1;
    Bow::Field<Bow::Vector<T, dim>>& activeNodesN2;


    void operator()()
    {
        BOW_TIMER_FLAG("collectGridData");

        activeNodesX.clear();
        activeNodesCauchy1.clear();
        activeNodesCauchy2.clear();
        activeNodesFi1.clear();
        activeNodesFi2.clear();
        activeNodesDG.clear();
        activeNodesV1.clear();
        activeNodesV2.clear();
        activeNodesFct1.clear();
        activeNodesFct2.clear();
        activeNodesM1.clear();
        activeNodesM2.clear();
        activeNodesSeparability1.clear();
        activeNodesSeparability2.clear();
        activeNodesSeparable.clear();
        activeNodesN1.clear();
        activeNodesN2.clear();

        //grid.countNumNodes();
        grid.iterateGridSerial([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            Vector<T, dim> xi = node.template cast<T>() * dx;
            activeNodesX.push_back(xi);
            activeNodesCauchy1.push_back(g.cauchy1);
            activeNodesCauchy2.push_back(g.cauchy2);
            activeNodesFi1.push_back(g.Fi1);
            activeNodesFi2.push_back(g.Fi2);
            activeNodesDG.push_back(g.gridDG);
            activeNodesV1.push_back(g.v1);
            activeNodesV2.push_back(g.v2);
            activeNodesFct1.push_back(g.fct1);
            activeNodesFct2.push_back(g.fct2);
            activeNodesM1.push_back(g.m1);
            activeNodesM2.push_back(g.m2);
            activeNodesSeparability1.push_back(g.gridSeparability[0]);
            activeNodesSeparability2.push_back(g.gridSeparability[1]);
            activeNodesSeparable.push_back(g.separable);
            activeNodesN1.push_back(g.n1);
            activeNodesN2.push_back(g.n2);
        });
    }
};
}
} // namespace Bow::DFGMPM