#pragma once

//#include <Bow/Simulation/MPM/DFGMPMGrid.h>
#include "../DFGMPM/DFGMPMGrid.h"
#include <Bow/IO/ply.h>
#include <Bow/Geometry/Hybrid/ParticlesLevelSet.h>
//#include <Bow/Math/LinearSolver/ConjugateGradient.h>
#include <Bow/Utils/Timer.h>
#include <Bow/Utils/Logging.h>
#include <tbb/tbb.h>
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include <Bow/Math/SVD.h>
#include <Bow/Math/PolarDecomposition.h>

using namespace SPGrid;

namespace Bow {
namespace CRAMP {

class AbstractOp {
};

/* Transfer particle quantities to the grid as well as explicit update for velocity*/
template <class T, int dim>
class ParticlesToGridOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    Field<Vector<T, dim>>& m_V;
    std::vector<T>& m_mass;
    Field<Matrix<T, dim, dim>>& m_C;
    Field<Matrix<T, dim, dim>>& stress;

    T gravity;

    Bow::Field<std::vector<int>>& particleAF;

    DFGMPM::DFGMPMGrid<T, dim>& grid;
    T dx;
    T dt;

    bool symplectic;
    bool useDFG;
    bool useAPIC;
    bool useImplicitContact;

    //Field<T> m_vol;

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

                //Grab volume
                //T vol = m_vol[i];
                
                grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, const Vector<T, dim>& dw, DFGMPM::GridState<T, dim>& g) {
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
                    if (g.separable != 1 || !useDFG) {
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

                        //Transfer volume so we can add our Mode 1 loading
                        //g.gridViYi1 += vol * w;
                    }
                    else if (g.separable == 1 && useDFG) {
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
                        }
                    }
                });
            }
        });

        grid.countNumNodes();

        /* Iterate grid to divide out the grid masses from momentum and then add gravity */
        grid.iterateGrid([&](const Vector<int, dim>& node, DFGMPM::GridState<T, dim>& g) {
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
            if (g.separable == 1) {
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

/* CRAMP Partitioned P2G: for each particle to grid mapping, check whether the line between them crosses the explicit crack path 
    if it does not, transfer particle quantity to field 1 and separable = 0
    if it does, transfer particle quantity either to field 1 or 2 (depending on CRAMP partitioning algorithm), and separable = 1 
    NOTE: we also need to set particleAF so we can map backwards in G2P */
template <class T, int dim>
class CRAMPPartitionedP2GOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    Field<Vector<T, dim>>& m_V;
    std::vector<T>& m_mass;
    Field<Matrix<T, dim, dim>>& m_C;
    Field<Matrix<T, dim, dim>>& stress;

    int topPlane_startIdx; //so we know the end point of the crackParticles

    Bow::Field<std::vector<int>>& particleAF;

    T gravity;

    DFGMPM::DFGMPMGrid<T, dim>& grid;
    T dx;
    T dt;

    bool symplectic;
    bool useAPIC;

    void operator()()
    {
        BOW_TIMER_FLAG("CRAMP Partitioned P2G");
        
        //Compute extent of crack segments
        Vector<T,dim> crackStart = m_X[grid.crackParticlesStartIdx];
        Vector<T,dim> crackTip = m_X[topPlane_startIdx - 1];
        T crackMinX = std::min(crackStart[0], crackTip[0]);
        T crackMaxX = std::max(crackStart[0], crackTip[0]);
        T crackMinY = std::min(crackStart[1], crackTip[1]);
        T crackMaxY = std::max(crackStart[1], crackTip[1]);
        
        grid.colored_for([&](int i) {
            if(!grid.crackInitialized || i < grid.crackParticlesStartIdx){ //skip crack particles if we have them
                const Vector<T, dim> x1 = m_X[i];
                const Vector<T, dim> v = m_V[i];
                const T mass = m_mass[i];
                const Matrix<T, dim, dim> C = m_C[i] * mass; //C * m_p
                const Vector<T, dim> momentum = mass * v; //m_p * v_p
                const Matrix<T, dim, dim> delta_t_tmp_force = -dt * stress[i]; //stress holds Vp^0 * PF^T
                BSplineWeights<T, dim> spline(x1, dx);
                
                grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, const Vector<T, dim>& dw, DFGMPM::GridState<T, dim>& g) {
                    Vector<T, dim> x2 = node.template cast<T>() * dx;
                    Vector<T, dim> xi_minus_xp = x2 - x1;
                    Vector<T, dim> velocity_term_APIC = Vector<T, dim>::Zero();
                    velocity_term_APIC = momentum + C * xi_minus_xp; //mv + (C @ dpos)
                    Vector<T, dim> stress_term_dw = Vector<T, dim>::Zero();
                    if (symplectic) {
                        stress_term_dw = delta_t_tmp_force * dw; //only add forces if symplectic, else added in residual
                    }
                    Vector<T, dim> delta_APIC = w * velocity_term_APIC + stress_term_dw;
                    Vector<T, dim> delta_FLIP = w * momentum + stress_term_dw;
                    Vector<T, dim> delta_vn = w * momentum; //we'll use this to compute v1^n and v2^n for FLIP

                    //Now we need to determine for this particle/node pairing whether we should transfer to field 1 or 2 and whether this node is separable or not.

                    //SUBTASK 1: determine if rectangle defined by x1 and x2 intersects the extent of the crack
                    T minX = std::min(x1[0], x2[0]);
                    T maxX = std::max(x1[0], x2[0]);
                    T minY = std::min(x1[1], x2[1]);
                    T maxY = std::max(x1[1], x2[1]);

                    bool intersectCrack = true;
                    if(crackMinX >= maxX || minX >= crackMaxX){
                        intersectCrack = false;
                    }
                    if(crackMinY >= maxY || minY >= crackMaxY){
                        intersectCrack = false;
                    }

                    //SUBTASKS 2-4
                    int numCrossings = 0;
                    bool aboveCrack = false;
                    bool belowCrack = false;
                    if(intersectCrack){
                        
                        //SUBTASK 2/3: for each crack segment with endpoints x3 and x4, compute the signs of the areas of triangles 123, 124, 341, and 342
                        for(int j = grid.crackParticlesStartIdx; j < topPlane_startIdx - 1; j++){
                            Vector<T,dim> x3 = m_X[j];
                            Vector<T,dim> x4 = m_X[j+1];
                            int a1 = signedTriangleArea(x1, x2, x3); //123
                            int a2 = signedTriangleArea(x1, x2, x4); //124
                            
                            if(a1 == 1 && a2 == 1){
                                intersectCrack = false; //if these are +,+ there is never an intersect
                            }
                            else if(a1 == -1 && a2 == -1){
                                intersectCrack = false; // -,- never intersects
                            }
                            else if(a1 == 0 && a2 == 0){
                                intersectCrack = false; // 0,0 never intersects
                            }
                            else{
                                int a3 = signedTriangleArea(x3, x4, x1); //341
                                int a4 = signedTriangleArea(x3, x4, x2); //342

                                //check for the exact 8 cases that indicate intersection
                                numCrossings++; //pre-emptively increase this, we will decrease it if there was no intersection
                                if(a1 == -1 && a2 == 1 && a3 == 1 && a4 == -1){         //-++-
                                    aboveCrack = true;
                                }
                                else if(a1 == -1 && a2 == 1 && a3 == 1 && a4 == 0){     //-++0
                                    aboveCrack = true;
                                }
                                else if(a1 == 0 && a2 == 1 && a3 == 1 && a4 == 0){      //0++0
                                    aboveCrack = true;
                                }
                                else if(a1 == -1 && a2 == 0 && a3 == 1 && a4 == 0){     //-0+0
                                    aboveCrack = true;
                                }
                                else if(a1 == 1 && a2 == -1 && a3 == -1 && a4 == 1){    //+--+
                                    belowCrack = true;
                                }
                                else if(a1 == 1 && a2 == -1 && a3 == -1 && a4 == 0){    //+--0
                                    belowCrack = true;
                                }
                                else if(a1 == 0 && a2 == -1 && a3 == -1 && a4 == 0){    //0--0
                                    belowCrack = true;
                                }
                                else if(a1 == 1 && a2 == 0 && a3 == -1 && a4 == 0){     //+0-0
                                    belowCrack = true;
                                }
                                else{
                                    numCrossings--; //no intersect
                                    intersectCrack = false;
                                }
                            }
                        }

                        //SUBTASK 3/4: determine whether there was an intersection or not based on crossing count
                        if(numCrossings % 2 == 0){ //if number of crossings was even, ignore the intersections
                            intersectCrack = false;
                        }
                    }

                    //TODO: if we want CRAMP partitioning, we need to completely change some other parts of the MPM data flow because it requires THREE fields, not just two
                    // //Now, set separable, particleAF, and do two field P2G all based on intersectCrack and above/belowCrack!
                    // if(intersectCrack){
                    //     if(aboveCrack){ //particle goes into field 1

                    //     }
                    //     else if(belowCrack){ //particle goes into field 2

                    //     }
                    //     else{
                    //         std::cout << "ERROR: Shouldn't get here... :c" << std::endl;
                    //     }
                    // }
                    // else{
                    //     //this particle goes into this field 0
                    // }

                    // //Notice we treat single-field and two-field nodes differently
                    // //NOTE: remember we are also including explicit force here if symplectic!
                    // if (g.separable != 1 || !useDFG) {
                    //     //Single-field treatment if separable = 0 OR if we are using single field MPM
                    //     if (useAPIC) {
                    //         g.v1 += delta_APIC;
                    //     }
                    //     else {
                    //         g.v1 += delta_FLIP;
                    //         g.vn1 += delta_vn; //transfer momentum to compute v^n
                    //     }

                    //     //Now transfer mass if we aren't using DFG (this means we skipped massP2G earlier)
                    //     if (!useDFG) {
                    //         g.m1 += mass * w;
                    //     }

                    //     //Transfer volume so we can add our Mode 1 loading
                    //     //g.gridViYi1 += vol * w;
                    // }
                    // else if (g.separable == 1 && useDFG) {
                    //     //Treat node as having two fields
                    //     int fieldIdx = particleAF[i][oidx]; //grab the field that this particle belongs in for this grid node (oidx)
                    //     if (fieldIdx == 0) {
                    //         if (useAPIC) {
                    //             g.v1 += delta_APIC;
                    //         }
                    //         else {
                    //             g.v1 += delta_FLIP;
                    //             g.vn1 += delta_vn; //transfer momentum to compute v^n
                    //         }
                    //         //Compute normal for field 1 particles
                    //         g.n1 += mass * dw; //remember to normalize this later!
                    //     }
                    //     else if (fieldIdx == 1) {
                    //         if (useAPIC) {
                    //             g.v2 += delta_APIC;
                    //         }
                    //         else {
                    //             g.v2 += delta_FLIP;
                    //             g.vn2 += delta_vn; //transfer momentum to compute v^n
                    //         }
                    //         //Compute normal for field 2 particles
                    //         g.n2 += mass * dw; //remember to normalize this later!
                    //     }
                    // }
                });
            }
        });

        // grid.countNumNodes();

        // /* Iterate grid to divide out the grid masses from momentum and then add gravity */
        // grid.iterateGrid([&](const Vector<int, dim>& node, DFGMPM::GridState<T, dim>& g) {
        //     Vector<T, dim> gravity_term = Vector<T, dim>::Zero();
        //     if (symplectic) {
        //         gravity_term[1] = gravity * dt; //only nonzero if symplectic, else we add gravity in residual of implicit solve
        //     }
        //     T mass1 = g.m1;
        //     Vector<T, dim> alpha1;
        //     alpha1 = Vector<T, dim>::Ones() * ((T)1 / mass1);
        //     g.v1 = g.v1.cwiseProduct(alpha1);
        //     g.v1 += gravity_term;
        //     g.vn1 = g.vn1.cwiseProduct(alpha1); // this is how we get v1^n
        //     g.x1 = node.template cast<T>() * dx; //put nodal position in x1 regardless of separability
        //     if (g.separable == 1) {
        //         T mass2 = g.m2;
        //         Vector<T, dim> alpha2;
        //         alpha2 = Vector<T, dim>::Ones() * ((T)1 / mass2);
        //         g.v2 = g.v2.cwiseProduct(alpha2);
        //         g.v2 += gravity_term;
        //         g.vn2 = g.vn2.cwiseProduct(alpha2); //this is how we get v2^n
        //     }
        // });
    }
};

//Compute signed triangle area
template<class T, int dim>
int signedTriangleArea(Vector<T,dim> _x1, Vector<T,dim> _x2, Vector<T,dim> _x3){
    T x1 = _x1[0];
    T x2 = _x2[0];
    T x3 = _x3[0];
    T y1 = _x1[1];
    T y2 = _x2[1];
    T y3 = _x3[1];
    T area = (x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2));
    
    if(area > 0){
        return 1;
    }
    else if(area < 0){
        return -1;
    }
    
    return 0;
}

/*Mark particles for loading, takes in upper and lower boundaries for Mode 1 loading */
template <class T, int dim>
class MarkParticlesForLoadingOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    Field<int>& m_marker;
    T y1;
    T y2;

    DFGMPM::DFGMPMGrid<T, dim>& grid;

    void operator()()
    {
        BOW_TIMER_FLAG("Mark Particles for Mode 1 Loading");

        //Iterate particles and mark as either 4 (up) or 5 (down)
        grid.parallel_for([&](int i) {
            if(!grid.crackInitialized || i < grid.crackParticlesStartIdx){ //skip crack particles if we have them
                const Vector<T, dim> pos = m_X[i];
                if(pos[1] > y1 || pos[1] < y2){ //ONLY set to 4 or 5 if particle is above y1 and below y2!
                    m_marker[i] = 4;
                    if(pos[1] < y2){
                        m_marker[i] = 5; //set as 5 for particles below y2
                    }
                }
            }
        });
    }
};

/*Iterate grid to apply a mode I loading to the configuration based on y1, y2, and sigmaA */
template <class T, int dim>
class ApplyMode1LoadingOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    Field<int> m_marker;
    T scaledSigmaA;

    T dx;
    T dt;

    DFGMPM::DFGMPMGrid<T, dim>& grid;

    Field<T> m_vol;

    void operator()()
    {
        BOW_TIMER_FLAG("applyMode1Loading");

        //Using particle volume, compute per particle forces using the scaledSigmaA passed in, then transfer this force to the grid and apply it
        T stress = scaledSigmaA;
        grid.colored_for([&](int i) {
            if(!grid.crackInitialized || i < grid.crackParticlesStartIdx){ //skip crack particles if we have them
                const Vector<T, dim> pos = m_X[i];
                const int marker = m_marker[i];
                stress = scaledSigmaA;

                if(marker == 4 || marker == 5){ //ONLY apply this force to particles above y1 and below y2!
                    if(marker == 5){
                        stress *= -1; //apply negative here for particles below y2
                    }

                    T fp = m_vol[i] * stress; //particle force (working simply with y direction magnitude, not full vector)

                    //std::cout << "particle idx:" << i << "fp:" << fp << std::endl;

                    BSplineWeights<T, dim> spline(pos, dx);
                    grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, const Vector<T, dim>& dw, DFGMPM::GridState<T, dim>& g) {
                        //Store f_i in gridViYi since we're not using it anyway
                        g.gridViYi1 += fp * w;
                    });
                }
            }
        });
        
        /* Iterate grid to apply these loadings to the velocities */
        grid.iterateGrid([&](const Vector<int, dim>& node, DFGMPM::GridState<T, dim>& g) {

            T fi = g.gridViYi1; //stored fi in here
            if(fi != 0.0){
                g.v1[1] += (fi / g.m1) * dt;
            }

        });
    }
};

/* Evolve crack planes -- we need to use v1, v2, and v_cm and do a G2P style transfer from nodes to crack plane particles, then perform a simple particle advection */
template <class T, int dim>
class EvolveCrackPlanesOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;

    Field<Vector<T, dim>>& m_X;
    Field<Vector<T, dim>>& m_V;

    int topPlane_startIdx;
    int bottomPlane_startIdx;

    DFGMPM::DFGMPMGrid<T, dim>& grid;
    T dx;
    T dt;

    T flipPicRatio;
    bool useAPIC;
    
    void operator()()
    {
        //First let's iterate separable grid nodes and compute v_cm and store it in fi1 (since we won't apply dynamic impulses for cramp)
        BOW_TIMER_FLAG("evolveCrackPlanes");

        //set some indeces so we can always set the top and bottom plane tips to be equal to the central crack tip
        int crackTipIdx = topPlane_startIdx - 1;
        int topTipIdx = bottomPlane_startIdx - 1;
        int bottomTipIdx = m_X.size() - 1;

        //Iterate separable nodes and compute V_cm for each
        grid.iterateGrid([&](const Vector<int, dim>& node, DFGMPM::GridState<T, dim>& g) {
            //For separable nodes, compute the frictional contact forces for each field
            if (g.separable == 1) {
                //Grab momentum
                Vector<T, dim> q1 = g.v1 * g.m1;
                Vector<T, dim> q2 = g.v2 * g.m2;
                Vector<T, dim> q_cm = q1 + q2;

                //Grab mass
                T m_1 = g.m1;
                T m_2 = g.m2;
                T m_cm = m_1 + m_2;

                //Compute v_cm
                Vector<T, dim> v_cm = q_cm / m_cm; //THIS IS CRUCIAL we need to compute v_cm using momentum

                g.fi1 = v_cm; //store v_cm in fi1

                //Now compute old v_cm for use in FLIP blending
                Vector<T, dim> q1old = g.vn1 * g.m1;
                Vector<T, dim> q2old = g.vn2 * g.m2;
                Vector<T, dim> q_cm_old = q1old + q2old;

                g.fi2 = q_cm_old / m_cm; //store v_cm_old in fi2
            }
        });

        //CRACK PLANE
        //Now transfer v_cm from nodes to the crack plane particles
        grid.parallel_for([&](int i) {
            if(i >= grid.crackParticlesStartIdx && i < topPlane_startIdx){ //only process the crack plane particles
                Vector<T, dim>& Xp = m_X[i];
                Vector<T, dim> picV = Vector<T, dim>::Zero();
                Vector<T, dim> flipV = Vector<T, dim>::Zero();
                BSplineWeights<T, dim> spline(Xp, dx);

                //Vector<T, dim> picX = Vector<T, dim>::Zero();
                grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, Vector<T, dim> dw, DFGMPM::GridState<T, dim>& g) {
                    if (g.idx < 0) return;
                    // Vector<T, dim> g_v_new = g.v1;
                    // Vector<T, dim> g_v2_new = g.v2;
                    // Vector<T, dim> g_v_old = g.vn1;
                    // Vector<T, dim> g_v2_old = g.vn2;
                    
                    //these steps depend on which field the particle is in
                    if (g.separable != 1) {
                        //treat as single field node
                        picV += w * g.v1;
                        flipV += w * (g.v1 - g.vn1); 
                        //picX += w * g.x1;
                    }
                    else if (g.separable == 1) {
                        //treat as two-field node -> use our computed v_cm (new and old) values in fi1 and fi2
                        picV += w * g.fi1;
                        flipV += w * (g.fi1 - g.fi2);
                        //picX += w * g.x1;
                    }
                });
                if(useAPIC){
                    m_V[i] = picV;
                }
                else{
                    //Finish computing FLIP velocity: v_p^n+1 = v_p^n + dt (v_i^n+1 - v_i^n) * wip
                    flipV = m_V[i] + (dt * flipV);
                    m_V[i] = (flipPicRatio * flipV) + ((1.0 - flipPicRatio) * picV); //blended velocity
                }
                
                //m_X[i] = picX; //use PIC for advection
                m_X[i] += m_V[i] * dt; //advect using velocity either from PIC (APIC or full PIC) or from FLIP (PIC blend FLIP)
            }
        });

        //TOP PLANE
        //Now use the field velocities to update the top and bottom crack planes --> strategy for now is to apply all upward field velocities to top plane, and all downward velocities to bottom plane (this is naive and only for SENT) 
        grid.parallel_for([&](int i) {
            if(i >= topPlane_startIdx && i < bottomPlane_startIdx){ //only process the top plane particles
                Vector<T, dim>& Xp = m_X[i];
                Vector<T, dim> picV = Vector<T, dim>::Zero();
                Vector<T, dim> flipV = Vector<T, dim>::Zero();
                BSplineWeights<T, dim> spline(Xp, dx);

                grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, Vector<T, dim> dw, DFGMPM::GridState<T, dim>& g) {
                    if (g.idx < 0) return;                    
                    //these steps depend on which field the particle is in
                    if (g.separable != 1) {
                        //treat as single field node -> if upward, add contribution for top crack
                        if(g.v1[1] > 0){
                            picV += w * g.v1;
                            flipV += w * (g.v1 - g.vn1);
                        }
                    }
                    else if (g.separable == 1) {
                        //add contributions from ALL upward vel fields
                        if(g.v1[1] > 0){
                            picV += w * g.v1;
                            flipV += w * (g.v1 - g.vn1);
                        }
                        if(g.v2[1] > 0){
                            picV += w * g.v2;
                            flipV += w * (g.v2 - g.vn2);
                        }
                    }
                });
                if(useAPIC){
                    m_V[i] = picV;
                }
                else{
                    //Finish computing FLIP velocity: v_p^n+1 = v_p^n + dt (v_i^n+1 - v_i^n) * wip
                    flipV = m_V[i] + (dt * flipV);
                    m_V[i] = (flipPicRatio * flipV) + ((1.0 - flipPicRatio) * picV); //blended velocity
                }
                
                m_X[i] += m_V[i] * dt; //advect using velocity either from PIC (APIC or full PIC) or from FLIP (PIC blend FLIP)

                //if this is the top plane tip, set it equal to the updated crackTip
                if(i == topTipIdx){
                    m_X[topTipIdx] = m_X[crackTipIdx];
                }
            }
        });

        //BOTTOM PLANE
        //Now use the field velocities to update the top and bottom crack planes --> strategy for now is to apply all upward field velocities to top plane, and all downward velocities to bottom plane (this is naive and only for SENT) 
        grid.parallel_for([&](int i) {
            if(i >= bottomPlane_startIdx){ //only process the top plane particles
                Vector<T, dim>& Xp = m_X[i];
                Vector<T, dim> picV = Vector<T, dim>::Zero();
                Vector<T, dim> flipV = Vector<T, dim>::Zero();
                BSplineWeights<T, dim> spline(Xp, dx);

                grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, Vector<T, dim> dw, DFGMPM::GridState<T, dim>& g) {
                    if (g.idx < 0) return;                    
                    //these steps depend on which field the particle is in
                    if (g.separable != 1) {
                        //treat as single field node -> if downard, add contribution for bottom crack
                        if(g.v1[1] < 0){
                            picV += w * g.v1;
                            flipV += w * (g.v1 - g.vn1);
                        }
                    }
                    else if (g.separable == 1) {
                        //add contributions from ALL downward vel fields
                        if(g.v1[1] < 0){
                            picV += w * g.v1;
                            flipV += w * (g.v1 - g.vn1);
                        }
                        if(g.v2[1] < 0){
                            picV += w * g.v2;
                            flipV += w * (g.v2 - g.vn2);
                        }
                    }
                });
                if(useAPIC){
                    m_V[i] = picV;
                }
                else{
                    //Finish computing FLIP velocity: v_p^n+1 = v_p^n + dt (v_i^n+1 - v_i^n) * wip
                    flipV = m_V[i] + (dt * flipV);
                    m_V[i] = (flipPicRatio * flipV) + ((1.0 - flipPicRatio) * picV); //blended velocity
                }
                
                m_X[i] += m_V[i] * dt; //advect using velocity either from PIC (APIC or full PIC) or from FLIP (PIC blend FLIP)

                //if this is the bottom plane tip, set it equal to the updated crackTip
                if(i == bottomTipIdx){
                    m_X[bottomTipIdx] = m_X[crackTipIdx];
                }
            }
        });
        
    }
};

/*Iterate particles to compute stress at various distances from the crack tip - we want to plot sigma_yy vs r */
template <class T, int dim>
class StressSnapshotOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    
    Vector<T,dim> crackTip;

    Field<Matrix<T, dim, dim>>& stress;

    DFGMPM::DFGMPMGrid<T, dim>& grid;
    T dx;

    Field<T>& p_sigmaYY;
    Field<T>& p_r;
    Field<T>& p_posX;
    Field<int>& p_idx;

    T halfEnvelope;

    void operator()()
    {
        BOW_TIMER_FLAG("takeStressSnapshot");

        p_sigmaYY.clear();
        p_r.clear();
        p_posX.clear();
        p_idx.clear();

        //Iterate particles in serial
        grid.serial_for([&](int i) {
            if(!grid.crackInitialized || i < grid.crackParticlesStartIdx){ //skip crack particles if we have them
                T radius = 0;
                T sigmaYY = 0;
                const Vector<T, dim> pos = m_X[i];
                const Matrix<T, dim, dim> sigma = stress[i];

                //std::cout << "crack tip:" << crackTip << std::endl;
                //Only evaluate theta = 0, so within some y-trheshold from the central crack AND only ahead of the crack
                T maxY = crackTip[1] + halfEnvelope;
                T minY = crackTip[1] - halfEnvelope;
                T minX = crackTip[0] + (2.0 * dx);
                if(pos[1] < maxY && pos[1] > minY && pos[0] > minX){
                    radius = std::sqrt((pos[0] - crackTip[0])*(pos[0] - crackTip[0]) + (pos[1] - crackTip[1])*(pos[1] - crackTip[1]));
                    sigmaYY = sigma(1,1);

                    p_sigmaYY.push_back(sigmaYY);
                    p_r.push_back(radius);
                    p_posX.push_back(pos[0] - crackTip[0]);
                    p_idx.push_back(i);
                }
            }
        });
    }
};

/*Simple particle based damping (occurs after G2P) */
template <class T, int dim>
class SimpleDampingOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_V;
    
    T simpleDampingFactor;

    DFGMPM::DFGMPMGrid<T, dim>& grid;

    void operator()()
    {
        BOW_TIMER_FLAG("applySimpleDamping");

        //Iterate particles in serial
        grid.parallel_for([&](int i) {
            if(!grid.crackInitialized || i < grid.crackParticlesStartIdx){ //skip crack particles if we have them
                m_V[i] *= simpleDampingFactor;
            }
        });
    }
};

/* Compute the J Integral using a rectangular path of grid nodes centered on the closest node to the crack tip */
template <class T, int dim>
class ComputeJIntegralOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    
    Vector<T,dim> crackTip;
    int topPlane_startIdx;
    int bottomPlane_startIdx;

    Field<Matrix<T, dim, dim>>& stress;

    DFGMPM::DFGMPMGrid<T, dim>& grid;
    T dx;
    T dt;

    T mu;
    T la;

    void operator()(int contourRadius, std::ofstream& file)
    {
        BOW_TIMER_FLAG("computeJIntegral");

        //NOTE: This routine is designed for HORIZONTAL LEFT SIDE CRACKS (b.c. of contour intersection assumptions --> clockwise path hits bottom of crack first)

        //STEP 0: Grab an iterable list of unprocessed contour points
        //Iterate a rectangular contour around the closest node to the crack tip --> COUNTER CLOCKWISE STARTING FROM TOP LEFT NODE
        std::vector<Vector<T,dim>> contourPoints;
        std::vector<DFGMPM::GridState<T,dim>*> contourGridStates;
        BSplineWeights<T, dim> spline(crackTip, dx);
        grid.iterateRectangularContour(spline, contourRadius, [&](const Vector<int, dim>& node, DFGMPM::GridState<T, dim>& g) {
            Vector<T,dim> xi = node.template cast<T>() * dx;
            //Grab data
            contourPoints.push_back(xi);
            contourGridStates.push_back(&g); //hold pointers to grid node data

            //std::cout << "x_i: (" << xi[0] << "," << xi[1] << ")" << std::endl;
        });

        //STEP 1: Construct ordered list of contour points starting with intersection with bottom of crack, counter-clockwise around contour, and then ending with intersection with top of crack
        bool foundBottomIntersection = false;
        bool foundTopIntersection = false;
        //store the first point index for the segment intersecting with bottom and top crack planes
        int bottomIntersectionIdx = -1;
        int topIntersectionIdx = -1; 
        Vector<T, dim> bottomIntersection, topIntersection;
        //STEP 1a: find top intersection
        for(int i = 0; i < (int)contourPoints.size() - 1; i++){
            //check each contour line segment for intersections with bottom crack
            Vector<T, dim> A = contourPoints[i];
            Vector<T, dim> B = contourPoints[i+1];
            
            for(int j = topPlane_startIdx; j < bottomPlane_startIdx - 1; j++){
                Vector<T, dim> C = m_X[j];
                Vector<T, dim> D = m_X[j+1];

                if(intersect(A, B, C, D)){
                    topIntersectionIdx = i;
                    getIntersection(A, B, C, D, topIntersection);
                    foundTopIntersection = true;
                    break;
                }
            }
            if(foundTopIntersection){
                break;
            }            
        }

        //STEP 1b: Now look for the lower intersection -> start with topIntersectionIdx since this is the first segment that can have an intersection with the top points
        for(int i = topIntersectionIdx; i < (int)contourPoints.size() - 1; i++){
            //check each contour line segment for intersections with top crack
            Vector<T, dim> A = contourPoints[i];
            Vector<T, dim> B = contourPoints[i+1];
            
            for(unsigned int j = bottomPlane_startIdx; j < m_X.size() - 1; j++){
                Vector<T, dim> C = m_X[j];
                Vector<T, dim> D = m_X[j+1];

                if(intersect(A, B, C, D)){
                    bottomIntersectionIdx = i;
                    getIntersection(A, B, C, D, bottomIntersection);
                    foundBottomIntersection = true;
                    break;
                }
            }
            if(foundBottomIntersection){
                break;
            }  
        }

        //STEP 1c: Check that crack is actually open (if it is not, we cannot compute J integral)
        std::vector<Vector<T,dim>> finalContourPoints;
        std::vector<DFGMPM::GridState<T,dim>*> finalContourGridStates;
        T epsilon = 1e-9;
        if(topIntersection[1] - bottomIntersection[1] < epsilon){
            std::cout << "Crack is not opened at the intersection points, cannot compute J-integral!" << std::endl;
        }
        else{
            //Else we construct all of the pieces we need to finally compute the J integral!
            //Specifically, a list of contour points starting with bottom intersection and going counter-clockwise around to the top intersection

            //Bottom intersection
            finalContourPoints.push_back(bottomIntersection);
            finalContourGridStates.push_back(contourGridStates[bottomIntersectionIdx]); //store the grid state for the START point of the contour segment that intersected the crack

            //now add points counter-clockwise until hit end of list (then we will start from beginning until bottom intersecting segment)
            for(int i = bottomIntersectionIdx + 1; i < (int)contourPoints.size(); ++i){ //end of list
                finalContourPoints.push_back(contourPoints[i]);
                finalContourGridStates.push_back(contourGridStates[i]);
            }
            for(int i = 0; i < (int)topIntersectionIdx + 1; ++i){ //begin of list
                finalContourPoints.push_back(contourPoints[i]);
                finalContourGridStates.push_back(contourGridStates[i]);
            }

            //top intersection
            finalContourPoints.push_back(topIntersection);
            finalContourGridStates.push_back(contourGridStates[topIntersectionIdx]); //store grid state from START of top intersecting segment
        }

        //STEP 2: Compute J-integral!
        
        //DEBUG: Setup a bunch of lists to hold intermediate data for debugging
        std::vector<T> Fsum_I_List;
        std::vector<T> DeltaI_List;
        std::vector<T> Fm_I_SegmentList; //Store Fm1 and Fm2 for each line segment so we can check them!
        std::vector<T> Fm_I_NormalX;
        std::vector<T> Fm_I_NormalY;
        std::vector<T> Fm_I_W;
        std::vector<T> Fm_I_termTwo;
        std::vector<T> blendRatios;
        std::vector<Matrix<T,dim,dim>> m_Fi; //collect reconstructed Fi's

        T J_I = 0; //set J integral mode I to 0 for now
        T J_II = 0; //set J integral mode II to 0 for now
        for(int i = 0; i < (int)finalContourPoints.size() - 1; ++i){ //iterate contour segments
            T Fsum_I = 0; //this is what we focus on setting for each segment (three cases below)
            T Fsum_II = 0; //mode II
            Vector<T,dim> x1 = finalContourPoints[i];
            Vector<T,dim> x2 = finalContourPoints[i+1];
            if(i == 0){ //first segment
                //If first segment, first we compute interpolations for bottom intersection point
                Vector<T, dim> xi1 = contourPoints[bottomIntersectionIdx];
                Vector<T, dim> xi2 = contourPoints[bottomIntersectionIdx + 1];
                DFGMPM::GridState<T,dim>* gi1 = contourGridStates[bottomIntersectionIdx]; //NOTE: this will very likely be a separable node!!!! only use field 1 for interpolating bottom intersect
                DFGMPM::GridState<T,dim>* gi2 = contourGridStates[bottomIntersectionIdx + 1]; //grab the two grid states to interpolate between
                //Compute Fm for these two points, then we'll interpolate between them for our intersection point
                std::vector<T> Fmi1_I = computeFm(gi1, x2 - x1, 0, m_Fi, true, 1); //store F and use field 1 for bottom intersect
                std::vector<T> Fmi2_I = computeFm(gi2, x2 - x1, 0, m_Fi); 
                std::vector<T> Fmi1_II = computeFm(gi1, x2 - x1, 1, m_Fi, false, 1); //dont store F for J_II but we want to use field 1 for bottom intersect
                std::vector<T> Fmi2_II = computeFm(gi2, x2 - x1, 1, m_Fi);
                T blendRatio = abs(x1[1] - xi1[1]) / abs(xi2[1] - xi1[1]);
                T FmIntersect_I = (Fmi1_I[0] * (1 - blendRatio)) + (Fmi2_I[0] * blendRatio);
                T FmIntersect_II = (Fmi1_II[0] * (1 - blendRatio)) + (Fmi2_II[0] * blendRatio);

                //Compute Fm2 (from actual second point)
                DFGMPM::GridState<T,dim>* g2 = finalContourGridStates[i+1];
                std::vector<T> Fm2_I = computeFm(g2, x2 - x1, 0, m_Fi, true); //storeF
                std::vector<T> Fm2_II = computeFm(g2, x2 - x1, 1, m_Fi);

                //Compute Fsum
                Fsum_I = FmIntersect_I + Fm2_I[0];
                Fsum_II = FmIntersect_II + Fm2_II[0];

                //Store Fm, normal, W, and termTwo of end points
                Fm_I_SegmentList.push_back(FmIntersect_I);
                Fm_I_NormalX.push_back(Fmi1_I[1]);
                Fm_I_NormalY.push_back(Fmi1_I[2]);
                Fm_I_W.push_back(Fmi1_I[3]);
                Fm_I_termTwo.push_back(Fmi1_I[4]); //these are all specifically for the FIRST point used to interpolate this intersection point

                Fm_I_SegmentList.push_back(Fm2_I[0]);
                Fm_I_NormalX.push_back(Fm2_I[1]);
                Fm_I_NormalY.push_back(Fm2_I[2]);
                Fm_I_W.push_back(Fm2_I[3]);
                Fm_I_termTwo.push_back(Fm2_I[4]);

                blendRatios.push_back(blendRatio);

            }
            else if(i == (int)finalContourPoints.size() - 2){ //last segment
                //Compute Fm1 (from actual first point)
                DFGMPM::GridState<T,dim>* g1 = finalContourGridStates[i];
                std::vector<T> Fm1_I = computeFm(g1, x2 - x1, 0, m_Fi, true); //store F
                std::vector<T> Fm1_II = computeFm(g1, x2 - x1, 1, m_Fi);
                
                //compute interpolations for top intersection point (second endpoint)
                Vector<T, dim> xi1 = contourPoints[topIntersectionIdx];
                Vector<T, dim> xi2 = contourPoints[topIntersectionIdx + 1];
                DFGMPM::GridState<T,dim>* gi1 = contourGridStates[topIntersectionIdx];
                DFGMPM::GridState<T,dim>* gi2 = contourGridStates[topIntersectionIdx + 1]; //grab the two grid states to interpolate between -> this one is very likely separable!! use field 2 only for interpolating
                //Compute Fm for these two points, then we'll interpolate between them for our intersection point
                std::vector<T> Fmi1_I = computeFm(gi1, x2 - x1, 0, m_Fi);
                std::vector<T> Fmi2_I = computeFm(gi2, x2 - x1, 0, m_Fi, true, 2); //store F, this way the F we print for idx 0 = for max idx
                std::vector<T> Fmi1_II = computeFm(gi1, x2 - x1, 1, m_Fi);
                std::vector<T> Fmi2_II = computeFm(gi2, x2 - x1, 1, m_Fi, false, 2); //dont store F, but only use field 2 for bottom intersection interpolation
                T blendRatio = abs(x2[1] - xi1[1]) / abs(xi2[1] - xi1[1]);
                T FmIntersect_I = (Fmi1_I[0] * (1 - blendRatio)) + (Fmi2_I[0] * blendRatio);
                T FmIntersect_II = (Fmi1_II[0] * (1 - blendRatio)) + (Fmi2_II[0] * blendRatio);

                //Compute Fsum
                Fsum_I = Fm1_I[0] + FmIntersect_I;
                Fsum_II = Fm1_II[0] + FmIntersect_II;

                Fm_I_SegmentList.push_back(Fm1_I[0]);
                Fm_I_NormalX.push_back(Fm1_I[1]);
                Fm_I_NormalY.push_back(Fm1_I[2]);
                Fm_I_W.push_back(Fm1_I[3]);
                Fm_I_termTwo.push_back(Fm1_I[4]);

                Fm_I_SegmentList.push_back(FmIntersect_I);
                Fm_I_NormalX.push_back(Fmi2_I[1]);
                Fm_I_NormalY.push_back(Fmi2_I[2]);
                Fm_I_W.push_back(Fmi2_I[3]);
                Fm_I_termTwo.push_back(Fmi2_I[4]);

                blendRatios.push_back(blendRatio);
            }
            else{ //rest of the non-intersect segments
                DFGMPM::GridState<T,dim>* g1 = finalContourGridStates[i];
                DFGMPM::GridState<T,dim>* g2 = finalContourGridStates[i+1];
                std::vector<T> Fm1_I = computeFm(g1, x2 - x1, 0, m_Fi, true); //store F
                std::vector<T> Fm2_I = computeFm(g2, x2 - x1, 0, m_Fi, true); //store F
                std::vector<T> Fm1_II = computeFm(g1, x2 - x1, 1, m_Fi);
                std::vector<T> Fm2_II = computeFm(g2, x2 - x1, 1, m_Fi);
                Fsum_I = Fm1_I[0] + Fm2_I[0];
                Fsum_II = Fm1_II[0] + Fm2_II[0];

                Fm_I_SegmentList.push_back(Fm1_I[0]);
                Fm_I_NormalX.push_back(Fm1_I[1]);
                Fm_I_NormalY.push_back(Fm1_I[2]);
                Fm_I_W.push_back(Fm1_I[3]);
                Fm_I_termTwo.push_back(Fm1_I[4]);

                Fm_I_SegmentList.push_back(Fm2_I[0]);
                Fm_I_NormalX.push_back(Fm2_I[1]);
                Fm_I_NormalY.push_back(Fm2_I[2]);
                Fm_I_W.push_back(Fm2_I[3]);
                Fm_I_termTwo.push_back(Fm2_I[4]);
            }

            //store Fsum_I for debugging
            Fsum_I_List.push_back(Fsum_I);

            //Now after computing Fsum using one of three cases, we can add this contribution to the Jintegral!
            T deltaI = std::sqrt((x1[0] - x2[0])*(x1[0] - x2[0]) + (x1[1] - x2[1])*(x1[1] - x2[1])); //compute distance from x1 to x2 (the end points of current segment)
            J_I += Fsum_I * (deltaI / 2.0);
            J_II += Fsum_II * (deltaI / 2.0);

            DeltaI_List.push_back(deltaI);
        }
        Fsum_I_List.push_back(0.0); //dummy for final endpoint
        DeltaI_List.push_back(0.0);

        //STEP 3: Compute thetaC and G
        Vector<T, dim> x1 = m_X[topPlane_startIdx - 2];
        Vector<T, dim> x2 = m_X[topPlane_startIdx - 1];
        Vector<T, dim> crackTipDirection = (x2 - x1).normalized();
        Vector<T, dim> xDir(1,0);
        T thetaC = acos(crackTipDirection.dot(xDir));
        T G = (J_I * cos(thetaC)) + (J_II * sin(thetaC));

        //STEP 4: Compute K_I and K_II
        x1 = m_X[bottomPlane_startIdx - 2]; //second to last top plane
        x2 = m_X[m_X.size() - 2]; //second to last bottom plane
        T xDisplacement = abs(x2[0] - x1[0]);
        T yDisplacement = abs(x2[1] - x1[1]);
        T magnitude = sqrt((xDisplacement * xDisplacement) + (yDisplacement * yDisplacement));
        T E = (mu*(3*la + 2*mu)) / (la + mu);
        T nu = la / (2 * (la + mu));
        T planeStressFactor = sqrt(G * E);
        T planeStrainFactor = sqrt((G*E) / (1 - (nu * nu)));
        T K_I_factor = yDisplacement / magnitude;
        T K_II_factor = xDisplacement / magnitude;
        T K_I_planeStress = K_I_factor * planeStressFactor;
        T K_I_planeStrain = K_I_factor * planeStrainFactor;
        T K_II_planeStress = K_II_factor * planeStressFactor;
        T K_II_planeStrain = K_II_factor * planeStrainFactor;

        //Print it all out (later write to a simple file)
        file << "=== J-Integral Computation using " << contourRadius << "x" << contourRadius << " Contour ===\n";
        file << "Bottom Intersection | Idx: " << bottomIntersectionIdx << " Point: (" << bottomIntersection[0] << "," << bottomIntersection[1] << "), Blend Ratio: " << blendRatios[0] << " \n";
        file << "Top Intersection | Idx: " << topIntersectionIdx << " Point: (" << topIntersection[0] << "," << topIntersection[1] << "), Blend Ratio: " << blendRatios[1] << "\n";
        // for(int i = 0; i < (int)finalContourPoints.size(); ++i){
        //     file << "idx:" << i << ", Point: (" << finalContourPoints[i][0] << "," << finalContourPoints[i][1] << "), Fsum_I: " << Fsum_I_List[i] << " \n";
        // }
        for(int i = 0; i < (int)finalContourPoints.size() - 1; ++i){
            file << "--<Line Segment " << i << ", Fsum_I: " << Fsum_I_List[i] << ", Delta_I: " << DeltaI_List[i] << ", J_I Contribution: " << Fsum_I_List[i] * (DeltaI_List[i] / 2.0) << ">--\n";
            file << "idx1: " << i << ", Point: (" << finalContourPoints[i][0] << "," << finalContourPoints[i][1] << "), Fm_I: " << Fm_I_SegmentList[i*2] << ", Normal: [" << Fm_I_NormalX[i*2] << "," << Fm_I_NormalY[i*2] << "], W: " << Fm_I_W[i*2] << ", termTwo: " << Fm_I_termTwo[i*2] << "\nFi1: " << m_Fi[i*2] << " \n";
            file << "idx2: " << i+1 << ", Point: (" << finalContourPoints[i+1][0] << "," << finalContourPoints[i+1][1] << "), Fm_I: " << Fm_I_SegmentList[(i*2) + 1] << ", Normal: [" << Fm_I_NormalX[(i*2) + 1] << "," << Fm_I_NormalY[(i*2) + 1] << "], W: " << Fm_I_W[(i*2) + 1] << ", termTwo: " << Fm_I_termTwo[(i*2) + 1] << "\nFi1: " << m_Fi[(i*2)+1] << " \n";
        }
        // for(int i = 0; i < (int)finalContourPoints.size(); ++i){
        //     file << "idx " << i << " pointer: " << finalContourGridStates[i] << "\n";
        // }
        // for(int i = 0; i < (int)finalContourPoints.size(); ++i){
        //     file << "idx " << i << " position: (" << finalContourPoints[i][0] << "," << finalContourPoints[i][1] << ")\n";
        // }
        // for(int i = 0; i < (int)contourPoints.size(); ++i){
        //     file << "idx " << i << " pointer: " << contourGridStates[i] << "\n";
        // }
        // for(int i = 0; i < (int)contourPoints.size(); ++i){
        //     file << "idx " << i << " position: (" << contourPoints[i][0] << "," << contourPoints[i][1] << ")\n";
        // }
        file << "J_I: " << J_I << "\n"; 
        file << "J_II: " << J_II << "\n"; 
        file << "thetaC: " << thetaC << "\n"; 
        file << "G: " << G << "\n"; 
        file << "K_I (plane stress): " << K_I_planeStress << "\n"; 
        file << "K_I (plane strain): " << K_I_planeStrain << "\n"; 
        file << "K_II (plane stress): " << K_II_planeStress << "\n"; 
        file << "K_II (plane strain): " << K_II_planeStrain << "\n";
        file << "\n";
    }

    //Compute Fm based on grid data at node i, the line segment between the nodes, and the mode (0 for x, 1 for y)
    std::vector<T> computeFm(DFGMPM::GridState<T,dim>* g, Vector<T, dim> lineSegment, int mode, std::vector<Matrix<T,dim,dim>>& m_Fi, bool storeF = false, int intersect = 0){
        //intersect = 0 for regular call to Fm (we don't expect this to ever be a separable node)
        //intersect = 1 for a bottom intersect interpolation with the separable node -> use only Field 1
        //intersect = 2 for a top intersect interpolation with the separable node -> use only Field 2
        
        std::vector<T> FmResults;
        
        T Fm = 0;
        
        //Compute normal
        Vector<T, dim> normal;
        if(lineSegment[0] == 0){ //normal = left or right
            normal[1] = 0;
            if(lineSegment[1] > 0){ //normal = right
                normal[0] = 1;
            }
            else{ //normal = left
                normal[0] = -1;
            }
        }
        else{ //normal = up or down
            normal[0] = 0;
            if(lineSegment[0] > 0){ //normal = down
                normal[1] = -1;
            }
            else{ //normal = up
                normal[1] = 1;
            }
        }

        //NOTE: for separable nodes we will ONLY use one or the other field, the field that points in the direction of the current segment
        //NOTE: so, for example, we use field 1 (points down from crack) for interpolating the bottom intersect, and field 2 (points up from crack) for the top intersect
        //Now we must reconstruct our deformation gradient from the singular values and quaternion rotations F = U Sigma V^T
        Matrix<T,dim,dim> Fi1, Fi2;
        Matrix<T,dim,dim> U1, U2, V1, V2, Sigma1, Sigma2;
        Eigen::Quaternion<T> rotUreconstruct1(g->Uquat1);
        Eigen::Quaternion<T> rotVreconstruct1(g->Vquat1);
        rotUreconstruct1.normalize();
        rotVreconstruct1.normalize();
        Matrix<T,3,3> Ureconstruct1 = rotUreconstruct1.toRotationMatrix();
        Matrix<T,3,3> Vreconstruct1 = rotVreconstruct1.toRotationMatrix();
        U1 = Ureconstruct1.topLeftCorner(2,2);
        V1 = Vreconstruct1.topLeftCorner(2,2);
        Sigma1 = g->sigma1.asDiagonal();
        Fi1 = U1 * Sigma1 * V1.transpose();
        if(g->separable == 1){
            Eigen::Quaternion<T> rotUreconstruct2(g->Uquat2);
            Eigen::Quaternion<T> rotVreconstruct2(g->Vquat2);
            rotUreconstruct2.normalize();
            rotVreconstruct2.normalize();
            Matrix<T,3,3> Ureconstruct2 = rotUreconstruct2.toRotationMatrix();
            Matrix<T,3,3> Vreconstruct2 = rotVreconstruct2.toRotationMatrix();
            U2 = Ureconstruct2.topLeftCorner(2,2);
            V2 = Vreconstruct2.topLeftCorner(2,2);
            Sigma2 = g->sigma2.asDiagonal();
            Fi2 = U2 * Sigma2 * V2.transpose();
        }
        if(storeF){
            if(g->separable == 1 && intersect == 2){
                m_Fi.push_back(Fi2); //grab this Fi2
            }
            else{
                m_Fi.push_back(Fi1); //grab this Fi1
            }
        }

        //Compute strain energy density, W (elastic potential energy density)
        //NOTE: It is MUCH easier to hardcode the elasticity model here, so if we change the constitutive model we NEED to change this here as well!!
        T W = 0;
        T termTwo = 0;
        int elasticityMode = 1; //0 = LINEAR, 1 = FCR
        if(elasticityMode == 0){
            //LINEAR ELASTICITY
            Matrix<T, dim, dim> epsilon1 = 0.5 * (Fi1 + Fi1.transpose()) - Matrix<T, dim, dim>::Identity();
            Matrix<T, dim, dim> epsilon2 = 0.5 * (Fi2 + Fi2.transpose()) - Matrix<T, dim, dim>::Identity();
            T tr_epsilon1 = epsilon1.diagonal().sum();
            T tr_epsilon2 = epsilon2.diagonal().sum();
            W = mu * epsilon1.squaredNorm() + la * 0.5 * tr_epsilon1 * tr_epsilon1; // W = psi(epsilon), field 1 -> this is correct for intersect = 0 and intersect = 1
            if(g->separable == 1 && intersect == 2){
                W = mu * epsilon2.squaredNorm() + la * 0.5 * tr_epsilon2 * tr_epsilon2; // W += psi(epsilon), field 2 -> for intersect = 2 (top intersect)
            }

            //Compute Piola Kirchhoff stress
            Matrix<T, dim, dim> Pi1, Pi2;
            Matrix<T, dim, dim> R = Matrix<T, dim, dim>::Identity();
            Pi1.noalias() = 2 * mu * R * epsilon1 + la * tr_epsilon1 * R;
            
            //Compute term two
            termTwo = (Pi1 * normal).dot(Fi1.col(mode)); //intersect == 0 or 1
            if(g->separable == 1 && intersect == 2){
                Pi2.noalias() = 2 * mu * R * epsilon2 + la * tr_epsilon2 * R;
                termTwo = (Pi2 * normal).dot(Fi2.col(mode)); //intersect == 2 (top intersect)
            }
        }
        else if(elasticityMode == 1){
            //FIXED COROTATED ELASTICITY
            Matrix<T, dim, dim> U1, V1, U2, V2;
            Vector<T, dim> sigma1, sigma2;
            Math::svd(Fi1, U1, sigma1, V1);
            W = mu * (sigma1 - Vector<T, dim>::Ones()).squaredNorm() + T(0.5) * la * std::pow(sigma1.prod() - T(1), 2); //intersect == 0 or 1
            if(g->separable == 1 && intersect == 2){
                Math::svd(Fi2, U2, sigma2, V2);
                W = mu * (sigma2 - Vector<T, dim>::Ones()).squaredNorm() + T(0.5) * la * std::pow(sigma2.prod() - T(1), 2); //intersect == 2
            }

            //Compute Piola Kirchhoff stress
            Matrix<T, dim, dim> Pi1, Pi2;
            T J1 = Fi1.determinant();
            Matrix<T, dim, dim> JFinvT1, JFinvT2;
            Math::cofactor(Fi1, JFinvT1);
            Matrix<T, dim, dim> R1, S1, R2, S2;
            Math::polar_decomposition(Fi1, R1, S1);
            Pi1 = T(2) * mu * (Fi1 - R1) + la * (J1 - 1) * JFinvT1;

            //Compute term two
            termTwo = (Pi1 * normal).dot(Fi1.col(mode)); //intersect == 0 or 1
            if(g->separable == 1 && intersect == 2){
                T J2 = Fi2.determinant();
                Math::cofactor(Fi2, JFinvT2);
                Math::polar_decomposition(Fi2, R2, S2);
                Pi2 = T(2) * mu * (Fi2 - R2) + la * (J2 - 1) * JFinvT2;
                termTwo = (Pi2 * normal).dot(Fi2.col(mode)); //intersect == 2 (top intersect)
            }
        }
        
        //Add first term -> mode = 0 or 1
        Fm += W * normal[mode];

        //Add second term (field 1)
        Fm -= termTwo;

        FmResults.push_back(Fm);
        FmResults.push_back(normal[0]);
        FmResults.push_back(normal[1]);
        FmResults.push_back(W);
        FmResults.push_back(termTwo);

        return FmResults;
    }

    //Elegant line segment intersection check from https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
    //NOTE: this ignores intersections in the colinear case
    bool ccw(Vector<T, dim>& A, Vector<T, dim>& B, Vector<T, dim>& C){
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1]-A[1]) * (C[0]-A[0]);
    }

    //return true if line segments AB and CD intersect
    bool intersect(Vector<T, dim>& A, Vector<T, dim>& B, Vector<T, dim>& C, Vector<T, dim>& D){
        return ccw(A,C,D) != ccw(B,C,D) && ccw(A,B,C) != ccw(A,B,D);
    }

    //Function to compute the intersection point (we'll only use this if we know there is an intersection) - from here: https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    void getIntersection(Vector<T, dim>& A, Vector<T, dim>& B, Vector<T, dim>& C, Vector<T, dim>& D, Vector<T,dim>& intersection){
        T s1x = B[0] - A[0];
        T s1y = B[1] - A[1];
        T s2x = D[0] - C[0];
        T s2y = D[1] - C[1];
        T t = (s2x * (A[1] - C[1]) - s2y * (A[0] - C[0])) / (-s2x * s1y + s1x * s2y);
        intersection[0] = A[0] + (t * s1x);
        intersection[1] = A[1] + (t * s1y);
    }
};

/* Transfer particle stress, nominal stress, and def grad to grid so we can compute J Integral*/
template <class T, int dim>
class CollectJIntegralGridDataOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    Field<Matrix<T, dim, dim>>& m_stress; //holds Vp^0 * PF^T 
    Field<Matrix<T,dim,dim>>& m_F; //holds particle def grad F

    Bow::Field<std::vector<int>>& particleAF;

    DFGMPM::DFGMPMGrid<T, dim>& grid;
    T dx;
    T dt;

    bool useDFG;

    void operator()()
    {
        BOW_TIMER_FLAG("collectJIntegralGridData");
        
        //Now we transfer to the grid!
        grid.colored_for([&](int i) {
            if(!grid.crackInitialized || i < grid.crackParticlesStartIdx){ //skip crack particles if we have them
                const Vector<T, dim> pos = m_X[i];
                //const Matrix<T, dim, dim> stress = m_stress[i];
                const Matrix<T, dim, dim> F = m_F[i];
                
                //We will transfer the deformation gradient through transferring its singular values and rotations (as quaternions) -> three separate intrinsic transfers here: sigma, Uquat, Vquat
                Matrix<T, dim, dim> U, V;
                Vector<T, dim> sigma;
                Vector<T, 4> Uquat, Vquat; //quaternion coefficients for U and V
                Math::svd(F, U, sigma, V);
                
                //Now convert U and V to quaternions
                Matrix<T, 3,3> Upad = Matrix<T,3,3>::Identity();
                Matrix<T, 3,3> Vpad = Matrix<T,3,3>::Identity();
                Upad.topLeftCorner(2,2) = U;
                Vpad.topLeftCorner(2,2) = V; //pad these to be 3x3 for quaternion
                Eigen::Quaternion<T> rotU(Upad);
                Eigen::Quaternion<T> rotV(Vpad);
                rotU.normalize();
                rotV.normalize(); //normalize our quaternions!
                Uquat = rotU.coeffs();
                Vquat = rotV.coeffs();

                //Compute spline
                BSplineWeights<T, dim> spline(pos, dx);
                
                grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, const Vector<T, dim>& dw, DFGMPM::GridState<T, dim>& g) {
                    
                    ///NOTES
                    //Storing grid force in fi1 and fi2 (if separable) -> outdated
                    //We will intrinsically transfer here the singular values and quaternions for U and V rotations from F = UsigmaV^T
                    //Storing accumulated weights in gridSeparability[0] (field 1 weight sum) and gridSeparability[1] (field 2 weight sum)

                    //Notice we treat single-field and two-field nodes differently
                    if (g.separable != 1 || !useDFG) {
                        //Single-field treatment if separable = 0 OR if we are using single field MPM
                        
                        //g.fi1 += stress * dw; //transfer stress to grid, fi
                        g.sigma1 += sigma * w;
                        g.Uquat1 += Uquat * w;
                        g.Vquat1 += Vquat * w;
                        g.gridSeparability[0] += w; //sum up the total weight
                    }
                    else if (g.separable == 1 && useDFG) {
                        //Treat node as having two fields
                        int fieldIdx = particleAF[i][oidx]; //grab the field that this particle belongs in for this grid node (oidx)
                        if (fieldIdx == 0) {
                            //g.fi1 += stress * dw; //transfer stress to grid, fi
                            g.sigma1 += sigma * w;
                            g.Uquat1 += Uquat * w;
                            g.Vquat1 += Vquat * w;
                            g.gridSeparability[0] += w; //sum up the total weight
                        }
                        else if (fieldIdx == 1) {
                            //g.fi2 += stress * dw; //transfer stress to grid, fi (second field)
                            g.sigma2 += sigma * w;
                            g.Uquat2 += Uquat * w;
                            g.Vquat2 += Vquat * w;
                            g.gridSeparability[1] += w; //sum up the total weight (second field)
                        }
                    }
                });
            }
        });

        /* Iterate grid to divide out the total weight sums for P and F since we transfer these intrinsically */
        grid.iterateGrid([&](const Vector<int, dim>& node, DFGMPM::GridState<T, dim>& g) {
            if(g.gridSeparability[0] != 0){
                g.sigma1 /= g.gridSeparability[0];
                g.Uquat1 /= g.gridSeparability[0];
                g.Vquat1 /= g.gridSeparability[0]; //divide by field 1 weight sum
            }
            if (g.separable == 1) {
                if(g.gridSeparability[1] != 0){
                    g.sigma2 /= g.gridSeparability[1];
                    g.Uquat2 /= g.gridSeparability[1];
                    g.Vquat2 /= g.gridSeparability[1]; //divide by field 2 weight sum
                }
            }
        });
    }
};

}
} // namespace Bow::DFGMPM