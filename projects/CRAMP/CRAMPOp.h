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

/*Iterate grid to apply a mode I loading to the configuration based on y1, y2, and sigmaA */
template <class T, int dim>
class ApplyMode1LoadingOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    T y1;
    T y2;
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
                stress = scaledSigmaA;

                if(pos[1] > y1 || pos[1] < y2){ //ONLY apply this force to particles above y1 and below y2!
                    if(pos[1] < y2){
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

}
} // namespace Bow::DFGMPM