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
    Field<Vector<T, dim>>& m_Xinitial;
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
    int elasticityDegradationType;

    Field<T> m_currentVolume;
    Field<Matrix<T, dim, dim>>& m_scaledCauchy;
    Field<int>& m_marker;

    bool computeJIntegral;

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
                Matrix<T, dim, dim> delta_t_tmp_force = -dt * stress[i]; //stress holds Vp^0 * PF^T
                Matrix<T, dim, dim> tmp_force = -stress[i];
                BSplineWeights<T, dim> spline(pos, dx);

                //Compute scaled stress forces (if using elasticity degradation)
                if(elasticityDegradationType == 1){
                    //Use damage scaled Cauchy stress for grid forces
                    delta_t_tmp_force = -dt * m_currentVolume[i] * m_scaledCauchy[i]; //NOTE: from Eq. 190 in MPM course notes
                    tmp_force = -m_currentVolume[i] * m_scaledCauchy[i]; 
                }
                
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

                        //Transfer particle stresses to compute nodal force (for computing Work done by Traction BCs)
                        g.fi1 += tmp_force * dw;

                        //Transfer displacement if we are computing the Jintegral
                        if(computeJIntegral){
                            g.u1 += (pos - m_Xinitial[i]) * mass * w;
                        }

                        //Transfer volume so we can add our Mode 1 loading
                        //g.gridViYi1 += vol * w;
                    }
                    else if (g.separable != 0 && useDFG) {
                        //Treat node as having two fields

                        if(g.separable == 3){ //coupling case, always transfer solid to field 1 and fluid to field 2
                            int materialIdx = m_marker[i];
                            if(materialIdx == 0){
                                g.m1 += mass * w; //have to do this here since we couldn't earlier without interfering with DFG partitioning
                                
                                if (useAPIC) {
                                    g.v1 += delta_APIC;
                                }
                                else {
                                    g.v1 += delta_FLIP;
                                    g.vn1 += delta_vn; //transfer momentum to compute v^n
                                }
                                //Compute normal for field 1 (solid) particles
                                g.n1 += mass * dw; //remember to normalize this later!

                                //Transfer particle stresses to compute nodal force (for computing Work done by Traction BCs)
                                g.fi1 += tmp_force * dw;

                                //Transfer displacement if we are computing the Jintegral
                                if(computeJIntegral){
                                    g.u1 += (pos - m_Xinitial[i]) * mass * w;
                                }
                            }
                            else if(materialIdx == 4){ //transfer fluid particles to field 2
                                g.m2 += mass * w; //have to do this here since we couldn't earlier without interfering with DFG partitioning
                                
                                if (useAPIC) {
                                    g.v2 += delta_APIC;
                                }
                                else {
                                    g.v2 += delta_FLIP;
                                    g.vn2 += delta_vn; //transfer momentum to compute v^n
                                }
                                //Compute normal for field 2 (fluid) particles
                                g.n2 += mass * dw; //remember to normalize this later!

                                //Transfer particle stresses to compute nodal force (for computing Work done by Traction BCs)
                                g.fi2 += tmp_force * dw;

                                //Transfer displacement if we are computing the Jintegral
                                if(computeJIntegral){
                                    g.u2 += (pos - m_Xinitial[i]) * mass * w;
                                }
                            }
                        }
                        else{ //regular two field transfer from DFG
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

                                //Transfer particle stresses to compute nodal force (for computing Work done by Traction BCs)
                                g.fi1 += tmp_force * dw;

                                //Transfer displacement if we are computing the Jintegral
                                if(computeJIntegral){
                                    g.u1 += (pos - m_Xinitial[i]) * mass * w;
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

                                //Transfer particle stresses to compute nodal force (for computing Work done by Traction BCs)
                                g.fi2 += tmp_force * dw;

                                //Transfer displacement if we are computing the Jintegral
                                if(computeJIntegral){
                                    g.u2 += (pos - m_Xinitial[i]) * mass * w;
                                }
                            }
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

            if(computeJIntegral){
                g.u1.cwiseProduct(alpha1); //divide out m_i
            }

            g.v1 += gravity_term;
            g.vn1 = g.vn1.cwiseProduct(alpha1); // this is how we get v1^n
            g.x1 = node.template cast<T>() * dx; //put nodal position in x1 regardless of separability
            g.fi1 += mass1 * (gravity_term / dt); //add gravity term to nodal force
            if (g.separable != 0) {
                T mass2 = g.m2;
                Vector<T, dim> alpha2;
                alpha2 = Vector<T, dim>::Ones() * ((T)1 / mass2);
                g.v2 = g.v2.cwiseProduct(alpha2);

                if(computeJIntegral){
                    g.u2.cwiseProduct(alpha2); //divide out m_i
                }

                g.v2 += gravity_term;
                g.vn2 = g.vn2.cwiseProduct(alpha2); //this is how we get v2^n
                g.fi2 += mass2 * (gravity_term / dt); //add gravity term to nodal force, field 2
            }
        });
    }
};

/*Simple Linear Tension Elasticity Degradation (Homel2016 Eq. 26, 27) */
template <class T, int dim>
class SimpleLinearTensionElasticityDegOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    Field<Matrix<T, dim, dim>>& m_cauchy;
    Field<Matrix<T, dim, dim>>& m_scaledCauchy;
    std::vector<T>& m_Dp;
    T degAlpha;
    DFGMPM::DFGMPMGrid<T, dim>& grid;
    Field<int>& m_marker;

    void operator()()
    {
        BOW_TIMER_FLAG("simpleLinearTensionElasticityDegradation");
        grid.parallel_for([&](int i) {
            if(m_marker[i] == 0){
                //Compute updated damage and the associated scaled Cauchy stress (Homel 2016 eq. 26 and 27) 
                Matrix<T, dim, dim> sigmaScaled = Matrix<T, dim, dim>::Zero();
                Vector<T, dim> eigenVec;
                T eigenVal = 0.0;
                Eigen::EigenSolver<Matrix<T, dim, dim>> es(m_cauchy[i]);

                //Compute Scaled Cauchy (Homel2016 Eq. 26,27)
                for (int j = 0; j < dim; j++) {
                    for (int k = 0; k < dim; k++) {
                        eigenVec(k) = es.eigenvectors().col(j)(k).real(); //get the real parts of each eigenvector
                    }
                    eigenVal = es.eigenvalues()(j).real();
                    if(eigenVal > 0){
                        eigenVal *= std::pow((1 - m_Dp[i]), degAlpha);
                    }

                    sigmaScaled += eigenVal * (eigenVec * eigenVec.transpose());
                }
                m_scaledCauchy[i] = sigmaScaled;
            }
            else if(m_marker[i] == 4){
                m_scaledCauchy[i] = m_cauchy[i]; //no degradation of fluid particles
            }
        });
    }
};

/* Update Rankine Damage */
template <class T, int dim>
class UpdateRankineDamageOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    Field<Matrix<T, dim, dim>>& m_cauchy;
    std::vector<T>& m_Dp;
    DFGMPM::DFGMPMGrid<T, dim>& grid;

    std::vector<T>& m_sigmaC; //particle sigmaC
    std::vector<T>& Hs;

    Field<bool> m_useDamage;

    void operator()()
    {
        BOW_TIMER_FLAG("updateRankineDamage");
        grid.parallel_for([&](int i) {
            if(m_useDamage[i]){
                //Compute updated damage
                Vector<T, dim> eigenVec;
                T eigenVal = 0.0;
                T maxEigVal = -10000000.0;
                Eigen::EigenSolver<Matrix<T, dim, dim>> es(m_cauchy[i]);
                
                //Step 1.) compute maxEigVal
                for (int j = 0; j < dim; j++) {
                    eigenVal = es.eigenvalues()(j).real();
                    maxEigVal = (maxEigVal > eigenVal) ? maxEigVal : eigenVal;
                }

                //Step 2.) Update Damage based on maxEigVal
                if(maxEigVal > m_sigmaC[i]){
                    T newD = (1 + Hs[i]) * (1 - (m_sigmaC[i] / maxEigVal)); 
                    m_Dp[i] = std::max(m_Dp[i], std::min(1.0, newD));
                }
            }
        });
    }
};

/* Update Hyperbolic Tangent Damage */
template <class T, int dim>
class UpdateTanhDamageOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    Field<Matrix<T, dim, dim>>& m_F;
    std::vector<T>& m_Dp;
    DFGMPM::DFGMPMGrid<T, dim>& grid;

    T lamC;
    T tanhWidth;

    Field<bool> m_useDamage;
    Field<T> m_lamMax;

    void operator()()
    {
        BOW_TIMER_FLAG("updateTanhDamage");
        
        //Compute updated damage using d = 0.5 + 0.5tanh((lamMax - lamC) / tanhWidth)
        grid.parallel_for([&](int i) {
            if(m_useDamage[i]){            
                //Update damage values
                T newD = 0.5 + (0.5 * tanh((m_lamMax[i] - lamC)/tanhWidth));
                m_Dp[i] = std::max(m_Dp[i], newD); //function will always be between 0 and 1, so we just have to make sure it's monotonically increasing
            }
        });
    }
};

/* Compute Lambda Max for each particle */
template <class T, int dim>
class ComputeLamMaxOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    DFGMPM::DFGMPMGrid<T, dim>& grid;
    Field<Matrix<T, dim, dim>>& m_F;
    Field<T>& m_lamMax;

    void operator()()
    {
        BOW_TIMER_FLAG("comuteLamMax");
        
        grid.serial_for([&](int i) {
            //Compute polar decomposition so we can compute maximum stretch, lamMax
            // Matrix<T, dim, dim> R, S;
            // Math::polar_decomposition(m_F[i], R, S);
            // m_lamMax[i] = S(0,0); //NOTE: this assumes SVs were sorted in SVD
            
            Matrix<T,dim, dim> C;
            C = m_F[i].transpose() * m_F[i];
            T eigenVal = 0.0;
            T maxEigVal = -10000000.0;
            Eigen::EigenSolver<Matrix<T, dim, dim>> es(C);
            
            //compute maxEigVal of C
            for (int j = 0; j < dim; j++) {
                eigenVal = es.eigenvalues()(j).real();
                maxEigVal = (maxEigVal > eigenVal) ? maxEigVal : eigenVal;
            }
            m_lamMax[i] = sqrt(maxEigVal);
            //std::cout << "S(0,0): " << S(0,0) << ", S(1,1): " << S(1,1) << ", sqrt(maxEigVal of C): " << sqrt(maxEigVal) << std::endl;
        });
    }
};


/* Transfer Cauchy stress and deformation gradient to the grid */
template <class T, int dim>
class TensorP2GOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    std::vector<T>& m_mass;
    Field<Matrix<T, dim, dim>>& m_cauchy;
    Field<Matrix<T, dim, dim>>& m_F;

    Bow::Field<std::vector<int>>& particleAF;

    DFGMPM::DFGMPMGrid<T, dim>& grid;
    T dx;

    bool useDFG;
    Field<int>& m_marker;

    void operator()()
    {
        BOW_TIMER_FLAG("tensorP2G");
        grid.colored_for([&](int i) {
            if((!grid.crackInitialized || i < grid.crackParticlesStartIdx) && m_marker[i] == 0){ //skip crack particles if we have them and only process SOLID particles!
                const Vector<T, dim> pos = m_X[i];
                const T mass = m_mass[i];
                const Matrix<T, dim, dim> cauchyXmass = m_cauchy[i] * mass; //cauchy * m_p
                const Matrix<T, dim, dim> defGradXmass = m_F[i] * mass; //F * m_p
                BSplineWeights<T, dim> spline(pos, dx);
                
                grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, const Vector<T, dim>& dw, DFGMPM::GridState<T, dim>& g) {
                    //Notice we treat single-field and two-field nodes differently
                    if (g.separable == 0 || !useDFG) {
                        //Single-field treatment if separable = 0 OR if we are using single field MPM
                        g.cauchy1 += cauchyXmass * w;
                        g.Fi1 += defGradXmass * w;
                    }
                    else if (g.separable != 0 && useDFG) {

                        if(g.separable == 3){ //coupling case, always transfer solid to field 1 and fluid to field 2 (here we already know it's solid)
                            g.cauchy1 += cauchyXmass * w;
                            g.Fi1 += defGradXmass * w;
                        }
                        else{ //regular two field transfer from DFG
                            int fieldIdx = particleAF[i][oidx]; //grab the field that this particle belongs in for this grid node (oidx)
                            if (fieldIdx == 0) {
                                g.cauchy1 += cauchyXmass * w;
                                g.Fi1 += defGradXmass * w;
                            }
                            else if (fieldIdx == 1) {
                                g.cauchy2 += cauchyXmass * w;
                                g.Fi2 += defGradXmass * w;
                            }
                        }
                    }
                });
            }
        });

        /* Iterate grid to divide out the grid masses from cauchy and defGrad */
        grid.iterateGrid([&](const Vector<int, dim>& node, DFGMPM::GridState<T, dim>& g) {
            g.cauchy1 /= g.m1;
            g.Fi1 /= g.m1;
            if (g.separable != 0 && g.separable != 3) { //don't treat field 2 if coupling case (solid-fluid is sep = 3)
                g.cauchy2 /= g.m2;
                g.Fi2 /= g.m2;
            }
        });
    }
};

/* Transfer Cauchy stress and deformation gradient back from the grid for a smoothed cauchy and F! */
template <class T, int dim>
class TensorG2POp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    Field<Matrix<T, dim, dim>>& m_cauchySmoothed;
    Field<Matrix<T, dim, dim>>& m_FSmoothed;

    Bow::Field<std::vector<int>>& particleAF;

    DFGMPM::DFGMPMGrid<T, dim>& grid;
    T dx;

    bool useDFG;
    Field<int>& m_marker;

    void operator()()
    {
        BOW_TIMER_FLAG("tensorG2P");
        grid.parallel_for([&](int i) {
            if((!grid.crackInitialized || i < grid.crackParticlesStartIdx) && m_marker[i] == 0){ //skip crack particles if we have them and only process SOLID particles!
                const Vector<T, dim> pos = m_X[i];
                BSplineWeights<T, dim> spline(pos, dx);

                Matrix<T, dim, dim> cauchySmooth = Matrix<T, dim, dim>::Zero();
                Matrix<T, dim, dim> FSmooth = Matrix<T, dim, dim>::Zero();
                
                grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, const Vector<T, dim>& dw, DFGMPM::GridState<T, dim>& g) {
                    //Notice we treat single-field and two-field nodes differently
                    if (g.separable == 0 || !useDFG) {
                        //Single-field treatment if separable = 0 OR if we are using single field MPM
                        cauchySmooth += g.cauchy1 * w;
                        FSmooth += g.Fi1 * w;
                    }
                    else if (g.separable != 0 && useDFG) {

                        if(g.separable == 3){ //coupling case, always transfer solid to field 1 and fluid to field 2 (here we already know it's solid)
                            cauchySmooth += g.cauchy1 * w;
                            FSmooth += g.Fi1 * w;
                        }
                        else{
                            //Treat node as having two fields
                            int fieldIdx = particleAF[i][oidx]; //grab the field that this particle belongs in for this grid node (oidx)
                            if (fieldIdx == 0) {
                                cauchySmooth += g.cauchy1 * w;
                                FSmooth += g.Fi1 * w;
                            }
                            else if (fieldIdx == 1) {
                                cauchySmooth += g.cauchy2 * w;
                                FSmooth += g.Fi2 * w;
                            }
                        }
                    }
                });

                //std::cout << "Before setting values in tensorG2P for index: " << i << std::endl;
                m_cauchySmoothed[i] = cauchySmooth;
                m_FSmoothed[i] = FSmooth;
                //std::cout << "Finished tensorG2P for index: " << i << std::endl;
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
                    // if (g.separable == 0 || !useDFG) {
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
                    // else if (g.separable != 0 && useDFG) {
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
        //     if (g.separable != 0) {
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

    bool nodalLoading;

    T width;

    T y1;
    T y2;

    T x1;
    T x2;

    T dx;
    T dt;

    DFGMPM::DFGMPMGrid<T, dim>& grid;

    Field<T> m_vol;

    T ppc;

    T& totalWork;

    void operator()()
    {
        BOW_TIMER_FLAG("applyMode1Loading");

        T stress = scaledSigmaA;
        if(!nodalLoading){
            //Using particle volume, compute per particle forces using the scaledSigmaA passed in, then transfer this force to the grid and apply it
            grid.colored_for([&](int i) {
                if(!grid.crackInitialized || i < grid.crackParticlesStartIdx){ //skip crack particles if we have them
                    const Vector<T, dim> pos = m_X[i];
                    const int marker = m_marker[i];
                    stress = scaledSigmaA;

                    if(marker == 4 || marker == 5){ //ONLY apply this force to particles above y1 and below y2!
                        if(marker == 5){
                            stress *= -1; //apply negative here for particles below y2
                        }

                        //m_vol = dx^2 / PPC, however, need to compute Ap = dx / sqrt(PPC)
                        T ppcSqrt = std::pow(ppc, (T)1 / (T)dim);
                        T Ap = m_vol[i] * (ppcSqrt / dx);
                        T fp = Ap * stress; //particle force (working simply with y direction magnitude, not full vector)

                        //std::cout << "particle idx:" << i << "fp:" << fp << std::endl;

                        BSplineWeights<T, dim> spline(pos, dx);
                        grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, const Vector<T, dim>& dw, DFGMPM::GridState<T, dim>& g) {
                            //Store f_i in gridViYi since we're not using it anyway
                            g.gridViYi1 += fp * w;
                        });
                    }
                }
            });
        }
        else{
            //F_total = sigmaA * width * thickness
            //Distribute into N pieces with N = width / dx
            //F_nodal = F_total / N = (sigmaA * width * thickness) / (width / dx) -> we use t = 1 here
            //F_nodal = sigmaA * thickness * dx -> t = 1
            stress *= dx; //F_nodal (ends get half of this)
        }
        
        /* Iterate grid to apply these loadings to the velocities */
        grid.iterateGrid([&](const Vector<int, dim>& node, DFGMPM::GridState<T, dim>& g) {
            Vector<T,dim> xi = node.template cast<T>() * dx;
            T fi = 0.0;
            if(!nodalLoading){
                fi = g.gridViYi1; //stored fi in here
            }
            else{
                T eps = dx * 0.05;
                T midX1 = x1 + eps;
                T midX2 = x2 - eps; //for middle node loading
                // T endX1 = x1 - eps;
                // T endX2 = x2 + eps;
                Vector<T,dim> ti = Vector<T,dim>::Zero();
                if(xi[0] > midX1 && xi[0] < midX2){ //only within the x range of the material!
                    if(std::abs(xi[1] - y1) < eps){ //top pulls up
                        fi = stress;
                        ti[1] = fi;
                        //g.fi1 -= ti; //add traction to nodal force //ti = potential energy force, don't need to add it again!
                    }
                    else if(std::abs(xi[1] - y2) < eps){ //bottom pulls down
                        fi = -1 * stress;
                        ti[1] = fi;
                        //g.fi1 -= ti; //add traction to nodal force
                    }
                }
                else if(std::abs(xi[0] - x1) < eps || std::abs(xi[0] - x2) < eps){ //end nodes
                    if(std::abs(xi[1] - y1) < eps){ //top pulls up
                        fi = 0.5 * stress;
                        ti[1] = fi;
                        //g.fi1 -= ti; //add traction to nodal force
                    }
                    else if(std::abs(xi[1] - y2) < eps){ //bottom pulls down
                        fi = -0.5 * stress;
                        ti[1] = fi;
                        //g.fi1 -= ti; //add traction to nodal force
                    }
                }
                else{
                    //if not inside the traction BC region, zero out these nodal force values
                    g.fi1 = Vector<T,dim>::Zero();
                    g.fi2 = Vector<T,dim>::Zero();
                }
            }
            g.v1[1] += (fi / g.m1) * dt; //update velocity based on our computed forces
        });

        //Now compute traction BC work done
        grid.iterateGridSerial([&](const Vector<int, dim>& node, DFGMPM::GridState<T, dim>& g) {
            if(g.fi1.norm() > 0){
                totalWork += -g.fi1.dot(dt * g.v1);
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

    int crackType;
    
    void operator()()
    {
        //First let's iterate separable grid nodes and compute v_cm and store it in fi1 (since we won't apply dynamic impulses for cramp)
        BOW_TIMER_FLAG("evolveCrackPlanes");

        //set some indeces so we can always set the top and bottom plane tips to be equal to the central crack tip
        int crackTipRightIdx = topPlane_startIdx - 1;
        int topTipRightIdx = bottomPlane_startIdx - 1;
        int bottomTipRightIdx = m_X.size() - 1;
        int crackTipLeftIdx = grid.crackParticlesStartIdx;
        int topTipLeftIdx = topPlane_startIdx;
        int bottomTipLeftIdx = bottomPlane_startIdx;

        //Iterate separable nodes and compute V_cm for each
        grid.iterateGrid([&](const Vector<int, dim>& node, DFGMPM::GridState<T, dim>& g) {
            if (g.separable != 0) {
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
                    if (g.separable == 0) {
                        //treat as single field node
                        picV += w * g.v1;
                        flipV += w * (g.v1 - g.vn1); 
                        //picX += w * g.x1;
                    }
                    else if (g.separable != 0) {
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
                    if (g.separable == 0) {
                        //treat as single field node -> if upward, add contribution for top crack
                        if(g.v1[1] > 0){
                            picV += w * g.v1;
                            flipV += w * (g.v1 - g.vn1);
                        }
                    }
                    else if (g.separable != 0) {
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

                //if this is the top plane tip, set it equal to the updated crackTip (left side and middle ONLY)
                if(i == topTipRightIdx && (crackType == 0 || crackType == 1)){
                    m_X[topTipRightIdx] = m_X[crackTipRightIdx];
                }
                if(i == topTipLeftIdx && (crackType == 1 || crackType == 2)){ //only for middle and right side cracks!
                    m_X[topTipLeftIdx] = m_X[crackTipLeftIdx];
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
                    if (g.separable == 0) {
                        //treat as single field node -> if downard, add contribution for bottom crack
                        if(g.v1[1] < 0){
                            picV += w * g.v1;
                            flipV += w * (g.v1 - g.vn1);
                        }
                    }
                    else if (g.separable != 0) {
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

                //if this is the bottom plane tip, set it equal to the updated crackTip (left cracks and middle cracks ONLY)
                if(i == bottomTipRightIdx && (crackType == 0 || crackType == 1)){
                    m_X[bottomTipRightIdx] = m_X[crackTipRightIdx];
                }
                if(i == bottomTipLeftIdx && (crackType == 1 || crackType == 2)){
                    m_X[bottomTipLeftIdx] = m_X[crackTipLeftIdx];
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
class ComputeJIntegralLineTermOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    
    int topPlane_startIdx;
    int bottomPlane_startIdx;

    Field<Matrix<T, dim, dim>>& stress;

    DFGMPM::DFGMPMGrid<T, dim>& grid;
    T dx;
    T dt;

    T mu;
    T la;

    bool useDFG;

    T operator()(Vector<T,dim> center, Vector<int,4> contour, bool containsCrackTip, bool trackContributions, std::ofstream& file, std::ofstream& file2)
    {
        BOW_TIMER_FLAG("computeJIntegralLineTerm");

        T J_I = 0; //set J integral mode I to 0 for now
        T J_II = 0; //set J integral mode II to 0 for now

        //NOTE: This routine is designed for HORIZONTAL LEFT SIDE CRACKS (b.c. of contour intersection assumptions --> counter clockwise path hits top of crack first)

        //STEP 0: Grab an iterable list of unprocessed contour points
        //Iterate a rectangular contour around the closest node to the crack tip --> COUNTER CLOCKWISE STARTING FROM TOP LEFT NODE
        std::vector<Vector<T,dim>> contourPoints;
        std::vector<DFGMPM::GridState<T,dim>*> contourGridStates;
        BSplineWeights<T, dim> spline(center, dx);
        grid.iterateRectangularContour(spline, contour[0], contour[1], contour[2], contour[3], [&](const Vector<int, dim>& node, DFGMPM::GridState<T, dim>& g) {
            Vector<T,dim> xi = node.template cast<T>() * dx;
            //Grab data
            contourPoints.push_back(xi);
            contourGridStates.push_back(&g); //hold pointers to grid node data

            //std::cout << "x_i: (" << xi[0] << "," << xi[1] << ")" << std::endl;
        });
        // file << "Initial Contour List:\n";
        // for(int i = 0; i < (int)contourPoints.size(); ++i){
        //     file << "idx: " << i <<  ", xi: (" << contourPoints[i][0] << "," << contourPoints[i][1] << "), separable:" << contourGridStates[i]->separable << "\n";
        // }

        //STEP 1: Construct ordered list of contour points starting with intersection with bottom of crack, counter-clockwise around contour, and then ending with intersection with top of crack
        bool foundBottomIntersection = false;
        bool foundTopIntersection = false;
        bool foundIntersection = false;
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
        if(foundTopIntersection){
            foundIntersection = true; //Now we have to make sure we treat this NON-INTERSECTING case differently (and we expect J = 0)
        }

        //NOTE: Now this code greatly diverges between two cases, an INTERSECTING J!= 0 case, and a NON-INTERSECTING J = 0 case
        //INTERSECTION CASE
        if(useDFG && foundIntersection && containsCrackTip){
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

            //EDGE CASE: if crack intersect is exactly at a node
            T epsilon = 1e-9;
            if(std::abs(bottomIntersection[1] - contourPoints[bottomIntersectionIdx + 1][1]) < epsilon){
                bottomIntersectionIdx++;
            }

            //STEP 1c: Check that crack is actually open (if it is not, we cannot compute J integral)
            std::vector<Vector<T,dim>> finalContourPoints;
            std::vector<DFGMPM::GridState<T,dim>*> finalContourGridStates;
            if(std::abs(topIntersection[1] - bottomIntersection[1]) < epsilon){
                std::cout << "Crack is not opened at the intersection points, cannot compute J-integral!" << std::endl;

                //We construct all of the pieces we need to finally compute the J integral noting that the crack is closed here
                //Specifically, a list of contour points starting with crack intersection and going counter-clockwise around back to the crack intersection 

                //Crack intersection
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
            std::vector<Matrix<T,dim,dim>> m_Pi; //collest computed Piola Kirchhoff Stresses
            std::vector<Matrix<T,dim,dim>> m_Fi_Interpolated; //collect the reconstructed Fi's that were interpolated between for x_bottom and x_top
            
            for(int i = 0; i < (int)finalContourPoints.size() - 1; ++i){ //iterate contour segments
                T Fsum_I = 0; //this is what we focus on setting for each segment (three cases below)
                T Fsum_II = 0; //mode II
                Vector<T,dim> x1 = finalContourPoints[i];
                Vector<T,dim> x2 = finalContourPoints[i+1];
                Matrix<T,dim,dim> Fi1, Fi2, Finterp1, Finterp2, Pi1, Pi2;
                if(i == 0){ //first segment
                    //If first segment, first we compute interpolations for bottom intersection point
                    Vector<T, dim> xi1 = contourPoints[bottomIntersectionIdx];
                    Vector<T, dim> xi2 = contourPoints[bottomIntersectionIdx + 1];
                    T blendRatio = abs(x1[1] - xi1[1]) / abs(xi2[1] - xi1[1]);

                    //Interpolate the deformation gradient ingredients! --> USE FIELD 1 FOR BOTTOM INTERSECT!!!
                    DFGMPM::GridState<T,dim>* gi1 = contourGridStates[bottomIntersectionIdx]; //NOTE: this will very likely be a separable node!!!! only use field 1 for interpolating bottom intersect
                    DFGMPM::GridState<T,dim>* gi2 = contourGridStates[bottomIntersectionIdx + 1]; //grab the two grid states to interpolate between

                    //Save these Fs for viewing
                    Finterp1 = gi1->Fi1;
                    Finterp2 = gi2->Fi1;
                    m_Fi_Interpolated.push_back(Finterp1);
                    m_Fi_Interpolated.push_back(Finterp2); //save these for viewing
                    //Now interpolate b/w the Fs
                    Fi1 = ((gi1->Fi1 * gi1->m1 * (1 - blendRatio)) + (gi2->Fi1 * gi2->m1 * blendRatio)) / ((gi1->m1*(1-blendRatio)) + (gi2->m1 * blendRatio)); // mass weighted 1-D interpolation, USING FIELD 1 VALUES FOR BOTTOM INTERSECT
                    m_Fi.push_back(Fi1);
                    std::vector<T> Fm1_I = computeFm(Fi1, x2 - x1, 0);
                    std::vector<T> Fm1_II = computeFm(Fi1, x2 - x1, 1);

                    //Compute Fm2 (from actual second point)
                    DFGMPM::GridState<T,dim>* g2 = finalContourGridStates[i+1];
                    // Fi2 = computeF(g2->sigma1, g2->Uquat1, g2->Vquat1);
                    Fi2 = g2->Fi1;
                    m_Fi.push_back(Fi2);
                    std::vector<T> Fm2_I = computeFm(Fi2, x2 - x1, 0);
                    std::vector<T> Fm2_II = computeFm(Fi2, x2 - x1, 1);

                    //Compute Fsum
                    Fsum_I = Fm1_I[0] + Fm2_I[0];
                    Fsum_II = Fm1_II[0] + Fm2_II[0];

                    //Store Fm, normal, W, and termTwo of end points
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

                    blendRatios.push_back(blendRatio);

                    //Store Piola Kirchhoff Stresses
                    Pi1(0,0) = Fm1_I[5];
                    Pi1(0,1) = Fm1_I[6];
                    Pi1(1,0) = Fm1_I[7];
                    Pi1(1,1) = Fm1_I[8];
                    Pi2(0,0) = Fm2_I[5];
                    Pi2(0,1) = Fm2_I[6];
                    Pi2(1,0) = Fm2_I[7];
                    Pi2(1,1) = Fm2_I[8];
                    m_Pi.push_back(Pi1);
                    m_Pi.push_back(Pi2);
                }
                else if(i == (int)finalContourPoints.size() - 2){ //last segment
                    //Compute Fm1 (from actual first point)
                    DFGMPM::GridState<T,dim>* g1 = finalContourGridStates[i];
                    //Fi1 = computeF(g1->sigma1, g1->Uquat1, g1->Vquat1);
                    Fi1 = g1->Fi1;
                    m_Fi.push_back(Fi1);
                    std::vector<T> Fm1_I = computeFm(Fi1, x2 - x1, 0);
                    std::vector<T> Fm1_II = computeFm(Fi1, x2 - x1, 1);
                    
                    //compute interpolations for top intersection point (second endpoint)
                    Vector<T, dim> xi1 = contourPoints[topIntersectionIdx];
                    Vector<T, dim> xi2 = contourPoints[topIntersectionIdx + 1];
                    T blendRatio = abs(x2[1] - xi1[1]) / abs(xi2[1] - xi1[1]);

                    //Interpolate the deformation gradient ingredients! --> USE FIELD 2 FOR TOP INTERSECT!!!
                    DFGMPM::GridState<T,dim>* gi1 = contourGridStates[topIntersectionIdx];
                    DFGMPM::GridState<T,dim>* gi2 = contourGridStates[topIntersectionIdx + 1]; //grab the two grid states to interpolate between -> this one is very likely separable!! use field 2 only for interpolating
                    Finterp1 = gi1->Fi1;
                    m_Fi_Interpolated.push_back(Finterp1);

                    if(gi2->separable == 1){
                        file << "Top Intersect Interpolated Using Field 2\n";
                        Finterp2 = gi2->Fi2; //NOTE WE USE FIELD 2 VALUES FOR TOP INTERSECT!!
                        Fi2 = ((gi1->Fi1 * gi1->m1 * (1 - blendRatio)) + (gi2->Fi2 * gi2->m2 * blendRatio)) / ((gi1->m1*(1-blendRatio)) + (gi2->m2 * blendRatio)); // mass weighted 1-D interpolation, using field 2
                    }
                    else{
                        file << "Top Intersect Interpolated Using Field 1\n";
                        Finterp2 = gi2->Fi1; //if not separable, just use field 1
                        Fi2 = ((gi1->Fi1 * gi1->m1 * (1 - blendRatio)) + (gi2->Fi1 * gi2->m1 * blendRatio)) / ((gi1->m1*(1-blendRatio)) + (gi2->m1 * blendRatio)); // mass weighted 1-D interpolation, using field 1
                    }
                    m_Fi_Interpolated.push_back(Finterp2);
                    m_Fi.push_back(Fi2);

                    //Compute Fm for this interpolated Fi2            
                    std::vector<T> Fm2_I = computeFm(Fi2, x2 - x1, 0);
                    std::vector<T> Fm2_II = computeFm(Fi2, x2 - x1, 1);

                    //Compute Fsum
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

                    blendRatios.push_back(blendRatio);

                    //Store Piola Kirchhoff Stresses
                    Pi1(0,0) = Fm1_I[5];
                    Pi1(0,1) = Fm1_I[6];
                    Pi1(1,0) = Fm1_I[7];
                    Pi1(1,1) = Fm1_I[8];
                    Pi2(0,0) = Fm2_I[5];
                    Pi2(0,1) = Fm2_I[6];
                    Pi2(1,0) = Fm2_I[7];
                    Pi2(1,1) = Fm2_I[8];
                    m_Pi.push_back(Pi1);
                    m_Pi.push_back(Pi2);
                }
                else{ //rest of the non-intersect segments
                    DFGMPM::GridState<T,dim>* g1 = finalContourGridStates[i];
                    DFGMPM::GridState<T,dim>* g2 = finalContourGridStates[i+1];

                    //Compute F for each endpoint --> NEITHER should be separable in this case
                    Fi1 = g1->Fi1;
                    Fi2 = g2->Fi1;
                    m_Fi.push_back(Fi1);
                    m_Fi.push_back(Fi2);

                    std::vector<T> Fm1_I = computeFm(Fi1, x2 - x1, 0);
                    std::vector<T> Fm2_I = computeFm(Fi2, x2 - x1, 0);
                    std::vector<T> Fm1_II = computeFm(Fi1, x2 - x1, 1);
                    std::vector<T> Fm2_II = computeFm(Fi2, x2 - x1, 1);
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

                    //Store Piola Kirchhoff Stresses
                    Pi1(0,0) = Fm1_I[5];
                    Pi1(0,1) = Fm1_I[6];
                    Pi1(1,0) = Fm1_I[7];
                    Pi1(1,1) = Fm1_I[8];
                    Pi2(0,0) = Fm2_I[5];
                    Pi2(0,1) = Fm2_I[6];
                    Pi2(1,0) = Fm2_I[7];
                    Pi2(1,1) = Fm2_I[8];
                    m_Pi.push_back(Pi1);
                    m_Pi.push_back(Pi2);
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
            file << "====================================================== J-Integral Computation using LxDxRxU = " << contour[0] << "x" << contour[1] << "x" << contour[2] << "x" << contour[3] << "Contour Centered at (" << center[0] <<  "," << center[1] << ") ======================\n";
            file << "INTERSECTING CONTOUR CASE (J != 0)" << "\n";
            file << "Bottom Intersection | Idx: " << bottomIntersectionIdx << " Point: (" << bottomIntersection[0] << "," << bottomIntersection[1] << "), Blend Ratio: " << blendRatios[0] << " \n";
            file << "Top Intersection | Idx: " << topIntersectionIdx << " Point: (" << topIntersection[0] << "," << topIntersection[1] << "), Blend Ratio: " << blendRatios[1] << "\n";
            // for(int i = 0; i < (int)finalContourPoints.size(); ++i){
            //     file << "idx:" << i << ", Point: (" << finalContourPoints[i][0] << "," << finalContourPoints[i][1] << "), Fsum_I: " << Fsum_I_List[i] << " \n";
            // }
            file << "------Bottom Intersection Interpolation (first point in contour)------\n";
            file << "Finterp1:" << m_Fi_Interpolated[0] << "\n";
            file << "Finterp2:" << m_Fi_Interpolated[1] << "\n";
            for(int i = 0; i < (int)finalContourPoints.size() - 1; ++i){
                file << "-----------------<Line Segment " << i << ", Fsum_I: " << Fsum_I_List[i] << ", Delta_I: " << DeltaI_List[i] << ", J_I Contribution: " << Fsum_I_List[i] * (DeltaI_List[i] / 2.0) << ">-----------------\n";
                file << "idx1: " << i << ", Point: (" << finalContourPoints[i][0] << "," << finalContourPoints[i][1] << "), Fm_I: " << Fm_I_SegmentList[i*2] << ", Normal: [" << Fm_I_NormalX[i*2] << "," << Fm_I_NormalY[i*2] << "], W: " << Fm_I_W[i*2] << ", termTwo: " << Fm_I_termTwo[i*2] << "\nFi1: " << m_Fi[i*2] << "\nPi1: " << m_Pi[i*2] << " \n";
                file << "-----\n"; 
                file << "idx2: " << i+1 << ", Point: (" << finalContourPoints[i+1][0] << "," << finalContourPoints[i+1][1] << "), Fm_I: " << Fm_I_SegmentList[(i*2) + 1] << ", Normal: [" << Fm_I_NormalX[(i*2) + 1] << "," << Fm_I_NormalY[(i*2) + 1] << "], W: " << Fm_I_W[(i*2) + 1] << ", termTwo: " << Fm_I_termTwo[(i*2) + 1] << "\nFi2: " << m_Fi[(i*2)+1] << "\nPi2: " << m_Pi[(i*2)+1] << " \n";
            }
            file << "------Top Intersection Interpolation (last point in contour)------\n";
            file << "Finterp1:" << m_Fi_Interpolated[2] << "\n";
            file << "Finterp2:" << m_Fi_Interpolated[3] << "\n";
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

            //Write second file with all J_I Contributions
            if(trackContributions){
                file2 << "====================================================== J-Integral Computation using LxDxRxU = " << contour[0] << "x" << contour[1] << "x" << contour[2] << "x" << contour[3] << "Contour Centered at (" << center[0] <<  "," << center[1] << ") ======================\n";
                file2 << "Line Segment Index, J_I Contribution, Fi_11, Fi_12, Fi_21, Fi_22, Pi_22, W \n";
                for(int i = 0; i < (int)finalContourPoints.size() - 1; ++i){
                    file2 << i << ", " << Fsum_I_List[i] * (DeltaI_List[i] / 2.0) << ", " << m_Fi[i*2](0,0) << ", " << m_Fi[i*2](0,1) << ", " << m_Fi[i*2](1,0) << ", " << m_Fi[i*2](1,1) << ", " <<  m_Pi[i*2](1,1) << ", " << Fm_I_W[i*2] << "\n";
                }
                int i = (int)finalContourPoints.size() - 2;
                file2 << i+1 << ", " << 0.0 << ", " << m_Fi[(i*2)+1](0,0) << ", " << m_Fi[(i*2)+1](0,1) << ", " << m_Fi[(i*2)+1](1,0) << ", " << m_Fi[(i*2)+1](1,1) << ", " <<  m_Pi[(i*2)+1](1,1) << ", " << Fm_I_W[(i*2)+1] << "\n";
            }
        }
        //NON-INTERSECTING CASE (J = 0) ==============================================================================
        else if(useDFG && !foundIntersection && !containsCrackTip){
            //STEP 1c: Construct our contour list- in this NON-INTERSECTING case, we simply must ensure that we start and end with the same point
            std::vector<Vector<T,dim>> finalContourPoints;
            std::vector<DFGMPM::GridState<T,dim>*> finalContourGridStates;
            for(int i = 0; i < (int)contourPoints.size(); ++i){
                finalContourPoints.push_back(contourPoints[i]);
                finalContourGridStates.push_back(contourGridStates[i]);
            }
            //Additionally, re-add the first point to the end
            finalContourPoints.push_back(contourPoints[0]);
            finalContourGridStates.push_back(contourGridStates[0]);

            //STEP 2: Compute J-integral!
            
            //DEBUG: Setup a bunch of lists to hold intermediate data for debugging
            std::vector<T> Fsum_I_List;
            std::vector<T> DeltaI_List;
            std::vector<T> Fm_I_SegmentList; //Store Fm1 and Fm2 for each line segment so we can check them!
            std::vector<T> Fm_I_NormalX;
            std::vector<T> Fm_I_NormalY;
            std::vector<T> Fm_I_W;
            std::vector<T> Fm_I_termTwo;
            std::vector<Matrix<T,dim,dim>> m_Fi; //collect reconstructed Fi's
            std::vector<Matrix<T,dim,dim>> m_Pi; //collect computed Piola Kirchhoff Stresses
            
            for(int i = 0; i < (int)finalContourPoints.size() - 1; ++i){ //iterate contour segments
                T Fsum_I = 0; //this is what we focus on setting for each segment (three cases below)
                T Fsum_II = 0; //mode II
                Vector<T,dim> x1 = finalContourPoints[i];
                Vector<T,dim> x2 = finalContourPoints[i+1];
                Matrix<T,dim,dim> Fi1, Fi2, Finterp1, Finterp2, Pi1, Pi2;
                DFGMPM::GridState<T,dim>* g1 = finalContourGridStates[i];
                DFGMPM::GridState<T,dim>* g2 = finalContourGridStates[i+1];

                //Compute F for each endpoint --> NEITHER should be separable in this case
                Fi1 = g1->Fi1;
                Fi2 = g2->Fi1;
                m_Fi.push_back(Fi1);
                m_Fi.push_back(Fi2);

                std::vector<T> Fm1_I = computeFm(Fi1, x2 - x1, 0);
                std::vector<T> Fm2_I = computeFm(Fi2, x2 - x1, 0);
                std::vector<T> Fm1_II = computeFm(Fi1, x2 - x1, 1);
                std::vector<T> Fm2_II = computeFm(Fi2, x2 - x1, 1);
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

                //Store Piola Kirchhoff Stresses
                Pi1(0,0) = Fm1_I[5];
                Pi1(0,1) = Fm1_I[6];
                Pi1(1,0) = Fm1_I[7];
                Pi1(1,1) = Fm1_I[8];
                Pi2(0,0) = Fm2_I[5];
                Pi2(0,1) = Fm2_I[6];
                Pi2(1,0) = Fm2_I[7];
                Pi2(1,1) = Fm2_I[8];
                m_Pi.push_back(Pi1);
                m_Pi.push_back(Pi2);

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

            //Print it all out (later write to a simple file)
            file << "====================================================== J-Integral Computation using LxDxRxU = " << contour[0] << "x" << contour[1] << "x" << contour[2] << "x" << contour[3] << "Contour Centered at (" << center[0] <<  "," << center[1] << ") ======================\n";
            file << "NON-INTERSECTING CONTOUR CASE (J == 0)" << "\n";
            for(int i = 0; i < (int)finalContourPoints.size() - 1; ++i){
                file << "-----------------<Line Segment " << i << ", Fsum_I: " << Fsum_I_List[i] << ", Delta_I: " << DeltaI_List[i] << ", J_I Contribution: " << Fsum_I_List[i] * (DeltaI_List[i] / 2.0) << ">-----------------\n";
                file << "idx1: " << i << ", Point: (" << finalContourPoints[i][0] << "," << finalContourPoints[i][1] << "), Fm_I: " << Fm_I_SegmentList[i*2] << ", Normal: [" << Fm_I_NormalX[i*2] << "," << Fm_I_NormalY[i*2] << "], W: " << Fm_I_W[i*2] << ", termTwo: " << Fm_I_termTwo[i*2] << "\nFi1: " << m_Fi[i*2] << "\nPi1: " << m_Pi[i*2] << " \n";
                file << "-----\n"; 
                file << "idx2: " << i+1 << ", Point: (" << finalContourPoints[i+1][0] << "," << finalContourPoints[i+1][1] << "), Fm_I: " << Fm_I_SegmentList[(i*2) + 1] << ", Normal: [" << Fm_I_NormalX[(i*2) + 1] << "," << Fm_I_NormalY[(i*2) + 1] << "], W: " << Fm_I_W[(i*2) + 1] << ", termTwo: " << Fm_I_termTwo[(i*2) + 1] << "\nFi2: " << m_Fi[(i*2)+1] << "\nPi2: " << m_Pi[(i*2)+1] << " \n";
            }
            file << "J_I: " << J_I << "\n"; 
            file << "J_II: " << J_II << "\n";
            file << "\n";

            //Write second file with all J_I Contributions
            if(trackContributions){
                file2 << "====================================================== J-Integral Computation using LxDxRxU = " << contour[0] << "x" << contour[1] << "x" << contour[2] << "x" << contour[3] << "Contour Centered at (" << center[0] <<  "," << center[1] << ") ======================\n";
                file2 << "Line Segment Index, J_I Contribution, Fi_11, Fi_12, Fi_21, Fi_22, Pi_22, W \n";
                for(int i = 0; i < (int)finalContourPoints.size() - 1; ++i){
                    file2 << i << ", " << Fsum_I_List[i] * (DeltaI_List[i] / 2.0) << ", " << m_Fi[i*2](0,0) << ", " << m_Fi[i*2](0,1) << ", " << m_Fi[i*2](1,0) << ", " << m_Fi[i*2](1,1) << ", " <<  m_Pi[i*2](1,1) << ", " << Fm_I_W[i*2] << "\n";
                }
                int i = (int)finalContourPoints.size() - 2;
                file2 << i+1 << ", " << 0.0 << ", " << m_Fi[(i*2)+1](0,0) << ", " << m_Fi[(i*2)+1](0,1) << ", " << m_Fi[(i*2)+1](1,0) << ", " << m_Fi[(i*2)+1](1,1) << ", " <<  m_Pi[(i*2)+1](1,1) << ", " << Fm_I_W[(i*2)+1] << "\n";
            }
        }
        // INTERSECTING CASE WITH MATERIAL DISCONTINUITY (with dx gap modeling crack) ==============================================================================
        else if(!useDFG || (useDFG && containsCrackTip)){
            //STEP 1c: Construct our contour list- in this SINGLE FIELD INTERSECTING case, we must make sure we start and end at the right points
            std::vector<Vector<T,dim>> finalContourPoints;
            std::vector<DFGMPM::GridState<T,dim>*> finalContourGridStates;
            //Crack intersection
            int U = contour[3];
            if(!useDFG){ //for 4*dx wide cracks using single field MPM
                topIntersectionIdx = U - 2; //this excludes the non-material grid point that still has mass
                bottomIntersectionIdx = U + 2; //again exludes the non-material grd point that has mass, ALSO NOTE this requires the crack width to be exactly 4*dx
            }
            else if(useDFG){ //for the 2*dx wide crack that DFG can handle!
                topIntersectionIdx = U - 1; 
                bottomIntersectionIdx = U + 1;
            }
            
            //bottom intersect
            finalContourPoints.push_back(contourPoints[bottomIntersectionIdx]);
            finalContourGridStates.push_back(contourGridStates[bottomIntersectionIdx]);

            //now add points counter-clockwise until hit end of list (then we will start from beginning until bottom intersecting segment)
            for(int i = bottomIntersectionIdx + 1; i < (int)contourPoints.size(); ++i){ //end of list
                finalContourPoints.push_back(contourPoints[i]);
                finalContourGridStates.push_back(contourGridStates[i]);
            }
            for(int i = 0; i < (int)topIntersectionIdx + 1; ++i){ //begin of list
                finalContourPoints.push_back(contourPoints[i]);
                finalContourGridStates.push_back(contourGridStates[i]);
            }

            //STEP 2: Compute J-integral!
            
            //DEBUG: Setup a bunch of lists to hold intermediate data for debugging
            std::vector<T> Fsum_I_List;
            std::vector<T> DeltaI_List;
            std::vector<T> Fm_I_SegmentList; //Store Fm1 and Fm2 for each line segment so we can check them!
            std::vector<T> Fm_I_NormalX;
            std::vector<T> Fm_I_NormalY;
            std::vector<T> Fm_I_W;
            std::vector<T> Fm_I_K; //kinetic energy
            std::vector<T> Fm_I_termTwo;
            std::vector<Matrix<T,dim,dim>> m_Fi; //collect reconstructed Fi's
            std::vector<Matrix<T,dim,dim>> m_Pi; //collect computed Piola Kirchhoff Stresses
            
            for(int i = 0; i < (int)finalContourPoints.size() - 1; ++i){ //iterate contour segments
                T Fsum_I = 0; //this is what we focus on setting for each segment (three cases below)
                T Fsum_II = 0; //mode II
                Vector<T,dim> x1 = finalContourPoints[i];
                Vector<T,dim> x2 = finalContourPoints[i+1];
                Matrix<T,dim,dim> Fi1, Fi2, Finterp1, Finterp2, Pi1, Pi2;
                DFGMPM::GridState<T,dim>* g1 = finalContourGridStates[i];
                DFGMPM::GridState<T,dim>* g2 = finalContourGridStates[i+1];

                //Compute F for each endpoint --> NEITHER should be separable in this case
                Fi1 = g1->Fi1;
                Fi2 = g2->Fi1;
                m_Fi.push_back(Fi1);
                m_Fi.push_back(Fi2);

                //Compute Kinetic Energy for each endpoint
                T KE1, KE2;
                KE1 = 0.5 * g1->m1 * (g1->v1.dot(g1->v1)); //KE = 1/2 * m * (v dot v)
                KE2 = 0.5 * g2->m1 * (g2->v1.dot(g2->v1));
                Fm_I_K.push_back(KE1);
                Fm_I_K.push_back(KE2);

                std::vector<T> Fm1_I = computeFm(Fi1, x2 - x1, 0, KE1);
                std::vector<T> Fm2_I = computeFm(Fi2, x2 - x1, 0, KE2);
                std::vector<T> Fm1_II = computeFm(Fi1, x2 - x1, 1, KE1);
                std::vector<T> Fm2_II = computeFm(Fi2, x2 - x1, 1, KE2);
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

                //Store Piola Kirchhoff Stresses
                Pi1(0,0) = Fm1_I[5];
                Pi1(0,1) = Fm1_I[6];
                Pi1(1,0) = Fm1_I[7];
                Pi1(1,1) = Fm1_I[8];
                Pi2(0,0) = Fm2_I[5];
                Pi2(0,1) = Fm2_I[6];
                Pi2(1,0) = Fm2_I[7];
                Pi2(1,1) = Fm2_I[8];
                m_Pi.push_back(Pi1);
                m_Pi.push_back(Pi2);

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

            //Print it all out (later write to a simple file)
            file << "====================================================== J-Integral Computation using LxDxRxU = " << contour[0] << "x" << contour[1] << "x" << contour[2] << "x" << contour[3] << "Contour Centered at (" << center[0] <<  "," << center[1] << ") ======================\n";
            if(!useDFG){
                file << "SINGLE FIELD MPM - INTERSECTING CONTOUR CASE (J != 0)" << "\n";
            }
            else if(useDFG){
                file << "TWO-FIELD MPM - 2*DX INTERSECTING CONTOUR CASE (J != 0)" << "\n";
            }
            for(int i = 0; i < (int)finalContourPoints.size() - 1; ++i){
                file << "-----------------<Line Segment " << i << ", Fsum_I: " << Fsum_I_List[i] << ", Delta_I: " << DeltaI_List[i] << ", J_I Contribution: " << Fsum_I_List[i] * (DeltaI_List[i] / 2.0) << ">-----------------\n";
                file << "idx1: " << i << ", Point: (" << finalContourPoints[i][0] << "," << finalContourPoints[i][1] << "), Fm_I: " << Fm_I_SegmentList[i*2] << ", Normal: [" << Fm_I_NormalX[i*2] << "," << Fm_I_NormalY[i*2] << "], W: " << Fm_I_W[i*2] << ", KE: " << Fm_I_K[i*2] << ", termTwo: " << Fm_I_termTwo[i*2] << "\nFi1: " << m_Fi[i*2] << "\nPi1: " << m_Pi[i*2] << " \n";
                file << "-----\n"; 
                file << "idx2: " << i+1 << ", Point: (" << finalContourPoints[i+1][0] << "," << finalContourPoints[i+1][1] << "), Fm_I: " << Fm_I_SegmentList[(i*2) + 1] << ", Normal: [" << Fm_I_NormalX[(i*2) + 1] << "," << Fm_I_NormalY[(i*2) + 1] << "], W: " << Fm_I_W[(i*2) + 1] << ", KE: " << Fm_I_K[(i*2) + 1] << ", termTwo: " << Fm_I_termTwo[(i*2) + 1] << "\nFi2: " << m_Fi[(i*2)+1] << "\nPi2: " << m_Pi[(i*2)+1] << " \n";
            }
            file << "J_I: " << J_I << "\n"; 
            file << "J_II: " << J_II << "\n";
            file << "\n";

            //Write second file with all J_I Contributions
            if(trackContributions){
                file2 << "====================================================== J-Integral Computation using LxDxRxU = " << contour[0] << "x" << contour[1] << "x" << contour[2] << "x" << contour[3] << "Contour Centered at (" << center[0] <<  "," << center[1] << ") ======================\n";
                file2 << "Line Segment Index, J_I Contribution, Fi_11, Fi_12, Fi_21, Fi_22, Pi_22, W \n";
                for(int i = 0; i < (int)finalContourPoints.size() - 1; ++i){
                    file2 << i << ", " << Fsum_I_List[i] * (DeltaI_List[i] / 2.0) << ", " << m_Fi[i*2](0,0) << ", " << m_Fi[i*2](0,1) << ", " << m_Fi[i*2](1,0) << ", " << m_Fi[i*2](1,1) << ", " <<  m_Pi[i*2](1,1) << ", " << Fm_I_W[i*2] << "\n";
                }
                int i = (int)finalContourPoints.size() - 2;
                file2 << i+1 << ", " << 0.0 << ", " << m_Fi[(i*2)+1](0,0) << ", " << m_Fi[(i*2)+1](0,1) << ", " << m_Fi[(i*2)+1](1,0) << ", " << m_Fi[(i*2)+1](1,1) << ", " <<  m_Pi[(i*2)+1](1,1) << ", " << Fm_I_W[(i*2)+1] << "\n";
            }
        }

        return J_I;
    }

    //Compute F_i based on singular values and quaternion rotations from F = U * Sigma * V^T
    Matrix<T,dim,dim> computeF(Vector<T,dim> singularValues, Vector<T,4> Uquat, Vector<T,4> Vquat){
        Matrix<T,dim,dim> U, V, Sigma;
        Eigen::Quaternion<T> rotUreconstruct(Uquat);
        Eigen::Quaternion<T> rotVreconstruct(Vquat);
        rotUreconstruct.normalize();
        rotVreconstruct.normalize();
        Matrix<T,3,3> Ureconstruct = rotUreconstruct.toRotationMatrix();
        Matrix<T,3,3> Vreconstruct = rotVreconstruct.toRotationMatrix();
        U = Ureconstruct.topLeftCorner(2,2);
        V = Vreconstruct.topLeftCorner(2,2);
        Sigma = singularValues.asDiagonal();
        return U * Sigma * V.transpose();
    }

    //Compute Fm based on grid data at node i, the line segment between the nodes, and the mode (0 for x, 1 for y), optional to add Kinetic Energy
    std::vector<T> computeFm(Matrix<T,dim,dim> Fi, Vector<T, dim> lineSegment, int mode, T KE = 0){
        
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

        //Compute strain energy density, W (elastic potential energy density)
        //NOTE: It is MUCH easier to hardcode the elasticity model here, so if we change the constitutive model we NEED to change this here as well!!
        T W = 0;
        T termTwo = 0;
        Matrix<T, dim, dim> Pi;
        int elasticityMode = 1; //0 = LINEAR, 1 = FCR
        if(elasticityMode == 0){
            //LINEAR ELASTICITY
            Matrix<T, dim, dim> epsilon = 0.5 * (Fi + Fi.transpose()) - Matrix<T, dim, dim>::Identity();
            T tr_epsilon = epsilon.diagonal().sum();
            W = mu * epsilon.squaredNorm() + la * 0.5 * tr_epsilon * tr_epsilon; // W = psi(epsilon)

            //Compute Piola Kirchhoff stress
            Matrix<T, dim, dim> R = Matrix<T, dim, dim>::Identity();
            Pi.noalias() = 2 * mu * R * epsilon + la * tr_epsilon * R;
            
            //Compute term two
            termTwo = (Pi * normal).dot(Fi.col(mode));
        }
        else if(elasticityMode == 1){
            //FIXED COROTATED ELASTICITY
            Matrix<T, dim, dim> U, V;
            Vector<T, dim> sigma;
            Math::svd(Fi, U, sigma, V);
            W = mu * (sigma - Vector<T, dim>::Ones()).squaredNorm() + T(0.5) * la * std::pow(sigma.prod() - T(1), 2);

            //Compute Piola Kirchhoff stress
            T J = Fi.determinant();
            Matrix<T, dim, dim> JFinvT;
            Math::cofactor(Fi, JFinvT);
            Matrix<T, dim, dim> R, S;
            Math::polar_decomposition(Fi, R, S);
            Pi = T(2) * mu * (Fi - R) + la * (J - 1) * JFinvT;

            //Compute term two
            termTwo = (Pi * normal).dot(Fi.col(mode));
        }
        
        //Add first term -> mode = 0 or 1
        Fm += (W + KE) * normal[mode];

        //Add second term (field 1)
        Fm -= termTwo;

        FmResults.push_back(Fm);
        FmResults.push_back(normal[0]);
        FmResults.push_back(normal[1]);
        FmResults.push_back(W);
        FmResults.push_back(termTwo);
        FmResults.push_back(Pi(0,0));
        FmResults.push_back(Pi(0,1));
        FmResults.push_back(Pi(1,0));
        FmResults.push_back(Pi(1,1)); //components of Pi

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

/* Compute nodal deformation gradients, Fi, using nodal displacements and B-spline interpolation */
template <class T, int dim>
class ConstructNodalDeformationGradientsOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;

    DFGMPM::DFGMPMGrid<T, dim>& grid;
    T dx;
    T rp;
    int neighborRadius;

    void operator()()
    {
        BOW_TIMER_FLAG("computeFiUsingNodalDisplacements");

        //Now iterate over all active grid nodes so we can compute their deformation gradients!
        grid.iterateGridSerial([&](const Vector<int, dim>& node, DFGMPM::GridState<T, dim>& g) {
            Vector<T, dim> pos_i = node.template cast<T>() * dx; //compute current nodal position to center our B-Spline on
            T Ux = 0.0; //x-displacement field
            T Uy = 0.0; //y-displacement field
            T S = 0.0;
            Vector<T, dim> nablaUx = Vector<T, dim>::Zero();
            Vector<T, dim> nablaUy = Vector<T, dim>::Zero();
            Vector<T, dim> nablaS = Vector<T, dim>::Zero();
            
            //Now construct displacement fields for x and y dimensions (using B-spline interpolation we used for damage gradients)
            BSplineWeights<T, dim> spline(pos_i, dx);
            grid.iterateNeighbors_ClosestNode(spline, neighborRadius, [&](const Vector<int, dim>& node2, DFGMPM::GridState<T, dim>& g2) {
                //Now iterate the neighboring grid nodes to our current node
                Vector<T, dim> pos_j = node2.template cast<T>() * dx;
                T dist = (pos_i - pos_j).norm();
                if(dist > 0){ //exclude the dist = 0 case
                    T rBar = dist / rp;
                    Vector<T, dim> rBarGrad = (pos_i - pos_j) * (1.0 / (rp * dist));

                    T omega = 1 - (3 * rBar * rBar) + (2 * rBar * rBar * rBar);
                    T omegaPrime = 6 * ((rBar * rBar) - rBar);
                    if (rBar <= 0.0 || rBar > 1.0) {
                        omega = 0.0;
                        omegaPrime = 0.0;
                    }

                    //Now we need to figure out what displacements to use in constructing the displacement field
                    T u_x = 0.0;
                    T u_y = 0.0;
                    if(g.gridDG.dot(g2.gridDG) >= 0){
                        //DGs in same direction, always use field 1 (even if separable, field 1 is correct since DGs same direction)
                        u_x = g2.u1[0];
                        u_y = g2.u1[1];
                    }
                    else{
                        //DGs opposite directions, use field 2 if separable, otherwise use 0!
                        if(g2.separable == 1){
                            u_x = g2.u2[0];
                            u_y = g2.u2[1];
                        }
                    }

                    Ux += u_x * omega;
                    Uy += u_y * omega;
                    S += omega;

                    nablaUx += (u_x * omegaPrime * rBarGrad);
                    nablaUy += (u_y * omegaPrime * rBarGrad);
                    nablaS += (omegaPrime * rBarGrad);

                    //std::cout << "rBar: " << rBar << " rBarGrad: " << rBarGrad << " omegaPrime: " << omegaPrime << std::endl;
                }
            });

            //std::cout << "Ux: " << Ux << " Uy: " << Uy << " nablaUx: " << nablaUx << " nablaUy: " << nablaUy << std::endl;
            //std::cout << "S: " << S << " nablaS: " << nablaS << std::endl;

            //Now construct the displacement gradient
            Vector<T, dim> nablaUxBar;
            Vector<T, dim> nablaUyBar;
            if(S == 0){
                nablaUxBar = Vector<T, dim>::Zero();
                nablaUyBar = Vector<T, dim>::Zero();
            }
            else{
                nablaUxBar = (nablaUx * S - Ux * nablaS) / (S * S);
                nablaUyBar = (nablaUy * S - Uy * nablaS) / (S * S);
            }
            Matrix<T, dim, dim> displacementGradient;
            displacementGradient.col(0) = nablaUxBar.transpose();
            displacementGradient.col(1) = nablaUyBar.transpose();

            //Finally, compute def grad!
            g.Fi1 = displacementGradient + Matrix<T,dim,dim>::Identity();
        });

        return;
    }
};

/* Compute the Dynamic J-Integral's area integral term by integrating over all particles enclosed in a rectangular path of grid nodes centered on the closest node to the crack tip */
template <class T, int dim>
class ComputeJIntegralAreaTermOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    Field<Vector<T, dim>>& m_V;
    Field<Vector<T, dim>>& m_Vprevious;
    std::vector<T>& m_mass;
    Field<T> m_initialVolume;

    Field<Matrix<T, dim, dim>>& m_F;
    Field<Matrix<T, dim, dim>>& m_Fprevious;

    DFGMPM::DFGMPMGrid<T, dim>& grid;
    T dx;
    T dt;

    T operator()(Vector<T,dim> center, Vector<int,4> contour, std::ofstream& file)
    {
        BOW_TIMER_FLAG("computeJIntegralAreaTerm");

        T J_I = 0; //set J integral mode I to 0 for now

        //Calculate contour boundaries, contour defined by (L,D,R,U) each indicating the number of nodes Left, Down, Right, Up from the center point (x,y)
        T left, down, right, up;
        left = center[0] - (contour[0] * dx);
        down = center[1] - (contour[1] * dx);
        right = center[0] + (contour[2] * dx);
        up = center[1] + (contour[3] * dx);
        
        //Setup lists to save intermediate data to write out
        std::vector<int> particleIndeces;
        std::vector<T> contributions;
        //std::vector<Matrix<T,dim,dim>> m_Fi;


        //Now iterate over all particles enclosed in the contour and sum up their contributions to the area integral
        grid.serial_for([&](int i) {
            if(!grid.crackInitialized || i < grid.crackParticlesStartIdx){ 
                const Vector<T, dim> pos = m_X[i];
                T integrand = 0;
                if(pos[0] > left && pos[0] < right && pos[1] > down && pos[1] < up){ //only process particles inside contour
                    integrand = 0;
                    
                    Vector<T,dim> a_p = (m_V[i] - m_Vprevious[i]) / dt; // particle acceleration
                    
                    Matrix<T, dim, dim> F = m_F[i];
                    Matrix<T, dim, dim> Fdot = (F - m_Fprevious[i]) / dt;
                    Matrix<T, dim, dim> Finv = F.inverse();
                    Matrix<T, dim, dim> L_p = Fdot * Finv; //velocity gradient, nabla v = L = Fdot * Finv

                    integrand = (m_mass[i] / m_initialVolume[i]) * (a_p.dot(F.col(0)) - m_V[i].dot(L_p.col(0))); // area integral integrand = (mp/vp0) * (ap dot F.col(0) - vp dot L.col(0))
                    T contribution = integrand * m_initialVolume[i];

                    J_I += contribution; //sum integrand weighted by initial volume

                    particleIndeces.push_back(i);
                    contributions.push_back(contribution);
                }
            }
        });

        //Write soem intermediate data to a file
        file << "====================================================== J-Integral Dynamic Area Integral Computation using LxDxRxU = " << contour[0] << "x" << contour[1] << "x" << contour[2] << "x" << contour[3] << "Contour Centered at (" << center[0] <<  "," << center[1] << ") ======================\n";
        // for(int i = 0; i < (int)particleIndeces.size(); ++i){
        //     T idx = particleIndeces[i];
        //     file << "Particle Idx: " << idx << ", Position: (" << m_X[idx][0] << "," << m_X[idx][1] << "), Contribution: " << contributions[idx] << " \n";
        // }
        file << "Number of Particles Integrated Over: " << (int)particleIndeces.size() << "\n";
        file << "Total Area of Region: " << ((T)contour[0]+(T)contour[2])*((T)contour[1]+(T)contour[3])*dx*dx << "\n";
        file << "J_I: " << J_I << "\n"; 
        file << "\n";

        return J_I;
    }
};

/* Compute system energies by iterating over particles (PE_solid, PE_fluid, KE_solid, KE_fluid, GPE_solid, GPE_fluid, Work by BCs, Time) */
/*                                                     (   0,         1,        2,        3,        4,         5,          6,        7 ) */
template <class T, int dim>
class ComputeSystemEnergyOp : public AbstractOp {
public:
    using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    Field<Vector<T, dim>>& m_V;
    std::vector<T>& m_mass;
    Field<T> m_energy;
    Field<int> m_marker;
    T gravity;

    DFGMPM::DFGMPMGrid<T, dim>& grid;
    T dt;
    T totalWork;

    void operator()(Vector<T,8>& energies)
    {
        BOW_TIMER_FLAG("computeSystemEnergy");

        //Now iterate particles and sum up energy contributions
        grid.serial_for([&](int i) {
            if(!grid.crackInitialized || i < grid.crackParticlesStartIdx){ 
                if(m_marker[i] == 0){
                    energies[0] += m_energy[i]; //solid PE
                    energies[2] += 0.5 * m_mass[i] * (m_V[i].dot(m_V[i])); //solid KE
                    energies[4] += gravity * m_mass[i] * m_X[i][1]; //solid GPE
                }
                else if(m_marker[i] == 4){
                    energies[1] += m_energy[i]; //fluid PE
                    energies[3] += 0.5 * m_mass[i] * (m_V[i].dot(m_V[i])); //fluid KE
                    energies[5] += gravity * m_mass[i] * m_X[i][1]; //fluid GPE
                }
            }
        });

        //Now take a snapshot of the total work at this time!
        energies[6] = totalWork; 

        return;
    }
};


/* Transfer particle stress, nominal stress, and def grad to grid so we can compute J Integral*/
// template <class T, int dim>
// class CollectJIntegralGridDataOp : public AbstractOp {
// public:
//     using SparseMask = typename DFGMPM::DFGMPMGrid<T, dim>::SparseMask;
//     Field<Vector<T, dim>>& m_X;
//     Field<Matrix<T, dim, dim>>& m_stress; //holds Vp^0 * PF^T 
//     Field<Matrix<T,dim,dim>>& m_F; //holds particle def grad F

//     Bow::Field<std::vector<int>>& particleAF;

//     DFGMPM::DFGMPMGrid<T, dim>& grid;
//     T dx;
//     T dt;

//     bool useDFG;

//     void operator()()
//     {
//         BOW_TIMER_FLAG("collectJIntegralGridData");
        
//         //Now we transfer to the grid!
//         grid.colored_for([&](int i) {
//             if(!grid.crackInitialized || i < grid.crackParticlesStartIdx){ //skip crack particles if we have them
//                 const Vector<T, dim> pos = m_X[i];
//                 //const Matrix<T, dim, dim> stress = m_stress[i];
//                 const Matrix<T, dim, dim> F = m_F[i];
                
//                 //We will transfer the deformation gradient through transferring its singular values and rotations (as quaternions) -> three separate intrinsic transfers here: sigma, Uquat, Vquat
//                 Matrix<T, dim, dim> U, V;
//                 Vector<T, dim> sigma;
//                 Vector<T, 4> Uquat, Vquat; //quaternion coefficients for U and V
//                 Math::svd(F, U, sigma, V);
                
//                 //Now convert U and V to quaternions
//                 Matrix<T, 3,3> Upad = Matrix<T,3,3>::Identity();
//                 Matrix<T, 3,3> Vpad = Matrix<T,3,3>::Identity();
//                 Upad.topLeftCorner(2,2) = U;
//                 Vpad.topLeftCorner(2,2) = V; //pad these to be 3x3 for quaternion
//                 Eigen::Quaternion<T> rotU(Upad);
//                 Eigen::Quaternion<T> rotV(Vpad);
//                 rotU.normalize();
//                 rotV.normalize(); //normalize our quaternions!
//                 Uquat = rotU.coeffs();
//                 Vquat = rotV.coeffs();

//                 //Compute spline
//                 BSplineWeights<T, dim> spline(pos, dx);
                
//                 grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, const Vector<T, dim>& dw, DFGMPM::GridState<T, dim>& g) {
                    
//                     ///NOTES
//                     //Storing grid force in fi1 and fi2 (if separable) -> outdated
//                     //We will intrinsically transfer here the singular values and quaternions for U and V rotations from F = UsigmaV^T
//                     //Storing accumulated weights in gridSeparability[0] (field 1 weight sum) and gridSeparability[1] (field 2 weight sum)

//                     //Notice we treat single-field and two-field nodes differently
//                     if (g.separable == 0 || !useDFG) {
//                         //Single-field treatment if separable = 0 OR if we are using single field MPM
                        
//                         //g.fi1 += stress * dw; //transfer stress to grid, fi
//                         g.sigma1 += sigma * w;
//                         g.Uquat1 += Uquat * w;
//                         g.Vquat1 += Vquat * w;
//                         g.gridSeparability[0] += w; //sum up the total weight
//                     }
//                     else if (g.separable != 0 && useDFG) {
//                         //Treat node as having two fields
//                         int fieldIdx = particleAF[i][oidx]; //grab the field that this particle belongs in for this grid node (oidx)
//                         if (fieldIdx == 0) {
//                             //g.fi1 += stress * dw; //transfer stress to grid, fi
//                             g.sigma1 += sigma * w;
//                             g.Uquat1 += Uquat * w;
//                             g.Vquat1 += Vquat * w;
//                             g.gridSeparability[0] += w; //sum up the total weight
//                         }
//                         else if (fieldIdx == 1) {
//                             //g.fi2 += stress * dw; //transfer stress to grid, fi (second field)
//                             g.sigma2 += sigma * w;
//                             g.Uquat2 += Uquat * w;
//                             g.Vquat2 += Vquat * w;
//                             g.gridSeparability[1] += w; //sum up the total weight (second field)
//                         }
//                     }
//                 });
//             }
//         });

//         /* Iterate grid to divide out the total weight sums for P and F since we transfer these intrinsically */
//         grid.iterateGrid([&](const Vector<int, dim>& node, DFGMPM::GridState<T, dim>& g) {
//             if(g.gridSeparability[0] != 0){
//                 g.sigma1 /= g.gridSeparability[0];
//                 g.Uquat1 /= g.gridSeparability[0];
//                 g.Vquat1 /= g.gridSeparability[0]; //divide by field 1 weight sum
//             }
//             if (g.separable != 0) {
//                 if(g.gridSeparability[1] != 0){
//                     g.sigma2 /= g.gridSeparability[1];
//                     g.Uquat2 /= g.gridSeparability[1];
//                     g.Vquat2 /= g.gridSeparability[1]; //divide by field 2 weight sum
//                 }
//             }
//         });
//     }
// };

}
} // namespace Bow::DFGMPM