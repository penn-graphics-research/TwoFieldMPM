#include "TwoFieldHodgepodgeEnergy.h"

namespace Bow::DFGMPM {

template <class T, int dim, class StorageIndex, int interpolation_degree>
void TwoFieldHodgepodgeEnergy<T, dim, StorageIndex, interpolation_degree>::precompute(const Field<Vector<T, dim>>& x)
{
    Field<TM> m_gradXp(m_X.size(), TM::Zero());
    tbb::parallel_for(size_t(0), m_X.size(), [&](size_t i) {
        const Vector<T, dim>& Xp = m_X[i];
        BSplineWeights<T, dim, interpolation_degree> spline(Xp, dx);
        Matrix<T, dim, dim> gradXp = Matrix<T, dim, dim>::Identity();
        grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, Vector<T, dim> dw, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            Vector<T, dim> xn = node.template cast<T>() * dx;
            
            //G2P style to compute gradXp for each particle
            if(g.separable != 1 || !useDFG){ //single field
                gradXp.noalias() += (x[g.idx] - xn) * dw.transpose();
            }
            else if(g.separable == 1){
                int fieldIdx = particleAF[i][oidx];
                if(fieldIdx == 0){ //two field: if field 1
                    gradXp.noalias() += (x[g.idx] - xn) * dw.transpose();
                }
                else if(fieldIdx == 1){ //two field: if field 2
                    gradXp.noalias() += (x[grid.num_nodes + g.sep_idx] - xn) * dw.transpose();
                }
            }
        });
        m_gradXp[i] = gradXp;
    });
    for (auto& model : elasticity_models)
        model->trial_strain(m_gradXp);
}

template <class T, int dim, class StorageIndex, int interpolation_degree>
T TwoFieldHodgepodgeEnergy<T, dim, StorageIndex, interpolation_degree>::energy(const Field<Vector<T, dim>>& x)
{
    T total_energy = 0;
    int ndof = grid.num_nodes;
    int sdof = useDFG ? grid.separable_nodes : 0;
    //KINETIC AND GRAVITATIONAL: field 1
    grid.iterateGridSerial([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
        T m = g.m1;
        TV x_n = node.template cast<T>() * dx;
        TV v_n = g.v1;
        TV a_n = g.a1;
        TV x_tilde = x_n + v_n * dt + tsParam[tsMethod][0] * (1 - 2 * tsParam[tsMethod][1]) * a_n * dt * dt;
        total_energy += (T)0.5 * m * (x[g.idx] - x_tilde).squaredNorm();
        total_energy -= 2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt * m * gravity.dot(x[g.idx] - x_n);
    });
    //KINETIC AND GRAVITATIONAL: field 2 if have it
    if(sdof > 0){
        grid.iterateSeparableNodesSerial([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            T m = g.m2;
            TV x_n = node.template cast<T>() * dx;
            TV v_n = g.v2;
            TV a_n = g.a2;
            TV x_tilde = x_n + v_n * dt + tsParam[tsMethod][0] * (1 - 2 * tsParam[tsMethod][1]) * a_n * dt * dt;
            total_energy += (T)0.5 * m * (x[ndof + g.sep_idx] - x_tilde).squaredNorm();
            total_energy -= 2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt * m * gravity.dot(x[ndof + g.sep_idx] - x_n);
        });
    }

    //BARRIER ENERGY -- TAG=BARRIER
    if(useDFG && useImplicitContact && sdof > 0){
        grid.iterateSeparableNodesSerial([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            T ci = g.gridCi;    //NOTE we should have computed ci by now since we do it after each UpdateDV
            T B = computeB(ci, chat);
            T ViYi1 = g.gridViYi1;
            T ViYi2 = g.gridViYi2;
            total_energy += (ViYi1 + ViYi2) * B * factor * energyFactor;
        });
    }

    //ELASTICITY
    Field<T> t_energy(m_X.size(), (T)0);
    for (auto& model : elasticity_models) {
        model->trial_energy(t_energy);
    }
    for (size_t i = 0; i < m_X.size(); ++i) {
        total_energy += 2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt * t_energy[i];
    }
    return this->energy_scale * total_energy;
}

template <class T, int dim, class StorageIndex, int interpolation_degree>
void TwoFieldHodgepodgeEnergy<T, dim, StorageIndex, interpolation_degree>::gradient(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& grad)
{
    int ndof = grid.num_nodes;
    int sdof = useDFG ? grid.separable_nodes : 0;
    grad.assign(ndof + sdof, Vector<T, dim>::Zero());
    
    //KINETIC AND GRAVITATIONAL: field 1
    grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
        if (g.idx < 0) return;
        auto idx = g.idx;
        T m = g.m1;
        TV x_n = node.template cast<T>() * dx;
        TV v_n = g.v1;
        TV a_n = g.a1;
        TV x_tilde = x_n + v_n * dt + tsParam[tsMethod][0] * (1 - 2 * tsParam[tsMethod][1]) * a_n * dt * dt;
        grad[idx] += this->energy_scale * m * (x[idx] - x_tilde);
        grad[idx] -= this->energy_scale * 2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt * m * gravity;
    });
    //KINETIC AND GRAVITATIONAL: field 2
    if(sdof > 0){
        grid.iterateSeparableNodes([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            if (ndof + g.sep_idx < 0) return;
            auto idx2 = ndof + g.sep_idx;
            T m = g.m2;
            TV x_n = node.template cast<T>() * dx;
            TV v_n = g.v2;
            TV a_n = g.a2;
            TV x_tilde = x_n + v_n * dt + tsParam[tsMethod][0] * (1 - 2 * tsParam[tsMethod][1]) * a_n * dt * dt;
            grad[idx2] += this->energy_scale * m * (x[idx2] - x_tilde);
            grad[idx2] -= this->energy_scale * 2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt * m * gravity;
        });
    }

    //BARRIER ENERGY GRADIENT -- TAG=BARRIER
    if(useDFG && useImplicitContact && sdof > 0){
        grid.iterateSeparableNodes([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            auto idx = g.idx;
            auto idx2 = ndof + g.sep_idx;
            
            T ViYi1 = g.gridViYi1;
            T ViYi2 = g.gridViYi2;

            T bPrime = computeBPrime(g.gridCi, chat);
            bPrime *= (ViYi1 + ViYi2);

            TV n_cm1 = g.n1; //this assumes we computed normals even for implicit
            TV nablaB1 = bPrime * -n_cm1 * (factor * rhsFactor);
            TV nablaB2 = bPrime * n_cm1 * (factor * rhsFactor);

            grad[idx] += nablaB1;
            grad[idx2] += nablaB2;
        });
    }

    //ELASTICITY
    Field<TM> t_gradient(m_X.size(), TM::Zero());
    for (auto& model : elasticity_models) {
        model->trial_gradient(t_gradient);
    }
    grid.colored_for([&](int i) {
        Vector<T, dim>& Xp = m_X[i];
        BSplineWeights<T, dim, interpolation_degree> spline(Xp, dx);
        Matrix<T, dim, dim> stress = t_gradient[i];
        grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, Vector<T, dim> dw, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            //P2G style
            if(g.separable != 1 || !useDFG){
                //single field
                auto idx = g.idx;
                grad[idx] += this->energy_scale * 2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt * stress * dw;
            }
            else if(g.separable == 1 && useDFG){
                //two field
                int fieldIdx = particleAF[i][oidx];
                if(fieldIdx == 0){
                    //if field 1
                    auto idx = g.idx;
                    grad[idx] += this->energy_scale * 2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt * stress * dw;
                }
                else if(fieldIdx == 1){
                    //if field 2
                    auto idx2 = ndof + g.sep_idx;
                    grad[idx2] += this->energy_scale * 2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt * stress * dw;
                }
            }
        });
    });
}

template <class T, int dim, class StorageIndex, int interpolation_degree>
void TwoFieldHodgepodgeEnergy<T, dim, StorageIndex, interpolation_degree>::multiply(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& Ax, bool project_pd)
{
    int ndof = grid.num_nodes;
    int sdof = useDFG ? grid.separable_nodes : 0;

    //Set Ax
    Ax.assign(ndof + sdof, Vector<T, dim>::Zero());
    grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
        auto idx = g.idx;
        T m = g.m1;
        Ax[idx] += m * x[idx];
    });
    if(sdof > 0){
        grid.iterateSeparableNodes([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            auto idx2 = ndof + g.sep_idx;
            T m = g.m2;
            Ax[idx2] += m * x[idx2];
        });
    }

    Field<TM> d_F(m_X.size(), TM::Zero());
    Field<TM> t_differential(m_X.size(), TM::Zero());
    grid.parallel_for([&](int i) {
        const Vector<T, dim>& Xp = m_X[i];
        BSplineWeights<T, dim, interpolation_degree> spline(Xp, dx);
        TM gradVp = TM::Zero();
        grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, Vector<T, dim> dw, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            //G2P style
            if(g.separable != 1 || !useDFG){
                //single field
                gradVp.noalias() += x[g.idx] * dw.transpose();
            }
            else if(g.separable == 1 && useDFG){
                //two field
                int fieldIdx = particleAF[i][oidx];
                if(fieldIdx == 0){
                    //if field 1
                    gradVp.noalias() += x[g.idx] * dw.transpose();
                }
                else if(fieldIdx == 1){
                    //if field 2
                    gradVp.noalias() += x[ndof + g.sep_idx] * dw.transpose();
                }
            }
        });
        d_F[i] = gradVp;
    });
    for (auto& model : elasticity_models) {
        model->trial_differential(d_F, t_differential, project_pd);
    }
    grid.colored_for([&](int i) {
        const Vector<T, dim>& Xp = m_X[i];
        BSplineWeights<T, dim, interpolation_degree> spline(Xp, dx);
        grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, Vector<T, dim> dw, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            //P2G style
            if(g.separable != 1 || !useDFG){
                //single field
                auto idx = g.idx;
                Ax[idx] += this->energy_scale * 2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt * t_differential[i] * dw;
            }
            else if(g.separable == 1 && useDFG){
                //two field
                int fieldIdx = particleAF[i][oidx];
                if(fieldIdx == 0){
                    //if field 1
                    auto idx = g.idx;
                    Ax[idx] += this->energy_scale * 2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt * t_differential[i] * dw;
                }
                else if(fieldIdx == 1){
                    //if field 2
                    auto idx2 = ndof + g.sep_idx;
                    Ax[idx2] += this->energy_scale * 2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt * t_differential[i] * dw;
                }
            }
        });
    });
}

template <class T, int dim, class StorageIndex, int interpolation_degree>
void TwoFieldHodgepodgeEnergy<T, dim, StorageIndex, interpolation_degree>::precondition(Field<Vector<T, dim>>& diagonal)
{
    int ndof = grid.num_nodes;
    int sdof = useDFG ? grid.separable_nodes : 0;
    diagonal.assign(ndof + sdof, Vector<T, dim>::Zero());
    grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
        diagonal[g.idx] += g.m1 * TV::Ones();
    });
    if(sdof > 0){
        grid.iterateSeparableNodes([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            diagonal[ndof + g.sep_idx] += g.m2 * TV::Ones();
        });
    }
}

// TODO: optimize neasted loop
// https://github.com/penn-graphics-research/ziran/blob/LBFGSAMG/Projects/multigrid/ImplicitSolver.h
// https://github.com/penn-graphics-research/ziran/blob/LBFGSAMG/Projects/multigrid/ImplicitSolver_prev.h
template <class T, int dim, class StorageIndex, int interpolation_degree>
void TwoFieldHodgepodgeEnergy<T, dim, StorageIndex, interpolation_degree>::hessian(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess, bool project_pd)
{
    int ndof = grid.num_nodes;
    int sdof = useDFG ? grid.separable_nodes : 0;
    int tdof = ndof + sdof;

    int nNbr = kernel_size;
    // if(useDFG){
    //     nNbr *= 2; //NOTE: we'll use this extra space as a scratch space for second field contributions
    //     nNbr += sdof; //add space for our barrier contributions to separable nodes
    // }

    std::vector<int> entryRow(tdof * nNbr, 0);
    std::vector<int> entryCol(tdof * nNbr, 0);
    std::vector<TM> entryVal(tdof * nNbr, TM::Zero());    
    
    //INTERTIA TERM: field 1
    grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
        T m = g.m1;
        entryRow[g.idx * nNbr + kernelOffset(Vector<int, dim>::Zero())] = g.idx;
        entryCol[g.idx * nNbr + kernelOffset(Vector<int, dim>::Zero())] = g.idx;
        entryVal[g.idx * nNbr + kernelOffset(Vector<int, dim>::Zero())] += m * TM::Identity();
    });
    //INERTIA TERM: field 2
    if(sdof > 0){
        grid.iterateSeparableNodes([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            T m = g.m2;
            auto idx2 = ndof + g.sep_idx;
            entryRow[idx2 * nNbr + kernelOffset(Vector<int, dim>::Zero())] = idx2;
            entryCol[idx2 * nNbr + kernelOffset(Vector<int, dim>::Zero())] = idx2;
            entryVal[idx2 * nNbr + kernelOffset(Vector<int, dim>::Zero())] += m * TM::Identity();
        });
    }

    Field<Matrix<T, dim * dim, dim * dim>> t_hessian(m_X.size(), Matrix<T, dim * dim, dim * dim>::Zero());
    for (auto& model : elasticity_models) {
        model->trial_hessian(t_hessian, project_pd);
    }
    grid.colored_for([&](int i) {
        Vector<T, dim>& Xp = m_X[i];
        BSplineWeights<T, dim, interpolation_degree> spline(Xp, dx);
        Matrix<T, dim * dim, dim* dim> deformed_dPdF = t_hessian[i];
        
        //Cache the proper indeces to map this particle to each of its 3^dim stencil nodes
        p_cached_idx[i].clear(); //clear vector for curr particle, then fill it up!
        grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, Vector<T, dim> dw, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            if(g.separable != 1 || !useDFG){
                //single field
                p_cached_idx[i].push_back(g.idx); //if single field, just use field 1 idx
            }
            else if(g.separable == 1 && useDFG){
                //two field
                int fieldIdx = particleAF[i][oidx];
                if(fieldIdx == 0){
                    //if field 1
                    p_cached_idx[i].push_back(g.idx); //use field 1 DOF idx
                }
                else if(fieldIdx == 1){
                    //if field 2
                    p_cached_idx[i].push_back(ndof + g.sep_idx); //use field 2 DOF idx
                }
            }
        });

        //TODO: FIX THIS BECAUSE IT DOESNT WORK LMAO
        grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, Vector<T, dim> dw, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            //int dofi = p_cached_idx[i][oidx];
            grid.iterateKernel(spline, [&](const Vector<int, dim>& _node, int _oidx, T _w, Vector<T, dim> _dw, GridState<T, dim>& _g) {
                if (_g.idx < 0) return;
                //int dofj = p_cached_idx[i][_oidx];
                TM dFdX = TM::Zero();
                for (int u = 0; u < dim; ++u)
                    for (int p = 0; p < dim; ++p)
                        for (int x = 0; x < dim; ++x)
                            for (int y = 0; y < dim; ++y)
                                dFdX(u, p) += deformed_dPdF(u + x * dim, p + y * dim) * dw(x) * _dw(y);
                // entryRow[dofi * nNbr + kernelOffset(node - _node)] = dofi;
                // entryCol[dofi * nNbr + kernelOffset(node - _node)] = dofj;
                // entryVal[dofi * nNbr + kernelOffset(node - _node)] += 2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt * dFdX;
                entryRow[g.idx * nNbr + kernelOffset(node - _node)] = g.idx;
                entryCol[g.idx * nNbr + kernelOffset(node - _node)] = _g.idx;
                entryVal[g.idx * nNbr + kernelOffset(node - _node)] += 2 * tsParam[tsMethod][0] * tsParam[tsMethod][1] * dt * dt * dFdX;
            });
        });
    });

    //TODO: BARRIER HESSIAN STUFF -- TAG=BARRIER


    using IJK = Eigen::Triplet<T>;
    std::vector<IJK> coeffs;
    for (int i = 0; i < tdof * nNbr; ++i)
        for (int u = 0; u < dim; ++u)
            for (int v = 0; v < dim; ++v) {
                coeffs.push_back(IJK(entryRow[i] * dim + u, entryCol[i] * dim + v, this->energy_scale * entryVal[i](u, v)));
            }
    hess.derived().resize(tdof * dim, tdof * dim);
    hess.derived().setZero();
    hess.derived().setFromTriplets(coeffs.begin(), coeffs.end());
}

template <class T, int dim, class StorageIndex, int interpolation_degree>
T TwoFieldHodgepodgeEnergy<T, dim, StorageIndex, interpolation_degree>::stepsize_upperbound(const Field<Vector<T, dim>>& x, const Field<Vector<T, dim>>& direction)
{
    T alpha = 1.0;
    Field<TM> m_gradDXp(m_X.size(), TM::Zero());
    tbb::parallel_for(size_t(0), m_X.size(), [&](size_t i) {
        const Vector<T, dim>& Xp = m_X[i];
        BSplineWeights<T, dim, interpolation_degree> spline(Xp, dx);
        Matrix<T, dim, dim> gradDXp = Matrix<T, dim, dim>::Zero();
        grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, Vector<T, dim> dw, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            //G2P style to compute gradDXp for each particle
            if(g.separable != 1 || !useDFG){ //single field
                gradDXp.noalias() += direction[g.idx] * dw.transpose();
            }
            else if(g.separable == 1){
                int fieldIdx = particleAF[i][oidx];
                if(fieldIdx == 0){ //two field: if field 1
                    gradDXp.noalias() += direction[g.idx] * dw.transpose();
                }
                else if(fieldIdx == 1){ //two field: if field 2
                    gradDXp.noalias() += direction[grid.num_nodes + g.sep_idx] * dw.transpose();
                }
            }
        });
        m_gradDXp[i] = gradDXp;
    });
    for (auto& model : elasticity_models) {
        alpha = std::min(alpha, model->stepsize_upperbound(m_gradDXp));
    }
    return alpha;
}

template <class T, int dim, class StorageIndex, int interpolation_degree>
void TwoFieldHodgepodgeEnergy<T, dim, StorageIndex, interpolation_degree>::internal_force(const Field<Vector<T, dim>>& xn, const Field<Vector<T, dim>>& vn, Field<Vector<T, dim>>& force)
{
    int ndof = grid.num_nodes;
    int sdof = useDFG ? grid.separable_nodes : 0;
    force.assign(ndof + sdof, Vector<T, dim>::Zero());
    grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
        if (g.idx < 0) return;
        auto idx = g.idx;
        T m = g.m1;
        force[idx] -= m * gravity;
    });
    if(sdof > 0){
        grid.iterateSeparableNodes([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            auto idx2 = ndof + g.sep_idx;
            T m = g.m2;
            force[idx2] -= m * gravity;
        }); 
    }
    precompute(xn);
    Field<TM> t_gradient(m_X.size(), TM::Zero());
    for (auto& model : elasticity_models) {
        model->trial_gradient(t_gradient);
    }
    grid.colored_for([&](int i) {
        Vector<T, dim>& Xp = m_X[i];
        BSplineWeights<T, dim, interpolation_degree> spline(Xp, dx);
        Matrix<T, dim, dim> stress = t_gradient[i];
        grid.iterateKernel(spline, [&](const Vector<int, dim>& node, int oidx, T w, Vector<T, dim> dw, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            //P2G style
            if(g.separable != 1 || !useDFG){
                //single field
                auto idx = g.idx;
                force[idx] += stress * dw;
            }
            else if(g.separable == 1 && useDFG){
                //two field
                int fieldIdx = particleAF[i][oidx];
                if(fieldIdx == 0){
                    //if field 1
                    auto idx = g.idx;
                    force[idx] += stress * dw;
                }
                else if(fieldIdx == 1){
                    //if field 2
                    auto idx2 = ndof + g.sep_idx;
                    force[idx2] += stress * dw;
                }
            }
        });
    });
}

} // namespace Bow::MPM