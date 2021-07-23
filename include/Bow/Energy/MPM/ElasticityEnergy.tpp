#include "ElasticityEnergy.h"
#include <Bow/Utils/Timer.h>

namespace Bow::MPM {
template <class T, int dim, class StorageIndex, int interpolation_degree>
ElasticityEnergy<T, dim, StorageIndex, interpolation_degree>::ElasticityEnergy(MPMGrid<T, dim>& grid, const Field<Vector<T, dim>>& m_X, std::vector<std::shared_ptr<ElasticityOp<T, dim>>>& elasticity_models, const T dx, T energy_scale)
    : grid(grid)
    , m_X(m_X)
    , elasticity_models(elasticity_models)
    , dx(dx)
{
    this->energy_scale = energy_scale;
    this->name = "MPM-Elasticity";
}

template <class T, int dim, class StorageIndex, int interpolation_degree>
void ElasticityEnergy<T, dim, StorageIndex, interpolation_degree>::precompute(const Field<Vector<T, dim>>& x)
{
    Field<TM> m_gradXp(m_X.size(), TM::Zero());
    tbb::parallel_for(size_t(0), m_X.size(), [&](size_t i) {
        const Vector<T, dim>& Xp = m_X[i];
        BSplineWeights<T, dim, interpolation_degree> spline(Xp, dx);
        Matrix<T, dim, dim> gradXp = Matrix<T, dim, dim>::Identity();
        grid.iterateKernel(spline, [&](const Vector<int, dim>& node, T w, Vector<T, dim> dw, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            Vector<T, dim> xn = node.template cast<T>() * dx;
            gradXp.noalias() += (x[g.idx] - xn) * dw.transpose();
        });
        m_gradXp[i] = gradXp;
    });
    for (auto& model : elasticity_models)
        model->trial_strain(m_gradXp);
}

template <class T, int dim, class StorageIndex, int interpolation_degree>
T ElasticityEnergy<T, dim, StorageIndex, interpolation_degree>::energy(const Field<Vector<T, dim>>& x)
{
    T total_energy = 0;
    Field<T> t_energy(m_X.size(), (T)0);
    for (auto& model : elasticity_models) {
        model->trial_energy(t_energy);
    }
    for (size_t i = 0; i < m_X.size(); ++i) {
        total_energy += t_energy[i];
    }
    return this->energy_scale * total_energy;
}

template <class T, int dim, class StorageIndex, int interpolation_degree>
void ElasticityEnergy<T, dim, StorageIndex, interpolation_degree>::gradient(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& grad)
{
    grad.assign(grid.num_nodes, Vector<T, dim>::Zero());
    Field<TM> t_gradient(m_X.size(), TM::Zero());
    for (auto& model : elasticity_models) {
        model->trial_gradient(t_gradient);
    }
    grid.colored_for([&](int i) {
        const Vector<T, dim>& Xp = m_X[i];
        BSplineWeights<T, dim, interpolation_degree> spline(Xp, dx);
        Matrix<T, dim, dim> stress = t_gradient[i];
        grid.iterateKernel(spline, [&](const Vector<int, dim>& node, T w, Vector<T, dim> dw, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            auto idx = g.idx;
            grad[idx] += this->energy_scale * stress * dw;
        });
    });
}

template <class T, int dim, class StorageIndex, int interpolation_degree>
void ElasticityEnergy<T, dim, StorageIndex, interpolation_degree>::hessian(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess, bool project_pd)
{
    std::vector<int> entryRow(grid.num_nodes * kernel_size, 0);
    std::vector<int> entryCol(grid.num_nodes * kernel_size, 0);
    Field<TM> entryVal(grid.num_nodes * kernel_size, TM::Zero());
    // local dPdF
    Field<Matrix<T, dim * dim, dim * dim>> t_hessian(m_X.size(), Matrix<T, dim * dim, dim * dim>::Zero());
    for (auto& model : elasticity_models) {
        model->trial_hessian(t_hessian, project_pd);
    }
    grid.colored_for([&](int i) {
        const Vector<T, dim>& Xp = m_X[i];
        BSplineWeights<T, dim, interpolation_degree> spline(Xp, dx);
        Matrix<T, dim * dim, dim* dim> deformed_dPdF = t_hessian[i];
        grid.iterateKernel(spline, [&](const Vector<int, dim>& node, T w, Vector<T, dim> dw, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            grid.iterateKernel(spline, [&](const Vector<int, dim>& _node, T _w, Vector<T, dim> _dw, GridState<T, dim>& _g) {
                if (_g.idx < 0) return;
                TM dFdX = TM::Zero();
                for (int u = 0; u < dim; ++u)
                    for (int p = 0; p < dim; ++p)
                        for (int x = 0; x < dim; ++x)
                            for (int y = 0; y < dim; ++y)
                                dFdX(u, p) += deformed_dPdF(u + x * dim, p + y * dim) * dw(x) * _dw(y);
                entryRow[g.idx * kernel_size + kernelOffset(node - _node)] = g.idx;
                entryCol[g.idx * kernel_size + kernelOffset(node - _node)] = _g.idx;
                entryVal[g.idx * kernel_size + kernelOffset(node - _node)] += dFdX;
            });
        });
    });
    using IJK = Eigen::Triplet<T>;
    std::vector<IJK> coeffs;
    for (int i = 0; i < grid.num_nodes * kernel_size; ++i)
        for (int u = 0; u < dim; ++u)
            for (int v = 0; v < dim; ++v) {
                coeffs.push_back(IJK(entryRow[i] * dim + u, entryCol[i] * dim + v, this->energy_scale * entryVal[i](u, v)));
            }
    hess.derived().resize(grid.num_nodes * dim, grid.num_nodes * dim);
    hess.derived().setZero();
    hess.derived().setFromTriplets(coeffs.begin(), coeffs.end());
}

template <class T, int dim, class StorageIndex, int interpolation_degree>
T ElasticityEnergy<T, dim, StorageIndex, interpolation_degree>::stepsize_upperbound(const Field<Vector<T, dim>>& x, const Field<Vector<T, dim>>& direction)
{
    T alpha = 1.0;
    Field<TM> m_gradDXp(m_X.size(), TM::Zero());
    tbb::parallel_for(size_t(0), m_X.size(), [&](size_t i) {
        const Vector<T, dim>& Xp = m_X[i];
        BSplineWeights<T, dim, interpolation_degree> spline(Xp, dx);
        Matrix<T, dim, dim> gradDXp = Matrix<T, dim, dim>::Zero();
        grid.iterateKernel(spline, [&](const Vector<int, dim>& node, T w, Vector<T, dim> dw, GridState<T, dim>& g) {
            if (g.idx < 0) return;
            gradDXp.noalias() += direction[g.idx] * dw.transpose();
        });
        m_gradDXp[i] = gradDXp;
    });
    for (auto& model : elasticity_models) {
        alpha = std::min(alpha, model->stepsize_upperbound(m_gradDXp));
    }
    return alpha;
}
} // namespace Bow::MPM