#pragma once

#include <Bow/Geometry/Hybrid/MPMGrid.h>
#include <Bow/IO/ply.h>
#include <Bow/Utils/Timer.h>
#include <Bow/Utils/Logging.h>
#include <tbb/tbb.h>
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>

using namespace SPGrid;

namespace Bow {
namespace MPM {

class AbstractOp {
};

template <class T, int dim, bool symplectic = true, int interpolation_degree = 2>
class ParticlesToGridOp : public AbstractOp {
public:
    using SparseMask = typename MPMGrid<T, dim>::SparseMask;
    Field<Vector<T, dim>>& m_X;
    Field<Vector<T, dim>>& m_V;
    std::vector<T>& m_mass;
    Field<Matrix<T, dim, dim>>& m_C;
    Field<Matrix<T, dim, dim>>& stress;

    MPMGrid<T, dim>& grid;
    T dx;
    T dt;

    void operator()()
    {
        BOW_TIMER_FLAG("P2G");

        if (false) {
            // TODO: BUG: v_and_m of some grids are not zero with cubic kernel in 3D
            // to reproduce: ./bow_tests [MPM-Momentum]
            for (size_t i = 0; i < m_X.size(); ++i) {
                const Vector<T, dim> pos = m_X[i];
                BSplineWeights<T, dim, interpolation_degree> spline(pos, dx);
                grid.iterateKernel(spline, [&](const Vector<int, dim>& node, T w, const Vector<T, dim>& dw, GridState<T, dim>& g) {
                    if (g.v_and_m.norm() > 1e-16) {
                        Logging::error("g.v_and_m is not zero: ", g.v_and_m.transpose());
                    }
                });
            }
        }
        grid.colored_for([&](int i) {
            const Vector<T, dim> pos = m_X[i];
            const Vector<T, dim> v = m_V[i];
            const T mass = m_mass[i];
            const Matrix<T, dim, dim> C = m_C[i] * mass;
            const Vector<T, dim> momentum = mass * v;
            const Matrix<T, dim, dim> delta_t_tmp_force = -dt * stress[i];
            BSplineWeights<T, dim, interpolation_degree> spline(pos, dx);
            grid.iterateKernel(spline, [&](const Vector<int, dim>& node, T w, const Vector<T, dim>& dw, GridState<T, dim>& g) {
                Vector<T, dim> xi_minus_xp = node.template cast<T>() * dx - pos;
                Vector<T, dim + 1> velocity_term = Vector<T, dim + 1>::Zero();
                velocity_term.template topLeftCorner<dim, 1>() = momentum + C * xi_minus_xp;
                velocity_term(dim) = mass;
                Vector<T, dim + 1> stress_term_dw = Vector<T, dim + 1>::Zero();
                if constexpr (symplectic)
                    stress_term_dw.template topLeftCorner<dim, 1>() = delta_t_tmp_force * dw;
                Vector<T, dim + 1> delta = w * velocity_term + stress_term_dw;
                g.v_and_m += delta;
            });
        });
        grid.countNumNodes();
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            T mass = g.v_and_m(dim);
            Vector<T, dim + 1> alpha;
            alpha.template topLeftCorner<dim, 1>() = Vector<T, dim>::Ones() * ((T)1 / mass);
            alpha(dim) = T(1);
            g.v_and_m = g.v_and_m.cwiseProduct(alpha);
            g.old_v = g.v_and_m.template segment<dim>(0);
            g.x = node.template cast<T>() * dx;
        });
    }
};

template <class T, int dim, int interpolation_degree = 2>
class GridToParticlesOp : public AbstractOp {
public:
    Field<Vector<T, dim>>& m_X;
    Field<Vector<T, dim>>& m_V;
    Field<Matrix<T, dim, dim>>& m_C;

    MPMGrid<T, dim>& grid;
    T dx;
    T dt;
    T flip_pic_ratio = 0.98;

    Field<Matrix<T, dim, dim>> m_gradXp = Field<Matrix<T, dim, dim>>();

    template <bool apic>
    void grid_to_particle()
    {
        BOW_TIMER_FLAG("G2P");
        T D_inverse = 0;
        if constexpr (interpolation_degree == 2)
            D_inverse = (T)4 / (dx * dx);
        else if constexpr (interpolation_degree == 3)
            D_inverse = (T)3 / (dx * dx);
        m_gradXp.assign(m_X.size(), Matrix<T, dim, dim>());
        grid.parallel_for([&](int i) {
            Vector<T, dim>& Xp = m_X[i];
            Vector<T, dim> picV = Vector<T, dim>::Zero();
            BSplineWeights<T, dim, interpolation_degree> spline(Xp, dx);

            Matrix<T, dim, dim> gradXp = Matrix<T, dim, dim>::Identity();
            Vector<T, dim> oldV = Vector<T, dim>::Zero();
            Vector<T, dim> picX = Vector<T, dim>::Zero();
            grid.iterateKernel(spline, [&](const Vector<int, dim>& node, T w, Vector<T, dim> dw, GridState<T, dim>& g) {
                if (g.idx < 0) return;
                Vector<T, dim> new_v = g.v_and_m.template topLeftCorner<dim, 1>();
                picV += w * new_v;
                oldV += w * g.old_v;
                picX += w * g.x;
                Vector<T, dim> xn = node.template cast<T>() * dx;
                gradXp.noalias() += (g.x - xn) * dw.transpose();
            });
            if constexpr (apic) {
                Matrix<T, dim, dim> Bp = Matrix<T, dim, dim>::Zero();
                grid.iterateKernel(spline, [&](const Vector<int, dim>& node, T w, Vector<T, dim> dw, GridState<T, dim>& g) {
                    if (g.idx < 0) return;
                    Vector<T, dim> xn = dx * node.template cast<T>();
                    Vector<T, dim> new_v = g.v_and_m.template topLeftCorner<dim, 1>();
                    Bp += 0.5 * w * (new_v * (xn - m_X[i] + g.x - picX).transpose() + (xn - m_X[i] - g.x + picX) * new_v.transpose());
                });
                if constexpr (interpolation_degree == 1)
                    m_C[i] = (gradXp - Matrix<T, dim, dim>::Identity()) / dt;
                else
                    m_C[i] = Bp * D_inverse;
                m_V[i] = picV;
            }
            else {
                m_C[i].setZero();
                m_V[i] *= flip_pic_ratio;
                m_V[i] += picV - flip_pic_ratio * oldV;
            }
            m_X[i] = picX;

            m_gradXp[i] = gradXp;
        });
    }

    void operator()(bool apic = true)
    {
        if (apic)
            grid_to_particle<true>();
        else
            grid_to_particle<false>();
    }
};
}
} // namespace Bow::MPM