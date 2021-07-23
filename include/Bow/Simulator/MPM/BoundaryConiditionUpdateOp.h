#include <Bow/Types.h>
#include <Bow/Simulator/BoundaryConditionManager.h>
#include <Bow/Geometry/Hybrid/MPMGrid.h>

namespace Bow::MPM {

template <class T, int dim>
class BoundaryConditionUpdateOp : public AbstractOp {
public:
    using SparseMask = typename MPMGrid<T, dim>::SparseMask;
    MPMGrid<T, dim>& grid;
    Vector<T, dim>& gravity;
    BoundaryConditionManager<T, dim>& BC;
    T dx;
    T dt;

    void operator()()
    {
        BOW_TIMER_FLAG("grid update");
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            g.v_and_m.template head<dim>() += gravity * dt;
            Vector<T, dim> new_v = g.v_and_m.template head<dim>();
            BC.mpm_explicit_update(node.template cast<T>() * dx, new_v);
            g.v_and_m.template head<dim>() = new_v;
            g.old_v = new_v;
            g.x = node.template cast<T>() * dx + dt * new_v;
        });
    }
};

template <class T, int dim>
class ImplicitBoundaryConditionUpdateOp : public AbstractOp {
public:
    using TM = Matrix<T, dim, dim>;
    MPMGrid<T, dim>& grid;
    BoundaryConditionManager<T, dim>& BC;
    Field<Matrix<T, dim, dim>>& BC_basis;
    Field<int>& BC_order;
    T dx;

    void operator()()
    {
        BC_basis.assign(grid.num_nodes, TM::Identity());
        BC_order.assign(grid.num_nodes, 0);
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            BC.mpm_implicit_update(node.template cast<T>() * dx, BC_basis[g.idx], BC_order[g.idx]);
        });
    }
};

template <class T, int dim>
class ImplicitMovingBoundaryConditionUpdateOp : public AbstractOp {
public:
    using TM = Matrix<T, dim, dim>;
    MPMGrid<T, dim>& grid;
    BoundaryConditionManager<T, dim>& BC;
    Field<Matrix<T, dim, dim>>& BC_basis;
    Field<int>& BC_order;
    Field<Vector<T, dim>>& BC_target;
    T dx;
    T dt;

    void operator()()
    {
        BC_basis.assign(grid.num_nodes, TM::Identity());
        BC_order.assign(grid.num_nodes, 0);
        BC_target.resize(grid.num_nodes);
        grid.iterateGrid([&](const Vector<int, dim>& node, GridState<T, dim>& g) {
            uint8_t _fixed;
            BC.init_update(node.template cast<T>() * dx, BC_basis[g.idx], BC_order[g.idx], BC_target[g.idx]);
            BC.level_set_based_update(node.template cast<T>() * dx, dt, BC_basis[g.idx], BC_order[g.idx], BC_target[g.idx], _fixed);
        });
    }
};

} // namespace Bow::MPM