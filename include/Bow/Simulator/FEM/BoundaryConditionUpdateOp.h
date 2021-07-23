#ifndef FEM_BOUNDARY_CONDITION_UPDATE_OP_H
#define FEM_BOUNDARY_CONDITION_UPDATE_OP_H

#include <oneapi/tbb.h>
#include <Bow/Types.h>
#include <Bow/Simulator/BoundaryConditionManager.h>

namespace Bow::FEM {
template <class T, int dim>
class BoundaryConditionUpdateOp {
public:
    BoundaryConditionManager<T, dim>& BC;
    const Field<Vector<T, dim>>& m_x;
    Field<Matrix<T, dim, dim>>& BC_basis;
    Field<int>& BC_order;
    Field<Vector<T, dim>>& BC_target;
    Field<uint8_t>& BC_fixed;
    T dt;

    void operator()()
    {
        tbb::parallel_for(size_t(0), m_x.size(), [&](size_t i) {
            BC.init_update(m_x[i], BC_basis[i], BC_order[i], BC_target[i]);
            BC.level_set_based_update(m_x[i], dt, BC_basis[i], BC_order[i], BC_target[i], BC_fixed[i]);
            BC.index_based_update(i, m_x[i], dt, BC_basis[i], BC_order[i], BC_target[i], BC_fixed[i]);
        });
    }
};
} // namespace Bow::FEM

#endif