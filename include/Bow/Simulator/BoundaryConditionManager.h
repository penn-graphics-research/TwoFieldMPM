#ifndef BOUNDARY_CONDITION_MANAGER_H
#define BOUNDARY_CONDITION_MANAGER_H

#include <Bow/Types.h>
#include <Bow/Geometry/BoundaryConditions.h>

namespace Bow {

template <class T, int dim>
class BoundaryConditionManager {
public:
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    // levelset based
    std::vector<std::shared_ptr<Geometry::AnalyticalLevelSet<T, dim>>> level_set_objects;
    // index based
    std::map<int, std::vector<std::shared_ptr<Geometry::IndexBasedBoundaryCondition<T, dim>>>> node_bcs;
    void clear()
    {
        level_set_objects.clear();
        node_bcs.clear();
    }
    void add(std::shared_ptr<Geometry::AnalyticalLevelSet<T, dim>> ls_ptr)
    {
        level_set_objects.push_back(ls_ptr);
    }
    void add(std::shared_ptr<Geometry::IndexBasedBoundaryCondition<T, dim>> node_bc_ptr)
    {
        node_bcs[node_bc_ptr->index].push_back(node_bc_ptr);
    }
    bool update_index_based_bc_velocity(int node, const TV& velocity)
    {
        if (node_bcs.find(node) == node_bcs.end()) return false;
        for (auto bc : node_bcs[node]) {
            bc->update_rate(velocity.norm());
            bc->update_direction(velocity.normalized());
        }
        return true;
    }
    void update(T dt)
    {
        for (auto& object : level_set_objects) object->update(dt);
        for (auto& it : node_bcs)
            for (auto& object : it.second)
                object->update(dt);
    }
    void mpm_explicit_update(const TV& X, TV& V)
    {
        for (auto& ls_ptr : level_set_objects)
            if (ls_ptr->signed_distance(X) < 0) {
                TV v = ls_ptr->velocity(X);
                TV n = ls_ptr->normal(X);
                if (ls_ptr->type == Geometry::STICKY) {
                    V = v;
                }
                if (ls_ptr->type == Geometry::SLIP) {
                    V -= n.dot(V - v) * n;
                }
                if (ls_ptr->type == Geometry::SEPARATE) {
                    T dot_value = n.dot(V - v);
                    if (dot_value < 0)
                        V -= dot_value * n;
                }
            }
    }
    void mpm_implicit_update(const TV& X, TM& basis, int& order)
    {
        basis = Matrix<T, dim, dim>::Identity();
        order = 0;
        for (auto& ls_ptr : level_set_objects)
            if (ls_ptr->signed_distance(X) <= 0) {
                TV v = ls_ptr->velocity(X);
                TV n = ls_ptr->normal(X);
                if (v.norm()) { BOW_NOT_IMPLEMENTED }
                if (ls_ptr->type == Geometry::STICKY) {
                    basis = Matrix<T, dim, dim>::Identity();
                    order = dim;
                    return;
                }
                if (ls_ptr->type == Geometry::SLIP) {
                    for (int i = 0; i < order; ++i)
                        n -= n.dot(basis.col(i)) * basis.col(i);
                    if (n.norm()) {
                        n.normalize();
                        basis.col(order++) = n;
                    }
                }
                if (ls_ptr->type == Geometry::SEPARATE) {
                    BOW_NOT_IMPLEMENTED
                }
            }
        // no need to deal with order == 0
        if constexpr (dim == 2) {
            if (order == 1) {
                basis(0, 1) = basis(1, 0);
                basis(1, 1) = -basis(0, 0);
            }
        }
        if constexpr (dim == 3) {
            if (order == 1) {
                if (basis.col(0).dot(TV(1, 0, 0)) > 0.5)
                    basis.col(1) = basis.col(0).cross(TV(0, 1, 0));
                else
                    basis.col(1) = basis.col(0).cross(TV(1, 0, 0));
                basis.col(1).normalize();
            }
            if (order == 1 || order == 2) {
                basis.col(2) = basis.col(0).cross(basis.col(1));
                basis.col(2).normalize();
            }
        }
    }

    void init_update(const TV& x, TM& basis, int& order, TV& target_after_transform)
    {
        basis = Matrix<T, dim, dim>::Identity();
        order = 0;
        target_after_transform = x;
    }

    /** Must be called after init_update */
    bool level_set_based_update(const TV& x, const T dt, TM& basis, int& order, TV& target_after_transform, uint8_t& fixed)
    {
        bool valid = false;
        for (auto& ls_ptr : level_set_objects)
            if (ls_ptr->signed_distance(x) <= 0) {
                valid = true;
                TV v = ls_ptr->velocity(x);
                TV n = ls_ptr->normal(x);
                if (v.squaredNorm() == 0) { fixed = true; }
                if (ls_ptr->type == Geometry::STICKY) {
                    basis = Matrix<T, dim, dim>::Identity();
                    order = dim;
                    target_after_transform += dt * ls_ptr->velocity(x);
                    return true;
                }
                if (ls_ptr->type == Geometry::SLIP) {
                    for (int i = 0; i < order; ++i)
                        n -= n.dot(basis.col(i)) * basis.col(i);
                    if (n.norm()) {
                        n.normalize();
                        basis.col(order++) = n;
                    }
                }
                if (ls_ptr->type == Geometry::SEPARATE) {
                    BOW_NOT_IMPLEMENTED
                }
                target_after_transform += dt * ls_ptr->velocity(x).dot(n) * n;
            }
        // no need to deal with order == 0
        if constexpr (dim == 2) {
            if (order == 1) {
                basis(0, 1) = basis(1, 0);
                basis(1, 1) = -basis(0, 0);
            }
        }
        if constexpr (dim == 3) {
            if (order == 1) {
                if (basis.col(0).dot(TV(1, 0, 0)) > 0.5)
                    basis.col(1) = basis.col(0).cross(TV(0, 1, 0));
                else
                    basis.col(1) = basis.col(0).cross(TV(1, 0, 0));
                basis.col(1).normalize();
            }
            if (order == 1 || order == 2) {
                basis.col(2) = basis.col(0).cross(basis.col(1));
                basis.col(2).normalize();
            }
        }
        target_after_transform = basis.transpose() * target_after_transform;
        return valid;
    }

    /** Must be called after init_update */
    bool index_based_update(const int node, const TV& x, const T dt, TM& basis, int& order, TV& target_after_transform, uint8_t& fixed)
    {
        if (node_bcs.find(node) == node_bcs.end()) return false;
        target_after_transform = basis * target_after_transform; // transform back for further editting.
        for (auto bc : node_bcs[node]) {
            TV v = bc->velocity();
            if (v.squaredNorm() == 0) { fixed = true; }
            TV n = bc->normal();
            if (bc->type == Geometry::STICKY) {
                basis = Matrix<T, dim, dim>::Identity();
                order = dim;
                target_after_transform += dt * v;
                return true;
            }
            else if (bc->type == Geometry::SLIP) {
                for (int i = 0; i < order; ++i)
                    n -= n.dot(basis.col(i)) * basis.col(i);
                if (n.norm()) {
                    n.normalize();
                    basis.col(order++) = n;
                }
            }
            else if (bc->type == Geometry::SEPARATE) {
                BOW_NOT_IMPLEMENTED
            }
            target_after_transform += dt * v.dot(n) * n;
        }
        // no need to deal with order == 0
        if constexpr (dim == 2) {
            if (order == 1) {
                basis(0, 1) = basis(1, 0);
                basis(1, 1) = -basis(0, 0);
            }
        }
        if constexpr (dim == 3) {
            if (order == 1) {
                if (basis.col(0).dot(TV(1, 0, 0)) > 0.5)
                    basis.col(1) = basis.col(0).cross(TV(0, 1, 0));
                else
                    basis.col(1) = basis.col(0).cross(TV(1, 0, 0));
                basis.col(1).normalize();
            }
            if (order == 1 || order == 2) {
                basis.col(2) = basis.col(0).cross(basis.col(1));
                basis.col(2).normalize();
            }
        }

        target_after_transform = basis.transpose() * target_after_transform;
        return true;
    }
};

} // namespace Bow

#endif