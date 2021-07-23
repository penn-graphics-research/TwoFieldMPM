#ifndef QUERY_H
#define QUERY_H
#include <Bow/Macros.h>
#include <Bow/Types.h>
namespace Bow {
namespace Geometry {
template <class T, int dim>
void select(const Field<Vector<T, dim>>& points, const std::function<bool(const Vector<T, dim>&)>& criteria, std::vector<int>& indices)
{
    indices.clear();
    for (size_t i = 0; i < points.size(); ++i) {
        if (criteria(points[i]))
            indices.push_back(i);
    }
}

/**
 * \brief find boundary element of a triangle mesh or a tet mesh.
 *        A face is a boundary face iff that face has only one half-face.
 *        The orientation of each tet is assumed to be: the orientation of (0,1,2) points to 3.
*/
template <int dim>
BOW_INLINE void find_boundary(const Field<Vector<int, dim + 1>>& elements, Field<Vector<int, dim>>& boundary_elements);

/**
 * \brief find boundary element of a tet mesh.
 *        The orientation of each tet is assumed to be: the orientation of (0,1,2) points to 3.
*/
BOW_INLINE void find_boundary(const Field<Vector<int, 3>>& elements,
    Field<Vector<int, 2>>& boundary_edges,
    Field<int>& boundary_points);

/**
 * \brief helper function for 3D find_boundary()
 *        The orientation of each tet is assumed to be: the orientation of (0,1,2) points to 3.
*/
void collect_edge_and_point(
    const Field<Vector<int, 3>>& boundary_faces,
    Field<Vector<int, 2>>& boundary_edges,
    Field<int>& boundary_points);

/**
 * \brief find boundary element of a triangle mesh.
*/
BOW_INLINE void find_boundary(const Field<Vector<int, 4>>& elements,
    Field<Vector<int, 3>>& boundary_faces,
    Field<Vector<int, 2>>& boundary_edges,
    Field<int>& boundary_points);
} // namespace Geometry

template <class T>
BOW_INLINE bool inside_triangle(const Vector<T, 2>& v, const Vector<T, 2>& v0, const Vector<T, 2>& v1, const Vector<T, 2>& v2);

template <class T>
BOW_INLINE bool inside_convex_polygon(const Vector<T, 2>& v, const Field<Vector<T, 2>>& vertices);

} // namespace Bow

#ifndef BOW_STATIC_LIBRARY
#include "Query.cpp"
#endif

#endif