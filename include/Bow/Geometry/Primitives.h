#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include <Bow/Macros.h>
#include <Bow/Types.h>

namespace Bow {
namespace Geometry {
/**
     * Triangulate a 2D/3D cube with identical triangles.
*/
template <class T, int dim>
BOW_INLINE void cube(const Vector<int, dim> resolution, const T dx, Field<Vector<T, dim>>& points, Field<Vector<int, dim + 1>>& elements, const Vector<T, dim> center = Vector<T, dim>::Zero());

template <class T, int dim>
BOW_INLINE void cube(const Vector<T, dim> min_corner, const Vector<T, dim> max_corner, const Vector<T, dim> block_size, Field<Vector<T, dim>>& points, Field<Vector<int, dim + 1>>& elements)
{
    Vector<T, dim> center = 0.5 * (min_corner + max_corner);
    Vector<int, dim> resolution = ((max_corner - min_corner).array() / block_size.array()).round().template cast<int>();
    cube(resolution, block_size(0), points, elements);
    auto mat = Eigen::Map<Matrix<T, dim, -1>>(&points[0][0], dim, points.size());
    for (int d = 1; d < dim; ++d)
        mat.row(d) *= (block_size(d) / block_size(0));
    mat.colwise() += center;
}
}
} // namespace Bow::Geometry

#ifndef BOW_STATIC_LIBRARY
#include "Primitives.cpp"
#endif

#endif