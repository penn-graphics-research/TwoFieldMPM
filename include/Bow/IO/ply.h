#ifndef BOW_PLY_H
#define BOW_PLY_H
#include <Bow/Types.h>
#include <Bow/Macros.h>

namespace Bow {
namespace IO {
template <class T, int dim>
BOW_INLINE void read_ply(const std::string filename, Field<Vector<T, dim>>& vertices, Field<Vector<int, 3>>& faces);
template <class T, int dim>
BOW_INLINE void read_ply(const std::string filename, Field<Vector<T, dim>>& vertices);
template <class T, int dim>
BOW_INLINE void write_ply(const std::string filename, const Field<Vector<T, dim>>& vertices, const Field<Vector<int, 3>>& faces, const bool binary = true);
template <class T, int dim>
BOW_INLINE void write_ply(const std::string filename, const Field<Vector<T, dim>>& vertices, const Field<Vector<int, 3>>& faces, const Field<T>& info, const bool binary = true);
template <class T, int dim>
BOW_INLINE void write_ply(const std::string filename, const Field<Vector<T, dim>>& vertices, const bool binary = true);
template <class T, int dim>
BOW_INLINE void writeTwoField_particles_ply(const std::string filename, const Field<Vector<T, dim>>& vertices, const Field<Vector<T, dim>>& velocities, const Field<Vector<T, dim>>& damageGradients, const std::vector<T>& masses, const std::vector<T>& damage, const std::vector<int>& sp, const Field<int>& markers, const bool binary = true);
template <class T, int dim>
BOW_INLINE void writeTwoField_nodes_ply(const std::string filename, const Field<Vector<T, dim>>& vertices, const Field<Vector<T, dim>>& damageGradients, const Field<Vector<T, dim>>& v1, const Field<Vector<T, dim>>& v2, const Field<Vector<T, dim>>& fct1, const Field<Vector<T, dim>>& fct2, const std::vector<T>& m1, const std::vector<T>& m2, const std::vector<T>& sep1, const std::vector<T>& sep2, const std::vector<int>& separable, const bool binary = true);
}
} // namespace Bow::IO

#ifndef BOW_STATIC_LIBRARY
#include "ply.cpp"
#endif

#endif