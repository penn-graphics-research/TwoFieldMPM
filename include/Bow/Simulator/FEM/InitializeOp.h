#ifndef FEM_INITIALIZE_OP_H
#define FEM_INITIALIZE_OP_H

#include <oneapi/tbb.h>
#include <Bow/Types.h>

namespace Bow::FEM {

template <class T, int dim>
class InitializeOp {
public:
    // Inputs:
    const Field<Vector<T, dim>>& m_X; // material coordinate;
    const Field<Vector<int, dim + 1>>& m_elem; // element vertex indices
    const Field<T>& m_density; // element density
    const Field<Vector<int, dim>>& m_elem_codim1; // element vertex indices
    const Field<T>& m_thickness_codim1; // element thickness
    const Field<T>& m_density_codim1; // element density

    // Outputs:
    Field<T>& m_mass; // node mass
    Field<T>& m_vol; // element volume
    Field<Matrix<T, dim, dim>>& m_IB; // inverse basis
    Field<T>& m_vol_codim1; // element volume
    Field<Matrix<T, dim - 1, dim - 1>>& m_IB_codim1; // inverse basis

    inline void operator()();
};

/* InitializeOp */
template <class T, int dim>
inline void InitializeOp<T, dim>::operator()()
{
    // codim0 elements
    m_vol.resize(m_elem.size());
    m_IB.resize(m_elem.size());
    tbb::parallel_for(size_t(0), m_elem.size(), [&](size_t e) {
        const auto& indices = m_elem[e];
        const auto& X0 = m_X[indices[0]];
        const auto& X1 = m_X[indices[1]];
        const auto& X2 = m_X[indices[2]];
        Matrix<T, dim, dim> B;
        B.col(0) = X1 - X0;
        B.col(1) = X2 - X0;
        if constexpr (dim == 3) {
            const auto& X3 = m_X[indices[3]];
            B.col(2) = X3 - X0;
            m_vol[e] = B.determinant() / T(6);
        }
        else {
            m_vol[e] = B.determinant() / T(2);
        }
        m_IB[e] = B.inverse();
    });

    // codim1 elements
    m_vol_codim1.resize(m_elem_codim1.size());
    m_IB_codim1.resize(m_elem_codim1.size());
    tbb::parallel_for(size_t(0), m_elem_codim1.size(), [&](size_t e) {
        const auto& indices = m_elem_codim1[e];
        const auto& X0 = m_X[indices[0]];
        const auto& X1 = m_X[indices[1]];
        const auto& X2 = m_X[indices[2]];

        const Vector<T, dim> e01 = X1 - X0;
        const Vector<T, dim> e02 = X2 - X0;

        Matrix<T, dim - 1, dim - 1> B;
        B(0, 0) = e01.squaredNorm();
        B(1, 0) = B(0, 1) = e01.dot(e02);
        B(1, 1) = e02.squaredNorm();

        if constexpr (dim == 3) {
            m_vol_codim1[e] = e01.cross(e02).norm() / T(2) * m_thickness_codim1[e];
        }
        else {
            //TODO
        }
        m_IB_codim1[e] = B.inverse();
    });

    // compute node mass
    m_mass.resize(m_X.size());
    tbb::parallel_for(size_t(0), m_X.size(), [&](size_t i) {
        m_mass[i] = 0;
    });
    for (size_t e = 0; e < m_elem.size(); ++e) {
        const auto& indices = m_elem[e];
        T mass_contrib = m_density[e] * m_vol[e] / T(dim + 1);
        for (int i = 0; i < dim + 1; ++i) {
            m_mass[indices[i]] += mass_contrib;
        }
    }
    for (size_t e = 0; e < m_elem_codim1.size(); ++e) {
        const auto& indices = m_elem_codim1[e];
        T mass_contrib = m_density_codim1[e] * m_vol_codim1[e] / T(dim);
        for (int i = 0; i < dim; ++i) {
            m_mass[indices[i]] += mass_contrib;
        }
    }
}
} // namespace Bow::FEM

#endif