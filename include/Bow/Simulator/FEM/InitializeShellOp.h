#ifndef INITIALIZE_SHELL_OP_H
#define INITIALIZE_SHELL_OP_H
#include <Bow/Macros.h>
#include <Bow/Types.h>
#include <Bow/Geometry/DihedralAngle.h>

//TODO: combine with InitializeOp
namespace Bow::Shell {
template <class T>
class InitializeMembraneOp {
public:
    static const int dim = 3;
    static const int codim = 2;

    const Field<Vector<T, dim>>& m_X;
    const Field<Vector<int, codim + 1>>& m_elem;
    const T thickness;

    Field<Matrix<T, codim, codim>>& m_IB;
    Field<T>& m_vol;

    InitializeMembraneOp(const Field<Vector<T, dim>>& X, const Field<Vector<int, codim + 1>>& elem, const T thickness, Field<Matrix<T, codim, codim>>& IB, Field<T>& vol)
        : m_X(X), m_elem(elem), thickness(thickness), m_IB(IB), m_vol(vol) {}
    void operator()()
    {
        m_IB.resize(m_elem.size());
        m_vol.resize(m_elem.size());
        for (size_t e = 0; e < m_elem.size(); ++e) {
            const auto& vertices = m_elem[e];
            Matrix<T, dim, codim> basis;
            basis.col(0) = m_X[vertices[1]] - m_X[vertices[0]];
            basis.col(1) = m_X[vertices[2]] - m_X[vertices[0]];
            m_IB[e] = (basis.transpose() * basis).inverse();
            m_vol[e] = thickness * (basis.col(0).cross(basis.col(1))).norm() / T(2);
        }
    }
};

template <class T>
class InitializeBendingOp {
public:
    static const int dim = 3;
    static const int codim = 2;

    const Field<Vector<T, dim>>& m_X;
    const Field<Vector<int, 4>>& m_edge_stencil; // e0, e1 is the common edge
    Field<T>& m_e;
    Field<T>& m_h;
    Field<T>& m_rest_angle;

    InitializeBendingOp(const Field<Vector<T, dim>>& X, const Field<Vector<int, 4>>& edge_stencil, Field<T>& e, Field<T>& h, Field<T>& rest_angle)
        : m_X(X), m_edge_stencil(edge_stencil), m_e(e), m_h(h), m_rest_angle(rest_angle) {}
    void operator()()
    {
        m_e.resize(m_edge_stencil.size());
        m_h.resize(m_edge_stencil.size());
        m_rest_angle.resize(m_edge_stencil.size());
        for (size_t i = 0; i < m_edge_stencil.size(); ++i) {
            if (m_edge_stencil[i][3] < 0) continue;
            const auto& X0 = m_X[m_edge_stencil[i][2]];
            const auto& X1 = m_X[m_edge_stencil[i][0]];
            const auto& X2 = m_X[m_edge_stencil[i][1]];
            const auto& X3 = m_X[m_edge_stencil[i][3]];
            auto n1 = (X1 - X0).cross(X2 - X0);
            auto n2 = (X2 - X3).cross(X1 - X3);
            m_e[i] = (m_X[m_edge_stencil[i](1)] - m_X[m_edge_stencil[i](0)]).norm();
            m_h[i] = (n1.norm() + n2.norm()) / (m_e[i] * 6);
            m_rest_angle[i] = Bow::Geometry::dihedral_angle(m_X[m_edge_stencil[i](2)], m_X[m_edge_stencil[i](0)], m_X[m_edge_stencil[i](1)], m_X[m_edge_stencil[i](3)]);
        }
    }
};
} // namespace Bow::Shell

#endif