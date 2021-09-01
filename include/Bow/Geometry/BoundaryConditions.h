#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <Bow/Types.h>
#include <Bow/Utils/Logging.h>
#include <limits>
#include <memory>

namespace Bow::Geometry {

enum Type { STICKY,
    SLIP,
    SEPARATE
};

template <class T, int dim>
class AnalyticalLevelSet {
    using TV = Vector<T, dim>;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Type type;
    explicit AnalyticalLevelSet(Type type)
        : type(type) {}
    virtual T signed_distance(const TV& X) = 0;
    virtual TV normal(const TV& X) { BOW_NOT_IMPLEMENTED return TV::Unit(0); }
    virtual TV velocity(const TV& X) { return TV::Zero(); }
    virtual T moveTime() { return 0.0; }
    virtual TV closest_point(const TV& X) { BOW_NOT_IMPLEMENTED return TV::Unit(0); }
    virtual void update(const T& dt) { return; }
    virtual void setVelocity(const TV& newV) { return; }
};

template <class T, int dim>
class HalfSpaceLevelSet : public AnalyticalLevelSet<T, dim> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using TV = Vector<T, dim>;
    TV origin;
    TV outward_normal;
    TV v;
    T move_time;
    HalfSpaceLevelSet(Type type, const TV& origin, const TV& normal, TV v = TV::Zero(), T moveTime = 0.0);
    T signed_distance(const TV& X) override;
    TV normal(const TV& X) override;
    TV closest_point(const TV& X) override;
    TV velocity(const TV& X) override;
    T moveTime() override;
    void setVelocity(const TV& newV) override;
    virtual void update(const T& dt) override;
};

template <class T, int dim>
class BallLevelSet : public AnalyticalLevelSet<T, dim> {
    using TV = Vector<T, dim>;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    TV center;
    T radius;
    BallLevelSet(Type type, const TV& center, T radius);
    T signed_distance(const TV& X) override;
    TV normal(const TV& X) override;
    TV closest_point(const TV& X) override;
};

template <class T, int dim>
class InverseBallLevelSet : public AnalyticalLevelSet<T, dim> {
    using TV = Vector<T, dim>;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    TV center;
    T radius;
    InverseBallLevelSet(Type type, const TV& center, T radius);
    T signed_distance(const TV& X) override;
    TV normal(const TV& X) override;
    TV closest_point(const TV& X) override;
};

template <class T, int dim>
class MovingBallLevelSet : public BallLevelSet<T, dim> {
    using TV = Vector<T, dim>;
    using Base = BallLevelSet<T, dim>;
    using Base::center;
    using Base::radius;

public:
    TV center_velocity;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MovingBallLevelSet(Type type, const TV& center, T radius, const TV& center_velocity);
    virtual TV velocity(const TV& X) override;
    virtual void update(const T& dt) override;
};

template <class T, int dim>
class AlignedBoxLevelSet : public AnalyticalLevelSet<T, dim> {
    using TV = Vector<T, dim>;
    TV center;
    TV half_edges;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    AlignedBoxLevelSet(Type type, const TV& min_corner, const TV& max_corner);
    T signed_distance(const TV& X) override;
};

template <class T, int dim>
class BoxLevelSet : public AnalyticalLevelSet<T, dim> {
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    TV center;
    TV half_edges;
    TM R;
    Eigen::Quaternion<T> rotation;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    BoxLevelSet(Type type, const TV& min_corner, const TV& max_corner, const Bow::Vector<T, 4>& rot);
    T signed_distance(const TV& X) override;
    TV normal(const TV& X) override;
    TV closest_point(const TV& X) override;
};

template <class T, int dim>
class ScriptedLevelSet : public AnalyticalLevelSet<T, dim> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <class T, int dim>
class IndexBasedBoundaryCondition {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using TV = Vector<T, dim>;
    int index;
    Type type;
    TV m_n;
    T m_v;
    IndexBasedBoundaryCondition(int index, Type type, const TV& normal, const T velocity)
        : index(index), type(type), m_n(normal.normalized()), m_v(velocity) {}
    TV normal() { return m_n; }
    TV velocity() { return m_v * m_n; }
    virtual void update(const T& dt) { return; }
    void update_direction(const TV& n) { m_n = n.normalized(); }
    void update_rate(const T& v) { m_v = v; }
};

} // namespace Bow::Geometry

#ifndef BOW_STATIC_LIBRARY
#include "BoundaryConditions.cpp"
#endif

#endif