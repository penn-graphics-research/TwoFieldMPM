#include "BoundaryConditions.h"

namespace Bow::Geometry {

//USE MOVING BOUNDARIES WITH STICKY BC ONLY!!!!
template <class T, int dim>
HalfSpaceLevelSet<T, dim>::HalfSpaceLevelSet(Type type, const TV& origin, const TV& outward_normal, TV v, T moveTime)
    : AnalyticalLevelSet<T, dim>(type), origin(origin), outward_normal(outward_normal.normalized()), v(v), move_time(moveTime)
{
}

template <class T, int dim>
T HalfSpaceLevelSet<T, dim>::signed_distance(const TV& X)
{
    T result = (X - origin).dot(outward_normal);
    return result;
}

template <class T, int dim>
Vector<T, dim> HalfSpaceLevelSet<T, dim>::normal(const TV& X)
{
    return outward_normal;
}

template <class T, int dim>
Vector<T, dim> HalfSpaceLevelSet<T, dim>::closest_point(const TV& X)
{
    return X - X.dot(outward_normal) * outward_normal;
}

template <class T, int dim>
Vector<T, dim> HalfSpaceLevelSet<T, dim>::velocity(const TV& X)
{
    return v;
}

template <class T, int dim>
T HalfSpaceLevelSet<T, dim>::moveTime()
{
    return move_time;
}

template <class T, int dim>
void HalfSpaceLevelSet<T, dim>::setVelocity(const TV& newV)
{
    v = newV;
}

template <class T, int dim>
void HalfSpaceLevelSet<T, dim>::update(const T& dt)
{
    origin += v * dt;
    //USE MOVING BOUNDARIES WITH STICKY BC ONLY!!!!
    //std::cout << "updating HalfSpaceLevelSets, velocity:" << v << std::endl;
}

template <class T, int dim>
BallLevelSet<T, dim>::BallLevelSet(Type type, const TV& center, T radius)
    : AnalyticalLevelSet<T, dim>(type), center(center), radius(radius)
{
}

template <class T, int dim>
T BallLevelSet<T, dim>::signed_distance(const TV& X)
{
    T result = (X - center).norm() - radius;
    return result;
}

template <class T, int dim>
Vector<T, dim> BallLevelSet<T, dim>::normal(const TV& X)
{
    TV n = X - center;
    if (n.norm() < 1e-12) n = TV::Random();
    n.normalize();
    return n;
}

template <class T, int dim>
Vector<T, dim> BallLevelSet<T, dim>::closest_point(const TV& X)
{
    TV n = normal(X);
    return center + n * radius;
}

template <class T, int dim>
InverseBallLevelSet<T, dim>::InverseBallLevelSet(Type type, const TV& center, T radius)
    : AnalyticalLevelSet<T, dim>(type), center(center), radius(radius)
{
}

template <class T, int dim>
T InverseBallLevelSet<T, dim>::signed_distance(const TV& X)
{
    T result = radius - (center - X).norm();
    return result;
}

template <class T, int dim>
Vector<T, dim> InverseBallLevelSet<T, dim>::normal(const TV& X)
{
    TV n = center - X;
    if (n.norm() < 1e-12)
        n = TV::Unit(0);
    else
        n.normalize();
    return n;
}

template <class T, int dim>
Vector<T, dim> InverseBallLevelSet<T, dim>::closest_point(const TV& X)
{
    TV n = normal(X);
    return center - n * radius;
}

template <class T, int dim>
MovingBallLevelSet<T, dim>::MovingBallLevelSet(Type type, const TV& center, T radius, const TV& center_velocity)
    : BallLevelSet<T, dim>(type, center, radius), center_velocity(center_velocity)
{
}

template <class T, int dim>
Vector<T, dim> MovingBallLevelSet<T, dim>::velocity(const TV& X)
{
    return center_velocity;
}

template <class T, int dim>
void MovingBallLevelSet<T, dim>::update(const T& dt)
{
    center += center_velocity * dt;
}

template <class T, int dim>
AlignedBoxLevelSet<T, dim>::AlignedBoxLevelSet(Type type, const TV& min_corner, const TV& max_corner)
    : AnalyticalLevelSet<T, dim>(type)
{
    center = (min_corner + max_corner) * 0.5;
    half_edges = (max_corner - min_corner) * 0.5;
}

template <class T, int dim>
T AlignedBoxLevelSet<T, dim>::signed_distance(const TV& X)
{
    TV X_centered = X - center;
    TV d = X_centered.cwiseAbs() - half_edges;
    T dd = d.array().maxCoeff();
    TV qq = d;
    for (int i = 0; i < dim; i++)
        if (qq(i) < (T)0)
            qq(i) = (T)0;
    // min(dd, (T)0) is to deal with inside box case
    T result = std::min(dd, (T)0) + qq.norm();
    return result;
}

template <class T, int dim>
BoxLevelSet<T, dim>::BoxLevelSet(Type type, const TV& min_corner, const TV& max_corner, const Bow::Vector<T, 4>& rot)
    : AnalyticalLevelSet<T, dim>(type)
{
    center = (min_corner + max_corner) * 0.5;
    half_edges = (max_corner - min_corner) * 0.5;
    R = TM::Identity();

    if constexpr (dim == 3) {
        Eigen::Quaternion<T> rotation(rot.w(), rot.x(), rot.y(), rot.z());
        R = rotation.toRotationMatrix();
    }
    else {
        T theta = rot.x();
        R << std::cos(theta), -std::sin(theta), std::sin(theta), std::cos(theta);
    }
}

template <class T, int dim>
T BoxLevelSet<T, dim>::signed_distance(const TV& X)
{
    TV X_centered = R.transpose() * (X - center);
    TV d = X_centered.cwiseAbs() - half_edges;
    T dd = d.array().maxCoeff();
    TV qq = d;
    for (int i = 0; i < dim; i++)
        if (qq(i) < (T)0)
            qq(i) = (T)0;
    T result = std::min(dd, (T)0) + qq.norm();
    return result;
}

template <class T, int dim>
Vector<T, dim> BoxLevelSet<T, dim>::normal(const TV& X)
{
    TV n = TV::Zero();
    T eps = 1e-5;
    T dX = signed_distance(X);
    for (int i = 0; i < dim; i++) {
        TV Xh = X;
        Xh[i] += eps;
        T dXh = signed_distance(Xh);
        n[i] = (dXh - dX) / eps;
    }
    return n;
}

template <class T, int dim>
Vector<T, dim> BoxLevelSet<T, dim>::closest_point(const TV& X)
{
    TV n = normal(X);
    return center - n;
}

//MOVING BOX LEVEL SET
//USE MOVING BOUNDARIES WITH STICKY BC ONLY!!!!
template <class T, int dim>
MovingBoxLevelSet<T, dim>::MovingBoxLevelSet(Type type, const TV& min_corner, const TV& max_corner, const Bow::Vector<T, 4>& rot, TV _v, T _moveTime)
    : AnalyticalLevelSet<T, dim>(type)
{
    center = (min_corner + max_corner) * 0.5;
    half_edges = (max_corner - min_corner) * 0.5;
    R = TM::Identity();
    v = _v;
    move_time = _moveTime;

    if constexpr (dim == 3) {
        Eigen::Quaternion<T> rotation(rot.w(), rot.x(), rot.y(), rot.z());
        R = rotation.toRotationMatrix();
    }
    else {
        T theta = rot.x();
        R << std::cos(theta), -std::sin(theta), std::sin(theta), std::cos(theta);
    }
}

template <class T, int dim>
T MovingBoxLevelSet<T, dim>::signed_distance(const TV& X)
{
    TV X_centered = R.transpose() * (X - center);
    TV d = X_centered.cwiseAbs() - half_edges;
    T dd = d.array().maxCoeff();
    TV qq = d;
    for (int i = 0; i < dim; i++)
        if (qq(i) < (T)0)
            qq(i) = (T)0;
    T result = std::min(dd, (T)0) + qq.norm();
    return result;
}

template <class T, int dim>
Vector<T, dim> MovingBoxLevelSet<T, dim>::normal(const TV& X)
{
    TV n = TV::Zero();
    T eps = 1e-5;
    T dX = signed_distance(X);
    for (int i = 0; i < dim; i++) {
        TV Xh = X;
        Xh[i] += eps;
        T dXh = signed_distance(Xh);
        n[i] = (dXh - dX) / eps;
    }
    return n;
}

template <class T, int dim>
Vector<T, dim> MovingBoxLevelSet<T, dim>::closest_point(const TV& X)
{
    TV n = normal(X);
    return center - n;
}

template <class T, int dim>
Vector<T, dim> MovingBoxLevelSet<T, dim>::velocity(const TV& X)
{
    return v;
}

template <class T, int dim>
T MovingBoxLevelSet<T, dim>::moveTime()
{
    return move_time;
}

template <class T, int dim>
void MovingBoxLevelSet<T, dim>::setVelocity(const TV& newV)
{
    v = newV;
}

template <class T, int dim>
void MovingBoxLevelSet<T, dim>::update(const T& dt)
{
    center += v * dt;
    //USE MOVING BOUNDARIES WITH STICKY BC ONLY!!!!
    //std::cout << "updating BoxLevelSets, velocity:" << v << std::endl;
}

#ifdef BOW_STATIC_LIBRARY
#ifdef BOW_COMPILE_FLOAT
template class HalfSpaceLevelSet<float, 2>;
template class HalfSpaceLevelSet<float, 3>;
template class InverseBallLevelSet<float, 2>;
template class InverseBallLevelSet<float, 3>;
template class MovingBallLevelSet<float, 2>;
template class MovingBallLevelSet<float, 3>;
template class BoxLevelSet<float, 2>;
template class BoxLevelSet<float, 3>;
template class AlignedBoxLevelSet<float, 2>;
template class AlignedBoxLevelSet<float, 3>;
template class MovingBoxLevelSet<float, 2>;
template class MovingBoxLevelSet<float, 3>;
#endif
#ifdef BOW_COMPILE_DOUBLE
template class HalfSpaceLevelSet<double, 2>;
template class HalfSpaceLevelSet<double, 3>;
template class InverseBallLevelSet<double, 2>;
template class InverseBallLevelSet<double, 3>;
template class MovingBallLevelSet<double, 2>;
template class MovingBallLevelSet<double, 3>;
template class BoxLevelSet<double, 2>;
template class BoxLevelSet<double, 3>;
template class AlignedBoxLevelSet<double, 2>;
template class AlignedBoxLevelSet<double, 3>;
template class MovingBoxLevelSet<double, 2>;
template class MovingBoxLevelSet<double, 3>;
#endif
#endif

} // namespace Bow::Geometry
