#ifndef POISSON_DISK_H
#define POISSON_DISK_H

#include <Bow/Macros.h>
#include <Bow/Types.h>
#include <Bow/Utils/RandomNumber.h>
#include <Bow/Utils/MultiArray.h>
#include <ctime>

namespace Bow::Geometry {

/*
https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf 
*/

template <class T, int dim>
class PoissonDisk {
public:
    typedef Vector<T, dim> TV;
    typedef Vector<int, dim> IV;
    T min_distance; // the minimum distance r between samples
    int max_attempts;
    T h; // background cell size
    TV min_corner;
    TV max_corner;
    bool periodic;
    RandomNumber<T> rnd;

    PoissonDisk(const TV& min_corner, const TV& max_corner, const T min_distance, const unsigned seed = 123, int max_attempts = 30, bool periodic = false);
    PoissonDisk(const TV& min_corner, const TV& max_corner, const T dx, const T ppc, const unsigned seed = 123, int max_attempts = 30, bool periodic = false);
    ~PoissonDisk();

private:
    /**
    Convert position in the world space to its corresponding index space.
     */
    IV world_to_index_space(const TV& X) const;
    /**
    Generate one random point around center with a distance between .5 and 1 from center.
     */
    TV generate_random_point_around_annulus(const TV& center);
    /**
    Return true if the distance between the candidate point and any other points in samples are sufficiently far away (at least min_distance away), and false otherwise.
     */
    bool check_distance(const TV& point, const MultiArray<int, dim>& background_grid, const Field<TV>& samples) const;
    /**
    Set min_distance between particles based on the number of particles per cell and grid dx.
    ppc = particles per cell. 
     */
    void set_distance_by_ppc(T dx, T ppc);

public:
    /**
    Samples particles.
     */
    void sample(
        Field<TV>& samples, std::function<bool(TV)> feasible = [](TV) { return true; });
};

} // namespace Bow::Geometry

#ifndef BOW_STATIC_LIBRARY
#include "PoissonDisk.cpp"
#endif

#endif
