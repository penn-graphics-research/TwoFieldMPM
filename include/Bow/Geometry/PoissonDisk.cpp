#include "PoissonDisk.h"
#include <Bow/Math/MathTools.h>

namespace Bow::Geometry {

namespace internal {

template <class T, int dim>
struct Box {
    typedef Vector<T, dim> TV;
    // minimum and maximum corner of range
    TV min_corner;
    TV max_corner;
    TV side_length;
    T volume;
    Box(const TV& min_corner, const TV& max_corner)
        : min_corner(min_corner)
        , max_corner(max_corner)
    {
        side_length = max_corner - min_corner;
        volume = side_length.prod();
    }
};

template <int dim>
class MaxExclusiveBoxIterator {
public:
    typedef Vector<int, dim> IV;
    Box<int, dim> box;
    IV index;
    MaxExclusiveBoxIterator(const Box<int, dim>& box)
        : box(box)
    {
        index = box.min_corner;
    }
    // Prefix increment
    MaxExclusiveBoxIterator& operator++()
    {
        if (index == box.max_corner)
            return *this;
        else {
            int d = dim - 1;
            do {
                index(d)++;
                if (valid())
                    return *this;
                index(d) = box.min_corner(d);
                d--;
            } while (d > -1);
            index = box.max_corner;
            return *this;
        }
    }
    // Postfix increment
    MaxExclusiveBoxIterator operator++(int)
    {
        MaxExclusiveBoxIterator ret(*this);
        ++(*this);
        return ret;
    }
    bool valid()
    {
        bool ret = true;
        for (int d = 0; d < dim; ++d) {
            ret &= (index(d) >= box.min_corner(d) && index(d) < box.max_corner(d));
        }
        return ret;
    }
    int& operator()(int d)
    {
        return index(d);
    }
};
} // namespace internal

template <class T, int dim>
PoissonDisk<T, dim>::PoissonDisk(const TV& min_corner, const TV& max_corner, const T min_distance, const unsigned seed, int max_attempts, bool periodic)
    : min_distance(min_distance)
    , max_attempts(max_attempts)
    , h(min_distance / std::sqrt((T)dim))
    , min_corner(min_corner)
    , max_corner(max_corner)
    , periodic(periodic)
{
    rnd.resetSeed(seed);
    BOW_ASSERT_INFO(min_corner != max_corner, "min_corner == max_corner in PoissonDisk");
}

template <class T, int dim>
PoissonDisk<T, dim>::PoissonDisk(const TV& min_corner, const TV& max_corner, const T dx, const T ppc, const unsigned seed, int max_attempts, bool periodic)
    : max_attempts(max_attempts)
    , min_corner(min_corner)
    , max_corner(max_corner)
    , periodic(periodic)
{
    rnd.resetSeed(seed);
    set_distance_by_ppc(dx, ppc);
    BOW_ASSERT_INFO(min_corner != max_corner, "min_corner == max_corner in PoissonDisk");
}

template <class T, int dim>
PoissonDisk<T, dim>::~PoissonDisk()
{
}

template <class T, int dim>
typename PoissonDisk<T, dim>::IV PoissonDisk<T, dim>::world_to_index_space(const TV& X) const
{
    IV ijk;
    for (size_t d = 0; d < dim; ++d)
        // ijk(d) = (int)std::floor((X(d) - min_corner(d)) / h);
        ijk(d) = MATH_TOOLS::int_floor((X(d) - min_corner(d)) / h);

    return ijk;
}

template <class T, int dim>
typename PoissonDisk<T, dim>::TV PoissonDisk<T, dim>::generate_random_point_around_annulus(const TV& center)
{
    while (true) {
        TV v;
        rnd.fill(v, -1, 1);
        T mag2 = v.dot(v);
        if (mag2 >= 0.25 && mag2 <= 1)
            return v * min_distance * 2 + center;
    }
    return TV::Zero();
}

template <class T, int dim>
bool PoissonDisk<T, dim>::check_distance(const TV& point, const MultiArray<int, dim>& background_grid, const Field<TV>& samples) const
{
    IV index = world_to_index_space(point);
    // Check if we are outside of the background_grid. If so, return false
    for (int d = 0; d < dim; ++d) {
        if (index(d) < 0 || index(d) >= background_grid.size(d))
            return false;
    }
    // If there is already a particle in that cell, return false
    if (background_grid(index) != -1)
        return false;
    T min_distance_sqr = min_distance * min_distance;
    IV local_min_index = index.array() - 2;
    IV local_max_index = index.array() + 3;
    // If not periodic, clamp local_min_index and local_max_index to the size of the background grid
    if (!periodic) {
        local_min_index = local_min_index.cwiseMax(0);
        local_max_index = local_max_index.cwiseMin(background_grid.size);
    }
    // Create local_box for iterator purposes
    internal::Box<int, dim> local_box(local_min_index, local_max_index);
    if (!periodic)
        for (internal::MaxExclusiveBoxIterator<dim> it(local_box); it.valid(); ++it) {
            if (background_grid(it.index) == -1)
                continue;
            TV x = point - samples[background_grid(it.index)];
            if (x.dot(x) < min_distance_sqr) {
                return false;
            }
        }
    else {
        for (internal::MaxExclusiveBoxIterator<dim> it(local_box); it.valid(); ++it) {
            IV local_index = it.index;
            // Need to shift point in MultiArray if one of the indices is negative or greater than background_grid.size
            TV shift = TV::Zero();
            for (int d = 0; d < dim; ++d) {
                // If it.index < 0 update local index to all the way down to the right/top/back
                // If there is a point in that MultiArray index, we need to shift that point to the left/bottom/front
                if (it.index(d) < 0) {
                    local_index(d) = local_index(d) % background_grid.size(d) + background_grid.size(d);
                    shift(d) = min_corner(d) - max_corner(d);
                }
                // If it.index(d) >= background_grid(d) update local index to all the way down to the left/bottom/front
                // If there is a point in that MultiArray index, we need to shift that point to the left right/top/back
                else if (it.index(d) >= background_grid.size(d)) {
                    local_index(d) = local_index(d) % background_grid.size(d);
                    shift(d) = max_corner(d) - min_corner(d);
                }
            }
            if (background_grid(local_index) == -1)
                continue;
            TV x = point - (samples[background_grid(local_index)] + shift);
            if (x.dot(x) < min_distance_sqr) {
                return false;
            }
        }
    }
    return true;
}

template <class T, int dim>
void PoissonDisk<T, dim>::set_distance_by_ppc(T dx, T ppc)
{
    T v = std::pow(dx, dim) / (T)ppc;
    if (dim == 2) {
        min_distance = std::sqrt(v * ((T)2 / 3));
    }
    else if (dim == 3) {
        min_distance = std::pow(v * ((T)13 / 18), (T)1 / 3);
    }
    else {
        BOW_ASSERT_INFO(false, "Poisson disk only supports 2D and 3D");
    }
    h = min_distance / std::sqrt((T)dim);
}

template <class T, int dim>
void PoissonDisk<T, dim>::sample(Field<TV>& samples_out, std::function<bool(TV)> feasible)
{
    /*
    Set up background grid
    dx should be bounded by min_distance / sqrt(dim)
    background_grid is a MultiArray which keeps tracks the indices of points in samples.
    the value of background_grid is initialized to be -1, meaning that there is no particle
    in that cell
    */
    Field<TV> samples = samples_out;
    TV cell_numbers_candidate = max_corner - min_corner;
    IV cell_numbers;
    for (int d = 0; d < dim; ++d)
        cell_numbers(d) = std::ceil(cell_numbers_candidate(d) / h);
    MultiArray<int, dim> background_grid(cell_numbers, -1);
    // Set up active list
    std::vector<int> active_list;

    /*
    If X already contains points, then set the values of the background grid
    to keep track of these points
    */
    if (samples.size()) {
        for (size_t i = 0; i < samples.size(); ++i) {
            // Change the value of background_grid in that index to be i
            background_grid(world_to_index_space(samples[i])) = i;
            active_list.push_back(i);
        }
    }
    else {
        // Generate a random point within the range and append it to samples and active list
        TV first_point = (T).5 * (max_corner + min_corner);
        while (!feasible(first_point)) {
            for (int d = 0; d < dim; ++d) {
                T r = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
                first_point(d) = min_corner(d) + (max_corner(d) - min_corner(d)) * r;
            }
        }
        samples.push_back(first_point);
        background_grid(world_to_index_space(first_point)) = 0;
        active_list.push_back(0);
    }

    /*
    While active_list is non-zero, do step 2 in Bridson's proposed algorithm
    */
    while (active_list.size()) {
        // Get a random index from the active list and find the point corresponding to it
        int random_index = rnd.randInt(0, active_list.size() - 1);
        TV current_point = samples[active_list[random_index]];
        // Swap random index with the last element in the active list so that we can pop_back
        // if found_at_least_one is false at the end of this procedure
        iter_swap(active_list.begin() + random_index, active_list.end() - 1);
        // Generate up to max_attempts points in the annulus of radius r and 2r around current_point
        bool found_at_least_one = false;
        for (int i = 0; i < max_attempts; ++i) {
            TV new_point = generate_random_point_around_annulus(current_point);

            // If periodic and new_point is outside of the min_corner, max_corner shift it to be inside
            if (periodic) {
                for (int d = 0; d < dim; ++d) {
                    if (new_point(d) < min_corner(d))
                        new_point(d) += max_corner(d) - min_corner(d);
                    else if (new_point(d) > max_corner(d))
                        new_point(d) -= max_corner(d) - min_corner(d);
                }
            }

            bool outside = false;
            for (int d = 0; d < dim; ++d) {
                if (new_point(d) < min_corner(d) || new_point(d) > max_corner(d)) {
                    outside = true;
                    break;
                }
            }

            if (outside || !feasible(new_point))
                continue;

            if (check_distance(new_point, background_grid, samples)) {
                found_at_least_one = true;
                // Add new_point to samples
                samples.push_back(new_point);
                int index = samples.size() - 1;
                // Add new_point to active list
                active_list.push_back(index);
                // Update background_grid
                background_grid(world_to_index_space(new_point)) = index;
            }
        }
        // If not found at least one, remove random_index from active list
        if (!found_at_least_one) { //active_list.erase(active_list.begin()+random_index);}
            active_list.pop_back();
        }
    }

    // Remove points that are outside of Box(min_corner,max_corner)
    for (auto& point : samples) {
        bool outside = false;
        for (int d = 0; d < dim; ++d) {
            if (point(d) < min_corner(d) || point(d) > max_corner(d)) {
                outside = true;
                break;
            }
        }
        if (!outside)
            samples_out.push_back(point);
    }
}

#ifdef BOW_STATIC_LIBRARY
#ifdef BOW_COMPILE_FLOAT
template class PoissonDisk<float, 2>;
template class PoissonDisk<float, 3>;
#endif
#ifdef BOW_COMPILE_DOUBLE
template class PoissonDisk<double, 2>;
template class PoissonDisk<double, 3>;
#endif
#endif

} // namespace Bow::Geometry
