#ifndef RANDOM_SAMPLING_H
#define RANDOM_SAMPLING_H

#include <Bow/Macros.h>
#include <Bow/Types.h>
#include <Bow/Utils/RandomNumber.h>
#include <Bow/Utils/MultiArray.h>
#include <ctime>

namespace Bow::Geometry {

template <class T, int dim>
class RandomSampling {
public:
    typedef Vector<T, dim> TV;
    typedef Vector<int, dim> IV;
    TV min_corner;
    TV max_corner;
    unsigned count;
    RandomNumber<T> rnd;

    RandomSampling(const TV& min_corner, const TV& max_corner, const unsigned count, const unsigned seed = 256)
        : min_corner(min_corner), max_corner(max_corner), count(count)
    {
        rnd.resetSeed(seed);
        BOW_ASSERT_INFO(min_corner != max_corner, "min_corner == max_corner in RandomSampling");
    }

    ~RandomSampling() {}

public:
    /**
    Samples particles.
     */
    void sample(
        Field<TV>& samples, std::function<bool(TV)> feasible = [](TV) { return true; })
    {
        unsigned n = 0;
        while (n < count) {
            TV x = rnd.randInBox(min_corner, max_corner);
            if (feasible(x)) {
                samples.push_back(x);
                n++;
            }
        }
    }
};

} // namespace Bow::Geometry

#endif
