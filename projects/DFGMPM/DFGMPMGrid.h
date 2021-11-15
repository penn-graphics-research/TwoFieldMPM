#pragma once

#include <Bow/Macros.h>
#include <Bow/Types.h>
#include <Bow/Utils/Logging.h>
#include <Bow/Utils/Timer.h>
//#include <Bow/Physics/ConstitutiveModel.h>
#include <Bow/Math/BSpline.h>
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include <tbb/tbb.h>
#include <memory>

using namespace SPGrid;

namespace Bow {
namespace DFGMPM {

template <class T, int dim>
class GridState {
public:
    //Grid Node Position (We will advect this directly)
    Vector<T, dim> x1;
    Vector<T, dim> x2;
    
    //Two Field Simulation Structures
    T m1, m2; //mass
    Vector<T, 2> d1, d2; //damage
    Vector<T, dim> v1, v2; //velocity
    Vector<T, dim> vn1, vn2; //v^n
    Vector<T, dim> a1, a2; //acceleration used for implicit
    Vector<T, dim> n1, n2; //normals
    Vector<T, dim> fct1, fct2; //contact force
    Vector<T, dim> fi1, fi2; //impulse force

    //To get grid def grad, F_i, transfer singular values and quaternion coefficients for U and V rotations from SVD!
    Vector<T, dim> sigma1, sigma2;
    Vector<T, 4> Uquat1, Uquat2;
    Vector<T, 4> Vquat1, Vquat2;

    //To transfer cauchy stress and def grad to the grid
    Matrix<T, dim, dim> cauchy1, cauchy2;
    Matrix<T, dim, dim> Fi1, Fi2;

    //DOF Tracking
    typename std::conditional<std::is_same<T, float>::value, int32_t, int64_t>::type idx;
    typename std::conditional<std::is_same<T, float>::value, int32_t, int64_t>::type sep_idx;

    //DFG Specific Structures
    Vector<T, dim> gridDG; //damage gradient
    T gridMaxNorm; // maximum DG norm found for this grid node, this will later help to determine the grid DG
    Vector<T, 4> gridSeparability; //each grid node has a seperability condition for each field and we need to add up the numerator and denominator
    Vector<T, 2> gridMaxDamage; //store max damage from each field mapping to this node
    int separable; //0 = single field, 1 = two field

    //Barrier Functions (Implicit Only)
    T gridViYi1, gridViYi2, gridCi;

    //Neighbor Search
    std::vector<int> mappedParticles; //a list of particles that map to this grid node

    //PADDING to ensure GridState is power of 2
    //NOTE: if we already had a power of two, need to pad to the next one up still because can't conditionally do padding = 0 B
    
    //AFTER ADDING cauchy and Fi 10/5/21
    //Float2D: 384 B -> add 128 B -> 32 Ts
    //Float3D: 528 B -> add 496 B -> 124 Ts
    //Double2D: 768 B -> add 256 B -> 32 Ts
    //Double3D: 1056 B -> add 992 B -> 124 Ts
    Vector<T, (92 * dim) - 152> padding; //dim2 = 32 Ts, dim3 = 124 Ts --> y = 92x - 152

    GridState()
    {
        m1 = 0.0;
        m2 = 0.0;
        d1 = Vector<T, 2>::Zero();
        d2 = Vector<T, 2>::Zero();
        v1 = Vector<T, dim>::Zero();
        v2 = Vector<T, dim>::Zero();
        vn1 = Vector<T, dim>::Zero();
        vn2 = Vector<T, dim>::Zero();
        n1 = Vector<T, dim>::Zero();
        n2 = Vector<T, dim>::Zero();
        fct1 = Vector<T, dim>::Zero();
        fct2 = Vector<T, dim>::Zero();
        fi1 = Vector<T, dim>::Zero();
        fi2 = Vector<T, dim>::Zero();
        sigma1 = Vector<T, dim>::Zero();
        sigma2 = Vector<T, dim>::Zero();
        Uquat1 = Vector<T, 4>::Zero();
        Uquat2 = Vector<T, 4>::Zero();
        Vquat1 = Vector<T, 4>::Zero();
        Vquat2 = Vector<T, 4>::Zero();
        cauchy1 = Matrix<T, dim, dim>::Zero();
        cauchy2 = Matrix<T, dim, dim>::Zero();
        Fi1 = Matrix<T, dim, dim>::Zero();
        Fi2 = Matrix<T, dim, dim>::Zero();
        gridDG = Vector<T, dim>::Zero();
        gridMaxNorm = 0.0;
        gridSeparability = Vector<T, 4>::Zero();
        gridMaxDamage = Vector<T, 2>::Zero();
        separable = 0;
        gridViYi1 = 0.0;
        gridViYi2 = 0.0;
        gridCi = 0.0;
        mappedParticles.clear();

        idx = -1; //dummy idx
    }

    void getSizes()
    {
        std::cout << sizeof(m1) << std::endl;
        std::cout << sizeof(v1) << std::endl;
        std::cout << sizeof(d1) << std::endl;
        std::cout << sizeof(idx) << std::endl;
    }
};

inline constexpr bool is_power_of_two(size_t x)
{
    return x > 0 && (x & (x - 1)) == 0;
}
static_assert(is_power_of_two(sizeof(GridState<float, 2>)), "GridState<float, 2> size must be POT");
static_assert(is_power_of_two(sizeof(GridState<float, 3>)), "GridState<float, 3> size must be POT");
static_assert(is_power_of_two(sizeof(GridState<double, 2>)), "GridState<double, 2> size must be POT");
static_assert(is_power_of_two(sizeof(GridState<double, 3>)), "GridState<double, 3> size must be POT");

template <typename OP>
void iterateRegion(const Vector<int, 2>& region, const OP& target)
{
    for (int i = 0; i < region[0]; ++i)
        for (int j = 0; j < region[1]; ++j)
            target(Vector<int, 2>(i, j));
}

template <typename OP>
void iterateRegion(const Vector<int, 3>& region, const OP& target)
{
    for (int i = 0; i < region[0]; ++i)
        for (int j = 0; j < region[1]; ++j)
            for (int k = 0; k < region[2]; ++k)
                target(Vector<int, 3>(i, j, k));
}

template <class T, int dim>
class DFGMPMGrid {
public:
    static constexpr int log2_page = 12;
    // TODO: adaptive spgrid_size to accelerate particles sorting
    static constexpr int spgrid_size = 2048;
    static constexpr int half_spgrid_size = spgrid_size / 2;
    static constexpr int interpolation_degree = 2;
    static constexpr int kernel_size = (dim == 2)
        ? (interpolation_degree + 1) * (interpolation_degree + 1)
        : (interpolation_degree + 1) * (interpolation_degree + 1) * (interpolation_degree + 1);
    using SparseGrid = SPGrid_Allocator<GridState<T, dim>, dim, log2_page>;
    using SparseMask = typename SparseGrid::template Array_type<>::MASK;
    using PageMap = SPGrid_Page_Map<log2_page>;
    // Linear_Offset() is not a constexpr function to initialize this
    const uint64_t origin_offset;

    std::unique_ptr<SparseGrid> grid;
    std::unique_ptr<PageMap> page_map;
    std::unique_ptr<PageMap> fat_page_map;
    int num_nodes;
    int separable_nodes;

    bool useDFG = false;

    std::vector<int> particle_fence;
    std::vector<uint64_t> particle_sorter;

    //These are for CRAMP implementations using DFGMPMGrid
    int crackParticlesStartIdx = 1000000; //1 million to start
    bool crackInitialized = false;
    bool horizontalCrack = false;

public:
    DFGMPMGrid()
        : origin_offset(SparseMask::Linear_Offset(to_std_array<int, dim>(Vector<int, dim>(Vector<int, dim>::Ones() * half_spgrid_size).data())))
    {
        if constexpr (dim == 2) {
            grid = std::make_unique<SparseGrid>(spgrid_size, spgrid_size);
        }
        else {
            grid = std::make_unique<SparseGrid>(spgrid_size, spgrid_size, spgrid_size);
        }
        page_map = std::make_unique<PageMap>(*grid);
        fat_page_map = std::make_unique<PageMap>(*grid);
    }

    void sortParticles(const Field<Vector<T, dim>>& positions, T dx)
    {
        BOW_TIMER_FLAG("sort particles");
        constexpr int index_bits = (32 - SparseMask::block_bits);
        particle_sorter.resize(positions.size());
        auto grid_array = grid->Get_Array();
        {
            BOW_TIMER_FLAG("prepare array to sort");
            BOW_ASSERT(positions.size() <= (1u << index_bits));
            tbb::parallel_for(0, (int)positions.size(), [&](int i) {
                BSplineWeights<T, dim> spline(positions[i], dx);
                uint64_t biased_offset = SparseMask::Linear_Offset(to_std_array<int, dim>(spline.base_node.data()));
                uint64_t offset = SparseMask::Packed_Add(biased_offset, origin_offset);
                particle_sorter[i] = ((offset >> SparseMask::data_bits) << index_bits) + i;
            });
        }
        {
            BOW_TIMER_FLAG("tbb parallel sort");
            tbb::parallel_sort(particle_sorter.begin(), particle_sorter.begin() + particle_sorter.size());
        }
        {
            BOW_TIMER_FLAG("reset page map");
            page_map->Clear();
            for (int i = 0; i < (int)particle_sorter.size(); i++) {
                uint64_t offset = (particle_sorter[i] >> index_bits) << SparseMask::data_bits;
                page_map->Set_Page(offset);
            }
            page_map->Update_Block_Offsets();
        }
        // Update particle offset
        auto blocks = page_map->Get_Blocks();
        {
            BOW_TIMER_FLAG("fat page map");
            // Reset fat_page_map
            fat_page_map->Clear();
            for (int b = 0; b < (int)blocks.second; b++) {
                auto base_offset = blocks.first[b];
                if constexpr (dim == 2) {
                    auto x = 1 << SparseMask::block_xbits;
                    auto y = 1 << SparseMask::block_ybits;
                    auto c = SparseMask::LinearToCoord(base_offset);
                    for (int i = -1 + (c[0] == 0); i < 2; i++) {
                        for (int j = -1 + (c[1] == 0); j < 2; j++) {
                            fat_page_map->Set_Page(SparseMask::Packed_Add(
                                base_offset, SparseMask::Linear_Offset(x * i, y * j)));
                        }
                    }
                }
                else {
                    auto x = 1 << SparseMask::block_xbits;
                    auto y = 1 << SparseMask::block_ybits;
                    auto z = 1 << SparseMask::block_zbits;
                    auto c = SparseMask::LinearToCoord(base_offset);
                    for (int i = -1 + (c[0] == 0); i < 2; i++) {
                        for (int j = -1 + (c[1] == 0); j < 2; j++) {
                            for (int k = -1 + (c[2] == 0); k < 2; k++) {
                                fat_page_map->Set_Page(SparseMask::Packed_Add(
                                    base_offset, SparseMask::Linear_Offset(x * i, y * j, z * k)));
                            }
                        }
                    }
                }
            }
            fat_page_map->Update_Block_Offsets();
        }
        auto fat_blocks = fat_page_map->Get_Blocks();
        {
            BOW_TIMER_FLAG("reset grid");
            for (int i = 0; i < (int)fat_blocks.second; ++i) {
                auto offset = fat_blocks.first[i];
                std::memset((T*)&grid_array(offset), 0, 1 << log2_page);
                auto* g = reinterpret_cast<GridState<T, dim>*>(&grid_array(offset));
                for (int k = 0; k < (int)SparseMask::elements_per_block; ++k)
                    g[k].idx = -1;
            }
        }
        {
            BOW_TIMER_FLAG("block particle offset");
            particle_fence.clear();
            uint64_t last_offset = -1;
            for (uint32_t i = 0; i < particle_sorter.size(); i++) {
                if (last_offset != (particle_sorter[i] >> 32)) particle_fence.push_back(i);
                last_offset = particle_sorter[i] >> 32;
                particle_sorter[i] &= ((1ll << index_bits) - 1);
            }
            particle_fence.push_back((uint32_t)particle_sorter.size());
        }
        num_nodes = 0;
    }

    void countNumNodes()
    {
        num_nodes = 0;
        auto blocks = fat_page_map->Get_Blocks();
        auto grid_array = grid->Get_Array();
        for (int b = 0; b < (int)blocks.second; ++b) {
            GridState<T, dim>* g = reinterpret_cast<GridState<T, dim>*>(&grid_array(blocks.first[b]));
            for (int i = 0; i < (int)SparseMask::elements_per_block; ++i)
                if (g[i].m1 > 0)
                    g[i].idx = num_nodes++;
        }
        //std::cout << "num active nodes: " << num_nodes << std::endl;
    }

    void countSeparableNodes()
    {
        separable_nodes = 0;
        auto blocks = fat_page_map->Get_Blocks();
        auto grid_array = grid->Get_Array();
        for (int b = 0; b < (int)blocks.second; ++b) {
            GridState<T, dim>* g = reinterpret_cast<GridState<T, dim>*>(&grid_array(blocks.first[b]));
            for (int i = 0; i < (int)SparseMask::elements_per_block; ++i)
                if (g[i].m1 > 0 && g[i].separable == 1)
                    g[i].sep_idx = separable_nodes++;
        }
        //std::cout << "num sep nodes: " << separable_nodes << std::endl;
    }

    GridState<T, dim>& operator[](const Vector<int, dim>& v)
    {
        return grid->Get_Array()(to_std_array<int, dim>(Vector<int, dim>(v + Vector<int, dim>::Ones() * half_spgrid_size).data()));
    }

    const GridState<T, dim>& operator[](const Vector<int, dim>& v) const
    {
        return grid->Get_Array()(to_std_array<int, dim>(Vector<int, dim>(v + Vector<int, dim>::Ones() * half_spgrid_size).data()));
    }

    template <typename OP>
    inline void colored_for(const OP& target)
    {
        auto blocks = page_map->Get_Blocks();
        for (uint32_t color = 0; color < (1 << dim); ++color)
            tbb::parallel_for(size_t(0), size_t(blocks.second), [&](size_t b) {
                if (((blocks.first[b] >> log2_page) & ((1 << dim) - 1)) != color)
                    return;
                for (int idx = particle_fence[b]; idx < particle_fence[b + 1]; ++idx) {
                    int i = particle_sorter[idx];
                    target(i);
                }
            });
    }

    template <typename OP>
    inline void parallel_for(const OP& target)
    {
        auto blocks = page_map->Get_Blocks();
        tbb::parallel_for(size_t(0), size_t(blocks.second), [&](size_t b) {
            for (int idx = particle_fence[b]; idx < particle_fence[b + 1]; ++idx) {
                int i = particle_sorter[idx];
                target(i);
            }
        });
    }

    template <typename OP>
    inline void serial_for(const OP& target)
    {
        auto blocks = page_map->Get_Blocks();
        for (size_t b = 0; b < blocks.second; ++b) {
            for (int idx = particle_fence[b]; idx < particle_fence[b + 1]; ++idx) {
                int i = particle_sorter[idx];
                target(i);
            }
        }
    }

    inline bool existsNode(const Vector<int, dim>& node)
    {
        uint64_t biased_offset = SparseMask::Linear_Offset(to_std_array<int, dim>(node.data()));
        uint64_t offset = SparseMask::Packed_Add(biased_offset, origin_offset);
        return fat_page_map->Test_Page(offset);
    }

    inline bool existsKernel(const BSplineWeights<T, dim, interpolation_degree>& spline)
    {
        uint64_t biased_offset = SparseMask::Linear_Offset(to_std_array<int, dim>(spline.base_node.data()));
        uint64_t base_offset = SparseMask::Packed_Add(biased_offset, origin_offset);
        if constexpr (dim == 2) {
            for (int i = 0; i < interpolation_degree + 1; ++i) {
                for (int j = 0; j < interpolation_degree + 1; ++j) {
                    auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j));
                    if (!fat_page_map->Test_Page(offset)) return false;
                }
            }
        }
        else {
            for (int i = 0; i < interpolation_degree + 1; ++i) {
                for (int j = 0; j < interpolation_degree + 1; ++j) {
                    for (int k = 0; k < interpolation_degree + 1; ++k) {
                        auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j, k));
                        if (!fat_page_map->Test_Page(offset)) return false;
                    }
                }
            }
        }
        return true;
    }

    template <typename OP>
    inline void iterateKernel(const BSplineWeights<T, dim, interpolation_degree>& spline, const OP& target)
    {
        uint64_t biased_offset = SparseMask::Linear_Offset(to_std_array<int, dim>(spline.base_node.data()));
        uint64_t base_offset = SparseMask::Packed_Add(biased_offset, origin_offset);
        auto grid_array = grid->Get_Array();
        T one_over_dx = spline.one_over_dx;
        auto& w = spline.w;
        auto& dw = spline.dw;
        const Vector<int, dim>& base_coord = spline.base_node;
        Vector<int, dim> coord;
        int oidx = 0;
        if constexpr (dim == 2) {
            for (int i = 0; i < interpolation_degree + 1; ++i) {
                T wi = w[0](i);
                T dwidxi = one_over_dx * dw[0](i);
                coord[0] = base_coord[0] + i;
                for (int j = 0; j < interpolation_degree + 1; ++j) {
                    T wj = w[1](j);
                    T wij = wi * wj;
                    T dwijdxi = dwidxi * wj;
                    T dwijdxj = wi * one_over_dx * dw[1](j);
                    coord[1] = base_coord[1] + j;
                    auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j));
                    GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                    target(coord, oidx, wij, Vector<T, dim>{ dwijdxi, dwijdxj }, g);
                    ++oidx;
                }
            }
        }
        else {
            for (int i = 0; i < interpolation_degree + 1; ++i) {
                T wi = w[0](i);
                T dwidxi = one_over_dx * dw[0](i);
                coord[0] = base_coord[0] + i;
                for (int j = 0; j < interpolation_degree + 1; ++j) {
                    T wj = w[1](j);
                    T wij = wi * wj;
                    T dwijdxi = dwidxi * wj;
                    T dwijdxj = wi * one_over_dx * dw[1](j);
                    coord[1] = base_coord[1] + j;
                    for (int k = 0; k < interpolation_degree + 1; ++k) {
                        coord[2] = base_coord[2] + k;
                        T wk = w[2](k);
                        T wijk = wij * wk;
                        T wijkdxi = dwijdxi * wk;
                        T wijkdxj = dwijdxj * wk;
                        T wijkdxk = wij * one_over_dx * dw[2](k);
                        auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j, k));
                        GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                        target(coord, oidx, wijk, Vector<T, dim>{ wijkdxi, wijkdxj, wijkdxk }, g);
                        ++oidx;
                    }
                }
            }
        }
    }

    template <typename OP>
    inline void iterateKernelWithLaplacian(const BSplineWeightsWithSecondOrder<T, dim, interpolation_degree>& spline, const OP& target)
    {
        uint64_t biased_offset = SparseMask::Linear_Offset(to_std_array<int, dim>(spline.base_node.data()));
        uint64_t base_offset = SparseMask::Packed_Add(biased_offset, origin_offset);
        auto grid_array = grid->Get_Array();
        T one_over_dx = spline.one_over_dx;
        auto& w = spline.w;
        auto& dw = spline.dw;
        auto& ddw = spline.ddw;
        const Vector<int, dim>& base_coord = spline.base_node;
        Vector<int, dim> coord;
        int oidx = 0;
        if constexpr (dim == 2) {
            for (int i = 0; i < interpolation_degree + 1; ++i) {
                T wi = w[0](i);
                T dwidxi = one_over_dx * dw[0](i);
                coord[0] = base_coord[0] + i;
                for (int j = 0; j < interpolation_degree + 1; ++j) {
                    T wj = w[1](j);
                    T wij = wi * wj;
                    T dwijdxi = dwidxi * wj;
                    T dwijdxj = wi * one_over_dx * dw[1](j);

                    //Compute Laplacian
                    T ddwiwj = one_over_dx * one_over_dx * ddw[0](i) * wj;
                    T dwidwj = dwidxi * one_over_dx * dw[1](j);
                    T widdwj = one_over_dx * one_over_dx * wi * ddw[1](j);
                    Matrix<T, dim, dim> ddw;
                    ddw << ddwiwj, dwidwj, dwidwj, widdwj;
                    T laplacian = ddw(0, 0) + ddw(1, 1); //laplacian is the trace of the hessian

                    coord[1] = base_coord[1] + j;
                    auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j));
                    GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                    target(coord, oidx, wij, Vector<T, dim>{ dwijdxi, dwijdxj }, laplacian, g);
                    ++oidx;
                }
            }
        }
        else {
            for (int i = 0; i < interpolation_degree + 1; ++i) {
                T wi = w[0](i);
                T dwidxi = one_over_dx * dw[0](i);
                coord[0] = base_coord[0] + i;
                for (int j = 0; j < interpolation_degree + 1; ++j) {
                    T wj = w[1](j);
                    T wij = wi * wj;
                    T dwijdxi = dwidxi * wj;
                    T dwijdxj = wi * one_over_dx * dw[1](j);
                    coord[1] = base_coord[1] + j;
                    for (int k = 0; k < interpolation_degree + 1; ++k) {
                        coord[2] = base_coord[2] + k;
                        T wk = w[2](k);
                        T wijk = wij * wk;
                        T wijkdxi = dwijdxi * wk;
                        T wijkdxj = dwijdxj * wk;
                        T wijkdxk = wij * one_over_dx * dw[2](k);

                        //Compute laplacian
                        T first_term_in_laplacian = one_over_dx * one_over_dx * ddw[0](i) * wj * wk;
                        T second_term_in_laplacian = one_over_dx * one_over_dx * ddw[1](i) * wi * wk;
                        T third_term_in_laplacian = one_over_dx * one_over_dx * ddw[2](i) * wi * wj;
                        T laplacian = first_term_in_laplacian + second_term_in_laplacian + third_term_in_laplacian;

                        auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j, k));
                        GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                        target(coord, oidx, wijk, Vector<T, dim>{ wijkdxi, wijkdxj, wijkdxk }, laplacian, g);
                        ++oidx;
                    }
                }
            }
        }
    }

    template <typename OP>
    inline void iterateKernelWithoutWeights(const BSplineWeights<T, dim, interpolation_degree>& spline, const OP& target)
    {
        uint64_t biased_offset = SparseMask::Linear_Offset(to_std_array<int, dim>(spline.base_node.data()));
        uint64_t base_offset = SparseMask::Packed_Add(biased_offset, origin_offset);
        auto grid_array = grid->Get_Array();
        const Vector<int, dim>& base_coord = spline.base_node;
        Vector<int, dim> coord;
        if constexpr (dim == 2) {
            for (int i = 0; i < interpolation_degree + 1; ++i) {
                coord[0] = base_coord[0] + i;
                for (int j = 0; j < interpolation_degree + 1; ++j) {
                    coord[1] = base_coord[1] + j;
                    auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j));
                    GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                    target(coord, g);
                }
            }
        }
        else {
            for (int i = 0; i < interpolation_degree + 1; ++i) {
                coord[0] = base_coord[0] + i;
                for (int j = 0; j < interpolation_degree + 1; ++j) {
                    coord[1] = base_coord[1] + j;
                    for (int k = 0; k < interpolation_degree + 1; ++k) {
                        coord[2] = base_coord[2] + k;
                        auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j, k));
                        GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                        target(coord, g);
                    }
                }
            }
        }
    }

    template <typename OP>
    inline void iterateNeighbors(const BSplineWeights<T, dim, interpolation_degree>& spline, const OP& target)
    {
        uint64_t biased_offset = SparseMask::Linear_Offset(to_std_array<int, dim>(spline.base_node.data()));
        uint64_t base_offset = SparseMask::Packed_Add(biased_offset, origin_offset);
        auto grid_array = grid->Get_Array();
        const Vector<int, dim>& base_coord = spline.base_node;
        Vector<int, dim> coord;
        if constexpr (dim == 2) {
            for (int i = -1; i < 2; ++i) {
                coord[0] = base_coord[0] + i;
                for (int j = -1; j < 2; ++j) {
                    coord[1] = base_coord[1] + j;
                    auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j));
                    GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                    target(coord, g);
                }
            }
        }
        else {
            for (int i = -1; i < 2; ++i) {
                coord[0] = base_coord[0] + i;
                for (int j = -1; j < 2; ++j) {
                    coord[1] = base_coord[1] + j;
                    for (int k = -1; k < 2; ++k) {
                        coord[2] = base_coord[2] + k;
                        auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j, k));
                        GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                        target(coord, g);
                    }
                }
            }
        }
    }

    //Iterate a rectangular contour defined by 4 integers: L, M, N, O (L left of center, D down from center, R right of center, U up from center)
    template <typename OP>
    inline void iterateRectangularContour(const BSplineWeights<T, dim, interpolation_degree>& spline, int L, int D, int R, int U, const OP& target)
    {
        uint64_t biased_offset = SparseMask::Linear_Offset(to_std_array<int, dim>(spline.closest_node.data()));
        uint64_t base_offset = SparseMask::Packed_Add(biased_offset, origin_offset);
        auto grid_array = grid->Get_Array();
        const Vector<int, dim>& base_coord = spline.closest_node;
        //std::cout << "base_coord:" << base_coord << std::endl;
        Vector<int, dim> coord;
        if constexpr (dim == 2) {
            
            //Left side, starting with top left
            coord[0] = base_coord[0] - L;
            for (int j = U; j > (-1 * D) - 1; --j) {
                coord[1] = base_coord[1] + j;
                auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(-1 * L, j));
                GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                target(coord, g);
            }

            //Bottom (minus the left most)
            coord[1] = base_coord[1] - D;
            for (int i = -1 * L + 1; i < R + 1; ++i) {
                coord[0] = base_coord[0] + i;
                auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, -1 * D));
                GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                target(coord, g);
            }

            //Right side (minus the bottom most) -> setup to continue the contour counter-clockwise
            coord[0] = base_coord[0] + R;
            for (int j = (-1 * D) + 1; j < U + 1 ; ++j) {
                coord[1] = base_coord[1] + j;
                auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(R, j));
                GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                target(coord, g);
            }

            //Top (minus the right and left most) -> counter clockwise
            coord[1] = base_coord[1] + U;
            for (int i = R - 1; i > -1 * L; --i) {
                coord[0] = base_coord[0] + i;
                auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, U));
                GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                target(coord, g);
            }


        }
        else {
            std::cout << "3D J Integral Computation Not Implemented" << std::endl;
            // for (int i = -1; i < 2; ++i) {
            //     coord[0] = base_coord[0] + i;
            //     for (int j = -1; j < 2; ++j) {
            //         coord[1] = base_coord[1] + j;
            //         for (int k = -1; k < 2; ++k) {
            //             coord[2] = base_coord[2] + k;
            //             auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j, k));
            //             GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
            //             target(coord, g);
            //         }
            //     }
            // }
        }
    }

    template <typename OP>
    void iterateGrid(const OP& target)
    {
        auto blocks = fat_page_map->Get_Blocks();
        auto grid_array = grid->Get_Array();
        if constexpr (dim == 2) {
            tbb::parallel_for(0, (int)blocks.second, [&](int b) {
                uint64_t base_offset = blocks.first[b];
                auto stdarray_base_coord = SparseMask::LinearToCoord(base_offset);
                Vector<int, dim> base_coord{ stdarray_base_coord[0] - half_spgrid_size, stdarray_base_coord[1] - half_spgrid_size };
                auto x = 1 << SparseMask::block_xbits;
                auto y = 1 << SparseMask::block_ybits;
                for (int i = 0; i < x; ++i)
                    for (int j = 0; j < y; ++j) {
                        auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j));
                        GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                        if (g.m1 > 0)
                            target(base_coord + Vector<int, dim>{ i, j }, g);
                    }
            });
        }
        else {
            tbb::parallel_for(0, (int)blocks.second, [&](int b) {
                uint64_t base_offset = blocks.first[b];
                auto stdarray_base_coord = SparseMask::LinearToCoord(base_offset);
                Vector<int, dim> base_coord{ stdarray_base_coord[0] - half_spgrid_size, stdarray_base_coord[1] - half_spgrid_size, stdarray_base_coord[2] - half_spgrid_size };
                auto x = 1 << SparseMask::block_xbits;
                auto y = 1 << SparseMask::block_ybits;
                auto z = 1 << SparseMask::block_zbits;
                for (int i = 0; i < x; ++i)
                    for (int j = 0; j < y; ++j)
                        for (int k = 0; k < z; ++k) {
                            auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j, k));
                            GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                            if (g.m1 > 0)
                                target(base_coord + Vector<int, dim>{ i, j, k }, g);
                        }
            });
        }
    }

    template <typename OP>
    void iterateGridSerial(const OP& target)
    {
        auto blocks = fat_page_map->Get_Blocks();
        auto grid_array = grid->Get_Array();
        if constexpr (dim == 2) {
            for (unsigned b = 0; b < blocks.second; ++b) {
                uint64_t base_offset = blocks.first[b];
                auto stdarray_base_coord = SparseMask::LinearToCoord(base_offset);
                Vector<int, dim> base_coord{ stdarray_base_coord[0] - half_spgrid_size, stdarray_base_coord[1] - half_spgrid_size };
                auto x = 1 << SparseMask::block_xbits;
                auto y = 1 << SparseMask::block_ybits;
                for (int i = 0; i < x; ++i)
                    for (int j = 0; j < y; ++j) {
                        auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j));
                        GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                        if (g.m1 > 0)
                            target(base_coord + Vector<int, dim>{ i, j }, g);
                    }
            }
        }
        else {
            for (unsigned b = 0; b < blocks.second; ++b) {
                uint64_t base_offset = blocks.first[b];
                auto stdarray_base_coord = SparseMask::LinearToCoord(base_offset);
                Vector<int, dim> base_coord{ stdarray_base_coord[0] - half_spgrid_size, stdarray_base_coord[1] - half_spgrid_size, stdarray_base_coord[2] - half_spgrid_size };
                auto x = 1 << SparseMask::block_xbits;
                auto y = 1 << SparseMask::block_ybits;
                auto z = 1 << SparseMask::block_zbits;
                for (int i = 0; i < x; ++i)
                    for (int j = 0; j < y; ++j)
                        for (int k = 0; k < z; ++k) {
                            auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j, k));
                            GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                            if (g.m1 > 0)
                                target(base_coord + Vector<int, dim>{ i, j, k }, g);
                        }
            }
        }
    }

    template <typename OP>
    void iterateSeparableNodes(const OP& target)
    {
        auto blocks = fat_page_map->Get_Blocks();
        auto grid_array = grid->Get_Array();
        if constexpr (dim == 2) {
            tbb::parallel_for(0, (int)blocks.second, [&](int b) {
                uint64_t base_offset = blocks.first[b];
                auto stdarray_base_coord = SparseMask::LinearToCoord(base_offset);
                Vector<int, dim> base_coord{ stdarray_base_coord[0] - half_spgrid_size, stdarray_base_coord[1] - half_spgrid_size };
                auto x = 1 << SparseMask::block_xbits;
                auto y = 1 << SparseMask::block_ybits;
                for (int i = 0; i < x; ++i)
                    for (int j = 0; j < y; ++j) {
                        auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j));
                        GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                        if (g.m1 > 0 && g.separable == 1)
                            target(base_coord + Vector<int, dim>{ i, j }, g);
                    }
            });
        }
        else {
            tbb::parallel_for(0, (int)blocks.second, [&](int b) {
                uint64_t base_offset = blocks.first[b];
                auto stdarray_base_coord = SparseMask::LinearToCoord(base_offset);
                Vector<int, dim> base_coord{ stdarray_base_coord[0] - half_spgrid_size, stdarray_base_coord[1] - half_spgrid_size, stdarray_base_coord[2] - half_spgrid_size };
                auto x = 1 << SparseMask::block_xbits;
                auto y = 1 << SparseMask::block_ybits;
                auto z = 1 << SparseMask::block_zbits;
                for (int i = 0; i < x; ++i)
                    for (int j = 0; j < y; ++j)
                        for (int k = 0; k < z; ++k) {
                            auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j, k));
                            GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                            if (g.m1 > 0 && g.separable == 1)
                                target(base_coord + Vector<int, dim>{ i, j, k }, g);
                        }
            });
        }
    }

    template <typename OP>
    void iterateSeparableNodesSerial(const OP& target)
    {
        auto blocks = fat_page_map->Get_Blocks();
        auto grid_array = grid->Get_Array();
        if constexpr (dim == 2) {
            for (unsigned b = 0; b < blocks.second; ++b) {
                uint64_t base_offset = blocks.first[b];
                auto stdarray_base_coord = SparseMask::LinearToCoord(base_offset);
                Vector<int, dim> base_coord{ stdarray_base_coord[0] - half_spgrid_size, stdarray_base_coord[1] - half_spgrid_size };
                auto x = 1 << SparseMask::block_xbits;
                auto y = 1 << SparseMask::block_ybits;
                for (int i = 0; i < x; ++i)
                    for (int j = 0; j < y; ++j) {
                        auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j));
                        GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                        if (g.m1 > 0 && g.separable == 1)
                            target(base_coord + Vector<int, dim>{ i, j }, g);
                    }
            }
        }
        else {
            for (unsigned b = 0; b < blocks.second; ++b) {
                uint64_t base_offset = blocks.first[b];
                auto stdarray_base_coord = SparseMask::LinearToCoord(base_offset);
                Vector<int, dim> base_coord{ stdarray_base_coord[0] - half_spgrid_size, stdarray_base_coord[1] - half_spgrid_size, stdarray_base_coord[2] - half_spgrid_size };
                auto x = 1 << SparseMask::block_xbits;
                auto y = 1 << SparseMask::block_ybits;
                auto z = 1 << SparseMask::block_zbits;
                for (int i = 0; i < x; ++i)
                    for (int j = 0; j < y; ++j)
                        for (int k = 0; k < z; ++k) {
                            auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j, k));
                            GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                            if (g.m1 > 0 && g.separable == 1)
                                target(base_coord + Vector<int, dim>{ i, j, k }, g);
                        }
            }
        }
    }

    template <typename OP>
    void iterateWholeGridSerial(const OP& target)
    {
        auto blocks = fat_page_map->Get_Blocks();
        auto grid_array = grid->Get_Array();
        if constexpr (dim == 2) {
            for (unsigned b = 0; b < blocks.second; ++b) {
                uint64_t base_offset = blocks.first[b];
                auto stdarray_base_coord = SparseMask::LinearToCoord(base_offset);
                Vector<int, dim> base_coord{ stdarray_base_coord[0] - half_spgrid_size, stdarray_base_coord[1] - half_spgrid_size };
                auto x = 1 << SparseMask::block_xbits;
                auto y = 1 << SparseMask::block_ybits;
                for (int i = 0; i < x; ++i)
                    for (int j = 0; j < y; ++j) {
                        auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j));
                        GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                        target(base_coord + Vector<int, dim>{ i, j }, g);
                    }
            }
        }
        else {
            for (unsigned b = 0; b < blocks.second; ++b) {
                uint64_t base_offset = blocks.first[b];
                auto stdarray_base_coord = SparseMask::LinearToCoord(base_offset);
                Vector<int, dim> base_coord{ stdarray_base_coord[0] - half_spgrid_size, stdarray_base_coord[1] - half_spgrid_size, stdarray_base_coord[2] - half_spgrid_size };
                auto x = 1 << SparseMask::block_xbits;
                auto y = 1 << SparseMask::block_ybits;
                auto z = 1 << SparseMask::block_zbits;
                for (int i = 0; i < x; ++i)
                    for (int j = 0; j < y; ++j)
                        for (int k = 0; k < z; ++k) {
                            auto offset = SparseMask::Packed_Add(base_offset, SparseMask::Linear_Offset(i, j, k));
                            GridState<T, dim>& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
                            target(base_coord + Vector<int, dim>{ i, j, k }, g);
                        }
            }
        }
    }
};
}
} // namespace Bow::DFGMPM

//#ifndef BOW_STATIC_LIBRARY
//#include "MPMState.cpp"
//#endif
