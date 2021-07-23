#include "vtk.h"
#include <tbb/parallel_for.h>
#include <sstream>
#include <fstream>
#include <Bow/Utils/Logging.h>

namespace Bow {
namespace IO {

namespace internal {
template <class T>
inline void swap_endian(std::vector<T>& array)
{
    // vtk binary assumes big-endian byte order
    tbb::parallel_for(size_t(0), array.size(), [&](size_t i) {
        char* data = reinterpret_cast<char*>(&array[i]);
        for (long sub_i = 0; sub_i < static_cast<long>(sizeof(T) / 2); sub_i++)
            std::swap(data[sizeof(T) - 1 - sub_i], data[sub_i]);
    });
}
} // namespace internal

template <class T>
BOW_INLINE void write_vtk(const std::string filename, const Field<Vector<T, 3>>& _xyz, const Field<Vector<int, 4>>& _cells, const bool binary)
{
    int nPoints = _xyz.size();
    int nCells = _cells.size();
    std::vector<T> xyz(3 * nPoints);
    std::vector<uint32_t> cells(5 * nCells);
    std::vector<uint32_t> cell_types(nCells);
    tbb::parallel_for(0, nPoints, [&](int i) {
        for (int d = 0; d < 3; ++d) {
            xyz[3 * i + d] = _xyz[i][d];
        }
    });
    tbb::parallel_for(0, nCells, [&](int i) {
        cells[5 * i] = 4;
        for (int d = 0; d < 4; ++d) {
            cells[5 * i + d + 1] = _cells[i][d];
        }
        cell_types[i] = 10;
    });
    std::ofstream outstream;
    if (binary) {
        internal::swap_endian(xyz);
        internal::swap_endian(cells);
        internal::swap_endian(cell_types);
        outstream.open(filename, std::ios::out | std::ios::binary);
    }
    else
        outstream.open(filename, std::ios::out);

    if (outstream.fail()) throw std::runtime_error("failed to open " + filename);
    // const std::locale & fixLoc = std::locale("C");
    // outstream_binary.imbue(fixLoc);
    const std::locale& fixLoc = std::locale("C");
    outstream.imbue(fixLoc);
    outstream << "# vtk DataFile Version 2.0\n";
    outstream << "Visulaization output file\n";
    if (binary)
        outstream << "BINARY\n";
    else
        outstream << "ASCII\n";
    outstream << "DATASET UNSTRUCTURED_GRID\n";
    outstream << "POINTS " << nPoints;
    if (std::is_same<T, double>::value)
        outstream << " double\n";
    else
        outstream << " float\n";

    if (binary)
        outstream.write(reinterpret_cast<const char*>(xyz.data()), 3 * nPoints * sizeof(T));
    else {
        outstream << xyz[0];
        for (int i = 1; i < 3 * nPoints; ++i)
            outstream << " " << xyz[i];
    }
    outstream << "\n";
    outstream << "CELLS " << nCells << " " << 5 * nCells << "\n";
    if (binary)
        outstream.write(reinterpret_cast<char*>(cells.data()), 5 * nCells * sizeof(uint32_t));
    else {
        outstream << cells[0];
        for (int i = 1; i < 5 * nCells; ++i)
            outstream << " " << cells[i];
    }
    outstream << "\n";
    outstream << "CELL_TYPES " << nCells << "\n";
    if (binary)
        outstream.write(reinterpret_cast<char*>(cell_types.data()), nCells * sizeof(uint32_t));
    else {
        outstream << cell_types[0];
        for (int i = 1; i < nCells; ++i)
            outstream << " " << cell_types[i];
    }
    outstream << "\n";
    outstream.close();
}

template <class T>
BOW_INLINE void read_vtk(const std::string filePath, Field<Vector<T, 3>>& X, Field<Vector<int, 4>>& indices)
{
    std::ifstream in(filePath);
    if (!in.is_open()) {
        puts((filePath + " not found!").c_str());
        exit(-1);
    }

    auto initial_X_size = X.size();

    std::string line;
    Vector<T, 3> position;
    Vector<int, 4> tet;
    int position_index = 0;
    int tet_index = -1;

    bool reading_points = false;
    bool reading_tets = false;
    size_t n_points = 0;
    size_t n_tets = 0;

    while (std::getline(in, line)) {
        std::stringstream ss(line);
        if (line.size() == (size_t)(0)) {
        }
        else if (line.substr(0, 6) == "POINTS") {
            reading_points = true;
            reading_tets = false;
            ss.ignore(128, ' '); // ignore "POINTS"
            ss >> n_points;
        }
        else if (line.substr(0, 5) == "CELLS") {
            reading_points = false;
            reading_tets = true;
            ss.ignore(128, ' '); // ignore "CELLS"
            ss >> n_tets;
        }
        else if (line.substr(0, 10) == "CELL_TYPES") {
            reading_points = false;
            reading_tets = false;
        }
        else if (reading_points) {
            while (ss >> position(position_index)) {
                position_index += 1;
                if (position_index == 3) {
                    X.emplace_back(position);
                    position_index = 0;
                }
            }
        }
        else if (reading_tets) {
            int index_cache;
            while (ss >> index_cache) {
                if (tet_index == -1) { // ignore "4"
                    BOW_ASSERT_INFO(index_cache == 4, "Only support tetrahedral mesh!");
                    tet_index = 0;
                }
                else {
                    tet(tet_index) = index_cache;
                    tet_index += 1;
                    if (tet_index == 4) {
                        indices.emplace_back(tet);
                        tet_index = -1;
                    }
                }
            }
        }
    }
    in.close();

    BOW_ASSERT_INFO((n_points == X.size() - initial_X_size), "vtk read X count doesn't match.");
    BOW_ASSERT_INFO(((size_t)n_tets == indices.size()), "vtk read element count doesn't match.");
    Logging::info("#V: ", n_points, ", #C: ", n_tets);
}

#ifdef BOW_STATIC_LIBRARY
#ifdef BOW_COMPILE_FLOAT
template void write_vtk(const std::string filename, const Field<Vector<float, 3>>& xyz, const Field<Vector<int, 4>>& cells, const bool binary);
template void read_vtk(const std::string filePath, Field<Vector<float, 3>>& X, Field<Vector<int, 4>>& indices);
#endif
#ifdef BOW_COMPILE_DOUBLE
template void write_vtk(const std::string filename, const Field<Vector<double, 3>>& xyz, const Field<Vector<int, 4>>& cells, const bool binary);
template void read_vtk(const std::string filePath, Field<Vector<double, 3>>& X, Field<Vector<int, 4>>& indices);
#endif
#endif
}
} // namespace Bow::IO
