#include "ply.h"
#include "tinyply.h"
#include <tbb/parallel_for.h>
#include <sstream>
#include <fstream>
#include <Bow/Utils/Logging.h>

namespace Bow {
namespace IO {

namespace internal {
inline std::vector<uint8_t> read_file_binary(const std::string& pathToFile)
{
    std::ifstream file(pathToFile, std::ios::binary);
    std::vector<uint8_t> fileBufferBytes;

    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        size_t sizeBytes = file.tellg();
        file.seekg(0, std::ios::beg);
        fileBufferBytes.resize(sizeBytes);
        if (file.read((char*)fileBufferBytes.data(), sizeBytes)) return fileBufferBytes;
    }
    else
        throw std::runtime_error("could not open binary ifstream to path " + pathToFile);
    return fileBufferBytes;
}

struct memory_buffer : public std::streambuf {
    char* p_start{ nullptr };
    char* p_end{ nullptr };
    size_t size;

    memory_buffer(char const* first_elem, size_t size)
        : p_start(const_cast<char*>(first_elem)), p_end(p_start + size), size(size)
    {
        setg(p_start, p_start, p_end);
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override
    {
        if (dir == std::ios_base::cur)
            gbump(static_cast<int>(off));
        else
            setg(p_start, (dir == std::ios_base::beg ? p_start : p_end) + off, p_end);
        return gptr() - p_start;
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override
    {
        return seekoff(pos, std::ios_base::beg, which);
    }
};

struct memory_stream : virtual memory_buffer, public std::istream {
    memory_stream(char const* first_elem, size_t size)
        : memory_buffer(first_elem, size), std::istream(static_cast<std::streambuf*>(this)) {}
};

} // namespace internal

template <class T, int dim>
BOW_INLINE void read_ply(const std::string filepath, Field<Vector<T, dim>>& vertices_out)
{
    Logging::info("Reading: ", filepath);

    std::unique_ptr<std::istream> file_stream;
    std::vector<uint8_t> byte_buffer;

    try {
        // For most files < 1gb, pre-loading the entire file upfront and wrapping it into a
        // stream is a net win for parsing speed, about 40% faster.
        byte_buffer = internal::read_file_binary(filepath);
        file_stream.reset(new internal::memory_stream((char*)byte_buffer.data(), byte_buffer.size()));

        if (!file_stream || file_stream->fail()) throw std::runtime_error("file_stream failed to open " + filepath);

        tinyply::PlyFile file;
        file.parse_header(*file_stream);
        Logging::info("\t[ply_header] Type: ", (file.is_binary_file() ? "binary" : "ascii"));

        // Because most people have their own mesh types, tinyply treats parsed data as structured/typed byte buffers.
        // See examples below on how to marry your own application-specific data structures with this one.
        std::shared_ptr<tinyply::PlyData> vertices, faces;

        // The header information can be used to programmatically extract properties on elements
        // known to exist in the header prior to reading the data. For brevity of this sample, properties
        // like vertex position are hard-coded:
        try {
            vertices = file.request_properties_from_element("vertex", { "x", "y", "z" });
        }
        catch (const std::exception& e) {
            Logging::error("tinyply exception: ", e.what());
        }

        file.read(*file_stream);

        if (vertices) {
            Logging::info("\tRead ", vertices->count, " total vertices ");
            vertices_out.resize(vertices->count);
            if (vertices->t == tinyply::Type::FLOAT32) {
                float* verts_data = reinterpret_cast<float*>(vertices->buffer.get());
                tbb::parallel_for(size_t(0), vertices->count, [&](size_t i) {
                    for (int d = 0; d < dim; ++d)
                        vertices_out[i](d) = verts_data[3 * i + d];
                });
            }
            else if (vertices->t == tinyply::Type::FLOAT64) {
                double* verts_data = reinterpret_cast<double*>(vertices->buffer.get());
                tbb::parallel_for(size_t(0), vertices->count, [&](size_t i) {
                    for (int d = 0; d < dim; ++d)
                        vertices_out[i](d) = verts_data[3 * i + d];
                });
            }
        }
    }
    catch (const std::exception& e) {
        Logging::error("Caught tinyply exception: ", e.what());
    }
}

template <class T, int dim>
BOW_INLINE void read_ply(const std::string filepath, Field<Vector<T, dim>>& vertices_out, Field<Vector<int, 3>>& faces_out)
{
    Logging::info("Reading: ", filepath);

    std::unique_ptr<std::istream> file_stream;
    std::vector<uint8_t> byte_buffer;

    try {
        // For most files < 1gb, pre-loading the entire file upfront and wrapping it into a
        // stream is a net win for parsing speed, about 40% faster.
        byte_buffer = internal::read_file_binary(filepath);
        file_stream.reset(new internal::memory_stream((char*)byte_buffer.data(), byte_buffer.size()));

        if (!file_stream || file_stream->fail()) throw std::runtime_error("file_stream failed to open " + filepath);

        tinyply::PlyFile file;
        file.parse_header(*file_stream);
        Logging::info("\t[ply_header] Type: ", (file.is_binary_file() ? "binary" : "ascii"));

        // Because most people have their own mesh types, tinyply treats parsed data as structured/typed byte buffers.
        // See examples below on how to marry your own application-specific data structures with this one.
        std::shared_ptr<tinyply::PlyData> vertices, faces;

        // The header information can be used to programmatically extract properties on elements
        // known to exist in the header prior to reading the data. For brevity of this sample, properties
        // like vertex position are hard-coded:
        try {
            vertices = file.request_properties_from_element("vertex", { "x", "y", "z" });
        }
        catch (const std::exception& e) {
            Logging::error("tinyply exception: ", e.what());
        }

        // Providing a list size hint (the last argument) is a 2x performance improvement. If you have
        // arbitrary ply files, it is best to leave this 0.
        try {
            faces = file.request_properties_from_element("face", { "vertex_indices" }, 3);
        }
        catch (const std::exception& e) {
            Logging::error("tinyply exception: ", e.what());
        }

        file.read(*file_stream);

        if (vertices) {
            Logging::info("\tRead ", vertices->count, " total vertices ");
            vertices_out.resize(vertices->count);
            if (vertices->t == tinyply::Type::FLOAT32) {
                float* verts_data = reinterpret_cast<float*>(vertices->buffer.get());
                tbb::parallel_for(size_t(0), vertices->count, [&](size_t i) {
                    for (int d = 0; d < dim; ++d)
                        vertices_out[i](d) = verts_data[3 * i + d];
                });
            }
            else if (vertices->t == tinyply::Type::FLOAT64) {
                double* verts_data = reinterpret_cast<double*>(vertices->buffer.get());
                tbb::parallel_for(size_t(0), vertices->count, [&](size_t i) {
                    for (int d = 0; d < dim; ++d)
                        vertices_out[i](d) = verts_data[3 * i + d];
                });
            }
        }
        if (faces) {
            Logging::info("\tRead ", faces->count, " total faces (triangles) ");
            faces_out.resize(faces->count);
            if (faces->t == tinyply::Type::UINT32) {
                uint32_t* facess_data = reinterpret_cast<uint32_t*>(faces->buffer.get());
                tbb::parallel_for(size_t(0), faces->count, [&](size_t i) {
                    for (int d = 0; d < 3; ++d)
                        faces_out[i](d) = facess_data[3 * i + d];
                });
            }
            else if (faces->t == tinyply::Type::INT32) {
                int32_t* facess_data = reinterpret_cast<int32_t*>(faces->buffer.get());
                tbb::parallel_for(size_t(0), faces->count, [&](size_t i) {
                    for (int d = 0; d < 3; ++d)
                        faces_out[i](d) = facess_data[3 * i + d];
                });
            }
        }
    }
    catch (const std::exception& e) {
        Logging::error("Caught tinyply exception: ", e.what());
    }
} // namespace IO

template <class T, int dim>
BOW_INLINE void write_ply(const std::string filename, const Field<Vector<T, dim>>& _vertices, const Field<Vector<int, 3>>& faces, const bool binary)
{
    Logging::info("Writing: ", filename);
    Field<Vector<T, 3>> vertices(_vertices.size());
    if constexpr (dim == 2) {
        tbb::parallel_for((size_t)0, _vertices.size(), [&](size_t i) {
            vertices[i] = Vector<T, 3>(_vertices[i][0], _vertices[i][1], 0.0);
        });
    }
    else {
        tbb::parallel_for((size_t)0, _vertices.size(), [&](size_t i) {
            vertices[i] = _vertices[i];
        });
    }
    std::ofstream outstream_binary;
    if (binary)
        outstream_binary.open(filename, std::ios::out | std::ios::binary);
    else
        outstream_binary.open(filename, std::ios::out);
    tinyply::PlyFile file;
    if (std::is_same<T, float>::value)
        file.add_properties_to_element("vertex", { "x", "y", "z" },
            tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<uint8_t*>(vertices.data()), tinyply::Type::INVALID, 0);
    else
        file.add_properties_to_element("vertex", { "x", "y", "z" },
            tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<uint8_t*>(vertices.data()), tinyply::Type::INVALID, 0);
    file.add_properties_to_element("face", { "vertex_indices" },
        tinyply::Type::UINT32, faces.size(), reinterpret_cast<const uint8_t*>(faces.data()), tinyply::Type::UINT8, 3);
    if (binary)
        file.write(outstream_binary, true);
    else
        file.write(outstream_binary, false);
}

template <class T, int dim>
BOW_INLINE void write_ply(const std::string filename, const Field<Vector<T, dim>>& _vertices, const Field<Vector<int, 3>>& faces, const Field<T>& info, const bool binary)
{
    Logging::info("Writing: ", filename);
    Field<Vector<T, 3>> vertices(_vertices.size());
    if constexpr (dim == 2) {
        tbb::parallel_for((size_t)0, _vertices.size(), [&](size_t i) {
            vertices[i] = Vector<T, 3>(_vertices[i][0], _vertices[i][1], 0.0);
        });
    }
    else {
        tbb::parallel_for((size_t)0, _vertices.size(), [&](size_t i) {
            vertices[i] = _vertices[i];
        });
    }
    std::ofstream outstream_binary;
    if (binary)
        outstream_binary.open(filename, std::ios::out | std::ios::binary);
    else
        outstream_binary.open(filename, std::ios::out);
    tinyply::PlyFile file;
    if (std::is_same<T, float>::value) {
        file.add_properties_to_element("vertex", { "x", "y", "z" },
            tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<uint8_t*>(vertices.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "info" },
            tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(info.data()), tinyply::Type::INVALID, 0);
    }
    else {
        file.add_properties_to_element("vertex", { "x", "y", "z" },
            tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<uint8_t*>(vertices.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "info" },
            tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(info.data()), tinyply::Type::INVALID, 0);
    }
    file.add_properties_to_element("face", { "vertex_indices" },
        tinyply::Type::UINT32, faces.size(), reinterpret_cast<const uint8_t*>(faces.data()), tinyply::Type::UINT8, 3);
    if (binary)
        file.write(outstream_binary, true);
    else
        file.write(outstream_binary, false);
}

template <class T, int dim>
BOW_INLINE void write_ply(const std::string filename, const Field<Vector<T, dim>>& _vertices, const bool binary)
{
    Logging::info("Writing: ", filename);
    Field<Vector<T, 3>> vertices(_vertices.size());
    if constexpr (dim == 2) {
        tbb::parallel_for((size_t)0, _vertices.size(), [&](size_t i) {
            vertices[i] = Vector<T, 3>(_vertices[i][0], _vertices[i][1], 0.0);
        });
    }
    else {
        tbb::parallel_for((size_t)0, _vertices.size(), [&](size_t i) {
            vertices[i] = _vertices[i];
        });
    }

    std::ofstream outstream_binary;
    if (binary)
        outstream_binary.open(filename, std::ios::out | std::ios::binary);
    else
        outstream_binary.open(filename, std::ios::out);

    tinyply::PlyFile file;
    if constexpr (std::is_same<T, float>::value)
        file.add_properties_to_element("vertex", { "x", "y", "z" },
            tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<uint8_t*>(vertices.data()), tinyply::Type::INVALID, 0);
    else
        file.add_properties_to_element("vertex", { "x", "y", "z" },
            tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<uint8_t*>(vertices.data()), tinyply::Type::INVALID, 0);
    if (binary)
        file.write(outstream_binary, true);
    else
        file.write(outstream_binary, false);
}

template <class T, int dim>
BOW_INLINE void writeTwoField_particles_ply(const std::string filename, const Field<Vector<T, dim>>& _vertices, const Field<Vector<T, dim>>& velocities, const Field<Vector<T, dim>>& damageGradients, const std::vector<T>& masses, const std::vector<T>& damage, const std::vector<int>& _sp, const Field<int>& _markers, const Field<Matrix<T,dim,dim>>& _m_cauchy, const Field<Matrix<T,dim,dim>>& _m_F, const bool binary)
{
    Logging::info("Writing: ", filename);
    Field<Vector<T, 3>> vertices(_vertices.size());
    Field<T> velX(velocities.size());
    Field<T> velY(velocities.size());
    Field<T> velZ(velocities.size());
    Field<T> dgX(damageGradients.size());
    Field<T> dgY(damageGradients.size());
    Field<T> dgZ(damageGradients.size());
    Field<T> mass(masses.size());
    Field<T> d(damage.size());
    Field<int> sp(_sp.size());
    Field<int> markers(_markers.size());
    Field<T> sigma11(_vertices.size());
    Field<T> sigma22(_vertices.size());
    Field<T> sigma12(_vertices.size());
    Field<T> F11(_vertices.size());
    Field<T> F22(_vertices.size());
    Field<T> F12(_vertices.size());

    //formatting vector fields
    if constexpr (dim == 2) {
        tbb::parallel_for((size_t)0, _vertices.size(), [&](size_t i) {
            vertices[i] = Vector<T, 3>(_vertices[i][0], _vertices[i][1], 0.0);
            velX[i] = velocities[i][0];
            velY[i] = velocities[i][1];
            velZ[i] = 0.0;
            dgX[i] = damageGradients[i][0];
            dgY[i] = damageGradients[i][1];
            dgZ[i] = 0.0;
        });
    }
    else {
        tbb::parallel_for((size_t)0, _vertices.size(), [&](size_t i) {
            vertices[i] = _vertices[i];
            velX[i] = velocities[i][0];
            velY[i] = velocities[i][1];
            velZ[i] = velocities[i][2];
            dgX[i] = damageGradients[i][0];
            dgY[i] = damageGradients[i][1];
            dgZ[i] = damageGradients[i][2];
        });
    }

    //formatting the rest of the scalar fields
    tbb::parallel_for((size_t)0, _vertices.size(), [&](size_t i) {
        mass[i] = masses[i];
        d[i] = damage[i];
        sp[i] = _sp[i];
        markers[i] = _markers[i];
        if(_m_F.size() > 0){
            F11[i] = _m_F[i](0,0);
            F22[i] = _m_F[i](1,1);
            F12[i] = _m_F[i](0,1);
        }
        else{
            F11[i] = 1.0;
            F22[i] = 1.0;
            F12[i] = 0.0;
        }
        if(_m_cauchy.size() > 0){
            sigma11[i] = _m_cauchy[i](0,0);
            sigma22[i] = _m_cauchy[i](1,1);
            sigma12[i] = _m_cauchy[i](0,1);
        }
        else{
            sigma11[i] = 0.0;
            sigma22[i] = 0.0;
            sigma12[i] = 0.0;
        }
        
    });

    std::ofstream outstream_binary;
    if (binary)
        outstream_binary.open(filename, std::ios::out | std::ios::binary);
    else
        outstream_binary.open(filename, std::ios::out);
    tinyply::PlyFile file;
    if (std::is_same<T, float>::value) {
        file.add_properties_to_element("vertex", { "x", "y", "z" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<uint8_t*>(vertices.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "velX" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(velX.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "velY" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(velY.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "velZ" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(velZ.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "DGx" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(dgX.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "DGy" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(dgY.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "DGz" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(dgZ.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "mass" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(mass.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "damage" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(d.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "sp" }, tinyply::Type::INT32, vertices.size(), reinterpret_cast<const uint8_t*>(sp.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "marker" }, tinyply::Type::INT32, vertices.size(), reinterpret_cast<const uint8_t*>(markers.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "sigma11" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(sigma11.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "sigma22" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(sigma22.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "sigma12" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(sigma12.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "F11" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(F11.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "F22" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(F22.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "F12" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(F12.data()), tinyply::Type::INVALID, 0);
    }
    else {
        file.add_properties_to_element("vertex", { "x", "y", "z" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<uint8_t*>(vertices.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "velX" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(velX.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "velY" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(velY.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "velZ" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(velZ.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "DGx" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(dgX.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "DGy" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(dgY.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "DGz" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(dgZ.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "mass" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(mass.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "damage" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(d.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "sp" }, tinyply::Type::INT32, vertices.size(), reinterpret_cast<const uint8_t*>(sp.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "marker" }, tinyply::Type::INT32, vertices.size(), reinterpret_cast<const uint8_t*>(markers.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "sigma11" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(sigma11.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "sigma22" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(sigma22.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "sigma12" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(sigma12.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "F11" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(F11.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "F22" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(F22.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "F12" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(F12.data()), tinyply::Type::INVALID, 0);
    }

    if (binary)
        file.write(outstream_binary, true);
    else
        file.write(outstream_binary, false);
}

template <class T, int dim>
BOW_INLINE void writeTwoField_nodes_ply(const std::string filename, const Field<Vector<T, dim>>& _vertices, const Field<Matrix<T, dim, dim>>& cauchy1, const Field<Matrix<T, dim, dim>>& cauchy2, const Field<Matrix<T, dim, dim>>& Fi1, const Field<Matrix<T, dim, dim>>& Fi2, const Field<Vector<T, dim>>& damageGradients, const Field<Vector<T, dim>>& v1, const Field<Vector<T, dim>>& v2, const Field<Vector<T, dim>>& fct1, const Field<Vector<T, dim>>& fct2, const std::vector<T>& _m1, const std::vector<T>& _m2, const std::vector<T>& _sep1, const std::vector<T>& _sep2, const std::vector<int>& _separable, const bool binary)
{
    Logging::info("Writing: ", filename);
    Field<Vector<T, 3>> vertices(_vertices.size());
    Field<T> dgX(damageGradients.size());
    Field<T> dgY(damageGradients.size());
    Field<T> dgZ(damageGradients.size());
    Field<T> v1x(v1.size());
    Field<T> v1y(v1.size());
    Field<T> v1z(v1.size());
    Field<T> v2x(v2.size());
    Field<T> v2y(v2.size());
    Field<T> v2z(v2.size());
    Field<T> fct1x(fct1.size());
    Field<T> fct1y(fct1.size());
    Field<T> fct1z(fct1.size());
    Field<T> fct2x(fct2.size());
    Field<T> fct2y(fct2.size());
    Field<T> fct2z(fct2.size());
    Field<T> m1(_m1.size());
    Field<T> m2(_m2.size());
    Field<T> sep1(_sep1.size());
    Field<T> sep2(_sep2.size());
    Field<int> separable(_separable.size());

    Field<T> sigma11_f1(cauchy1.size());
    Field<T> sigma22_f1(cauchy1.size());
    Field<T> sigma12_f1(cauchy1.size());
    Field<T> F11_f1(Fi1.size());
    Field<T> F22_f1(Fi1.size());
    Field<T> F12_f1(Fi1.size());
    Field<T> sigma11_f2(cauchy2.size());
    Field<T> sigma22_f2(cauchy2.size());
    Field<T> sigma12_f2(cauchy2.size());
    Field<T> F11_f2(Fi2.size());
    Field<T> F22_f2(Fi2.size());
    Field<T> F12_f2(Fi2.size());

    //formatting vector fields
    if constexpr (dim == 2) {
        tbb::parallel_for((size_t)0, _vertices.size(), [&](size_t i) {
            vertices[i] = Vector<T, 3>(_vertices[i][0], _vertices[i][1], 0.0);
            dgX[i] = damageGradients[i][0];
            dgY[i] = damageGradients[i][1];
            dgZ[i] = 0.0;
            v1x[i] = v1[i][0];
            v1y[i] = v1[i][1];
            v1z[i] = 0.0;
            v2x[i] = v2[i][0];
            v2y[i] = v2[i][1];
            v2z[i] = 0.0;
            fct1x[i] = fct1[i][0];
            fct1y[i] = fct1[i][1];
            fct1z[i] = 0.0;
            fct2x[i] = fct2[i][0];
            fct2y[i] = fct2[i][1];
            fct2z[i] = 0.0;
        });
    }
    else {
        tbb::parallel_for((size_t)0, _vertices.size(), [&](size_t i) {
            vertices[i] = _vertices[i];
            dgX[i] = damageGradients[i][0];
            dgY[i] = damageGradients[i][1];
            dgZ[i] = damageGradients[i][2];
            v1x[i] = v1[i][0];
            v1y[i] = v1[i][1];
            v1z[i] = v1[i][2];
            v2x[i] = v2[i][0];
            v2y[i] = v2[i][1];
            v2z[i] = v2[i][2];
            fct1x[i] = fct1[i][0];
            fct1y[i] = fct1[i][1];
            fct1z[i] = fct1[i][2];
            fct2x[i] = fct2[i][0];
            fct2y[i] = fct2[i][1];
            fct2z[i] = fct2[i][2];
        });
    }

    //formatting the rest of the scalar fields
    tbb::parallel_for((size_t)0, _vertices.size(), [&](size_t i) {
        m1[i] = _m1[i];
        m2[i] = _m2[i];
        sep1[i] = _sep1[i];
        sep2[i] = _sep2[i];
        separable[i] = _separable[i];

        sigma11_f1[i] = cauchy1[i](0,0); //cauchy field 1
        sigma22_f1[i] = cauchy1[i](1,1); 
        sigma12_f1[i] = cauchy1[i](0,1);

        sigma11_f2[i] = cauchy2[i](0,0); //cauchy field 2
        sigma22_f2[i] = cauchy2[i](1,1);
        sigma12_f2[i] = cauchy2[i](0,1);

        F11_f1[i] = Fi1[i](0,0); //def grad field 1
        F22_f1[i] = Fi1[i](1,1);
        F12_f1[i] = Fi1[i](0,1);

        F11_f2[i] = Fi2[i](0,0); //def grad field 2
        F22_f2[i] = Fi2[i](1,1);
        F12_f2[i] = Fi2[i](0,1);
    });

    std::ofstream outstream_binary;
    if (binary)
        outstream_binary.open(filename, std::ios::out | std::ios::binary);
    else
        outstream_binary.open(filename, std::ios::out);
    tinyply::PlyFile file;
    if (std::is_same<T, float>::value) {
        file.add_properties_to_element("vertex", { "x", "y", "z" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<uint8_t*>(vertices.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "DGx" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(dgX.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "DGy" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(dgY.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "DGz" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(dgZ.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "v1x" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(v1x.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "v1y" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(v1y.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "v1z" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(v1z.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "v2x" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(v2x.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "v2y" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(v2y.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "v2z" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(v2z.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "fct1x" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(fct1x.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "fct1y" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(fct1y.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "fct1z" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(fct1z.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "fct2x" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(fct2x.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "fct2y" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(fct2y.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "fct2z" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(fct2z.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "m1" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(m1.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "m2" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(m2.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "sep1" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(sep1.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "sep2" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(sep2.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "separable" }, tinyply::Type::INT32, vertices.size(), reinterpret_cast<const uint8_t*>(separable.data()), tinyply::Type::INVALID, 0);

        file.add_properties_to_element("vertex", { "f1_sigma11" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(sigma11_f1.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "f1_sigma22" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(sigma22_f1.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "f1_sigma12" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(sigma12_f1.data()), tinyply::Type::INVALID, 0);

        file.add_properties_to_element("vertex", { "f2_sigma11" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(sigma11_f2.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "f2_sigma22" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(sigma22_f2.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "f2_sigma12" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(sigma12_f2.data()), tinyply::Type::INVALID, 0);

        file.add_properties_to_element("vertex", { "f1_F11" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(F11_f1.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "f1_F22" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(F22_f1.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "f1_F12" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(F12_f1.data()), tinyply::Type::INVALID, 0);

        file.add_properties_to_element("vertex", { "f2_F11" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(F11_f2.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "f2_F22" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(F22_f2.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "f2_F12" }, tinyply::Type::FLOAT32, vertices.size(), reinterpret_cast<const uint8_t*>(F12_f2.data()), tinyply::Type::INVALID, 0);   
    }
    else {
        file.add_properties_to_element("vertex", { "x", "y", "z" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<uint8_t*>(vertices.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "DGx" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(dgX.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "DGy" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(dgY.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "DGz" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(dgZ.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "v1x" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(v1x.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "v1y" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(v1y.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "v1z" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(v1z.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "v2x" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(v2x.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "v2y" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(v2y.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "v2z" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(v2z.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "fct1x" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(fct1x.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "fct1y" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(fct1y.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "fct1z" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(fct1z.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "fct2x" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(fct2x.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "fct2y" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(fct2y.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "fct2z" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(fct2z.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "m1" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(m1.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "m2" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(m2.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "sep1" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(sep1.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "sep2" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(sep2.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "separable" }, tinyply::Type::INT32, vertices.size(), reinterpret_cast<const uint8_t*>(separable.data()), tinyply::Type::INVALID, 0);

        file.add_properties_to_element("vertex", { "f1_sigma11" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(sigma11_f1.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "f1_sigma22" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(sigma22_f1.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "f1_sigma12" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(sigma12_f1.data()), tinyply::Type::INVALID, 0);

        file.add_properties_to_element("vertex", { "f2_sigma11" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(sigma11_f2.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "f2_sigma22" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(sigma22_f2.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "f2_sigma12" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(sigma12_f2.data()), tinyply::Type::INVALID, 0);

        file.add_properties_to_element("vertex", { "f1_F11" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(F11_f1.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "f1_F22" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(F22_f1.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "f1_F12" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(F12_f1.data()), tinyply::Type::INVALID, 0);

        file.add_properties_to_element("vertex", { "f2_F11" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(F11_f2.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "f2_F22" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(F22_f2.data()), tinyply::Type::INVALID, 0);
        file.add_properties_to_element("vertex", { "f2_F12" }, tinyply::Type::FLOAT64, vertices.size(), reinterpret_cast<const uint8_t*>(F12_f2.data()), tinyply::Type::INVALID, 0);
    }

    if (binary)
        file.write(outstream_binary, true);
    else
        file.write(outstream_binary, false);
}

#ifdef BOW_STATIC_LIBRARY
#ifdef BOW_COMPILE_FLOAT
template void read_ply(const std::string filename, Field<Vector<float, 2>>& vertices);
template void read_ply(const std::string filename, Field<Vector<float, 3>>& vertices);
template void read_ply(const std::string filename, Field<Vector<float, 2>>& vertices, Field<Vector<int, 3>>& faces);
template void read_ply(const std::string filename, Field<Vector<float, 3>>& vertices, Field<Vector<int, 3>>& faces);
template void write_ply(const std::string filename, const Field<Vector<float, 2>>& vertices, const Field<Vector<int, 3>>& faces, const bool);
template void write_ply(const std::string filename, const Field<Vector<float, 3>>& vertices, const Field<Vector<int, 3>>& faces, const bool);
template void write_ply(const std::string filename, const Field<Vector<float, 2>>& vertices, const bool);
template void write_ply(const std::string filename, const Field<Vector<float, 3>>& vertices, const bool);
template void write_ply(const std::string filename, const Field<Vector<float, 2>>& vertices, const Field<Vector<int, 3>>&, const Field<float>&, const bool);
template void write_ply(const std::string filename, const Field<Vector<float, 3>>& vertices, const Field<Vector<int, 3>>&, const Field<float>&, const bool);
template void writeTwoField_particles_ply(const std::string filename, const Field<Vector<float, 2>>& vertices, const Field<Vector<float, 2>>& velocities, const Field<Vector<float, 2>>& damageGradients, const std::vector<float>& masses, const std::vector<float>& damage, const std::vector<int>& sp, const Field<int>& markers, const Field<Matrix<float,2,2>>& m_cauchy, const Field<Matrix<float,2,2>>& m_F, const bool);
template void writeTwoField_particles_ply(const std::string filename, const Field<Vector<float, 3>>& vertices, const Field<Vector<float, 3>>& velocities, const Field<Vector<float, 3>>& damageGradients, const std::vector<float>& masses, const std::vector<float>& damage, const std::vector<int>& sp, const Field<int>& markers, const Field<Matrix<float,3,3>>& m_cauchy, const Field<Matrix<float,3,3>>& m_F, const bool);
template void writeTwoField_nodes_ply(const std::string filename, const Field<Vector<float, 2>>& vertices, const Field<Matrix<float, 2, 2>>& cauchy1, const Field<Matrix<float, 2, 2>>& cauchy2, const Field<Matrix<float, 2, 2>>& Fi1, const Field<Matrix<float, 2, 2>>& Fi2, const Field<Vector<float, 2>>& damageGradients, const Field<Vector<float, 2>>& v1, const Field<Vector<float, 2>>& v2, const Field<Vector<float, 2>>& fct1, const Field<Vector<float, 2>>& fct2, const std::vector<float>& m1, const std::vector<float>& m2, const std::vector<float>& sep1, const std::vector<float>& sep2, const std::vector<int>& separable, const bool);
template void writeTwoField_nodes_ply(const std::string filename, const Field<Vector<float, 3>>& vertices, const Field<Matrix<float, 3, 3>>& cauchy1, const Field<Matrix<float, 3, 3>>& cauchy2, const Field<Matrix<float, 3, 3>>& Fi1, const Field<Matrix<float, 3, 3>>& Fi2, const Field<Vector<float, 3>>& damageGradients, const Field<Vector<float, 3>>& v1, const Field<Vector<float, 3>>& v2, const Field<Vector<float, 3>>& fct1, const Field<Vector<float, 3>>& fct2, const std::vector<float>& m1, const std::vector<float>& m2, const std::vector<float>& sep1, const std::vector<float>& sep2, const std::vector<int>& separable, const bool);
#endif
#ifdef BOW_COMPILE_DOUBLE
template void read_ply(const std::string filename, Field<Vector<double, 2>>& vertices);
template void read_ply(const std::string filename, Field<Vector<double, 3>>& vertices);
template void read_ply(const std::string filename, Field<Vector<double, 2>>& vertices, Field<Vector<int, 3>>& faces);
template void read_ply(const std::string filename, Field<Vector<double, 3>>& vertices, Field<Vector<int, 3>>& faces);
template void write_ply(const std::string filename, const Field<Vector<double, 2>>& vertices, const Field<Vector<int, 3>>& faces, const bool);
template void write_ply(const std::string filename, const Field<Vector<double, 3>>& vertices, const Field<Vector<int, 3>>& faces, const bool);
template void write_ply(const std::string filename, const Field<Vector<double, 2>>& vertices, const bool);
template void write_ply(const std::string filename, const Field<Vector<double, 3>>& vertices, const bool);
template void write_ply(const std::string filename, const Field<Vector<double, 2>>& vertices, const Field<Vector<int, 3>>&, const Field<double>&, const bool);
template void write_ply(const std::string filename, const Field<Vector<double, 3>>& vertices, const Field<Vector<int, 3>>&, const Field<double>&, const bool);
template void writeTwoField_particles_ply(const std::string filename, const Field<Vector<double, 2>>& vertices, const Field<Vector<double, 2>>& velocities, const Field<Vector<double, 2>>& damageGradients, const std::vector<double>& masses, const std::vector<double>& damage, const std::vector<int>& sp, const Field<int>& markers, const Field<Matrix<double,2,2>>& m_cauchy, const Field<Matrix<double,2,2>>& m_F, const bool);
template void writeTwoField_particles_ply(const std::string filename, const Field<Vector<double, 3>>& vertices, const Field<Vector<double, 3>>& velocities, const Field<Vector<double, 3>>& damageGradients, const std::vector<double>& masses, const std::vector<double>& damage, const std::vector<int>& sp, const Field<int>& markers, const Field<Matrix<double,3,3>>& m_cauchy, const Field<Matrix<double,3,3>>& m_F, const bool);
template void writeTwoField_nodes_ply(const std::string filename, const Field<Vector<double, 2>>& vertices, const Field<Matrix<double, 2, 2>>& cauchy1, const Field<Matrix<double, 2, 2>>& cauchy2, const Field<Matrix<double, 2, 2>>& Fi1, const Field<Matrix<double, 2, 2>>& Fi2, const Field<Vector<double, 2>>& damageGradients, const Field<Vector<double, 2>>& v1, const Field<Vector<double, 2>>& v2, const Field<Vector<double, 2>>& fct1, const Field<Vector<double, 2>>& fct2, const std::vector<double>& m1, const std::vector<double>& m2, const std::vector<double>& sep1, const std::vector<double>& sep2, const std::vector<int>& separable, const bool);
template void writeTwoField_nodes_ply(const std::string filename, const Field<Vector<double, 3>>& vertices, const Field<Matrix<double, 3, 3>>& cauchy1, const Field<Matrix<double, 3, 3>>& cauchy2, const Field<Matrix<double, 3, 3>>& Fi1, const Field<Matrix<double, 3, 3>>& Fi2, const Field<Vector<double, 3>>& damageGradients, const Field<Vector<double, 3>>& v1, const Field<Vector<double, 3>>& v2, const Field<Vector<double, 3>>& fct1, const Field<Vector<double, 3>>& fct2, const std::vector<double>& m1, const std::vector<double>& m2, const std::vector<double>& sep1, const std::vector<double>& sep2, const std::vector<int>& separable, const bool);
#endif
#endif
}
} // namespace Bow::IO
