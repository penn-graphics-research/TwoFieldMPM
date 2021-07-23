#ifndef FEM_SIMULATOR_H
#define FEM_SIMULATOR_H

#include "FEMSimulator.h"

namespace Bow::FEM::IPC {
template <class T, int dim, class Optimizer = Optimization::AugmentedLagrangianNewton<T, dim, int>>
class IPCSimulator : public FEMSimulator<T, dim, Optimizer> {
public:
    using Base = FEMSimulator<T, dim, Optimizer>;
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    using TimeIntegrator = typename FEMSimulator<T, dim, Optimizer>::TimeIntegrator;
    Field<int> m_boundary_points;
    Field<Vector<int, 2>> m_boundary_edges;
    Field<Vector<int, 3>> m_boundary_faces;
    Field<T> m_boundary_point_area; // indexed by boundary point local id
    Field<std::set<int>> m_boundary_point_nb; // indexed by boundary point local id, stores boundary point global id
    Field<int8_t> m_boundary_point_type; // indexed by boundary point local id
    // 0: on codimensional segments/particles, 1: on the border of codimensional surfaces, 2: other (interior surface points)
    Field<T> m_boundary_edge_area; // indexed by boundary edge local id
    Field<std::set<int>> m_boundary_edge_pnb; // indexed by boundary edge local id and stores boundary point local id
    T dHat = 1e-3;
    T kappa = 1e4;
    bool improved_maxOp = false;
    //NOTE: improved_maxOp = true in 2D supports both codimensional and non-manifold meshes,
    // but in 3D it supports codimensional but not non-manifold meshes

    // friction related
    T mu = 0;
    T epsv = 1e-7;
    bool lag = false;

    // internal helper function
    virtual void append_contact_info(
        const Field<int>& new_boundary_points,
        const Field<Vector<int, 2>>& new_boundary_edges,
        const Field<Vector<int, 3>>& new_boundary_faces)
    {
        std::map<int, int> boundary_point_id_map;
        for (size_t i = 0; i < new_boundary_points.size(); ++i) {
            boundary_point_id_map[new_boundary_points[i]] = m_boundary_points.size() + i;
        }
        std::map<std::pair<int, int>, int> boundary_edge_id_map;
        for (size_t i = 0; i < new_boundary_edges.size(); ++i) {
            boundary_edge_id_map[std::pair<int, int>(new_boundary_edges[i][0], new_boundary_edges[i][1])] = m_boundary_edges.size() + i;
        }

        m_boundary_edges.insert(m_boundary_edges.end(), new_boundary_edges.begin(), new_boundary_edges.end());
        m_boundary_edge_area.resize(m_boundary_edges.size(), 0);
        m_boundary_edge_pnb.resize(m_boundary_edges.size());
        m_boundary_points.insert(m_boundary_points.end(), new_boundary_points.begin(), new_boundary_points.end());
        m_boundary_point_area.resize(m_boundary_points.size(), 0);
        m_boundary_point_nb.resize(m_boundary_points.size());
        m_boundary_point_type.resize(m_boundary_points.size(), 2);
        m_boundary_faces.insert(m_boundary_faces.end(), new_boundary_faces.begin(), new_boundary_faces.end());
        if constexpr (dim == 2) {
            for (const auto& eI : new_boundary_edges) {
                T eLen = (this->m_X[eI[0]] - this->m_X[eI[1]]).norm();
                auto finder0 = boundary_point_id_map.find(eI[0]);
                BOW_ASSERT_INFO(finder0 != boundary_point_id_map.end(), "cannot find boundary point id");
                auto finder1 = boundary_point_id_map.find(eI[1]);
                BOW_ASSERT_INFO(finder1 != boundary_point_id_map.end(), "cannot find boundary point id");
                m_boundary_point_area[finder0->second] += eLen / 2;
                m_boundary_point_area[finder1->second] += eLen / 2;
                m_boundary_point_nb[finder0->second].insert(eI[1]);
                m_boundary_point_nb[finder1->second].insert(eI[0]);
            }
        }
        else {
            for (const auto& fI : new_boundary_faces) {
                T face_area = (this->m_X[fI[1]] - this->m_X[fI[0]]).cross(this->m_X[fI[2]] - this->m_X[fI[0]]).norm() / 2;
                int bp_local_id[3];
                for (int i = 0; i < 3; ++i) {
                    auto finder = boundary_point_id_map.find(fI[i]);
                    BOW_ASSERT_INFO(finder != boundary_point_id_map.end(), "cannot find boundary point id");
                    bp_local_id[i] = finder->second;
                }

                for (int i = 0; i < 3; ++i) {
                    m_boundary_point_area[bp_local_id[i]] += face_area / 3;
                    m_boundary_point_nb[bp_local_id[i]].insert(fI[(i + 1) % 3]);
                    m_boundary_point_nb[bp_local_id[i]].insert(fI[(i + 2) % 3]);

                    auto finder = boundary_edge_id_map.find(std::pair<int, int>(fI[i], fI[(i + 1) % 3]));
                    if (finder == boundary_edge_id_map.end()) {
                        finder = boundary_edge_id_map.find(std::pair<int, int>(fI[(i + 1) % 3], fI[i]));
                        BOW_ASSERT_INFO(finder != boundary_edge_id_map.end(), "cannot find boundary edge id");
                    }
                    m_boundary_edge_area[finder->second] += face_area / 3;
                    m_boundary_edge_pnb[finder->second].insert(bp_local_id[(i + 2) % 3]);
                }
            }
            for (size_t beI = m_boundary_edge_pnb.size() - new_boundary_edges.size(); beI < m_boundary_edge_pnb.size(); ++beI) {
                if (m_boundary_edge_pnb[beI].size() <= 1) {
                    for (int i = 0; i < 2; ++i) {
                        auto finder = boundary_point_id_map.find(m_boundary_edges[beI][i]);
                        BOW_ASSERT_INFO(finder != boundary_point_id_map.end(), "cannot find boundary point id");
                        m_boundary_point_type[finder->second] = m_boundary_edge_pnb[beI].size();
                    }
                }
            }
        }
    }

    // codim0 elements
    virtual void append(const Field<TV>& x, const Field<Vector<int, dim + 1>>& elem, const int type, const T E, const T nu, const T density) override
    {
        Field<int> new_boundary_points;
        Field<Vector<int, 2>> new_boundary_edges;
        Field<Vector<int, 3>> new_boundary_faces;
        if constexpr (dim == 2)
            Geometry::find_boundary(elem, new_boundary_edges, new_boundary_points);
        else
            Geometry::find_boundary(elem, new_boundary_faces, new_boundary_edges, new_boundary_points);
        new_boundary_points += int(this->m_x.size());
        new_boundary_edges += Vector<int, 2>(this->m_x.size(), this->m_x.size());
        new_boundary_faces += Vector<int, 3>(this->m_x.size(), this->m_x.size(), this->m_x.size());

        // this will fill m_x (this->m_x.size() will change) and get ready for append_contact_info()
        Base::append(x, elem, type, E, nu, density);

        append_contact_info(new_boundary_points, new_boundary_edges, new_boundary_faces);
    }

    // codim1 elements
    virtual void append(const Field<TV>& x, const Field<Vector<int, dim>>& elem_codim1, const T thickness,
        const int type_memb, const T E_memb, const T nu_memb,
        const int type_bend, const T E_bend, const T nu_bend,
        const T density) override
    {
        if constexpr (dim == 3) {
            Field<int> new_boundary_points;
            Field<Vector<int, 2>> new_boundary_edges;
            Field<Vector<int, 3>> new_boundary_faces = elem_codim1;
            Geometry::collect_edge_and_point(new_boundary_faces, new_boundary_edges, new_boundary_points);
            new_boundary_points += int(this->m_x.size());
            new_boundary_edges += Vector<int, 2>(this->m_x.size(), this->m_x.size());
            new_boundary_faces += Vector<int, 3>(this->m_x.size(), this->m_x.size(), this->m_x.size());

            // this will fill m_x (this->m_x.size() will change) and get ready for append_contact_info()
            Base::append(x, elem_codim1, thickness, type_memb, E_memb, nu_memb, type_bend, E_bend, nu_bend, density);

            append_contact_info(new_boundary_points, new_boundary_edges, new_boundary_faces);
        }
    }

    void additional_options(TimeIntegrator& update_op, std::vector<std::shared_ptr<EnergyOp<T, dim>>>& energy_terms, const T dt) override
    {
        std::shared_ptr<IPC::IpcEnergyOp<T, dim>> ipc_energy;
        if constexpr (dim == 2)
            ipc_energy = std::make_shared<IPC::IpcEnergyOp<T, 2>>(m_boundary_points, m_boundary_edges, this->m_mass, m_boundary_point_area, m_boundary_point_nb);
        else
            ipc_energy = std::make_shared<IPC::IpcEnergyOp<T, 3>>(m_boundary_points, m_boundary_edges, m_boundary_faces,
                this->m_X, this->m_mass, m_boundary_point_area, m_boundary_point_type, m_boundary_point_nb,
                m_boundary_edge_area, m_boundary_edge_pnb);
        T tsParam[2][3] = { //TODO: use the one in FEMTimeIntegrator.h
            { 1, 0.5, 1 },
            { 0.5, 0.25, 0.5 }
        };
        ipc_energy->update_weight_and_xhat(this->m_x, this->m_v, this->m_a, this->m_x1, dt, tsParam[this->tsMethod], this->tsMethod);
        ipc_energy->initialize_friction(mu, epsv, dt);
        ipc_energy->dHat = dHat;
        ipc_energy->kappa = kappa;
        ipc_energy->improved_maxOp = improved_maxOp;
        energy_terms.push_back(ipc_energy);
        _ipc_ptr = ipc_energy.get();
        if (this->lag && !this->static_sim)
            update_op.lagging_callback = [&]() {
                _ipc_ptr->initialize_friction(mu, epsv, dt);
            };
    }

    virtual void dump_output(int frame_num) override
    {
        if constexpr (dim == 2)
            IO::write_ply(this->output_directory + "/" + std::to_string(frame_num) + ".ply", this->m_x, this->m_elem);
        else {
            IO::write_ply(this->output_directory + "/" + std::to_string(frame_num) + ".ply", this->m_x, m_boundary_faces);
        }
    }

protected:
    IPC::IpcEnergyOp<T, dim>* _ipc_ptr;
};
} // namespace Bow::FEM::IPC

#endif