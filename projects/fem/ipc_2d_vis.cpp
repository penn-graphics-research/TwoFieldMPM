#include <Bow/Types.h>
#include <Bow/Math/PolarDecomposition.h>
#include <Bow/Math/SVD.h>
#include <iostream>
#include <fstream>
#include <Bow/Geometry/Primitives.h>
#include <Bow/Utils/Timer.h>
#include <Bow/IO/ply.h>
#include <Bow/Geometry/Query.h>
#include <Bow/Utils/FileSystem.h>
#include <Bow/Utils/Logging.h>
#include <Bow/Simulator/FEM/IPCSimulator.h>
#include <Bow/Geometry/OrthogonalBasis.h>
#include <igl/opengl/glfw/Viewer.h>

int main(int argc, char* argv[])
{
    static const int dim = 2;
    using T = double;

    if (argc < 2 || argc > 6) {
        puts("ERROR: parameters");
        puts("USAGE: ./ipc_2d_vis bc_velocity <max_iter> <solver_tol> <time_step> <mu>");
        exit(0);
    }

    T bc_vel = std::atof(argv[1]);
    int max_iter = 300;
    if (argc > 2) {
        max_iter = std::atoi(argv[2]);
    }
    T tol = 1e-2;
    if (argc > 3) {
        tol = std::atof(argv[3]);
    }
    T h = 0.02;
    if (argc > 4) {
        h = std::atof(argv[4]);
    }
    T mu = 0.1;
    if (argc > 5) {
        mu = std::atof(argv[4]);
    }

    Bow::FEM::IPC::IPCSimulator<T, dim> simulator;

    // sim solve settings
    simulator.tol = tol;
    simulator.max_iter = max_iter;

    // body forces
    simulator.gravity = Bow::Vector<T, dim>(0, -5);

    // moduli, density
    T Y = 4.0e3, nu = 0.2, rho = 100.0;

    // barriers
    simulator.dHat = 1e-3;
    simulator.kappa = 0.1 * Y;

    // friction
    simulator.mu = mu;
    simulator.epsv = 1e-3;
    simulator.lag = false;

    // stepper
    simulator.suggested_dt = h; // TODO: add "hard" dt to simulator
    simulator.frame_dt = int(1.0 / 24 / simulator.suggested_dt) * simulator.suggested_dt; // TODO: add advance_frame() to simulator to take a step of run() per frame
    simulator.end_frame = std::ceil(5.0 / simulator.frame_dt);
    simulator.tsMethod = Bow::BE;

    // generate meshed domains:

    // collidees
    int num_verts = 0;
    Bow::Field<Bow::Vector<T, dim>> vertices1;
    Bow::Field<Bow::Vector<int, dim + 1>> elements1;
    Bow::IO::read_ply(EXAMPLE_DATA_PATH "EIPC/Sharkey.ply", vertices1, elements1);
    simulator.append(vertices1, elements1, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
    num_verts += vertices1.size();

    Bow::Field<Bow::Vector<T, dim>> vertices2;
    Bow::Field<Bow::Vector<int, dim + 1>> elements2;
    Bow::IO::read_ply(EXAMPLE_DATA_PATH "EIPC/Sharkey.ply", vertices2, elements2);
    for (auto& i : vertices2) {
        i[1] += 1;
    }
    simulator.append(vertices2, elements2, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
    num_verts += vertices2.size();

    // scripted cube
    Bow::Field<Bow::Vector<T, dim>> vertices_s;
    Bow::Field<Bow::Vector<int, dim + 1>> elements_s;
    Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1), 0.5, vertices_s, elements_s);
    for (auto& i : vertices_s) {
        i[1] -= 2 + simulator.dHat;
    }
    simulator.append(vertices_s, elements_s, Bow::ConstitutiveModel::NONE, Y, nu, rho);

    // set boundary conditions on scripted cube
    for (int i = num_verts; i < num_verts + int(vertices_s.size()); ++i) {
        simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(i, Bow::Geometry::STICKY, Bow::Vector<T, dim>(0, 1), bc_vel));
    }
    int scripted_verts_begin = num_verts;
    num_verts += int(vertices_s.size());
    int scripted_verts_end = num_verts;

    // fixed collider (floor)
    Bow::Field<Bow::Vector<T, dim>> vertices_f;
    Bow::Field<Bow::Vector<int, dim + 1>> elements_f;
    vertices_f.push_back(Bow::Vector<T, dim>(-5, -4));
    vertices_f.push_back(Bow::Vector<T, dim>(5, -4));
    vertices_f.push_back(Bow::Vector<T, dim>(5, -4.01));
    vertices_f.push_back(Bow::Vector<T, dim>(-5, -4.01));
    elements_f.push_back(Bow::Vector<int, dim + 1>(2, 1, 0));
    elements_f.push_back(Bow::Vector<int, dim + 1>(3, 2, 0));
    simulator.append(vertices_f, elements_f, Bow::ConstitutiveModel::NONE, Y, nu, rho);
    // set boundary conditions on fixed collider
    for (int i = num_verts; i < num_verts + int(vertices_f.size()); ++i) {
        simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(i, Bow::Geometry::STICKY, Bow::Vector<T, dim>(0, 1), 0));
    }

    // sim output
    simulator.output_directory = "outputs/ipc_2_viewer";

    // init sim
    simulator.initialize();

    // viewer
    igl::opengl::glfw::Viewer viewer;
    Eigen::Map<Eigen::MatrixXd> x_mat(reinterpret_cast<double*>(simulator.m_x.data()), dim, simulator.m_x.size());
    Eigen::Map<Eigen::MatrixXi> elem_mat(reinterpret_cast<int*>(simulator.m_elem.data()), (dim + 1), simulator.m_elem.size());
    viewer.data().set_mesh((x_mat.transpose()), (elem_mat.transpose()));
    viewer.data().set_face_based(true);

    T time_elapsed = 0;
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer& viewer) -> bool {
        if (viewer.core().is_animating) {
            simulator.advance(simulator.suggested_dt);
            viewer.data().set_vertices((x_mat.transpose()));
            time_elapsed += h;

            // update boundary conditions on fixed cube
            Bow::Vector<T, dim> new_vel;

            if (time_elapsed < 1) {
                new_vel << 0, 1;
                new_vel *= bc_vel;
                for (int i = scripted_verts_begin; i < scripted_verts_end; ++i) {
                    simulator.update_index_based_bc_velocity(i, new_vel);
                }
            }
            else if (time_elapsed < 2) {
                new_vel << 1, 0;
                new_vel *= bc_vel;
                for (int i = scripted_verts_begin; i < scripted_verts_end; ++i) {
                    simulator.update_index_based_bc_velocity(i, new_vel);
                }
            }
            else if (time_elapsed < 3) {
                new_vel << 0, -1;
                new_vel *= bc_vel;
                for (int i = scripted_verts_begin; i < scripted_verts_end; ++i) {
                    simulator.update_index_based_bc_velocity(i, new_vel);
                }
            }
            else if (time_elapsed < 4) {
                new_vel << -1, 0;
                new_vel *= bc_vel;
                for (int i = scripted_verts_begin; i < scripted_verts_end; ++i) {
                    simulator.update_index_based_bc_velocity(i, new_vel);
                }
            }
            else if (time_elapsed < 12) {
                new_vel << 0, 1;
                new_vel *= bc_vel;
                for (int i = scripted_verts_begin; i < scripted_verts_end; ++i) {
                    simulator.update_index_based_bc_velocity(i, new_vel);
                }
            }
            else {
                new_vel << 0, 0;
                for (int i = scripted_verts_begin; i < scripted_verts_end; ++i) {
                    simulator.update_index_based_bc_velocity(i, new_vel);
                }
            }
        }
        return false;
    };
    viewer.launch();
}