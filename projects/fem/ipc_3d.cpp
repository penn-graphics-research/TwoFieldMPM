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

int main(int argc, char* argv[])
{
    using T = double;
    static const int dim = 3;
    if (argc < 2) {
        puts("ERROR: please add parameters");
        puts("USAGE: ./fem_2d testcase");
        puts("       ./fem_2d testcase start_frame");
        exit(0);
    }
    int testcase = std::atoi(argv[1]);
    int start_frame = 0;
    if (argc == 3) {
        start_frame = std::atoi(argv[2]);
    }
    Bow::FEM::IPC::IPCSimulator<T, dim> simulator;

    if (testcase == 0) {
        // generate data
        T L = 2.0;
        int n_seg = 10;
        T dx = L / n_seg;
        // free cube
        Bow::Field<Bow::Vector<T, dim>> vertices;
        Bow::Field<Bow::Vector<int, dim + 1>> elements;
        Bow::Geometry::cube(Bow::Vector<int, dim>(n_seg, n_seg, n_seg), dx, vertices, elements);

        // fixed cube as ground
        Bow::Field<Bow::Vector<T, dim>> vertices2;
        Bow::Field<Bow::Vector<int, dim + 1>> elements2;
        Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1, 1), 3.0, vertices2, elements2);
        T dHat = 0.5 * dx;
        for (auto& i : vertices2) {
            i[0] *= 10;
            i[2] *= 10;
            i[1] -= 3 + dHat;
        }

        T Y = 4.0e4, nu = 0.2, rho = 1000.0;
        simulator.append(vertices, elements, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        simulator.append(vertices2, elements2, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        simulator.gravity = Bow::Vector<T, dim>(0, -10, 0);
        T nuC = 1.0;
        simulator.suggested_dt = nuC * dx / std::sqrt(Y / rho);
        simulator.frame_dt = int(1.0 / 24 / simulator.suggested_dt) * simulator.suggested_dt;
        simulator.end_frame = std::ceil(5.0 / simulator.frame_dt);
        simulator.dHat = dHat;
        for (int i = vertices.size(); i < int(vertices.size() + vertices2.size()); ++i) {
            simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(i, Bow::Geometry::STICKY, Bow::Vector<T, dim>(1, 0, 0), 0.0));
        }
        simulator.kappa = 0.1 * Y;
        simulator.tsMethod = Bow::BE;
        // Simulation
        simulator.output_directory = "outputs/ipc_3d_nSeg" + std::to_string(n_seg);
        simulator.initialize();
        simulator.run(start_frame);
    }

    // 3D max() approximation error
    if (testcase == 1) {
        // free cube
        int m_seg = 1;
        if (argc > 3) {
            m_seg = std::stoi(argv[3]);
            if (m_seg < 1) {
                Bow::Logging::error("mesh size invalid!");
                exit(-1);
            }
        }
        Bow::Field<Bow::Vector<T, dim>> vertices;
        Bow::Field<Bow::Vector<int, dim + 1>> elements;
        Bow::Geometry::cube(Bow::Vector<int, dim>(m_seg, m_seg, m_seg), 2.0 / m_seg, vertices, elements);

        // fixed cube as ground
        T L = 16.0;
        int n_seg = 8;
        if (argc > 4) {
            n_seg = std::stoi(argv[4]);
            if (n_seg < 1) {
                Bow::Logging::error("mesh size invalid!");
                exit(-1);
            }
        }
        T dx = L / n_seg;
        Bow::Field<Bow::Vector<T, dim>> vertices2;
        Bow::Field<Bow::Vector<int, dim + 1>> elements2;
        Bow::Geometry::cube(Bow::Vector<int, dim>(n_seg / 8 * 7, 1, n_seg / 8 * 3), dx, vertices2, elements2);
        // Bow::Geometry::cube(Bow::Vector<int, dim>(n_seg, 1, n_seg), dx, vertices2, elements2);
        T dHat = 0.1;
        if (argc > 5) {
            dHat = std::stod(argv[5]);
            if (dHat < 0) {
                Bow::Logging::error("dhat can't be negative!");
                exit(-1);
            }
        }
        for (auto& i : vertices2) {
            i[1] -= (dx + 2) / 2 + dHat;
            // i[2] += 4;
        }

        T Y = 2e11, nu = 0.3, rho = 8000.0;
        simulator.append(vertices, elements, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        for (unsigned int i = 0; i < simulator.m_v.size(); ++i) {
            simulator.m_v[i][0] = 1;
        }
        simulator.append(vertices2, elements2, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        simulator.gravity = Bow::Vector<T, dim>(0, -5, 0);
        simulator.suggested_dt = 1e-2;
        simulator.dHat = dHat;
        simulator.add_boundary_condition(new Bow::Geometry::HalfSpaceLevelSet<T, dim>(Bow::Geometry::STICKY, vertices2[1] + Bow::Vector<T, dim>(0, 0.001 * dHat, 0), Bow::Vector<T, dim>(0, 1, 0)));
        T kappa = 1e6;
        if (argc > 6) {
            kappa = std::stod(argv[6]);
            if (kappa <= 0) {
                Bow::Logging::error("kappa must be positive!");
                exit(-1);
            }
        }
        simulator.kappa = kappa;
        simulator.improved_maxOp = true;
        simulator.tsMethod = Bow::BE;
        // Simulation
        simulator.output_directory = "outputs/ipc_3d_" + std::to_string(testcase) + "_nSeg" + std::to_string(m_seg) + "_" + std::to_string(n_seg);
        simulator.initialize();
        simulator.timestep_callback = [&]() {
            FILE* out = fopen((simulator.output_directory + "/info.txt").c_str(), "a+");
            if (out) {
                Bow::Vector<T, dim> pos_com = Bow::Vector<T, dim>::Zero();
                for (unsigned int i = 0; i < vertices.size(); ++i) {
                    pos_com += simulator.m_x[i] * simulator.m_mass[i];
                }
                pos_com /= std::accumulate(simulator.m_mass.begin(), simulator.m_mass.begin() + vertices.size(), T(0));
                fprintf(out, "%.10le %.10le %.10le\n", pos_com[0], pos_com[1], pos_com[2]);
                fclose(out);
            }
            else {
                Bow::Logging::error("file creation error!");
                exit(-1);
            }
        };
        simulator.run(start_frame);
    }

    if (testcase == 2) {
        // free cube
        int m_seg = 1;
        if (argc > 3) {
            m_seg = std::stoi(argv[3]);
            if (m_seg < 1) {
                Bow::Logging::error("mesh size invalid!");
                exit(-1);
            }
        }
        Bow::Field<Bow::Vector<T, dim>> vertices;
        Bow::Field<Bow::Vector<int, dim + 1>> elements;
        Bow::Geometry::cube(Bow::Vector<int, dim>(m_seg, m_seg, m_seg), 2.0 / m_seg, vertices, elements);

        // fixed cube as ground
        T L = 16.0;
        int n_seg = 8;
        if (argc > 4) {
            n_seg = std::stoi(argv[4]);
            if (n_seg < 1) {
                Bow::Logging::error("mesh size invalid!");
                exit(-1);
            }
        }
        T dx = L / n_seg;
        Bow::Field<Bow::Vector<T, dim>> vertices2;
        Bow::Field<Bow::Vector<int, dim + 1>> elements2;
        Bow::Geometry::cube(Bow::Vector<int, dim>(n_seg / 8 * 7, 1, n_seg / 8 * 3), dx, vertices2, elements2);
        // Bow::Geometry::cube(Bow::Vector<int, dim>(n_seg, 1, n_seg), dx, vertices2, elements2);
        T dHat = 1e-3;
        if (argc > 5) {
            dHat = std::stod(argv[5]);
            if (dHat < 0) {
                Bow::Logging::error("dhat can't be negative!");
                exit(-1);
            }
        }
        for (auto& i : vertices2) {
            i[1] -= (dx + 2) / 2 + dHat;
            // i[2] += 4;
        }

        T Y = 2e11, nu = 0.3, rho = 8000.0;
        simulator.append(vertices, elements, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        for (unsigned int i = 0; i < simulator.m_v.size(); ++i) {
            simulator.m_v[i][0] = 1;
        }
        simulator.append(vertices2, elements2, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        simulator.gravity = Bow::Vector<T, dim>(1, -5, 0);
        simulator.suggested_dt = 1e-2;
        simulator.dHat = dHat;
        simulator.mu = 0.21;
        simulator.tol = 1e-6;
        simulator.epsv = 1e-5;
        simulator.add_boundary_condition(new Bow::Geometry::HalfSpaceLevelSet<T, dim>(Bow::Geometry::STICKY, vertices2[1] + Bow::Vector<T, dim>(0, 0.001 * dHat, 0), Bow::Vector<T, dim>(0, 1, 0)));
        T kappa = 1e6;
        if (argc > 6) {
            kappa = std::stod(argv[6]);
            if (kappa <= 0) {
                Bow::Logging::error("kappa must be positive!");
                exit(-1);
            }
        }
        simulator.kappa = kappa;
        simulator.improved_maxOp = true;
        simulator.tsMethod = Bow::BE;
        // Simulation
        simulator.output_directory = "outputs/ipc_3d_" + std::to_string(testcase) + "_friction/";
        simulator.initialize();
        simulator.run(start_frame);
    }

    if (testcase == 3) {
        Bow::Field<Bow::Vector<T, dim>> vertices_cube;
        Bow::Field<Bow::Vector<int, dim + 1>> elements_cube;
        Bow::Geometry::cube(Bow::Vector<int, dim>(10, 10, 10), 0.01, vertices_cube, elements_cube);
        for (auto& i : vertices_cube) {
            i += Bow::Vector<T, dim>(0.5, 1, -0.5);
        }

        simulator.append(vertices_cube, elements_cube, Bow::ConstitutiveModel::NEO_HOOKEAN, 1e5, 0.4, 1e3);

        Bow::Field<Bow::Vector<T, dim>> vertices;
        Bow::Field<Bow::Vector<int, dim>> elements;
        T length = 1;
        int nSeg = 20;
        for (int i = 0; i < nSeg + 1; ++i) {
            for (int j = 0; j < nSeg + 1; ++j) {
                vertices.emplace_back(Bow::Vector<T, dim>(i * (length / nSeg), 0, -j * (length / nSeg)));
            }
        }
        for (int i = 0; i < nSeg; ++i) {
            for (int j = 0; j < nSeg; ++j) {
                if ((i % 2) ^ (j % 2)) {
                    elements.emplace_back(Bow::Vector<int, dim>(
                        i * (nSeg + 1) + j, (i + 1) * (nSeg + 1) + j, i * (nSeg + 1) + j + 1));
                    elements.emplace_back(Bow::Vector<int, dim>(
                        (i + 1) * (nSeg + 1) + j, (i + 1) * (nSeg + 1) + j + 1, i * (nSeg + 1) + j + 1));
                }
                else {
                    elements.emplace_back(Bow::Vector<int, dim>(
                        i * (nSeg + 1) + j, (i + 1) * (nSeg + 1) + j + 1, i * (nSeg + 1) + j + 1));
                    elements.emplace_back(Bow::Vector<int, dim>(
                        i * (nSeg + 1) + j, (i + 1) * (nSeg + 1) + j, (i + 1) * (nSeg + 1) + j + 1));
                }
            }
        }

        simulator.append(vertices, elements,
            3e-4, // thickness
            Bow::ConstitutiveModel::NEO_HOOKEAN_MEMBRANE, 1e6, 0.3, // E and nu
            Bow::ConstitutiveModel::DISCRETE_HINGE_BENDING, 1e6, 0.3, // E and nu
            500); // density

        simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(vertices_cube.size() + 0, Bow::Geometry::STICKY, Bow::Vector<T, dim>(0, 0, 1), 0));
        simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(vertices_cube.size() + nSeg, Bow::Geometry::STICKY, Bow::Vector<T, dim>(0, 0, 1), 0));
        simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(vertices_cube.size() + vertices.size() - 1, Bow::Geometry::STICKY, Bow::Vector<T, dim>(0, 0, 1), 0));
        simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(vertices_cube.size() + vertices.size() - 1 - nSeg, Bow::Geometry::STICKY, Bow::Vector<T, dim>(0, 0, 1), 0));

        simulator.gravity = Bow::Vector<T, dim>(0, -9.81, 0);

        simulator.output_directory = "outputs/cube_on_cloth";
        simulator.initialize();
        simulator.run(start_frame);
    }
}