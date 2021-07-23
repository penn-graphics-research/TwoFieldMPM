#include <Bow/Types.h>
#include <Bow/Math/PolarDecomposition.h>
#include <Bow/Math/SVD.h>
#include <iostream>
#include <fstream>
#include <numeric>
#include <Bow/Geometry/Primitives.h>
#include <Bow/Utils/Timer.h>
#include <Bow/IO/ply.h>
#include <Bow/Geometry/Query.h>
#include <Bow/Utils/FileSystem.h>
#include <Bow/Utils/Logging.h>
#include <Bow/Simulator/FEM/FEMSimulator.h>
#include <Bow/Simulator/FEM/IPCSimulator.h>
#include <Bow/Geometry/OrthogonalBasis.h>
#include <Bow/Utils/ResultRecorder.h>

int main(int argc, char* argv[])
{
    using T = double;
    static const int dim = 2;
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

    // 2D Signorini benchmark
    if (testcase == 0) {
        // generate data
        T L = 2.0;
        int n_seg = 24;
        if (argc > 3) {
            n_seg = std::stoi(argv[3]);
            if (n_seg < 2 || n_seg % 2 == 1) {
                Bow::Logging::error("n_seg needs to be positive and even!");
                exit(-1);
            }
        }
        T dx = L / n_seg;
        // free cube
        Bow::Field<Bow::Vector<T, dim>> vertices;
        Bow::Field<Bow::Vector<int, dim + 1>> elements;
        Bow::Geometry::cube(Bow::Vector<int, dim>(n_seg, n_seg), dx, vertices, elements);

        // fixed cube as ground
        Bow::Field<Bow::Vector<T, dim>> vertices2;
        Bow::Field<Bow::Vector<int, dim + 1>> elements2;
        Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1), 3.0, vertices2, elements2);
        T dHat = 0.5 * dx * dx * 5;
        // T dHat = 0.5 / 80;
        // T dHat = 0.5 * dx;
        for (auto& i : vertices2) {
            i[1] -= 2.5 + dHat;
        }

        T Y = 4.0e3, nu = 0.2, rho = 100.0;
        simulator.append(vertices, elements, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        simulator.append(vertices2, elements2, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        simulator.gravity = Bow::Vector<T, dim>(0, -5);
        T nuC = 1;
        if (argc > 4) {
            nuC = std::stod(argv[4]);
            if (nuC <= 0) {
                Bow::Logging::error("nuC must be positive!");
                exit(-1);
            }
        }
        simulator.suggested_dt = nuC * 2 / 10 / std::sqrt(Y / rho);
        simulator.frame_dt = simulator.suggested_dt;
        simulator.end_frame = std::ceil(5.0 / simulator.frame_dt);
        simulator.frame_batch = std::max(1, int(1.0 / 24 / simulator.frame_dt));
        simulator.dHat = dHat;
        simulator.add_boundary_condition(new Bow::Geometry::HalfSpaceLevelSet<T, dim>(Bow::Geometry::STICKY, vertices2[1] + Bow::Vector<T, dim>(0, 0.001 * dHat), Bow::Vector<T, dim>(0, 1)));
        simulator.kappa = 0.1 * Y;
        simulator.tsMethod = Bow::BDF2;
        // Simulation
        simulator.output_directory = "outputs/ipc_2d_" + std::to_string(testcase) + "_nSeg" + std::to_string(n_seg);
        simulator.initialize();
        simulator.timestep_callback = [&]() {
            FILE* out = fopen((simulator.output_directory + "/info.txt").c_str(), "a+");
            if (out) {
                static const int vI = vertices.size() / 2 + n_seg / 2;
                fprintf(out, "%.10le %.10le\n", simulator.m_x[vI][0], simulator.m_x[vI][1]);
                fclose(out);
            }
            else {
                Bow::Logging::error("file creation error!");
                exit(-1);
            }
        };
        simulator.run(start_frame);
    }

    // 2D Signorini benchmark, an extrusion from 1D
    if (testcase == 1) {
        // generate data
        T L = 10.0;
        int n_seg = 24;
        if (argc > 3) {
            n_seg = std::stoi(argv[3]);
            if (n_seg < 2) {
                Bow::Logging::error("mesh size too small!");
                exit(-1);
            }
        }
        T dx = L / n_seg;

        // free cube
        Bow::Field<Bow::Vector<T, dim>> vertices;
        Bow::Field<Bow::Vector<int, dim + 1>> elements;
        Bow::Geometry::cube(Bow::Vector<int, dim>(n_seg, n_seg), dx, vertices, elements);

        // fixed cube as ground
        Bow::Field<Bow::Vector<T, dim>> vertices2;
        Bow::Field<Bow::Vector<int, dim + 1>> elements2;
        Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1), 20.0, vertices2, elements2);
        for (auto& i : vertices2) {
            i[1] -= 20;
        }
        simulator.add_boundary_condition(new Bow::Geometry::HalfSpaceLevelSet<T, dim>(Bow::Geometry::STICKY, vertices2[1] + Bow::Vector<T, dim>(0, 0.001), Bow::Vector<T, dim>(0, 1)));

        T Y = 1800, nu = 0.2, rho = 1.0;
        simulator.append(vertices, elements, Bow::ConstitutiveModel::LINEAR_ELASTICITY, Y, nu, rho);
        simulator.append(vertices2, elements2, Bow::ConstitutiveModel::LINEAR_ELASTICITY, Y, nu, rho);
        simulator.gravity = Bow::Vector<T, dim>(0, -10);
        simulator.dHat = 0.5 * dx;
        simulator.kappa = 0.1 * Y;

        simulator.tsMethod = Bow::BE;
        T nuC = 0.75;
        if (argc > 4) {
            nuC = std::stod(argv[4]);
            if (nuC <= 0) {
                Bow::Logging::error("nuC must be positive!");
                exit(-1);
            }
        }
        simulator.suggested_dt = nuC * dx / std::sqrt(Y / rho);
        simulator.frame_dt = simulator.suggested_dt;
        simulator.end_frame = std::ceil(20.0 / simulator.frame_dt);
        simulator.frame_batch = std::max(1, int(1.0 / 24 / simulator.frame_dt));

        simulator.output_directory = "outputs/ipc_2d_" + std::to_string(testcase) + "_nSeg" + std::to_string(n_seg);

        // Simulation
        simulator.initialize();
        simulator.run(start_frame);
    }

    // 2D max() approximation error
    if (testcase == 2) {
        // generate data

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
        Bow::Geometry::cube(Bow::Vector<int, dim>(m_seg, m_seg), 2.0 / m_seg, vertices, elements);

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
        Bow::Geometry::cube(Bow::Vector<int, dim>(n_seg, 1), dx, vertices2, elements2);
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
        }

        T Y = 2e11, nu = 0.3, rho = 8000.0;
        simulator.append(vertices, elements, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        for (unsigned int i = 0; i < simulator.m_v.size(); ++i) {
            simulator.m_v[i][0] = 1;
        }
        simulator.append(vertices2, elements2, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        simulator.gravity = Bow::Vector<T, dim>(0, -5);
        simulator.suggested_dt = 1e-2;
        simulator.dHat = dHat;
        simulator.add_boundary_condition(new Bow::Geometry::HalfSpaceLevelSet<T, dim>(Bow::Geometry::STICKY, vertices2[1] + Bow::Vector<T, dim>(0, 0.001 * dHat), Bow::Vector<T, dim>(0, 1)));
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
        simulator.output_directory = "outputs/ipc_2d_" + std::to_string(testcase) + "_nSeg" + std::to_string(m_seg) + "_" + std::to_string(n_seg);
        simulator.initialize();
        simulator.timestep_callback = [&]() {
            FILE* out = fopen((simulator.output_directory + "/info.txt").c_str(), "a+");
            if (out) {
                Bow::Vector<T, dim> pos_com = Bow::Vector<T, dim>::Zero();
                for (unsigned int i = 0; i < vertices.size(); ++i) {
                    pos_com += simulator.m_x[i] * simulator.m_mass[i];
                }
                pos_com /= std::accumulate(simulator.m_mass.begin(), simulator.m_mass.begin() + vertices.size(), T(0));
                fprintf(out, "%.10le %.10le\n", pos_com[0], pos_com[1]);
                fclose(out);
            }
            else {
                Bow::Logging::error("file creation error!");
                exit(-1);
            }
        };
        simulator.run(start_frame);
    }

    // 2D Signorini benchmark (static)
    if (testcase == 3) {
        // generate data
        T L = 2.0;
        int n_seg = 24;
        if (argc > 3) {
            n_seg = std::stoi(argv[3]);
            if (n_seg < 2 || n_seg % 2 == 1) {
                Bow::Logging::error("n_seg needs to be positive and even!");
                exit(-1);
            }
        }
        T dx = L / n_seg;
        // free cube
        Bow::Field<Bow::Vector<T, dim>> vertices;
        Bow::Field<Bow::Vector<int, dim + 1>> elements;
        Bow::Geometry::cube(Bow::Vector<int, dim>(n_seg, n_seg), dx, vertices, elements);

        // fixed cube as ground
        Bow::Field<Bow::Vector<T, dim>> vertices2;
        Bow::Field<Bow::Vector<int, dim + 1>> elements2;
        Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1), 3.0, vertices2, elements2);
        T dHat = 0.5 * dx * dx * 5;
        for (auto& i : vertices2) {
            i[1] -= 2.5 + dHat * 0.99;
        }

        T Y = 4.0e3, nu = 0.2, rho = 100.0;
        simulator.append(vertices, elements, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        simulator.append(vertices2, elements2, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        simulator.gravity = Bow::Vector<T, dim>(0, -5);
        simulator.suggested_dt = 1;
        simulator.frame_dt = 1;
        simulator.end_frame = 1;
        simulator.dHat = dHat;
        simulator.add_boundary_condition(new Bow::Geometry::HalfSpaceLevelSet<T, dim>(Bow::Geometry::STICKY, vertices2[1] + Bow::Vector<T, dim>(0, 0.001 * dHat), Bow::Vector<T, dim>(0, 1)));
        simulator.kappa = 0.1 * Y;
        simulator.static_sim = true;
        simulator.tsMethod = Bow::BE;
        simulator.tol = 1e-6;
        // Simulation
        simulator.output_directory = "outputs/ipc_2d_" + std::to_string(testcase) + "_nSeg" + std::to_string(n_seg);
        simulator.initialize();
        simulator.timestep_callback = [&]() {
            FILE* out = fopen((simulator.output_directory + "/info.txt").c_str(), "a+");
            if (out) {
                static const int vI = vertices.size() / 2 + n_seg / 2;
                fprintf(out, "%.10le %.10le\n", simulator.m_x[vI][0], simulator.m_x[vI][1]);
                fclose(out);
            }
            else {
                Bow::Logging::error("file creation error!");
                exit(-1);
            }
        };
        simulator.run(start_frame);
    }

    if (testcase == 100) {
        // generate data
        T L = 2.0;
        int n_seg = 24;
        T dx = L / n_seg;
        // free cube
        Bow::Field<Bow::Vector<T, dim>> vertices;
        Bow::Field<Bow::Vector<int, dim + 1>> elements;
        Bow::Geometry::cube(Bow::Vector<int, dim>(n_seg, n_seg), dx, vertices, elements);

        // fixed cube as ground
        Bow::Field<Bow::Vector<T, dim>> vertices2;
        Bow::Field<Bow::Vector<int, dim + 1>> elements2;
        Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1), 3.0, vertices2, elements2);
        T dHat = 0.5 * dx;
        for (auto& i : vertices2) {
            i[0] *= 10;
            i[1] -= 2.5 + dHat;
        }

        // transformation
        Bow::Matrix<T, 2, 2> transformation = Bow::Geometry::extend_basis(Bow::Vector<T, dim>(Bow::Vector<T, dim>(5.0, 1.0).normalized()));
        for (size_t i = 0; i < vertices.size(); ++i) {
            vertices[i] = transformation.transpose() * vertices[i];
        }
        for (size_t i = 0; i < vertices2.size(); ++i) {
            vertices2[i] = transformation.transpose() * vertices2[i];
        }

        T Y = 4.e6, nu = 0.2, rho = 100.0;
        simulator.append(vertices, elements, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        simulator.append(vertices2, elements2, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        simulator.gravity = Bow::Vector<T, dim>(0, -5);
        simulator.suggested_dt = 1 / 24.;
        simulator.dHat = dHat;
        simulator.add_boundary_condition(new Bow::Geometry::HalfSpaceLevelSet<T, dim>(Bow::Geometry::STICKY, vertices2[1] + Bow::Vector<T, dim>(0, 0.001 * dHat), Bow::Vector<T, dim>(1, 5)));
        simulator.kappa = 0.1 * Y;
        simulator.tsMethod = Bow::BE;
        // Simulation
        simulator.output_directory = "outputs/ipc_2d_nSeg" + std::to_string(n_seg);
        simulator.initialize();
        simulator.run(start_frame);
    }

    else if (testcase == 101) // friction
    {
        // generate data
        T L = 2.0;
        int n_seg = 24;
        T dx = L / n_seg;
        // free cube
        Bow::Field<Bow::Vector<T, dim>> vertices;
        Bow::Field<Bow::Vector<int, dim + 1>> elements;
        Bow::Geometry::cube(Bow::Vector<int, dim>(n_seg, n_seg), dx, vertices, elements);

        // fixed cube as ground
        Bow::Field<Bow::Vector<T, dim>> vertices2;
        Bow::Field<Bow::Vector<int, dim + 1>> elements2;
        Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1), 3.0, vertices2, elements2);
        T dHat = 0.5 * dx;
        for (auto& i : vertices2) {
            i[0] *= 10;
            i[1] -= 2.5 + dHat;
        }

        // transformation
        Bow::Matrix<T, 2, 2> transformation = Bow::Geometry::extend_basis(Bow::Vector<T, dim>(5.0, 1.0).normalized());
        for (size_t i = 0; i < vertices.size(); ++i) {
            vertices[i] = transformation.transpose() * vertices[i];
        }
        for (size_t i = 0; i < vertices2.size(); ++i) {
            vertices2[i] = transformation.transpose() * vertices2[i];
        }

        T Y = 4.e6, nu = 0.2, rho = 100.0;
        simulator.append(vertices, elements, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        simulator.append(vertices2, elements2, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        simulator.add_boundary_condition(new Bow::Geometry::HalfSpaceLevelSet<T, dim>(Bow::Geometry::STICKY, vertices2[1] + Bow::Vector<T, dim>(0, 0.001 * dHat), Bow::Vector<T, dim>(1, 5)));
        simulator.gravity = Bow::Vector<T, dim>(0, -5);
        //friction
        simulator.mu = 0.21;
        simulator.epsv = 1e-5;
        simulator.lag = true;

        simulator.suggested_dt = 1 / 24.;
        simulator.kappa = 0.1 * Y;
        simulator.dHat = dHat;
        simulator.tsMethod = Bow::BE;
        // Simulation
        simulator.output_directory = "outputs/ipc_2d_friction_nSeg" + std::to_string(n_seg);
        simulator.initialize();
        simulator.run(start_frame);
    }

    else if (testcase >= 102 && testcase <= 110) {
        simulator.static_sim = true;
        // generate data
        T L = 2.0;
        int n_seg = 2;
        T dx = L / n_seg;
        // free cube
        Bow::Field<Bow::Vector<T, dim>> vertices;
        Bow::Field<Bow::Vector<int, dim + 1>> elements;
        Bow::Geometry::cube(Bow::Vector<int, dim>(n_seg, n_seg), dx, vertices, elements);

        // fixed cube as ground
        Bow::Field<Bow::Vector<T, dim>> vertices2;
        Bow::Field<Bow::Vector<int, dim + 1>> elements2;
        Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1), 3.0, vertices2, elements2);
        // T dHat = 0.5 * dx;
        T dHat = 1e-3;
        for (auto& i : vertices2) {
            i[0] *= 100;
            i[0] += 50;
            i[1] -= 2.5 + 0.5 * dHat;
        }
        T Y = 1.e12, nu = 0.3, rho = 1.0;
        simulator.append(vertices, elements, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        simulator.append(vertices2, elements2, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        for (int i = vertices.size(); i < int(vertices.size() + vertices2.size()); ++i) {
            simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(i, Bow::Geometry::STICKY, Bow::Vector<T, dim>(1, 0), 0.0));
        }
        for (int i = 0; i < (int)vertices.size(); ++i) {
            if (vertices[i][1] < -0.999 && vertices[i][0] < -0.999)
                simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(i, Bow::Geometry::SLIP, Bow::Vector<T, dim>(1, 0), 0.0));
        }
        simulator.gravity = Bow::Vector<T, dim>(2, -10);
        //friction
        if (testcase == 102)
            simulator.mu = 0.20;
        else if (testcase == 103)
            simulator.mu = 0.1;
        else if (testcase == 104)
            simulator.mu = 0.21;
        else if (testcase == 105)
            simulator.mu = 0;
        simulator.lag = true;

        simulator.suggested_dt = 1e-3;
        simulator.epsv = 1e-5;
        simulator.frame_dt = 1 / 24.;
        simulator.kappa = 0.1 * Y;
        simulator.dHat = dHat;
        simulator.tol = 1e-12;
        simulator.timestep_callback = [&]() {
            if (simulator.sub_step == 0)
                simulator.static_sim = true;
            else if (simulator.sub_step == 1) {
                simulator.static_sim = false;
                simulator.m_v -= simulator.m_v;
                simulator.m_v += Bow::Vector<T, dim>(1e-6, 0);
                simulator.m_a -= simulator.m_a;
                simulator.BC.clear();
                for (int i = vertices.size(); i < int(vertices.size() + vertices2.size()); ++i) {
                    simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(i, Bow::Geometry::STICKY, Bow::Vector<T, dim>(1, 0), 0.0));
                }
                simulator.initialize();
                simulator.dump_output(-1);
            }
            if (simulator.sub_step >= 1) {
                T com = 0, vec = 0, acc = 0;
                T total_mass = 0;
                for (int i = 0; i < (int)vertices.size(); ++i) {
                    com += simulator.m_x[i](0) * simulator.m_mass[i];
                    vec += simulator.m_v[i](0) * simulator.m_mass[i];
                    acc += simulator.m_a[i](0) * simulator.m_mass[i];
                    total_mass += simulator.m_mass[i];
                }
                // total_pressure = std::abs(total_pressure);
                com /= total_mass;
                vec /= total_mass;
                acc /= total_mass;
                Bow::RESULT_RECORDER::record("center_of_mass_" + std::to_string(testcase), simulator.time_elapsed, com);
                Bow::RESULT_RECORDER::record("velocity_" + std::to_string(testcase), simulator.time_elapsed, vec);
                Bow::RESULT_RECORDER::record("acceleration_" + std::to_string(testcase), simulator.time_elapsed, acc);
                Bow::RESULT_RECORDER::record("gravity_" + std::to_string(testcase), simulator.time_elapsed, simulator.gravity[0] * total_mass);
            }
        };
        // Simulation
        if (testcase == 102)
            simulator.output_directory = "outputs/ipc_2d_static_friction_nSeg" + std::to_string(n_seg);
        else if (testcase == 103)
            simulator.output_directory = "outputs/ipc_2d_static_friction2_nSeg" + std::to_string(n_seg);
        else if (testcase == 104)
            simulator.output_directory = "outputs/ipc_2d_static_friction3_nSeg" + std::to_string(n_seg);
        else if (testcase == 105)
            simulator.output_directory = "outputs/ipc_2d_static_friction0_nSeg" + std::to_string(n_seg);
        simulator.initialize();
        simulator.end_frame = 480;
        simulator.run();
    }

    if (testcase == 333) { // rolling ball
        using T = double;
        static const int dim = 2;
        simulator.end_frame = 5000;
        simulator.suggested_dt = 1e-3;
        simulator.frame_dt = 1e-3;
        // simulator.dx = 0.2;
        simulator.gravity = Bow::Vector<T, dim>(10. / std::sqrt(2.), -10. / std::sqrt(2.));
        // simulator.symplectic = false;
        simulator.dHat = 1e-3;
        // simulator.Y = 4.2e7 * 32;
        // simulator.nu = 0.4;
        // simulator.rho = 1000.0;
        simulator.tol = 1e-4;
        // simulator.particle_vol = coupling.dx * coupling.dx / std::pow(2, dim);
        simulator.lag = true;
        // simulator.newton_iter = 100;
        // coupling.backward_euler = false;
        simulator.mu = 0.4;
        simulator.epsv = 1e-5;
        simulator.improved_maxOp = true;

        T mpm_fem_ratio = T(1) / 8;
        T Y = 1e10;
        T nu = 0.4;
        T rho = 1000.0;

        simulator.kappa = 1e9;

        simulator.output_directory = "outputs/rolling_ball/";
        {
            // FEM data
            // wall
            Bow::Field<Bow::Vector<T, dim>> vertices1;
            Bow::Field<Bow::Vector<int, dim + 1>> elements1;
            Bow::Geometry::cube(Bow::Vector<int, dim>(1, 1), 3.2, vertices1, elements1);
            for (auto& x : vertices1) {
                x(0) -= 9.6;
                x(1) += 1.8 + simulator.dHat;
            }
            simulator.append(vertices1, elements1, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
            for (int i = 0; i < int(vertices1.size()); ++i) {
                simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(i, Bow::Geometry::STICKY, Bow::Vector<T, dim>(1, 0), 0.0));
            }

            // slope: 0.625 * 0.05 * 0.05
            Bow::Field<Bow::Vector<T, dim>> vertices2;
            Bow::Field<Bow::Vector<int, dim + 1>> elements2;
            T fem_dx = 0.2 / mpm_fem_ratio;
            Bow::Geometry::cube(Bow::Vector<T, dim>(-16, -1.6), Bow::Vector<T, dim>(24, 0), Bow::Vector<T, dim>(fem_dx, fem_dx), vertices2, elements2);
            simulator.append(vertices2, elements2, Bow::ConstitutiveModel::NEO_HOOKEAN, Y * 10, nu, rho * 10);
            for (size_t i = 0; i < vertices2.size(); ++i) {
                if (vertices2[i](1) <= -0.049999) {
                    simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(vertices1.size() + i, Bow::Geometry::STICKY, Bow::Vector<T, dim>(1, 0), 0.0));
                }
            }

            Bow::Field<Bow::Vector<T, dim>> vertices3;
            Bow::Field<Bow::Vector<int, dim + 1>> elements3;
            Bow::IO::read_ply(EXAMPLE_DATA_PATH "EIPC/sphere.ply", vertices3, elements3);
            simulator.append(vertices3, elements3, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        }
        simulator.timestep_callback = [&]() {
            if (simulator.time_elapsed >= 1) {
                for (size_t i = 0; i < 4; ++i) {
                    simulator.update_index_based_bc_velocity(i, Bow::Vector<T, dim>{ 1000, 1000 });
                }
            }
        };
        simulator.initialize();
        simulator.run(start_frame);
    }

    if (testcase == 1001) {
        // sim solve settings
        simulator.tol = 1e-3;

        // body forces
        simulator.gravity = Bow::Vector<T, dim>(0, -5);

        // moduli, density
        T Y = 4.0e3, nu = 0.2, rho = 100.0;

        // barriers
        simulator.dHat = 1e-3;
        simulator.kappa = 0.1 * Y;

        // friction
        simulator.mu = 0.3;
        simulator.epsv = 1e-3;
        simulator.lag = true;

        // stepper
        simulator.suggested_dt = 1e-3;
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
            simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(i, Bow::Geometry::STICKY, Bow::Vector<T, dim>(0, 1), 1.0));
        }
        num_verts += int(vertices_s.size());

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
        simulator.output_directory = "outputs/sharkey";

        // init sim
        simulator.initialize();
        simulator.run(start_frame);
    }

    return 0;
}