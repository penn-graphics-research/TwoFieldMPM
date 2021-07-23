#include <Bow/Types.h>
#include <iostream>
#include <fstream>
#include <Bow/Geometry/Primitives.h>
#include <Bow/Utils/Timer.h>
#include <Bow/IO/ply.h>
#include <Bow/Geometry/Query.h>
#include <Bow/Utils/FileSystem.h>
#include <Bow/Geometry/OrthogonalBasis.h>
#include <Bow/Simulator/FEM/FEMSimulator.h>
#include <Bow/Geometry/BoundaryConditions.h>

int main(int argc, char** argv)
{
    using T = double;
    static const int dim = 2;
    if (argc < 2) {
        puts("ERROR: please add parameters");
        puts("USAGE: ./fem_2d testcase");
        puts("       ./fem_2d testcase start_frame");
        exit(0);
    }

    Bow::FEM::FEMSimulator<T, dim> simulator;

    int testcase = std::atoi(argv[1]);
    int start_frame = 0;
    if (argc == 3) {
        start_frame = std::atoi(argv[2]);
    }

    if (testcase == 1) {
        simulator.gravity = Bow::Vector<T, dim>(0, -98);
        simulator.output_directory = "fem_slip/";

        // generate data
        Bow::Field<Bow::Vector<T, dim>> vertices;
        Bow::Field<Bow::Vector<int, dim + 1>> elements;
        Bow::Geometry::cube(Bow::Vector<int, dim>(10, 10), 0.1, vertices, elements);
        // transformation
        Bow::Matrix<T, 2, 2> transformation = Bow::Geometry::extend_basis(Bow::Vector<T, dim>(1.0 / sqrt(2.0), 1.0 / sqrt(2.0)));
        for (size_t i = 0; i < vertices.size(); ++i) {
            vertices[i] = transformation * vertices[i];
        }

        simulator.append(vertices, elements, Bow::ConstitutiveModel::FIXED_COROTATED, 1e7, 0.3, 1000);
        simulator.dump_output(0);
        // slip boundary condition
        simulator.add_boundary_condition(new Bow::Geometry::HalfSpaceLevelSet<T, dim>(Bow::Geometry::SLIP, Eigen::Vector2d(-0.5 * sqrt(2) + 0.0001, 0), Eigen::Vector2d(1, 1)));
        simulator.initialize();
        // simulator.end_frame = 1;
        simulator.run(start_frame);
    }
    else if (testcase == 2) { // moving sticky
        simulator.suggested_dt = simulator.frame_dt;
        simulator.output_directory = "fem_moving_bc/";
        // simulator.gravity = Bow::Vector<T, dim>(1, -10);
        Bow::Field<Bow::Vector<T, dim>> vertices;
        Bow::Field<Bow::Vector<int, dim + 1>> elements;
        Bow::Geometry::cube<T, dim>(Bow::Vector<T, dim>::Zero(), Bow::Vector<T, dim>::Ones(), Bow::Vector<T, dim>::Constant(0.1), vertices, elements);
        simulator.append(vertices, elements, Bow::ConstitutiveModel::NEO_HOOKEAN, 1e4, 0.3, 1000);
        for (int i = 0; i < (int)vertices.size(); ++i) {
            if (vertices[i](0) > 0.9999) {
                simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(i, Bow::Geometry::STICKY, Bow::Vector<T, dim>(1, 0), 1.0));
            }
            else if (vertices[i](0) < 0.00001) {
                simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(i, Bow::Geometry::STICKY, Bow::Vector<T, dim>(1, 0), 0.0));
            }
        }
        simulator.initialize();
        simulator.run(start_frame);
    }
    else if (testcase == 3) { // moving slip
        simulator.suggested_dt = simulator.frame_dt;
        simulator.output_directory = "fem_moving_bc2/";
        simulator.gravity = Bow::Vector<T, dim>(1, -10);
        Bow::Field<Bow::Vector<T, dim>> vertices;
        Bow::Field<Bow::Vector<int, dim + 1>> elements;
        Bow::Geometry::cube<T, dim>(Bow::Vector<T, dim>::Zero(), Bow::Vector<T, dim>::Ones(), Bow::Vector<T, dim>::Constant(0.1), vertices, elements);
        simulator.append(vertices, elements, Bow::ConstitutiveModel::NEO_HOOKEAN, 1e4, 0.3, 1000);
        for (int i = 0; i < (int)vertices.size(); ++i) {
            if (vertices[i](1) < 0.000001) {
                simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(i, Bow::Geometry::SLIP, Bow::Vector<T, dim>(0, 1), 0.1));
            }
        }
        simulator.initialize();
        simulator.run(start_frame);
    }
    else if (testcase == 4) { // initially stretched
        int nSeg = 10;
        if (argc > 3) {
            nSeg = std::stoi(argv[3]);
            if (nSeg < 1 || nSeg % 2 != 0) {
                Bow::Logging::error("nSeg must be positive and even!");
                exit(-1);
            }
        }

        // generate data
        Bow::Field<Bow::Vector<T, dim>> vertices;
        Bow::Field<Bow::Vector<int, dim + 1>> elements;
        T dx = 1.0 / nSeg;
        Bow::Geometry::cube(Bow::Vector<int, dim>(nSeg, nSeg), dx, vertices, elements);
        T Y = 1e5, nu = 0.2, rho = 1000;
        simulator.append(vertices, elements, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);

        simulator.tsMethod = Bow::BDF2;
        T nuC = 1;
        if (argc > 4) {
            nuC = std::stod(argv[4]);
            if (nuC <= 0) {
                Bow::Logging::error("nuC must be positive!");
                exit(-1);
            }
        }
        simulator.suggested_dt = nuC / 10 / std::sqrt(Y / rho);

        simulator.frame_dt = simulator.suggested_dt;
        simulator.end_frame = std::ceil(1.0 / simulator.frame_dt);
        simulator.frame_batch = std::max(1, int(1.0 / 24 / simulator.frame_dt));

        simulator.output_directory = "outputs/fem_2d_stretched_nSeg" + std::to_string(nSeg) + "/";
        simulator.initialize();
        for (size_t i = 0; i < simulator.m_x.size(); ++i) {
            simulator.m_x[i][0] *= 1.2;
            simulator.m_x1[i][0] *= 1.2;
        }
        simulator.timestep_callback = [&]() {
            FILE* out = fopen((simulator.output_directory + "/info.txt").c_str(), "a+");
            if (out) {
                int vI = vertices.size() - 1 - nSeg / 2; // middle rightmost node
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
    else if (testcase == 5) { // initially stretched, static
        int nSeg = 10;
        if (argc > 3) {
            nSeg = std::stoi(argv[3]);
            if (nSeg < 1 || nSeg % 2 != 0) {
                Bow::Logging::error("nSeg must be positive and even!");
                exit(-1);
            }
        }
        simulator.output_directory = "outputs/fem_2d_stretchedStatic_nSeg" + std::to_string(nSeg) + "/";

        // generate data
        Bow::Field<Bow::Vector<T, dim>> vertices;
        Bow::Field<Bow::Vector<int, dim + 1>> elements;
        T dx = 1.0 / nSeg;
        Bow::Geometry::cube(Bow::Vector<int, dim>(nSeg, nSeg), dx, vertices, elements);
        T Y = 1e5, nu = 0.2, rho = 1000;
        simulator.append(vertices, elements, Bow::ConstitutiveModel::NEO_HOOKEAN, Y, nu, rho);
        simulator.add_boundary_condition(new Bow::Geometry::HalfSpaceLevelSet<T, dim>(Bow::Geometry::STICKY, vertices[0] + Bow::Vector<T, dim>(0, dx / 10), Bow::Vector<T, dim>(0, 1)));
        simulator.add_boundary_condition(new Bow::Geometry::HalfSpaceLevelSet<T, dim>(Bow::Geometry::STICKY, vertices.back() - Bow::Vector<T, dim>(0, dx / 10), Bow::Vector<T, dim>(0, -1)));

        simulator.tsMethod = Bow::BE;
        simulator.static_sim = true;
        simulator.tol = 1e-6;
        simulator.suggested_dt = 1;
        simulator.frame_dt = simulator.suggested_dt;
        simulator.end_frame = 1;

        simulator.initialize();
        for (auto& i : simulator.m_x) {
            i[0] *= 1.2;
        }
        simulator.timestep_callback = [&]() {
            FILE* out = fopen((simulator.output_directory + "/info.txt").c_str(), "a+");
            if (out) {
                int vI = vertices.size() - 1 - nSeg / 2; // middle rightmost node
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
    else if (testcase == 333) {
        using T = double;
        static const int dim = 2;
        simulator.end_frame = 3000;
        simulator.suggested_dt = 1e-3;
        simulator.frame_dt = 1e-3;
        simulator.gravity = Bow::Vector<T, dim>(10. / std::sqrt(2.), -10. / std::sqrt(2.));
        simulator.tol = 1e-3;

        T mpm_fem_ratio = T(1)/8;
        T Y = 4.2e7 * 32;
        T nu = 0.4;
        T rho = 1000.0;

        simulator.output_directory = "outputs/rolling_ball/";
        {
            Bow::Field<Bow::Vector<T, dim>> vertices2;
            Bow::Field<Bow::Vector<int, dim + 1>> elements2;
            T fem_dx = 0.2 / mpm_fem_ratio;
            Bow::Geometry::cube(Bow::Vector<T, dim>(-16, -1.6), Bow::Vector<T, dim>(24, 0), Bow::Vector<T, dim>(fem_dx, fem_dx), vertices2, elements2);
            simulator.append(vertices2, elements2, Bow::ConstitutiveModel::NEO_HOOKEAN, Y * 10, nu, rho * 10);
            for (size_t i = 0; i < vertices2.size(); ++i) {
                if (vertices2[i](1) <= -0.049999) {
                    simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(i, Bow::Geometry::STICKY, Bow::Vector<T, dim>(1, 0), 0.0));
                }
            }
        }
        simulator.initialize();
        simulator.run(start_frame);
    }
    return 0;
}