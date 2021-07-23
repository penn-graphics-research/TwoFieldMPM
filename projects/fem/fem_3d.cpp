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

int main(int argc, char* argv[])
{
    using T = double;
    static const int dim = 3;

    if (argc < 2) {
        puts("ERROR: please add parameters");
        puts("USAGE: ./fem_3d testcase");
        puts("       ./fem_3d testcase start_frame");
        exit(0);
    }

    int testcase = std::atoi(argv[1]);
    int start_frame = 0;
    if (argc == 3) {
        start_frame = std::atoi(argv[2]);
    }

    Bow::FEM::FEMSimulator<T, dim> simulator;

    if (testcase == 0) {
        Bow::Field<Bow::Vector<T, dim>> vertices;
        Bow::Field<Bow::Vector<int, dim + 1>> elements;
        Bow::Geometry::cube(Bow::Vector<int, dim>(10, 10, 10), 0.1, vertices, elements);
        simulator.append(vertices, elements, Bow::ConstitutiveModel::FIXED_COROTATED, 10000, 0.3, 1000);

        Eigen::Map<Eigen::VectorXd>(&(simulator.m_x[0][0]), dim * simulator.m_x.size()) *= 2;

        simulator.output_directory = "outputs/cube_stretched";
        simulator.initialize();
        simulator.run(start_frame);
    }
    else if (testcase == 1) {
        Bow::Field<Bow::Vector<T, dim>> vertices;
        Bow::Field<Bow::Vector<int, dim>> elements;
        T length = 0.5;
        int nSeg = 20;
        for (int i = 0; i < nSeg + 1; ++i) {
            for (int j = 0; j < nSeg + 1; ++j) {
                vertices.emplace_back(Bow::Vector<T, dim>(i * (length / nSeg), j * (length / nSeg), 0));
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
        simulator.tol = 1e-6;
        simulator.append(vertices, elements,
            3e-4, // thickness
            Bow::ConstitutiveModel::NEO_HOOKEAN_MEMBRANE, 1e5, 0.3, // E and nu
            Bow::ConstitutiveModel::DISCRETE_HINGE_BENDING, 1e5, 0.3, // E and nu
            500); // density

        simulator.add_boundary_condition(new Bow::Geometry::IndexBasedBoundaryCondition<T, dim>(nSeg, Bow::Geometry::STICKY, Bow::Vector<T, dim>(0, 0, 1), 0));

        simulator.gravity = Bow::Vector<T, dim>(0, -9.81, 0);

        simulator.output_directory = "outputs/cloth_pin";
        simulator.initialize();
        simulator.run(start_frame);
    }
    else {
        Bow::Logging::error("Invalid test case number!");
    }

    return 0;
}