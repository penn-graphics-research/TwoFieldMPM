#include <Bow/Math/LinearSolver/Amgcl.h>
#include <Bow/Math/LinearSolver/SparseCholesky.h>
#include <Bow/Math/LinearSolver/SparseQR.h>
#include <Eigen/SparseCholesky>
#include <Bow/Types.h>
#include <Bow/Math/Utils.h>
#include <Bow/Utils/Logging.h>
#include <Bow/Utils/Timer.h>
#include <Bow/Simulator/FEM/FEMSimulator.h>
#include <Bow/Geometry/Primitives.h>

template <typename T>
struct WrappedArray {
    WrappedArray(T* first, T* last)
        : begin_{ first }, end_{ last } {}
    WrappedArray(T* first, std::ptrdiff_t size)
        : WrappedArray{ first, first + size } {}

    T* begin() const noexcept { return begin_; }
    T* end() const noexcept { return end_; }

    T& operator[](const size_t i) { return begin_[i]; }
    const T& operator[](const size_t i) const { return begin_[i]; }

    T* begin_;
    T* end_;
};

template <typename DerivedA, typename DerivedB>
void sample_problem(
    const typename DerivedA::StorageIndex n,
    Eigen::SparseMatrixBase<DerivedA>& A,
    Eigen::MatrixBase<DerivedB>& b)
{
    using T = typename DerivedA::Scalar;
    const int dim = 3;
    Bow::FEM::FEMSimulator<T, dim> fem_data;
    fem_data.suggested_dt = 0.02;
    // test on a cube
    Bow::Field<Bow::Vector<T, dim>> pos;
    Bow::Field<Bow::Vector<int, dim + 1>> elem;
    Bow::Vector<T, dim> min_corner = Bow::Vector<T, dim>::Zero();
    Bow::Vector<T, dim> max_corner = Bow::Vector<T, dim>::Ones();
    Bow::Vector<T, dim> res = Bow::Vector<T, dim>::Ones() * (1. / n);
    Bow::Geometry::cube(min_corner, max_corner, res, pos, elem);
    fem_data.append(pos, elem, Bow::ConstitutiveModel::FIXED_COROTATED, 1e4, 0.3, 1000);

    // create operators
    Bow::FEM::InitializeOp<T, dim> initialize{ fem_data.m_X, fem_data.m_elem, fem_data.m_density,
        fem_data.m_elem_codim1, fem_data.m_thickness_codim1, fem_data.m_density_codim1,
        fem_data.m_mass, fem_data.m_vol, fem_data.m_IB, fem_data.m_vol_codim1, fem_data.m_IB_codim1 };
    Bow::FEM::InertialEnergyOp<T, dim> inertial_energy(fem_data.m_mass, fem_data.m_x_tilde);
    Bow::FEM::FixedCorotatedEnergyOp<T, dim> fcr_energy(fem_data.m_elem, fem_data.m_vol, fem_data.m_mu, fem_data.m_lam, fem_data.m_IB, fem_data.m_obj_divider[Bow::ConstitutiveModel::FIXED_COROTATED]);
    Bow::FEM::TimeIntegratorUpdateOp<T, dim> update(fem_data.BC_basis, fem_data.BC_order, fem_data.BC_target, fem_data.BC_fixed, fem_data.m_mass, fem_data.m_x, fem_data.m_v, fem_data.m_a, fem_data.m_x1, fem_data.m_v1, fem_data.m_x_tilde);
    update.dt = fem_data.suggested_dt;
    update.m_energy_terms.push_back(&inertial_energy);
    update.m_energy_terms.push_back(&fcr_energy);
    update.set_ts_weights();

    initialize();
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(&pos[0][0], pos.size() * dim);
    update.hessian(x, A.derived(), true);
    update.gradient(x, b.derived());
}

int main()
{
    int n = 30;
    Eigen::SparseMatrix<double> A;
    Bow::Vector<double, Eigen::Dynamic> b;
    sample_problem(n, A, b);
    Bow::Vector<double, Eigen::Dynamic> x;
    {
        BOW_TIMER_FLAG("Eigen CG");
        Bow::Logging::info("=================== Eigen CG ===================");
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
        {
            BOW_TIMER_FLAG("compute");
            solver.compute(A);
        }
        {
            BOW_TIMER_FLAG("solve");
            x = solver.solve(b);
        }
        Bow::Logging::info("residual: ", (A * x - b).norm());
    }

#ifdef BOW_SUITESPARSE
    {
        BOW_TIMER_FLAG("CholmodLLT(Supernodal)");
        Bow::Logging::info("=================== Supernodal LLT ===================");
        Bow::Math::LinearSolver::CholmodLLT<Eigen::SparseMatrix<double>> solver(CHOLMOD_SUPERNODAL);
        {
            BOW_TIMER_FLAG("compute");
            solver.compute(A);
        }
        {
            BOW_TIMER_FLAG("solve");
            x = solver.solve(b);
        }
        Bow::Logging::info("residual: ", (A * x - b).norm());
    }

    {
        BOW_TIMER_FLAG("CholmodLLT(Simplicial)");
        Bow::Logging::info("=================== Simplicial LLT ===================");
        Bow::Math::LinearSolver::CholmodLLT<Eigen::SparseMatrix<double>> solver(CHOLMOD_SIMPLICIAL);
        {
            BOW_TIMER_FLAG("compute");
            solver.compute(A);
        }
        {
            BOW_TIMER_FLAG("solve");
            x = solver.solve(b);
        }
        Bow::Logging::info("residual: ", (A * x - b).norm());
    }

    {
        BOW_TIMER_FLAG("SuiteSparseQR");
        Bow::Logging::info("=================== SuiteSparseQR ===================");
        Bow::Math::LinearSolver::SparseQR<Eigen::SparseMatrix<double>> solver;
        {
            BOW_TIMER_FLAG("compute");
            solver.compute(A);
        }
        {
            BOW_TIMER_FLAG("solve");
            x = solver.solve(b);
        }
        Bow::Logging::info("residual: ", (A * x - b).norm());
    }
#endif

#ifdef BOW_AMGCL
    {
        BOW_TIMER_FLAG("AMGCL");
        Bow::Logging::info("=================== AMGCL ===================");
        Bow::Math::LinearSolver::AmgclSolver<Eigen::SparseMatrix<double>, false> solver;
        {
            BOW_TIMER_FLAG("precond");
            solver.compute(A);
        }
        {
            BOW_TIMER_FLAG("solve");
            x = solver.solve(b);
        }
        Bow::Logging::info("residual: ", (A * x - b).norm());
    }
#endif

#ifdef ENABLE_AMGCL_CUDA
    {
        BOW_TIMER_FLAG("AMGCL(CUDA)");
        Bow::Logging::info("=================== AMGCL(CUDA) ===================");
        Bow::Math::LinearSolver::AmgclSolver<Eigen::SparseMatrix<double>, true> solver;
        {
            BOW_TIMER_FLAG("precond");
            solver.compute(A);
        }
        {
            BOW_TIMER_FLAG("solve");
            x = solver.solve(b);
        }
        Bow::Logging::info("residual: ", (A * x - b).norm());
    }
#endif

    Bow::Timer::flush();
    return 0;
}