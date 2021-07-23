#ifndef OPTIMIZER_LBFGS_H
#define OPTIMIZER_LBFGS_H

#include <Bow/Macros.h>
#include <Eigen/Eigen>
#include <Bow/Utils/Timer.h>
#include <Bow/Utils/Logging.h>
#include <functional>
#include <Bow/Types.h>

namespace Bow {
namespace Optimization {
template <class Scalar, int dim, class StorageIndex = int>
class LBFGS : public virtual OptimizerBase<Scalar, dim, StorageIndex> {
public:
    using Vec = Bow::Vector<Scalar, Eigen::Dynamic>;
    using Mat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;

    // store history
    Vec last_x;
    Vec last_g;
    Bow::Field<Vec> m_s;
    Bow::Field<Vec> m_y;
    Bow::Field<Scalar> m_rho;
    Bow::Field<Scalar> m_beta;
    Bow::Field<Scalar> m_gamma;
    Mat H0;

    int history_size = 3;

    LBFGS() { this->method = "LBFGS"; }

    virtual void initialize_optimizer(const Vec& x)
    {
        m_s.resize(history_size, Vec::Zero(x.size()));
        m_y.resize(history_size, Vec::Zero(x.size()));
        m_rho.resize(history_size, 0);
        m_beta.resize(history_size, 0);
        m_gamma.resize(history_size, 0);
        initialize_hessian(x);
    }

    virtual void initialize_hessian(const Vec& x_vec)
    {
        H0.resize(x_vec.size(), x_vec.size());
        H0.setIdentity();
    }

    virtual void search_direction(const Vec& x_vec, const Vec& grad, Vec& direction)
    {
        // https://handwiki.org/wiki/Limited-memory_BFGS
        // update Bk+1 and solve p^{k+1}
        if (this->iter_num == 0) {
            direction = -grad;
            last_x = x_vec;
            last_g = grad;
            return;
        }
        int index = (this->iter_num - 1) % history_size;
        m_s[index] = x_vec - last_x;
        m_y[index] = grad - last_g;
        Scalar ys = m_y[index].dot(m_s[index]);
        if (ys > 1e-10)
            m_rho[index] = Scalar(1) / ys;
        else {
            m_rho[index] = 0;
            direction = -grad;
            last_x = x_vec;
            last_g = grad;
            return;
        }

        Vec q = grad;
        for (int j = 0; j < history_size; ++j) {
            m_beta[j] = m_rho[j] * m_s[j].dot(q);
            q = q - m_beta[j] * m_y[j];
        }
        Vec p = H0 * q * (ys / m_y[index].squaredNorm());
        for (int j = 0; j < history_size; ++j) {
            m_gamma[index] = m_rho[j] * m_y[j].dot(p);
            p = p + (m_beta[j] - m_gamma[j]) * m_s[j];
        }
        direction = -p;

        last_x = x_vec;
        last_g = grad;
    }
};
}
} // namespace Bow::Optimization

#endif