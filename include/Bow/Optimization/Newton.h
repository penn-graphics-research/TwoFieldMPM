#ifndef OPTIMIZER_NEWTON_H
#define OPTIMIZER_NEWTON_H

#include <Bow/Macros.h>
#include <Eigen/Eigen>
#include <Bow/Utils/Timer.h>
#include <Bow/Utils/Logging.h>
#include <functional>
#include <Bow/Types.h>
#include "OptimizerBase.h"

namespace Bow {
namespace Optimization {
template <class Scalar, int dim, class StorageIndex = int>
class Newton : public virtual OptimizerBase<Scalar, dim, StorageIndex> {
public:
    using Vec = Bow::Vector<Scalar, Eigen::Dynamic>;
    using Mat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;

    Newton()
    {
        this->method = "PN";
    }

    virtual void search_direction(const Vec& x_vec, const Vec& grad, Vec& direction)
    {

        Mat hess;
        {
            BOW_TIMER_FLAG("Compute Hessian");
            this->hessian(x_vec, hess, this->project_pd);
        }
        direction = -this->linear_system(hess, grad);
    }

    virtual Scalar constraint_residual(const Vec& x, const Vec& cons)
    {
        return 0;
    }
};
}
} // namespace Bow::Optimization

#endif