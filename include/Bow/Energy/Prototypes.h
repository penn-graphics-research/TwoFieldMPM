#ifndef SIMULATION_OP_PROTOTYPES_H
#define SIMULATION_OP_PROTOTYPES_H
#include <Bow/Types.h>
#include <Bow/Utils/Logging.h>
#include <Eigen/SparseCore>

namespace Bow {

template <class T, int dim, class StorageIndex = int>
class EnergyOp {
public:
    T energy_scale = 1;
    /**
     * Naming convention: [UntitledType (FEM/MPM)]-[Energy Name]
     */
    std::string name = "UntitledType-UntitledName";
    std::vector<std::string> name_hierarchy()
    {
        std::vector<std::string> hierarchy;
        std::string s = name;
        std::string delimiter = "-";
        size_t pos = 0;
        std::string token;
        while ((pos = s.find(delimiter)) != std::string::npos) {
            token = s.substr(0, pos);
            hierarchy.push_back(token);
            s.erase(0, pos + delimiter.length());
        }
        hierarchy.push_back(s);
        return hierarchy;
    }
    /* called whenever x is changed */
    virtual void precompute(const Field<Vector<T, dim>>& x){};
    /* called before each iteration */
    virtual void callback(const Field<Vector<T, dim>>& xn){};
    virtual T stepsize_upperbound(const Field<Vector<T, dim>>& x, const Field<Vector<T, dim>>& dx) { return T(1); }
    virtual T energy(const Field<Vector<T, dim>>& x) { return 0; }
    virtual void gradient(const Field<Vector<T, dim>>& x, Field<Vector<T, dim>>& grad) { grad.assign(x.size(), Vector<T, dim>::Zero()); };
    virtual void hessian(const Field<Vector<T, dim>>& x, Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>& hess, bool project_pd = true)
    {
        hess.resize(x.size() * dim, x.size() * dim);
        hess.setZero();
    }
    virtual void internal_force(const Field<Vector<T, dim>>& xn, const Field<Vector<T, dim>>& vn, Field<Vector<T, dim>>& force)
    {
        force.assign(xn.size(), Vector<T, dim>::Zero());
    }
};
} // namespace Bow

#endif