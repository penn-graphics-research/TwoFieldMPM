#pragma once

#include <Bow/Types.h>
#include <oneapi/tbb.h>
#include <Bow/Utils/Logging.h>

namespace Bow {

template <class T, int dim>
class SpatialHash {
public: // data
    Vector<T, dim> leftBottomCorner, rightTopCorner;
    T one_div_voxelSize;
    Array<int, dim, 1> voxelCount;
    int voxelCount0x1;

    int surfEdgeStartInd, surfTriStartInd;

    std::unordered_map<int, Field<int>> voxel;
    Field<Field<int>> pointAndEdgeOccupancy; // for CCD

public: // constructor
    SpatialHash(void) {}

public: // API
    void build(const Field<Vector<T, dim>>& X,
        const Field<int>& boundaryNode,
        const Field<Vector<int, 2>>& boundaryEdge,
        const Field<Vector<int, 3>>& boundaryTri,
        T voxelSize)
    {
        Matrix<T, Eigen::Dynamic, dim> V(X.size(), dim);
        tbb::parallel_for(size_t(0), X.size(), [&](size_t i) {
            V.row(i) = X[i];
        });

        if (boundaryEdge.size()) {
            Vector<T, Eigen::Dynamic> eLen(boundaryEdge.size(), 1);
            tbb::parallel_for(size_t(0), boundaryEdge.size(), [&](size_t seCount) {
                const auto& seI = boundaryEdge[seCount];
                eLen[seCount] = (V.row(seI[0]) - V.row(seI[1])).norm();
            });
            voxelSize *= eLen.mean();
        }

        leftBottomCorner = V.colwise().minCoeff();
        rightTopCorner = V.colwise().maxCoeff();
        Array<T, dim, 1> range = rightTopCorner - leftBottomCorner;
        one_div_voxelSize = 1.0 / voxelSize;
        Array<long, dim, 1> vcl = (range * one_div_voxelSize).ceil().template cast<long>().max(Array<long, dim, 1>::Ones());
        long voxelAmt = vcl.matrix().prod();
        if (voxelAmt > 1e9) {
            voxelSize *= std::pow(voxelAmt / 1.0e9, 1.0 / 3);
            one_div_voxelSize = 1.0 / voxelSize;
        }
        voxelCount = (range * one_div_voxelSize).ceil().template cast<int>().max(Array<int, dim, 1>::Ones());
        if (voxelCount.minCoeff() <= 0) {
            // cast overflow due to huge search direction or tiny voxelSize
            one_div_voxelSize = 16.0 / (range.maxCoeff() * 1.01);
            voxelCount = (range * one_div_voxelSize).ceil().template cast<int>().max(Array<int, dim, 1>::Ones());
        }
        Bow::Logging::info("CCS SH voxel count ", voxelCount.prod());
        voxelCount0x1 = voxelCount[0] * voxelCount[1];

        surfEdgeStartInd = boundaryNode.size();
        surfTriStartInd = surfEdgeStartInd + boundaryEdge.size();

        // precompute svVAI
        Field<Array<int, dim, 1>> svVoxelAxisIndex(boundaryNode.size());
        Field<int> vI2SVI(X.size());
        tbb::parallel_for(size_t(0), boundaryNode.size(), [&](size_t svI) {
            int vI = boundaryNode[svI];
            locate_voxel_axis_index(V.row(vI).transpose(), svVoxelAxisIndex[svI]);
            vI2SVI[vI] = svI;
        });

        voxel.clear();

        for (int svI = 0; svI < (int)boundaryNode.size(); ++svI) {
            voxel[locate_voxel_index(V.row(boundaryNode[svI]).transpose())].emplace_back(svI);
        }

        Field<Field<int>> voxelLoc_e(boundaryEdge.size());
        tbb::parallel_for(size_t(0), boundaryEdge.size(), [&](size_t seCount) {
            const auto& seI = boundaryEdge[seCount];
            const Array<int, dim, 1>& voxelAxisIndex_first = svVoxelAxisIndex[vI2SVI[seI[0]]];
            const Array<int, dim, 1>& voxelAxisIndex_second = svVoxelAxisIndex[vI2SVI[seI[1]]];
            Array<int, dim, 1> mins = voxelAxisIndex_first.min(voxelAxisIndex_second);
            Array<int, dim, 1> maxs = voxelAxisIndex_first.max(voxelAxisIndex_second);
            voxelLoc_e[seCount].reserve((maxs - mins + 1).prod());
            if constexpr (dim == 3) {
                for (int iz = mins[2]; iz <= maxs[2]; ++iz) {
                    int zOffset = iz * voxelCount0x1;
                    for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                        int yzOffset = iy * voxelCount[0] + zOffset;
                        for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                            voxelLoc_e[seCount].emplace_back(ix + yzOffset);
                        }
                    }
                }
            }
            else {
                for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                    int yOffset = iy * voxelCount[0];
                    for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                        voxelLoc_e[seCount].emplace_back(ix + yOffset);
                    }
                }
            }
        });

        Field<Field<int>> voxelLoc_sf;
        if constexpr (dim == 3) {
            voxelLoc_sf.resize(boundaryTri.size());
            tbb::parallel_for(size_t(0), boundaryTri.size(), [&](size_t sfI) {
                const Array<int, dim, 1>& voxelAxisIndex0 = svVoxelAxisIndex[vI2SVI[boundaryTri[sfI][0]]];
                const Array<int, dim, 1>& voxelAxisIndex1 = svVoxelAxisIndex[vI2SVI[boundaryTri[sfI][1]]];
                const Array<int, dim, 1>& voxelAxisIndex2 = svVoxelAxisIndex[vI2SVI[boundaryTri[sfI][2]]];
                Array<int, dim, 1> mins = voxelAxisIndex0.min(voxelAxisIndex1).min(voxelAxisIndex2);
                Array<int, dim, 1> maxs = voxelAxisIndex0.max(voxelAxisIndex1).max(voxelAxisIndex2);
                voxelLoc_sf[sfI].reserve((maxs - mins + 1).prod());
                for (int iz = mins[2]; iz <= maxs[2]; ++iz) {
                    int zOffset = iz * voxelCount0x1;
                    for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                        int yzOffset = iy * voxelCount[0] + zOffset;
                        for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                            voxelLoc_sf[sfI].emplace_back(ix + yzOffset);
                        }
                    }
                }
            });
        }

        for (int seCount = 0; seCount < (int)voxelLoc_e.size(); ++seCount) {
            for (const auto& voxelI : voxelLoc_e[seCount]) {
                voxel[voxelI].emplace_back(seCount + surfEdgeStartInd);
            }
        }
        if constexpr (dim == 3) {
            for (int sfI = 0; sfI < (int)voxelLoc_sf.size(); ++sfI) {
                for (const auto& voxelI : voxelLoc_sf[sfI]) {
                    voxel[voxelI].emplace_back(sfI + surfTriStartInd);
                }
            }
        }
    }

    void query_point_for_triangles(const Vector<T, 3>& pos,
        T radius, Field<int>& triInds) const
    {
        Array<int, 3, 1> mins, maxs;
        locate_voxel_axis_index(pos.array() - radius, mins);
        locate_voxel_axis_index(pos.array() + radius, maxs);
        mins = mins.max(Array<int, 3, 1>::Zero());
        maxs = maxs.min(voxelCount - 1);

        triInds.resize(0);
        for (int iz = mins[2]; iz <= maxs[2]; ++iz) {
            int zOffset = iz * voxelCount0x1;
            for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                int yzOffset = iy * voxelCount[0] + zOffset;
                for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                    const auto voxelI = voxel.find(ix + yzOffset);
                    if (voxelI != voxel.end()) {
                        for (const auto& indI : voxelI->second) {
                            if (indI >= surfTriStartInd) {
                                triInds.emplace_back(indI - surfTriStartInd);
                            }
                        }
                    }
                }
            }
        }
        std::sort(triInds.begin(), triInds.end());
        triInds.erase(std::unique(triInds.begin(), triInds.end()), triInds.end());
    }

    void query_edge_for_edges(
        const Vector<T, dim>& vBegin, const Vector<T, dim>& vEnd,
        T radius, Field<int>& edgeInds, int eIq = -1) const
    {
        Vector<T, dim> leftBottom = vBegin.array().min(vEnd.array()) - radius;
        Vector<T, dim> rightTop = vBegin.array().max(vEnd.array()) + radius;
        Array<int, dim, 1> mins, maxs;
        locate_voxel_axis_index(leftBottom, mins);
        locate_voxel_axis_index(rightTop, maxs);
        mins = mins.max(Array<int, dim, 1>::Zero());
        maxs = maxs.min(voxelCount - 1);

        edgeInds.resize(0);
        if constexpr (dim == 3) {
            for (int iz = mins[2]; iz <= maxs[2]; ++iz) {
                int zOffset = iz * voxelCount0x1;
                for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                    int yzOffset = iy * voxelCount[0] + zOffset;
                    for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                        const auto voxelI = voxel.find(ix + yzOffset);
                        if (voxelI != voxel.end()) {
                            for (const auto& indI : voxelI->second) {
                                if (indI >= surfEdgeStartInd && indI < surfTriStartInd && indI - surfEdgeStartInd > eIq) {
                                    edgeInds.emplace_back(indI - surfEdgeStartInd);
                                }
                            }
                        }
                    }
                }
            }
        }
        else {
            for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                int yOffset = iy * voxelCount[0];
                for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                    const auto voxelI = voxel.find(ix + yOffset);
                    if (voxelI != voxel.end()) {
                        for (const auto& indI : voxelI->second) {
                            if (indI >= surfEdgeStartInd && indI - surfEdgeStartInd > eIq) {
                                edgeInds.emplace_back(indI - surfEdgeStartInd);
                            }
                        }
                    }
                }
            }
        }
        std::sort(edgeInds.begin(), edgeInds.end());
        edgeInds.erase(std::unique(edgeInds.begin(), edgeInds.end()), edgeInds.end());
    }

    void query_edge_for_points(
        const Vector<T, dim>& vBegin, const Vector<T, dim>& vEnd,
        T radius, Field<int>& pointInds) const
    {
        Vector<T, dim> leftBottom = vBegin.array().min(vEnd.array()) - radius;
        Vector<T, dim> rightTop = vBegin.array().max(vEnd.array()) + radius;
        Array<int, dim, 1> mins, maxs;
        locate_voxel_axis_index(leftBottom, mins);
        locate_voxel_axis_index(rightTop, maxs);
        mins = mins.max(Array<int, dim, 1>::Zero());
        maxs = maxs.min(voxelCount - 1);

        pointInds.resize(0);
        if constexpr (dim == 3) {
            for (int iz = mins[2]; iz <= maxs[2]; ++iz) {
                int zOffset = iz * voxelCount0x1;
                for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                    int yzOffset = iy * voxelCount[0] + zOffset;
                    for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                        const auto voxelI = voxel.find(ix + yzOffset);
                        if (voxelI != voxel.end()) {
                            for (const auto& indI : voxelI->second) {
                                if (indI < surfEdgeStartInd) {
                                    pointInds.emplace_back(indI);
                                }
                            }
                        }
                    }
                }
            }
        }
        else {
            for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                int yOffset = iy * voxelCount[0];
                for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                    const auto voxelI = voxel.find(ix + yOffset);
                    if (voxelI != voxel.end()) {
                        for (const auto& indI : voxelI->second) {
                            if (indI < surfEdgeStartInd) {
                                pointInds.emplace_back(indI);
                            }
                        }
                    }
                }
            }
        }
        std::sort(pointInds.begin(), pointInds.end());
        pointInds.erase(std::unique(pointInds.begin(), pointInds.end()), pointInds.end());
    }

    void query_point_for_edges(const Vector<T, dim>& pos,
        T radius, Field<int>& edgeInds) const
    {
        Array<int, dim, 1> mins, maxs;
        locate_voxel_axis_index(pos.array() - radius, mins);
        locate_voxel_axis_index(pos.array() + radius, maxs);
        mins = mins.max(Array<int, dim, 1>::Zero());
        maxs = maxs.min(voxelCount - 1);

        edgeInds.resize(0);
        if constexpr (dim == 3) {
            for (int iz = mins[2]; iz <= maxs[2]; ++iz) {
                int zOffset = iz * voxelCount0x1;
                for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                    int yzOffset = iy * voxelCount[0] + zOffset;
                    for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                        const auto voxelI = voxel.find(ix + yzOffset);
                        if (voxelI != voxel.end()) {
                            for (const auto& indI : voxelI->second) {
                                if (indI >= surfEdgeStartInd && indI < surfTriStartInd) {
                                    edgeInds.emplace_back(indI - surfEdgeStartInd);
                                }
                            }
                        }
                    }
                }
            }
        }
        else {
            for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                int yOffset = iy * voxelCount[0];
                for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                    const auto voxelI = voxel.find(ix + yOffset);
                    if (voxelI != voxel.end()) {
                        for (const auto& indI : voxelI->second) {
                            if (indI >= surfEdgeStartInd) {
                                edgeInds.emplace_back(indI - surfEdgeStartInd);
                            }
                        }
                    }
                }
            }
        }
        std::sort(edgeInds.begin(), edgeInds.end());
        edgeInds.erase(std::unique(edgeInds.begin(), edgeInds.end()), edgeInds.end());
    }

    void query_point_for_points(const Vector<T, dim>& pos,
        T radius, Field<int>& pointInds) const
    {
        Array<int, dim, 1> mins, maxs;
        locate_voxel_axis_index(pos.array() - radius, mins);
        locate_voxel_axis_index(pos.array() + radius, maxs);
        mins = mins.max(Array<int, dim, 1>::Zero());
        maxs = maxs.min(voxelCount - 1);

        pointInds.resize(0);
        if constexpr (dim == 3) {
            for (int iz = mins[2]; iz <= maxs[2]; ++iz) {
                int zOffset = iz * voxelCount0x1;
                for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                    int yzOffset = iy * voxelCount[0] + zOffset;
                    for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                        const auto voxelI = voxel.find(ix + yzOffset);
                        if (voxelI != voxel.end()) {
                            for (const auto& indI : voxelI->second) {
                                if (indI < surfEdgeStartInd) {
                                    pointInds.emplace_back(indI);
                                }
                            }
                        }
                    }
                }
            }
        }
        else {
            for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                int yOffset = iy * voxelCount[0];
                for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                    const auto voxelI = voxel.find(ix + yOffset);
                    if (voxelI != voxel.end()) {
                        for (const auto& indI : voxelI->second) {
                            if (indI < surfEdgeStartInd) {
                                pointInds.emplace_back(indI);
                            }
                        }
                    }
                }
            }
        }
        std::sort(pointInds.begin(), pointInds.end());
        pointInds.erase(std::unique(pointInds.begin(), pointInds.end()), pointInds.end());
    }

    void query_triangle_for_edges(
        const Vector<T, dim>& pos0,
        const Vector<T, dim>& pos1,
        const Vector<T, dim>& pos2,
        T radius, Field<int>& edgeInds) const
    {
        Array<int, dim, 1> mins, maxs;
        locate_voxel_axis_index(pos0.array().min(pos1.array()).min(pos2.array()) - radius, mins);
        locate_voxel_axis_index(pos0.array().max(pos1.array()).max(pos2.array()) + radius, maxs);
        mins = mins.max(Array<int, dim, 1>::Zero());
        maxs = maxs.min(voxelCount - 1);

        edgeInds.resize(0);
        if constexpr (dim == 3) {
            for (int iz = mins[2]; iz <= maxs[2]; ++iz) {
                int zOffset = iz * voxelCount0x1;
                for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                    int yzOffset = iy * voxelCount[0] + zOffset;
                    for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                        const auto voxelI = voxel.find(ix + yzOffset);
                        if (voxelI != voxel.end()) {
                            for (const auto& indI : voxelI->second) {
                                if (indI >= surfEdgeStartInd && indI < surfTriStartInd) {
                                    edgeInds.emplace_back(indI - surfEdgeStartInd);
                                }
                            }
                        }
                    }
                }
            }
        }
        else {
            for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                int yOffset = iy * voxelCount[0];
                for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                    const auto voxelI = voxel.find(ix + yOffset);
                    if (voxelI != voxel.end()) {
                        for (const auto& indI : voxelI->second) {
                            if (indI >= surfEdgeStartInd) {
                                edgeInds.emplace_back(indI - surfEdgeStartInd);
                            }
                        }
                    }
                }
            }
        }
        std::sort(edgeInds.begin(), edgeInds.end());
        edgeInds.erase(std::unique(edgeInds.begin(), edgeInds.end()), edgeInds.end());
    }

    // for CCD:
    void build(const Field<Vector<T, dim>>& X,
        const Field<int>& boundaryNode,
        const Field<Vector<int, 2>>& boundaryEdge,
        const Field<Vector<int, 3>>& boundaryTri,
        const Field<Vector<T, dim>>& searchDir,
        T& curMaxStepSize, T voxelSize, T thickness)
    {
        if (boundaryEdge.size()) {
            Vector<T, Eigen::Dynamic> eLen(boundaryEdge.size(), 1);
            tbb::parallel_for(size_t(0), boundaryEdge.size(), [&](size_t seCount) {
                const auto& seI = boundaryEdge[seCount];
                const Vector<T, dim>& v0 = X[seI[0]];
                const Vector<T, dim>& v1 = X[seI[1]];
                eLen[seCount] = (v0 - v1).norm();
            });
            voxelSize *= eLen.mean();
        }

        T pSize = 0;
        for (int svI = 0; svI < (int)boundaryNode.size(); ++svI) {
            int vI = boundaryNode[svI];
            pSize += std::abs(searchDir[vI][0]);
            pSize += std::abs(searchDir[vI][1]);
            if constexpr (dim == 3) {
                pSize += std::abs(searchDir[vI][2]);
            }
        }
        pSize /= boundaryNode.size() * dim;

        const T spanSize = curMaxStepSize * pSize / voxelSize;
        Bow::Logging::info("CCD spatial hash avg span size ", spanSize);
        if (spanSize > 1) {
            curMaxStepSize /= spanSize;
            Bow::Logging::info("curMaxStepSize reduced");
        }

        Matrix<T, Eigen::Dynamic, dim> SV(boundaryNode.size(), dim);
        Matrix<T, Eigen::Dynamic, dim> SVt(boundaryNode.size(), dim);
        std::unordered_map<int, int> vI2SVI;
        for (int svI = 0; svI < (int)boundaryNode.size(); ++svI) {
            int vI = boundaryNode[svI];
            vI2SVI[vI] = svI;
            SV.row(svI) = X[vI];
            SVt.row(svI) = X[vI] + curMaxStepSize * searchDir[vI];
        }

        leftBottomCorner = SV.colwise().minCoeff().array().min(SVt.colwise().minCoeff().array()) - thickness / 2;
        rightTopCorner = SV.colwise().maxCoeff().array().max(SVt.colwise().maxCoeff().array()) + thickness / 2;
        Array<T, dim, 1> range = rightTopCorner - leftBottomCorner;
        one_div_voxelSize = 1.0 / voxelSize;
        Array<long, dim, 1> vcl = (range * one_div_voxelSize).ceil().template cast<long>().max(Array<long, dim, 1>::Ones());
        long voxelAmt = vcl.matrix().prod();
        if (voxelAmt > 1e9) {
            voxelSize *= std::pow(voxelAmt / 1.0e9, 1.0 / 3);
            one_div_voxelSize = 1.0 / voxelSize;
        }
        voxelCount = (range * one_div_voxelSize).ceil().template cast<int>().max(Array<int, dim, 1>::Ones());
        if (voxelCount.minCoeff() <= 0) {
            // cast overflow due to huge search direction
            one_div_voxelSize = 16.0 / (range.maxCoeff() * 1.01);
            voxelCount = (range * one_div_voxelSize).ceil().template cast<int>().max(Array<int, dim, 1>::Ones());
        }
        Bow::Logging::info("CCD SH voxel count ", voxelCount.prod());
        voxelCount0x1 = voxelCount[0] * voxelCount[1];

        surfEdgeStartInd = boundaryNode.size();
        surfTriStartInd = surfEdgeStartInd + boundaryEdge.size();

        // precompute svVAI
        Field<Array<int, dim, 1>> svMinVAI(boundaryNode.size());
        Field<Array<int, dim, 1>> svMaxVAI(boundaryNode.size());
        tbb::parallel_for(size_t(0), boundaryNode.size(), [&](size_t svI) {
            Vector<T, dim> minCoord = SV.row(svI).array().min(SVt.row(svI).array()) - thickness / 2;
            Vector<T, dim> maxCoord = SV.row(svI).array().max(SVt.row(svI).array()) + thickness / 2;
            locate_voxel_axis_index(minCoord, svMinVAI[svI]);
            locate_voxel_axis_index(maxCoord, svMaxVAI[svI]);
        });

        voxel.clear(); //TODO: can try parallel insert
        pointAndEdgeOccupancy.resize(0);
        pointAndEdgeOccupancy.resize(surfTriStartInd);

        tbb::parallel_for(size_t(0), boundaryNode.size(), [&](size_t svI) {
            const Array<int, dim, 1>& mins = svMinVAI[svI];
            const Array<int, dim, 1>& maxs = svMaxVAI[svI];
            pointAndEdgeOccupancy[svI].reserve((maxs - mins + 1).prod());
            if constexpr (dim == 3) {
                for (int iz = mins[2]; iz <= maxs[2]; ++iz) {
                    int zOffset = iz * voxelCount0x1;
                    for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                        int yzOffset = iy * voxelCount[0] + zOffset;
                        for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                            pointAndEdgeOccupancy[svI].emplace_back(ix + yzOffset);
                        }
                    }
                }
            }
            else {
                for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                    int yOffset = iy * voxelCount[0];
                    for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                        pointAndEdgeOccupancy[svI].emplace_back(ix + yOffset);
                    }
                }
            }
        });

        tbb::parallel_for(size_t(0), boundaryEdge.size(), [&](size_t seCount) {
            int seIInd = seCount + surfEdgeStartInd;
            const auto& seI = boundaryEdge[seCount];

            Array<int, dim, 1> mins = svMinVAI[vI2SVI[seI[0]]].min(svMinVAI[vI2SVI[seI[1]]]);
            Array<int, dim, 1> maxs = svMaxVAI[vI2SVI[seI[0]]].max(svMaxVAI[vI2SVI[seI[1]]]);
            pointAndEdgeOccupancy[seIInd].reserve((maxs - mins + 1).prod());
            if constexpr (dim == 3) {
                for (int iz = mins[2]; iz <= maxs[2]; ++iz) {
                    int zOffset = iz * voxelCount0x1;
                    for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                        int yzOffset = iy * voxelCount[0] + zOffset;
                        for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                            pointAndEdgeOccupancy[seIInd].emplace_back(ix + yzOffset);
                        }
                    }
                }
            }
            else {
                for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                    int yOffset = iy * voxelCount[0];
                    for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                        pointAndEdgeOccupancy[seIInd].emplace_back(ix + yOffset);
                    }
                }
            }
        });

        Field<Field<int>> voxelLoc_sf;
        if constexpr (dim == 3) {
            voxelLoc_sf.resize(boundaryTri.size());
            tbb::parallel_for(size_t(0), boundaryTri.size(), [&](size_t sfI) {
                Array<int, dim, 1> mins = svMinVAI[vI2SVI[boundaryTri[sfI][0]]].min(svMinVAI[vI2SVI[boundaryTri[sfI][1]]]).min(svMinVAI[vI2SVI[boundaryTri[sfI][2]]]);
                Array<int, dim, 1> maxs = svMaxVAI[vI2SVI[boundaryTri[sfI][0]]].max(svMaxVAI[vI2SVI[boundaryTri[sfI][1]]]).max(svMaxVAI[vI2SVI[boundaryTri[sfI][2]]]);
                voxelLoc_sf[sfI].reserve((maxs - mins + 1).prod());
                for (int iz = mins[2]; iz <= maxs[2]; ++iz) {
                    int zOffset = iz * voxelCount0x1;
                    for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                        int yzOffset = iy * voxelCount[0] + zOffset;
                        for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                            voxelLoc_sf[sfI].emplace_back(ix + yzOffset);
                        }
                    }
                }
            });
        }

        for (int i = 0; i < (int)pointAndEdgeOccupancy.size(); ++i) {
            for (const auto& voxelI : pointAndEdgeOccupancy[i]) {
                voxel[voxelI].emplace_back(i);
            }
        }
        if constexpr (dim == 3) {
            for (int sfI = 0; sfI < (int)voxelLoc_sf.size(); ++sfI) {
                for (const auto& voxelI : voxelLoc_sf[sfI]) {
                    voxel[voxelI].emplace_back(sfI + surfTriStartInd);
                }
            }
        }
    }

    void query_point_for_primitives(int svI,
        Field<int>& pointInds,
        Field<int>& edgeInds,
        Field<int>& triInds) const
    {
        triInds.resize(0);
        edgeInds.resize(0);
        pointInds.resize(0);
        for (const auto& voxelInd : pointAndEdgeOccupancy[svI]) {
            const auto& voxelI = voxel.find(voxelInd);
            BOW_ASSERT_INFO(voxelI != voxel.end(), "cannot find spatial hash voxel");
            for (const auto& indI : voxelI->second) {
                if (indI >= surfTriStartInd) {
                    triInds.emplace_back(indI - surfTriStartInd);
                }
                else if (indI >= surfEdgeStartInd) {
                    edgeInds.emplace_back(indI - surfEdgeStartInd);
                }
                else {
                    pointInds.emplace_back(indI);
                }
            }
        }
        std::sort(triInds.begin(), triInds.end());
        triInds.erase(std::unique(triInds.begin(), triInds.end()), triInds.end());
        std::sort(edgeInds.begin(), edgeInds.end());
        edgeInds.erase(std::unique(edgeInds.begin(), edgeInds.end()), edgeInds.end());
        std::sort(pointInds.begin(), pointInds.end());
        pointInds.erase(std::unique(pointInds.begin(), pointInds.end()), pointInds.end());
    }

    // will only put edges with larger than seI index into edgeInds
    void query_edge_for_edges(int seI, Field<int>& edgeInds) const
    {
        edgeInds.resize(0);
        for (const auto& voxelInd : pointAndEdgeOccupancy[seI + surfEdgeStartInd]) {
            const auto& voxelI = voxel.find(voxelInd);
            BOW_ASSERT_INFO(voxelI != voxel.end(), "cannot find spatial hash voxel");
            for (const auto& indI : voxelI->second) {
                if (indI >= surfEdgeStartInd && indI < surfTriStartInd && indI - surfEdgeStartInd > seI) {
                    edgeInds.emplace_back(indI - surfEdgeStartInd);
                }
            }
        }
        std::sort(edgeInds.begin(), edgeInds.end());
        edgeInds.erase(std::unique(edgeInds.begin(), edgeInds.end()), edgeInds.end());
    }

    void query_point_for_edges(int svI, Field<int>& edgeInds) const
    {
        edgeInds.resize(0);
        for (const auto& voxelInd : pointAndEdgeOccupancy[svI]) {
            const auto& voxelI = voxel.find(voxelInd);
            BOW_ASSERT_INFO(voxelI != voxel.end(), "cannot find spatial hash voxel");
            for (const auto& indI : voxelI->second) {
                if (indI >= surfEdgeStartInd && indI < surfTriStartInd) {
                    edgeInds.emplace_back(indI - surfEdgeStartInd);
                }
            }
        }
        std::sort(edgeInds.begin(), edgeInds.end());
        edgeInds.erase(std::unique(edgeInds.begin(), edgeInds.end()), edgeInds.end());
    }

    void query_point_for_points(int svI, Field<int>& pointInds) const
    {
        pointInds.resize(0);
        for (const auto& voxelInd : pointAndEdgeOccupancy[svI]) {
            const auto& voxelI = voxel.find(voxelInd);
            BOW_ASSERT_INFO(voxelI != voxel.end(), "cannot find spatial hash voxel");
            for (const auto& indI : voxelI->second) {
                if (indI < surfEdgeStartInd) {
                    pointInds.emplace_back(indI);
                }
            }
        }
        std::sort(pointInds.begin(), pointInds.end());
        pointInds.erase(std::unique(pointInds.begin(), pointInds.end()), pointInds.end());
    }

public: // helper functions
    int locate_voxel_index(const Vector<T, dim>& pos) const
    {
        Array<int, dim, 1> voxelAxisIndex;
        locate_voxel_axis_index(pos, voxelAxisIndex);
        if constexpr (dim == 3) {
            return voxelAxisIndex[0] + voxelAxisIndex[1] * voxelCount[0] + voxelAxisIndex[2] * voxelCount0x1;
        }
        else {
            return voxelAxisIndex[0] + voxelAxisIndex[1] * voxelCount[0];
        }
    }
    void locate_voxel_axis_index(const Vector<T, dim>& pos,
        Array<int, dim, 1>& voxelAxisIndex) const
    {
        voxelAxisIndex = ((pos - leftBottomCorner) * one_div_voxelSize).array().floor().template cast<int>();
    }
};

} // namespace Bow