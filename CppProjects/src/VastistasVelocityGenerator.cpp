#include "VastistasVelocityGenerator.h"
#include <cmath>
#include <corecrt_math_defines.h>

VastistasVelocityGenerator::VastistasVelocityGenerator(int Xdim, int Ydim, Eigen::Vector2f minBondary, Eigen::Vector2f maxBondary, float rc, float n)
    : mgridDim_x(Xdim)
    , mgridDim_y(Ydim)
    , rc(rc)
    , n(n)
    , xmin(minBondary(0))
    , ymin(minBondary(1))
    , xmax(maxBondary(0))
    , ymax(maxBondary(1))
{

    gridIntervalX = (xmax - xmin) / (Xdim - 1);
    gridIntervalY = (ymax - ymin) / (Ydim - 1);
    // S1
    SiMatices_[0] << 1.0f, 0.0f, 0.0f, -1.0f;
    /*SiMatices_[0].row(0) = Eigen::RowVector2f{ 1.0f, 0.0f };
    SiMatices_[0].row(1) = Eigen::RowVector2f{ 0.0f, -1.0f };*/
    // S2
    SiMatices_[1] << 0.0f, 1.0f, -1.0f, 0.0f;
    /*SiMatices_[1].row(0)  = { 0.0f, 1.0f };
    SiMatices_[1].row(1)  = { -1.0f, 0.0f };*/
    // S3
    SiMatices_[2] << 0.0f, -1.0f, 1.0f, 0.0f;
    /*SiMatices_[2].row(0)  = { 0.0f, -1.0f };
    SiMatices_[2].row(1) = { 1.0f, 0.0f };*/
}

// resample VastistasVelocity to discrete grid
velocityFieldData VastistasVelocityGenerator::generateSteady(float sx, float sy, float theta, int Si) const noexcept
{

    std::vector<std::vector<Eigen::Vector2f>> data_(mgridDim_y, std::vector<Eigen::Vector2f>(mgridDim_x, Eigen::Vector2f { 0.0, 0.0 }));

    /*  const auto SiMat22 = SiMatices_[Si];*/
    const auto SiMat22 = SiMatices_[Si];

    for (size_t i = 0; i < mgridDim_y; i++)
        for (size_t j = 0; j < mgridDim_x; j++) {
            auto xy = getPosition(j, i);

            const float r = xy.norm();
            const float v0 = NormalizedVastistasV0(r);

            /*const float vp_row0 = SiMat22[0] * xy * v0;
            const float vp_row1 = SiMat22[1] * xy * v0;*/
            auto vp = SiMat22 * xy;
            data_[i][j] = vp;
        }
    return data_;
}

velocityFieldData VastistasVelocityGenerator::generateSteadyV2(float cx, float cy, float dx, float dy, float tx, float ty) const noexcept
{
    std::vector<std::vector<Eigen::Vector2f>> data_(mgridDim_y, std::vector<Eigen::Vector2f>(mgridDim_x, Eigen::Vector2f { 0.0, 0.0 }));

    const Eigen::Vector2f dx_cx = { dx, cx };
    const Eigen::Vector2f mcy_dy = { -cy, dy };

    const Eigen::Vector2f critial_point = { tx, ty };
    for (size_t i = 0; i < mgridDim_y; i++)
        for (size_t j = 0; j < mgridDim_x; j++) {
            auto xy = getPosition(j, i);
            Eigen::Vector2f xy_txy = xy - critial_point;
            const float v0 = NormalizedVastistasV0(xy_txy.norm());
            const float vp_row0 = dx_cx.dot(xy_txy) * v0;
            const float vp_row1 = mcy_dy.dot(xy_txy) * v0;
            data_[i][j] = { vp_row0, vp_row1 };
        }
    return data_;
}
