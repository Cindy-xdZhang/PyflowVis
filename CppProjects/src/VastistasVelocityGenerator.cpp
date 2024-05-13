#include "VastistasVelocityGenerator.h"
#include <cmath>
#include <corecrt_math_defines.h>

VastistasVelocityGenerator::VastistasVelocityGenerator(int Xdim, int Ydim, vec2d<float> minBondary, vec2d<float> maxBondary, float rc, float n)
    : mgridDim_x(Xdim)
    , mgridDim_y(Ydim)
    , rc(rc)
    , n(n)
    , xmin(minBondary.x)
    , ymin(minBondary.y)
    , xmax(maxBondary.x)
    , ymax(maxBondary.y)
{

    gridIntervalX = (xmax - xmin) / (Xdim - 1);
    gridIntervalY = (ymax - ymin) / (Ydim - 1);
    // S1
    SiMatices_[0][0] = { 1.0f, 0.0f };
    SiMatices_[0][1] = { 0.0f, -1.0f };
    // S2
    SiMatices_[1][0] = { 0.0f, 1.0f };
    SiMatices_[1][1] = { -1.0f, 0.0f };
    // S3
    SiMatices_[2][0] = { 0.0f, -1.0f };
    SiMatices_[2][1] = { 1.0f, 0.0f };
}

// resample VastistasVelocity to discrete grid
velocityFieldData VastistasVelocityGenerator::generateSteady(float sx, float sy, float theta, int Si) const noexcept
{

    std::vector<std::vector<vec2d<float>>> data_(mgridDim_y, std::vector<vec2d<float>>(mgridDim_x, vec2d<float> { 0.0, 0.0 }));

    /*  const auto SiMat22 = SiMatices_[Si];*/
    const auto SiMat22 = SiMatices_[Si];

    for (size_t i = 0; i < mgridDim_y; i++)
        for (size_t j = 0; j < mgridDim_x; j++) {
            vec2d xy = getPosition(j, i);

            const float r = xy.norm();
            const float v0 = NormalizedVastistasV0(r);

            const float vp_row0 = SiMat22[0] * xy * v0;
            const float vp_row1 = SiMat22[1] * xy * v0;

            data_[i][j] = { vp_row0, vp_row1 };
        }
    return data_;
}

velocityFieldData VastistasVelocityGenerator::generateSteadyV2(float cx, float cy, float dx, float dy, float tx, float ty) const noexcept
{
    std::vector<std::vector<vec2d<float>>> data_(mgridDim_y, std::vector<vec2d<float>>(mgridDim_x, vec2d<float> { 0.0, 0.0 }));

    const vec2d<float> dx_cx = { dx, cx };
    const vec2d<float> mcy_dy = { -cy, dy };

    const vec2d critial_point = { tx, ty };
    for (size_t i = 0; i < mgridDim_y; i++)
        for (size_t j = 0; j < mgridDim_x; j++) {
            vec2d xy = getPosition(j, i);
            vec2d xy_txy = xy - critial_point;
            const float v0 = NormalizedVastistasV0(xy_txy.norm());
            const float vp_row0 = dx_cx * xy_txy * v0;
            const float vp_row1 = mcy_dy * xy_txy * v0;
            data_[i][j] = { vp_row0, vp_row1 };
        }
    return data_;
}
