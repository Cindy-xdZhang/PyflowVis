#include "VastistasVelocityGenerator.h"
#include <cmath>
#include <corecrt_math_defines.h>

VastistasVelocityGenerator::VastistasVelocityGenerator(int Xdim, int Ydim, Eigen::Vector2d minBondary, Eigen::Vector2d maxBondary, double rc, double n)
    : mgridDim_x(Xdim)
    , mgridDim_y(Ydim)
    , rc(rc)
    , n(n)
    , domainBoundaryMin(minBondary)
    , domainBoundaryMax(maxBondary)
{
    const auto xmin = minBondary(0);
    const auto ymin = minBondary(1);
    const auto xmax = maxBondary(0);
    const auto ymax = maxBondary(1);
    const double gridIntervalX = (xmax - xmin) / (Xdim - 1);
    const double gridIntervalY = (ymax - ymin) / (Ydim - 1);
    spatialGridInterval = { gridIntervalX, gridIntervalY };
    // S1
    SiMatices_[0] << 1.0, 0.0, 0.0, -1.0;
    /*SiMatices_[0].row(0) = Eigen::RowVector2f{ 1.0f, 0.0f };
    SiMatices_[0].row(1) = Eigen::RowVector2f{ 0.0f, -1.0f };*/
    // S2
    SiMatices_[1] << 0.0, 1.0, -1.0, 0.0;
    /*SiMatices_[1].row(0)  = { 0.0f, 1.0f };
    SiMatices_[1].row(1)  = { -1.0f, 0.0f };*/
    // S3
    SiMatices_[2] << 0.0, -1.0, 1.0, 0.0;
    /*SiMatices_[2].row(0)  = { 0.0f, -1.0f };
    SiMatices_[2].row(1) = { 1.0f, 0.0f };*/
}

// resample VastistasVelocity to discrete grid
velocityFieldData VastistasVelocityGenerator::generateSteady(double sx, double sy, double theta, int Si) const noexcept
{

    std::vector<std::vector<Eigen::Vector2d>> data_(mgridDim_y, std::vector<Eigen::Vector2d>(mgridDim_x, Eigen::Vector2d { 0.0, 0.0 }));

    /*  const auto SiMat22 = SiMatices_[Si];*/
    const auto SiMat22 = SiMatices_[Si];

    for (size_t i = 0; i < mgridDim_y; i++)
        for (size_t j = 0; j < mgridDim_x; j++) {
            auto xy = getPosition(j, i);

            const double r = xy.norm();
            const double vastis = NormalizedVastistasV0(r);

            /*const double vp_row0 = SiMat22[0] * xy * v0;
            const double vp_row1 = SiMat22[1] * xy * v0;*/
            auto vp = SiMat22 * xy;
            data_[i][j] = vp * vastis;
        }
    return data_;
}
SteadyVectorField2D VastistasVelocityGenerator::generateSteadyField(double tx, double ty, double sx, double sy, double theta, int Si) const noexcept
{

    std::vector<std::vector<Eigen::Vector2d>> data_(mgridDim_y, std::vector<Eigen::Vector2d>(mgridDim_x, Eigen::Vector2d { 0.0, 0.0 }));

    /*  const auto SiMat22 = SiMatices_[Si];*/
    const auto SiMat22 = SiMatices_[Si];
    const Eigen::Vector2d critial_point = { tx, ty };
    for (size_t i = 0; i < mgridDim_y; i++)
        for (size_t j = 0; j < mgridDim_x; j++) {
            auto xy = getPosition(j, i);
            Eigen::Vector2d xy_txy = xy - critial_point;
            const double vastis = NormalizedVastistasV0(xy_txy.norm());

            /*const double vp_row0 = SiMat22[0] * xy * v0;
            const double vp_row1 = SiMat22[1] * xy * v0;*/
            auto vp = SiMat22 * xy_txy * vastis;
            data_[i][j] = vp;
        }
    Eigen::Vector2i XdimYdim = { mgridDim_x, mgridDim_y };
    SteadyVectorField2D steadyField {
        std::move(data_),
        this->domainBoundaryMin,
        this->domainBoundaryMax,
        XdimYdim
    };
    steadyField.analyticalFlowfunc_ = [this, SiMat22, critial_point](const Eigen::Vector2d& pos, double t) -> Eigen::Vector2d {
        Eigen::Vector2d xy_txy = pos - critial_point;
        const double vastis = NormalizedVastistasV0(xy_txy.norm());
        auto vp = SiMat22 * xy_txy * vastis;
        return vp;
    };

    return steadyField;
}
velocityFieldData VastistasVelocityGenerator::generateSteadyV2(double cx, double cy, double dx, double dy, double tx, double ty) const noexcept
{
    std::vector<std::vector<Eigen::Vector2d>> data_(mgridDim_y, std::vector<Eigen::Vector2d>(mgridDim_x, Eigen::Vector2d { 0.0, 0.0 }));

    const Eigen::Vector2d dx_cx = { dx, cx };
    const Eigen::Vector2d mcy_dy = { -cy, dy };

    const Eigen::Vector2d critial_point = { tx, ty };
    for (size_t i = 0; i < mgridDim_y; i++)
        for (size_t j = 0; j < mgridDim_x; j++) {
            auto xy = getPosition(j, i);
            Eigen::Vector2d xy_txy = xy - critial_point;
            const double v0 = NormalizedVastistasV0(xy_txy.norm());
            const double vp_row0 = dx_cx.dot(xy_txy) * v0;
            const double vp_row1 = mcy_dy.dot(xy_txy) * v0;
            data_[i][j] = { vp_row0, vp_row1 };
        }
    return data_;
}

AnalyticalFlowCreator::AnalyticalFlowCreator(Eigen::Vector2i grid_size, int time_steps /*= 10*/, Eigen::Vector2d domainBoundaryMin /*= Eigen::Vector2d(-2.0, -2.0)*/, Eigen::Vector2d domainBoundaryMax /*= Eigen::Vector2d(2.0, 2.0)*/,
    double tmin /*= 0.0f*/, double tmax /*=2 * M_PI*/)
    : grid_size(grid_size)
    , time_steps(time_steps)
    , domainBoundaryMin(domainBoundaryMin)
    , domainBoundaryMax(domainBoundaryMax)
    , tmin(tmin)
    , tmax(tmax)
{
    // Generate time steps
    t_interval = (tmax - tmin) / (time_steps - 1);

    t_values.resize(time_steps);
    for (int i = 0; i < time_steps; ++i) {
        t_values[i] = tmin + i * t_interval;
    }

    // Generate grid
    x_values = Eigen::VectorXd::LinSpaced(grid_size.x(), domainBoundaryMin.x(), domainBoundaryMax.x());
    y_values = Eigen::VectorXd::LinSpaced(grid_size.y(), domainBoundaryMin.y(), domainBoundaryMax.y());
    spatialGridInterval = { (domainBoundaryMax.x() - domainBoundaryMin.x()) / (grid_size.x() - 1),
        (domainBoundaryMax.y() - domainBoundaryMin.y()) / (grid_size.y() - 1) };
}

UnSteadyVectorField2D AnalyticalFlowCreator::createFlowField(std::function<Eigen::Vector2d(Eigen::Vector2d, double)> lambda_func)
{
    UnSteadyVectorField2D vectorField2d;
    vectorField2d.spatialDomainMinBoundary = domainBoundaryMin;
    vectorField2d.spatialDomainMaxBoundary = domainBoundaryMax;
    vectorField2d.timeSteps = time_steps;
    vectorField2d.tmin = tmin;
    vectorField2d.tmax = tmax;
    vectorField2d.spatialGridInterval = spatialGridInterval;

    vectorField2d.field.resize(time_steps, std::vector<std::vector<Eigen::Vector2d>>(grid_size.y(), std::vector<Eigen::Vector2d>(grid_size.x())));

    for (int t = 0; t < time_steps; ++t) {
        double time = t_values[t];
        for (int y = 0; y < grid_size.y(); ++y) {
            for (int x = 0; x < grid_size.x(); ++x) {
                Eigen::Vector2d pos(x_values[x], y_values[y]);
                vectorField2d.field[t][y][x] = lambda_func(pos, time);
            }
        }
    }

    return vectorField2d;
}
