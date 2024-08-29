#include "VastistasVelocityGenerator.h"
#include "flowGenerator.h"
#include <cmath>
#include <corecrt_math_defines.h>
#include <random>
std::mt19937 rng(static_cast<unsigned int>(std::time(0)));
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

SteadyVectorField2D VastistasVelocityGenerator::generateSteadyField_VortexBoundaryVIS2020(double tx, double ty, double sx, double sy, double theta, VastisVortexType Si) const noexcept
{

    std::vector<std::vector<Eigen::Vector2d>> data_(mgridDim_y, std::vector<Eigen::Vector2d>(mgridDim_x, Eigen::Vector2d { 0.0, 0.0 }));
    Eigen::Matrix2d deformMatA = Eigen::Matrix2d::Identity();
    deformMatA(0, 0) = sx * cos(theta);
    deformMatA(0, 1) = -sy * sin(theta);
    deformMatA(1, 0) = sx * sin(theta);
    deformMatA(1, 1) = sy * cos(theta);
    const auto SiMat22 = SiMatices_[static_cast<int>(Si)];
    const Eigen::Vector2d translation_t = { tx, ty };

    auto lambdaFunc = [this, SiMat22, translation_t, deformMatA](const Eigen::Vector2d& pos, double t) -> Eigen::Vector2d {
        auto originalPos = deformMatA.inverse() * (pos - translation_t);
        // get standard VastistasVelocity vector on originalPos (v(x=originalPos))
        auto standardVastistasVelocity = SiMat22 * originalPos * NormalizedVastistasV0(originalPos.norm());
        // get the transformed VastistasVelocity vector on pos (v(x=pos))
        auto transformedVastistasVelocity = deformMatA * standardVastistasVelocity;
        return transformedVastistasVelocity;
    };

    for (size_t i = 0; i < mgridDim_y; i++)
        for (size_t j = 0; j < mgridDim_x; j++) {
            auto pos = getPosition(j, i);
            data_[i][j] = lambdaFunc(pos, 0.0);
        }
    Eigen::Vector2i XdimYdim = { mgridDim_x, mgridDim_y };
    SteadyVectorField2D steadyField {
        std::move(data_),
        this->domainBoundaryMin,
        this->domainBoundaryMax,
        XdimYdim
    };
    steadyField.analyticalFlowfunc_ = lambdaFunc;
    return steadyField;
}

SteadyVectorField2D VastistasVelocityGenerator::generateSteadyFieldMixture(int mixture) const noexcept
{
    std::vector<std::vector<Eigen::Vector2d>> data_(mgridDim_y, std::vector<Eigen::Vector2d>(mgridDim_x, Eigen::Vector2d { 0.0, 0.0 }));

    std::normal_distribution<double> genTx(0.0, 0.59);
    std::normal_distribution<double> genTy(0.0, 0.62);
    std::normal_distribution<double> genCx(0.0, 0.49);
    std::normal_distribution<double> genDx(0.0, 0.35);
    std::normal_distribution<double> genCy(0.0, 0.47);
    std::normal_distribution<double> genDy(0.0, 0.25);

    std::normal_distribution<double> dist_rc(1.87, 0.37); // mean = 1.87, stddev = 0.34
    std::normal_distribution<double> dist_n(1.96, 0.61); // mean = 1.96, stddev = 0.61
    // random generate mixture pairs of sx,sy, theta
    std::vector<Eigen::Matrix2d> mixturesdeformMats;
    std::vector<Eigen::Vector2d> txyParams;
    std::vector<Eigen::Vector2d> n_rcParams;
    for (int i = 0; i < mixture; i++) {
        double tx = genTx(rng);
        double ty = genTy(rng);
        double cx = genCx(rng);
        double dx = genDx(rng);
        double cy = genCy(rng);
        double dy = genDy(rng);

        Eigen::Vector2d tmp_n_rc;
        tmp_n_rc(0) = dist_n(rng);
        tmp_n_rc(1) = dist_rc(rng);
        n_rcParams.emplace_back(tmp_n_rc);

        Eigen::Matrix2d deformMat = Eigen::Matrix2d::Identity();
        deformMat(0, 0) = dx;
        deformMat(0, 1) = cx;
        deformMat(1, 0) = -cy;
        deformMat(1, 1) = dy;
        mixturesdeformMats.emplace_back(deformMat);
        txyParams.emplace_back(Eigen::Vector2d(tx, ty));
    }

    auto lambdaFunc = [this, mixturesdeformMats, txyParams, n_rcParams, mixture](const Eigen::Vector2d& pos, double t) -> Eigen::Vector2d {
        Eigen::Vector2d result = Eigen::Vector2d::Zero();
        for (int i = 0; i < mixture; i++) {
            const auto tmp_n_rc = n_rcParams[i];
            const auto deformMat = mixturesdeformMats[i];
            const auto critial_point = txyParams[i];

            Eigen::Vector2d xy_txy = pos - critial_point;
            const double vastis = NormalizedVastistasV0_Fn(xy_txy.norm(), tmp_n_rc(0), tmp_n_rc(1));
            auto vp = deformMat * xy_txy * vastis; // eq (6) of paper Robust Reference Frame Extraction
            result += vp;
        }
        return result;
    };

    Eigen::Vector2i XdimYdim = { mgridDim_x, mgridDim_y };
    for (size_t i = 0; i < mgridDim_y; i++)
        for (size_t j = 0; j < mgridDim_x; j++) {
            auto pos = getPosition(j, i);
            data_[i][j] = lambdaFunc(pos, 0.0);
        }

    SteadyVectorField2D steadyField {
        std::move(data_),
        this->domainBoundaryMin,
        this->domainBoundaryMax,
        XdimYdim
    };
    steadyField.analyticalFlowfunc_ = lambdaFunc;
    return steadyField;
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
    t_values = Eigen::VectorXd::LinSpaced(time_steps, tmin, tmax);
    // Generate grid
    x_values = Eigen::VectorXd::LinSpaced(grid_size.x(), domainBoundaryMin.x(), domainBoundaryMax.x());
    y_values = Eigen::VectorXd::LinSpaced(grid_size.y(), domainBoundaryMin.y(), domainBoundaryMax.y());
}

UnSteadyVectorField2D AnalyticalFlowCreator::sampleAnalyticalFunctionAsFlowField(AnalyticalFlowFunc2D lambda_func)
{
    UnSteadyVectorField2D vectorField2d;
    vectorField2d.spatialDomainMinBoundary = domainBoundaryMin;
    vectorField2d.spatialDomainMaxBoundary = domainBoundaryMax;
    vectorField2d.timeSteps = time_steps;
    vectorField2d.tmin = tmin;
    vectorField2d.tmax = tmax;
    vectorField2d.spatialGridInterval = { (domainBoundaryMax.x() - domainBoundaryMin.x()) / (grid_size.x() - 1),
        (domainBoundaryMax.y() - domainBoundaryMin.y()) / (grid_size.y() - 1) };

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
    vectorField2d.analyticalFlowfunc_ = lambda_func;
    return vectorField2d;
}

UnSteadyVectorField2D AnalyticalFlowCreator::createRFC(double alt /*= 1.0*/, double maxV /*= 1.0*/, double iscale /*= 8.0*/)
{

    // Define a lambda function for rotating flow
    auto rotatingFourCenter = [=](Eigen::Vector2d p, double t) {
        const double x = p(0);
        const double y = p(1);
        const double al_t = alt;
        const double scale = iscale;
        const double maxVelocity = maxV;

        double u = exp(-y * y - x * x) * (al_t * y * exp(y * y + x * x) - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * y * y * y + (12.0 * scale * (cos(al_t * t) * cos(al_t * t)) - 6.0 * scale) * x * y * y + (6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * x + 6.0 * scale * cos(al_t * t) * sin(al_t * t)) * y + (3.0 * scale - 6.0 * scale * (cos(al_t * t) * cos(al_t * t))) * x);
        double v = -exp(-y * y - x * x) * (al_t * x * exp(y * y + x * x) - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * y * y + ((12.0 * scale * (cos(al_t * t) * cos(al_t * t)) - 6.0 * scale) * x * x - 6.0 * scale * (cos(al_t * t) * cos(al_t * t)) + 3.0 * scale) * y + 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * x * x - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x);

        double vecU = maxVelocity * u;
        double vecV = maxVelocity * v;

        Eigen::Vector2d components;
        components << vecU, vecV;
        return components;
    };
    return sampleAnalyticalFunctionAsFlowField(rotatingFourCenter);
}

UnSteadyVectorField2D AnalyticalFlowCreator::createBeadsFlow()
{
    auto lambda = getAnalyticalFlowFieldFunction("beads (Weinkauf and Theisel 2010)", tmin, tmax, time_steps);
    return sampleAnalyticalFunctionAsFlowField(lambda);
}
UnSteadyVectorField2D AnalyticalFlowCreator::createBeadsFlowNoContraction()
{
    AnalyticalFlowFunc2D lambda = [this](const Eigen::Vector2d& p, double t) {
        double x = p(0);
        double y = p(1);
        double u = -2 * y + (2 / 3) * sin(t);
        double v = 2 * x - (2 / 3) * cos(t);
        const Eigen::Vector2d result({ u, v });
        return result;
    };

    return sampleAnalyticalFunctionAsFlowField(lambda);
}
UnSteadyVectorField2D AnalyticalFlowCreator::createUnsteadyGyre()
{
    auto lambda = getAnalyticalFlowFieldFunction("unsteady gyre (Haller2023)", tmin, tmax, time_steps);
    return sampleAnalyticalFunctionAsFlowField(lambda);
}

AnalyticalFlowFunc2D AnalyticalFlowCreator::getAnalyticalFlowFieldFunction(const std::string& name, double tMin, double tMax, int numberOfTimeSteps)
{
    AnalyticalFlowFunc2D fieldFunction;
    if (name == "bickley jet") {
        fieldFunction = [tMin, tMax, numberOfTimeSteps](const Eigen::Vector2d& p, double t) {
            double x = p(0);
            double y = p(1);

            double U = 62.66e-6;
            double L = 1770e-3;
            double c2 = 0.205 * U;
            double c3 = 0.461 * U;
            double eps1 = 0.0075;
            double eps2 = 0.15;
            double eps3 = 0.3;
            double r0 = 6371e-3;
            double k1 = 2 / r0;
            double k2 = 4 / r0;
            double k3 = 6 / r0;
            double c1 = c3 + ((sqrt(5) - 1) / 2) * (k2 / k1) * (c2 - c3);
            double sechYLsquared = 1.0 / cosh(y / L);
            sechYLsquared = sechYLsquared * sechYLsquared;
            double u = U * sechYLsquared + (2 * eps1 * U * cos(k1 * (x - c1 * t)) + 2 * eps2 * U * cos(k2 * (x - c2 * t)) + 2 * eps3 * U * cos(k3 * (x - c3 * t))) * tanh(y / L) * sechYLsquared;

            double v = -(eps1 * k1 * U * L * sin(k1 * (x - c1 * t)) + eps2 * k2 * U * L * sin(k2 * (x - c2 * t)) + eps3 * k3 * U * L * sin(k3 * (x - c3 * t))) * sechYLsquared;
            const Eigen::Vector2d result({ u, v });
            return result;
        };
    }
    if (name == "beads (Weinkauf and Theisel 2010)") {
        fieldFunction = [tMin, tMax, numberOfTimeSteps](const Eigen::Vector2d& p, double t) {
            double x = p(0);
            double y = p(1);

            double u = -1.0 * (y - 1.0 / 3.0 * sin(t)) - (x - 1.0 / 3.0 * cos(t));
            double v = (x - 1.0 / 3.0 * cos(t)) - (y - 1.0 / 3.0 * sin(t));
            const Eigen::Vector2d result({ u, v });
            return result;
        };
    }

    if (name == "unsteady gyre (Haller2023)") {
        fieldFunction = [tMin, tMax, numberOfTimeSteps](const Eigen::Vector2d& p, double t) {
            double x = p(0);
            double y = p(1);
            double w = 4;

            double u = w * y - cos(t * w) * ((x * x) + (y * y) - 1) - 2 * y * (y * cos(t * w) + x * sin(t * w));
            double v = sin(t * w) * ((x * x) + (y * y) - 1) - w * x + 2 * x * (y * cos(t * w) + x * sin(t * w));

            const Eigen::Vector2d result({ u, v });

            return result;
        };
    }

    if (name == "eq33 (Haller2023)") {
        fieldFunction = [tMin, tMax, numberOfTimeSteps](const Eigen::Vector2d& p, double t) {
            double x = p(0);
            double y = p(1);

            double u = x * sin(4 * t) + (x * ((x * x) - 3 * (y * y))) / 200 + y * (cos(4 * t) + 2);
            double v = x * (cos(4 * t) - 2) - y * sin(4 * t) - (x * (3 * (x * x) - (y * y))) / 200;

            const Eigen::Vector2d result({ u, v });

            return result;
        };
    }

    if (name == "eq34 (Haller2023)") {
        fieldFunction = [tMin, tMax, numberOfTimeSteps](const Eigen::Vector2d& p, double t) {
            double x = p(0);
            double y = p(1);

            double u = (3 * (y * y)) / 200 + (cos(4 * t) + 1 / 2) * y + x * sin(4 * t) - (3 * (x * x)) / 200;
            double v = (3 * x * y) / 100 - y * sin(4 * t) + x * (cos(4 * t) - 1 / 2);

            const Eigen::Vector2d result({ u, v });

            return result;
        };
    }

    if (name == "observed sheer flow") {
        fieldFunction = [tMin, tMax, numberOfTimeSteps](const Eigen::Vector2d& p, double t) {
            double x = p(0);
            double y = p(1);
            double partialX = 2 * (-20 * pow(t, 3) + 60 * pow(t, 4) - 60 * pow(t, 5) + 20 * pow(t, 6) + x);
            double partialY = 4 * y;
            double u = -partialY;
            double v = partialX;

            const Eigen::Vector2d result({ u, v });
            return result;
        };
    }

    if (name == "bernstein polynomial translation") {
        fieldFunction = [tMin, tMax, numberOfTimeSteps](const Eigen::Vector2d& p, double t) {
            double x = p(0);
            double y = p(1);
            double u = 0;
            double v = 2 * (-20 * pow(t, 3) + 60 * pow(t, 4) - 60 * pow(t, 5) + 20 * pow(t, 6));
            const Eigen::Vector2d result({ u, v });
            return result;
        };
    }

    if (name == "sheer flow") {
        fieldFunction = [tMin, tMax, numberOfTimeSteps](const Eigen::Vector2d& p, double t) {
            double x = p(0);
            double y = p(1);
            double u = -2 * y;
            double v = 0;
            const Eigen::Vector2d result({ u, v });
            return result;
        };
    }

    if (name == "steady rotation") {
        fieldFunction = [tMin, tMax, numberOfTimeSteps](const Eigen::Vector2d& p, double t) {
            double x = p(0);
            double y = p(1);
            double partialX = x;
            double partialY = y;
            double u = -partialY;
            double v = partialX;
            const Eigen::Vector2d result({ u, v });
            return result;
        };
    }

    if (name == "steady vortex") {
        fieldFunction = [tMin, tMax, numberOfTimeSteps](const Eigen::Vector2d& p, double t) {
            double x = p(0);
            double y = p(1);
            double partialX = x;
            double partialY = y;
            double u = -partialY * exp(-x * x - y * y);
            double v = partialX * exp(-x * x - y * y);
            const Eigen::Vector2d result({ u, v });
            return result;
        };
    }

    if (name == "steady inverse vortex") {
        fieldFunction = [tMin, tMax, numberOfTimeSteps](const Eigen::Vector2d& p, double t) {
            double x = p(0);
            double y = p(1);
            double partialX = x;
            double partialY = y;
            double u = -partialY * (1. - exp(-x * x - y * y));
            double v = partialX * (1. - exp(-x * x - y * y));
            const Eigen::Vector2d result({ u, v });
            return result;
        };
    }

    if (name == "steady ellipse") {
        fieldFunction = [tMin, tMax, numberOfTimeSteps](const Eigen::Vector2d& p, double t) {
            double x = p(0);
            double y = p(1);
            double partialX = 2 * x;
            double partialY = 4 * y;
            double u = -partialY;
            double v = partialX;
            const Eigen::Vector2d result({ u, v });
            return result;
        };
    }

    if (name == "steady example") {
        fieldFunction = [tMin, tMax, numberOfTimeSteps](const Eigen::Vector2d& p, double t) {
            double x = p(0);
            double y = p(1);

            double u = 2 * y * y + x - 4;
            double v = cos(x);
            const Eigen::Vector2d result({ u, v });
            return result;
        };
    }

    if (name == "steady rotation") {
        fieldFunction = [tMin, tMax, numberOfTimeSteps](const Eigen::Vector2d& p, double t) {
            double x = p(0);
            double y = p(1);
            double u = -y;
            double v = x;

            const Eigen::Vector2d result({ u, v });
            return result;
        };
    }

    if (name == "center moving on circle") {
        fieldFunction = [tMin, tMax, numberOfTimeSteps](const Eigen::Vector2d& p, double t) {
            double x = p(0);
            double y = p(1);

            double u = -2.0 * y + 2.0 / 3.0 * sin(t);
            double v = 2.0 * x - 2.0 / 3.0 * cos(t);
            const Eigen::Vector2d result({ u, v });
            return result;
        };
    }

    return fieldFunction;
}
