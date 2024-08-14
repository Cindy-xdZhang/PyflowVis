
#pragma once
#ifndef VASTISTASVELOCITYGENERATOR_H
#define VASTISTASVELOCITYGENERATOR_H
#include "VectorFieldCompute.h"
#include "cereal/cereal.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <corecrt_math_defines.h>
#include <string>
#include <vector>

namespace cereal {

// Specialization for saving as array
template <class Archive>
void save(Archive& ar, const Eigen::Vector2d& vec)
{
    ar(vec.x(), vec.y());
}

template <class Archive>
void save(Archive& ar, const Eigen::Vector3d& vec)
{
    ar(vec.x(), vec.y(), vec.z());
}
template <class Archive>
void save(Archive& ar, const Eigen::Matrix2d& vec)
{
    ar(vec(0, 0), vec(0, 1), vec(1, 0), vec(1, 1));
}
}

class KillingAbcField : public IUnsteadField2D {
public:
    KillingAbcField(std::function<Eigen::Vector3d(double)>& func, int n, double tmin, double tmax)
    {
        this->tmin = tmin;
        this->tmax = tmax;
        this->timeSteps = n;
        dt = (tmax - tmin) / (double)(n - 1);
        assert(n > 1);
        assert(dt > 0.0);
        centerPos = { 0.0, 0.0 };
        this->killingABCfunc_ = func;
    }
    double dt;
    Eigen::Vector2d centerPos;
    std::function<Eigen::Vector3d(double)> killingABCfunc_ = nullptr;
    template <class Archive>
    void serialize(Archive& ar)
    {
        std::vector<Eigen::Vector3d> abcs_;
        abcs_.reserve(timeSteps);
        for (size_t i = 0; i < timeSteps; i++) {
            double time = this->tmin + i * this->dt;
            Eigen::Vector3d abc = this->killingABCfunc_(time);
            abcs_.push_back(abc);
        }
        ar(CEREAL_NVP(tmin), CEREAL_NVP(tmax), CEREAL_NVP(timeSteps), CEREAL_NVP(dt) /*, CEREAL_NVP(centerPos)*/);
        ar(CEREAL_NVP(abcs_));
    }

    // give a curve of a(t),b(t),c(t), killing vector(x,t) = a(t)+b(t)+ [0,-c(t);c(t),0]*(x-o), where o is the center of the plane.
    inline Eigen::Vector2d getKillingVector(const Eigen::Vector2d& queryPos, double t) const
    {
        if (this->killingABCfunc_)
            [[likely]] {
            Eigen::Vector3d abc = this->killingABCfunc_(t);

            Eigen::Vector2d uv = { abc(0), abc(1) };
            auto ra = queryPos - centerPos;

            const auto c = abc(2);
            //[0,-c(t);c(t),0]*(ra)
            Eigen::Vector2d c_componnet = { ra(1) * -c, ra(0) * c };
            return uv + c_componnet;
        }
        assert(false);
        return {};
    }
    virtual Eigen::Vector2d getVector(int x, int y, int t) const
    {
        // we don't need this function
        assert(false);
        return getKillingVector({ (double)x, (double)y }, t);
    };

    virtual Eigen::Vector2d getVector(double x, double y, double t) const
    {
        return getKillingVector({ (double)x, (double)y }, t);
    };
    UnSteadyVectorField2D resample2UnsteadyField(
        const Eigen::Vector2i& gridSize,
        const Eigen::Vector2d& domainBoundaryMin,
        const Eigen::Vector2d& domainBoundaryMax)
    {
        UnSteadyVectorField2D vectorField2d;

        vectorField2d.spatialDomainMinBoundary = domainBoundaryMin;
        vectorField2d.spatialDomainMaxBoundary = domainBoundaryMax;
        vectorField2d.timeSteps = this->timeSteps;
        vectorField2d.tmin = this->tmin;
        vectorField2d.tmax = this->tmax;

        Eigen::Vector2d spatialGridInterval;
        spatialGridInterval(0) = (domainBoundaryMax(0) - domainBoundaryMin(0)) / (gridSize(0) - 1);
        spatialGridInterval(1) = (domainBoundaryMax(1) - domainBoundaryMin(1)) / (gridSize(1) - 1);
        vectorField2d.spatialGridInterval = spatialGridInterval;

        vectorField2d.field.resize(this->timeSteps, std::vector<std::vector<Eigen::Vector2d>>(gridSize.y(), std::vector<Eigen::Vector2d>(gridSize.x())));

        for (int t = 0; t < timeSteps; ++t) {
            double time = this->tmin + t * this->dt;
            for (int y = 0; y < gridSize.y(); ++y) {
                double posY = domainBoundaryMin(1) + y * spatialGridInterval(1);
                for (int x = 0; x < gridSize.x(); ++x) {
                    double posX = domainBoundaryMin(0) + x * spatialGridInterval(0);
                    vectorField2d.field[t][y][x] = getVector(posX, posY, time);
                }
            }
        }

        return vectorField2d;
    }
};
enum class VastisVortexType : unsigned char {
    saddle = 0,
    center_cw = 1,
    center_ccw = 2,
};
class VastistasVelocityGenerator {
public:
    VastistasVelocityGenerator(int Xdim, int Ydim, Eigen::Vector2d minBondary, Eigen::Vector2d maxBondary, double rc, double n);

    // generate a deformed steady velocity slice  following the paper: Vortex Boundary Identification using Convolutional Neural Network
    SteadyVectorField2D generateSteadyField_VortexBoundaryVIS2020(double tx, double ty, double sx, double sy, double theta, VastisVortexType Si) const noexcept;

    // support mixture of multiple Vastistas profile
    SteadyVectorField2D generateSteadyFieldMixture(int mixture) const noexcept;

    inline auto NormalizedVastistasV0(const double r) const noexcept
    {
        const auto v0_r = r / (2 * M_PI * (rc * rc) * std::pow(std::pow(r / rc, 2 * n) + 1, 1 / n));
        return v0_r / (r);
    }

    inline auto getPosition(int xid, int yid) const noexcept
    {
        double x = domainBoundaryMin.x() + xid * spatialGridInterval.x();
        double y = domainBoundaryMin.y() + yid * spatialGridInterval.y();
        return Eigen::Vector2d { x, y };
    }

private:
    int mgridDim_x;
    int mgridDim_y;
    double rc, n;

    Eigen::Vector2d spatialGridInterval;
    Eigen::Vector2d domainBoundaryMin;
    Eigen::Vector2d domainBoundaryMax;
    Eigen::Matrix2d SiMatices_[3];
};

// AnalyticalFlowCreator  is the helper class for creating UnSteadyVectorField2D
class AnalyticalFlowCreator {
public:
    AnalyticalFlowCreator(Eigen::Vector2i grid_size, int time_steps,
        Eigen::Vector2d domainBoundaryMin = Eigen::Vector2d(-2.0, -2.0),
        Eigen::Vector2d domainBoundaryMax = Eigen::Vector2d(2.0, 2.0), double tmin = 0.0f, double tmax = 2 * M_PI);

    // Method to create the flow field using a lambda function
    UnSteadyVectorField2D sampleAnalyticalFunctionAsFlowField(AnalyticalFlowFunc2D lambda_func);
    UnSteadyVectorField2D createRFC(double alt = 1.0, double maxV = 1.0, double scale = 8.0);
    UnSteadyVectorField2D createBeadsFlow();
    UnSteadyVectorField2D createBeadsFlowNoContraction();
    UnSteadyVectorField2D createUnsteadyGyre();
    AnalyticalFlowFunc2D getAnalyticalFlowFieldFunction(const std::string& name, double tMin, double tMax, int numberOfTimeSteps);

private:
    Eigen::Vector2i grid_size;
    int time_steps;
    Eigen::Vector2d domainBoundaryMin;
    Eigen::Vector2d domainBoundaryMax;
    double tmin, tmax, t_interval;
    Eigen::VectorXd t_values;
    Eigen::VectorXd x_values;
    Eigen::VectorXd y_values;
};

#endif