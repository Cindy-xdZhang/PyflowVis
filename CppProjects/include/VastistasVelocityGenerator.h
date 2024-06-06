
#pragma once
#ifndef VASTISTASVELOCITYGENERATOR_H
#define VASTISTASVELOCITYGENERATOR_H
#include "cereal/cereal.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <corecrt_math_defines.h>
#include <string>
#include <vector>

using velocityFieldData = std::vector<std::vector<Eigen::Vector2d>>;

template <int Component>
__forceinline int VecComponentAddressTrans(const int x, const int y, const int mgridDim_x)
{
    static_assert(Component == 1 || Component == 2);
    const int GridPointFlatternIdx = x + y * mgridDim_x;
    if constexpr (Component == 1) {
        return 2 * GridPointFlatternIdx;
    } else if constexpr (Component == 2) {
        return 2 * GridPointFlatternIdx + 1;
    }
};

template <typename T>
T bilinear_interpolate(const std::vector<std::vector<T>>& vector_field, double x, double y)
{
    x = std::clamp(x, double(0), double(vector_field[0].size() - 1));
    y = std::clamp(y, double(0), double(vector_field.size() - 1));

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);

    int x1 = std::min(x0 + 1, static_cast<int>(vector_field[0].size() - 1));
    int y1 = std::min(y0 + 1, static_cast<int>(vector_field.size() - 1));

    double tx = x - x0;
    double ty = y - y0;

    T v00 = vector_field[y0][x0];
    T v01 = vector_field[y0][x1];
    T v10 = vector_field[y1][x0];
    T v11 = vector_field[y1][x1];
    T a = v00 * (1 - tx) + v01 * tx;
    T b = v10 * (1 - tx) + v11 * tx;
    return a * (1 - ty) + b * ty;
}

struct SteadyVectorField2D {
    std::vector<std::vector<Eigen::Vector2d>> field; // first y dimension then x dimension
    Eigen::Vector2d spatialDomainMinBoundary;
    Eigen::Vector2d spatialDomainMaxBoundary;
    Eigen::Vector2d spatialGridInterval;
    Eigen::Vector2i XdimYdim;
    std::function<Eigen::Vector2d(const Eigen::Vector2d& pos, double)> analyticalFlowfunc_ = nullptr;

    Eigen::Vector2d getVector(int x, int y) const
    {
        return field[y][x];
    }
    Eigen::Vector2d getVector(const Eigen::Vector2d& pos) const
    {
        return getVector(pos.x(), pos.y());
    }
    // paramter x,y are the physical positon in the vector field domian.
    Eigen::Vector2d getVector(double x, double y) const
    {
        const double inverse_grid_interval_x = 1.0f / (double)spatialGridInterval(0);
        const double inverse_grid_interval_y = 1.0f / (double)spatialGridInterval(1);
        double doubleIndicesX = (x - spatialDomainMinBoundary(0)) * inverse_grid_interval_x;
        double doubleIndicesY = (y - spatialDomainMinBoundary(1)) * inverse_grid_interval_y;
        return bilinear_interpolate(field, doubleIndicesX, doubleIndicesY);
    }
    // SteadyVectorField2D  might have analytical expression, then when query value from out of boundary is return valid value.

    inline Eigen::Vector2d getVectorAnalytical(const Eigen::Vector2d& pos, double t_is_useless = 0.0) const
    {
        assert(this->analyticalFlowfunc_);
        return this->analyticalFlowfunc_(pos, t_is_useless);
    }
};

struct IUnsteadField2D {
public:
    virtual Eigen::Vector2d
    getVector(int x, int y, int t) const
        = 0;
    virtual Eigen::Vector2d getVector(double x, double y, double t) const = 0;

public:
    double tmin, tmax;
    int timeSteps = -1;
};

class UnSteadyVectorField2D : public IUnsteadField2D {
public:
    std::vector<std::vector<std::vector<Eigen::Vector2d>>> field; // first t, then y dimension then x dimension
    Eigen::Vector2d spatialDomainMinBoundary;
    Eigen::Vector2d spatialDomainMaxBoundary;
    Eigen::Vector2d spatialGridInterval;
    Eigen::Vector2i XdimYdim;
    // UnSteadyVectorField2D  might have analytical expression, then when query value from out of boundary is return valid value.
    std::function<Eigen::Vector2d(const Eigen::Vector2d& pos, double)> analyticalFlowfunc_ = nullptr;

    inline Eigen::Vector2d getVectorAnalytical(const Eigen::Vector2d& pos, double t) const
    {
        assert(this->analyticalFlowfunc_);
        return this->analyticalFlowfunc_(pos, t);
    }

    Eigen::Vector2d getVector(int x, int y, int t) const
    {
        return field[t][y][x];
    }
    Eigen::Vector2d getVector(double x, double y, double t) const
    {
        const double inverse_grid_interval_t = (double)(timeSteps) / (tmax - tmin);

        const double inverse_grid_interval_x = 1.0f / (double)spatialGridInterval(0);
        const double inverse_grid_interval_y = 1.0f / (double)spatialGridInterval(1);
        double floatIndicesX = (x - spatialDomainMinBoundary(0)) * inverse_grid_interval_x;
        double floatIndicesY = (y - spatialDomainMinBoundary(1)) * inverse_grid_interval_y;
        double floatIndicesT = (t - tmin) * inverse_grid_interval_t;

        // Trilinear interpolation
        auto floorX = static_cast<int>(std::floor(floatIndicesX));
        auto floorY = static_cast<int>(std::floor(floatIndicesY));
        auto floorT = static_cast<int>(std::floor(floatIndicesT));
        // if tx is bigger than one need to clamp
        double tx = floatIndicesX - floorX;
        double ty = floatIndicesY - floorY;
        double tt = floatIndicesT - floorT;

        int x0 = std::clamp(floorX, 0, static_cast<int>(field[0][0].size() - 1));
        int x1 = std::clamp(floorX + 1, 0, static_cast<int>(field[0][0].size() - 1));
        int y0 = std::clamp(floorY, 0, static_cast<int>(field[0].size() - 1));
        int y1 = std::clamp(floorY + 1, 0, static_cast<int>(field[0].size() - 1));
        int t0 = std::clamp(floorT, 0, static_cast<int>(field.size() - 1));
        int t1 = std::clamp(floorT + 1, 0, static_cast<int>(field.size() - 1));

        const auto& c000 = field[t0][y0][x0];
        const auto& c100 = field[t0][y0][x1];
        const auto& c010 = field[t0][y1][x0];
        const auto& c110 = field[t0][y1][x1];
        const auto& c001 = field[t1][y0][x0];
        const auto& c101 = field[t1][y0][x1];
        const auto& c011 = field[t1][y1][x0];
        const auto& c111 = field[t1][y1][x1];

        const auto c00 = c000 * (1 - tx) + c100 * tx;
        const auto c10 = c010 * (1 - tx) + c110 * tx;
        const auto c01 = c001 * (1 - tx) + c101 * tx;
        const auto c11 = c011 * (1 - tx) + c111 * tx;

        Eigen::Vector2d c0 = c00 * (1 - ty) + c10 * ty;
        Eigen::Vector2d c1 = c01 * (1 - ty) + c11 * ty;

        return c0 * (1 - tt) + c1 * tt;
    }

    inline Eigen::Vector2d getVector(Eigen::Vector2d pos, double t) const
    {
        return getVector(pos(0), pos(1), t);
    }

    inline const SteadyVectorField2D getVectorfieldSliceAtTime(int t) const
    {
        SteadyVectorField2D vecfield;
        vecfield.spatialDomainMinBoundary = spatialDomainMinBoundary;
        vecfield.spatialDomainMaxBoundary = spatialDomainMaxBoundary;
        vecfield.spatialGridInterval = spatialGridInterval;
        if (t >= 0 && t < timeSteps) {
            vecfield.field = field[static_cast<int>(t)];
        } else {
            assert(false);
        }

        return vecfield;
    }
    inline bool resampleFromAnalyticalExpression()
    {
        if (this->analyticalFlowfunc_) {

            // this->timeSteps = timeSteps;
            // this->spatialGridInterval = (this->spatialDomainMaxBoundary - this->spatialDomainMinBoundary).cwiseQuotient(Eigen::Vector2d(GridX - 1, GridY - 1));
            this->field.resize(this->timeSteps);
            const auto xdim = this->XdimYdim.x();
            const auto ydim = this->XdimYdim.y();
            const auto dt = (tmax - tmin) / (double)(timeSteps - 1);
            for (size_t i = 0; i < this->timeSteps; i++) {
                // time slice i
                this->field[i].resize(ydim);
                const double physical_t_this_slice = tmin + i * dt;
                for (size_t j = 0; j < ydim; j++) {
                    // y slice
                    this->field[i][j].resize(xdim);
                    for (size_t k = 0; k < xdim; k++) {
                        Eigen::Vector2d pos = { k * this->spatialGridInterval(0) + this->spatialDomainMinBoundary(0), j * this->spatialGridInterval(1) + this->spatialDomainMinBoundary(1) };
                        Eigen::Vector2d analyticalVec = this->analyticalFlowfunc_(pos, physical_t_this_slice);

                        this->field[i][j][k] = analyticalVec;
                    }
                }
            }

            return true;
        }
        return false;
    }
};
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
        this->func_ = func;
    }
    double dt;
    Eigen::Vector2d centerPos;
    std::function<Eigen::Vector3d(double)> func_ = nullptr;
    template <class Archive>
    void serialize(Archive& ar)
    {
        std::vector<Eigen::Vector3d> abcs_;
        abcs_.reserve(timeSteps);
        for (size_t i = 0; i < timeSteps; i++) {
            double time = this->tmin + i * this->dt;
            Eigen::Vector3d abc = this->func_(time);
            abcs_.push_back(abc);
        }
        ar(CEREAL_NVP(tmin), CEREAL_NVP(tmax), CEREAL_NVP(timeSteps), CEREAL_NVP(dt) /*, CEREAL_NVP(centerPos)*/);
        ar(CEREAL_NVP(abcs_));
    }

    // give a curve of a(t),b(t),c(t), killing vector(x,t) = a(t)+b(t)+ [0,-c(t);c(t),0]*(x-o), where o is the center of the plane.
    inline Eigen::Vector2d getKillingVector(const Eigen::Vector2d& queryPos, double t) const
    {
        if (this->func_)
            [[likely]] {
            Eigen::Vector3d abc = this->func_(t);

            Eigen::Vector2d uv = { abc(0), abc(1) };
            auto ra = queryPos - centerPos;

            const auto c = abc(2);
            //[0,-c(t);c(t),0]*(ra)
            Eigen::Vector2d c_componnet = { ra(1) * -c, ra(0) * c };
            return uv + c_componnet;
        }
        assert(false);
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

class VastistasVelocityGenerator {
public:
    VastistasVelocityGenerator(int Xdim, int Ydim, Eigen::Vector2d minBondary, Eigen::Vector2d maxBondary, double rc, double n);

    // generate a deformed steady velocity slice  following the paper: Vortex Boundary Identification using Convolutional Neural Network
    /*[[deprecated]] */ velocityFieldData generateSteady(double sx, double sy, double theta, int Si) const noexcept;

    SteadyVectorField2D generateSteadyField(double tx, double ty, double sx, double sy, double theta, int Si) const noexcept;
    // generate a deformed Unsteady velocity slice  following the paper: Robust Reference Frame Extraction from Unsteady 2D Vector
    // Fields with Convolutional Neural Networks

    // tx,ty is the critial point,   c = (cx, cy) describes the vortical
    // motion and d = (dx, dy)denotes the in - flow and out - flow
    velocityFieldData generateSteadyV2(double cx, double cy, double dx, double dy, double tx, double ty) const noexcept;

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

// Definition of the AnalyticalFlowCreator class
class AnalyticalFlowCreator {
public:
    AnalyticalFlowCreator(Eigen::Vector2i grid_size, int time_steps,
        Eigen::Vector2d domainBoundaryMin = Eigen::Vector2d(-2.0, -2.0),
        Eigen::Vector2d domainBoundaryMax = Eigen::Vector2d(2.0, 2.0), double tmin = 0.0f, double tmax = 2 * M_PI);

    // Method to create the flow field using a lambda function
    UnSteadyVectorField2D createFlowField(std::function<Eigen::Vector2d(Eigen::Vector2d, double)> lambda_func);

private:
    Eigen::Vector2i grid_size;
    int time_steps;
    Eigen::Vector2d domainBoundaryMin;
    Eigen::Vector2d domainBoundaryMax;
    double tmin, tmax, t_interval;
    std::vector<double> t_values;
    Eigen::VectorXd x_values;
    Eigen::VectorXd y_values;
    Eigen::Vector2d spatialGridInterval;
};

#endif