
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

using velocityFieldData = std::vector<std::vector<Eigen::Vector2f>>;

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
T bilinear_interpolate(const std::vector<std::vector<T>>& vector_field, float x, float y)
{
    x = std::clamp(x, float(0), float(vector_field[0].size() - 1));
    y = std::clamp(y, float(0), float(vector_field.size() - 1));

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);

    int x1 = std::min(x0 + 1, static_cast<int>(vector_field[0].size() - 1));
    int y1 = std::min(y0 + 1, static_cast<int>(vector_field.size() - 1));

    float tx = x - x0;
    float ty = y - y0;

    T v00 = vector_field[y0][x0];
    T v01 = vector_field[y0][x1];
    T v10 = vector_field[y1][x0];
    T v11 = vector_field[y1][x1];
    T a = v00 * (1 - tx) + v01 * tx;
    T b = v10 * (1 - tx) + v11 * tx;
    return a * (1 - ty) + b * ty;
}

struct SteadyVectorField2D {
    std::vector<std::vector<Eigen::Vector2f>> field; // first y dimension then x dimension
    Eigen::Vector2f spatialDomainMinBoundary;
    Eigen::Vector2f spatialDomainMaxBoundary;
    Eigen::Vector2f spatialGridInterval;
    Eigen::Vector2f getVector(int x, int y) const
    {
        return field[y][x];
    }
    // paramter x,y are the physical positon in the vector field domian.
    Eigen::Vector2f getVector(float x, float y) const
    {
        const float inverse_grid_interval_x = 1.0f / (float)spatialGridInterval(0);
        const float inverse_grid_interval_y = 1.0f / (float)spatialGridInterval(1);
        float floatIndicesX = (x - spatialDomainMinBoundary(0)) * inverse_grid_interval_x;
        float floatIndicesY = (y - spatialDomainMinBoundary(1)) * inverse_grid_interval_y;
        return bilinear_interpolate(field, floatIndicesX, floatIndicesY);
    }
};

struct UnSteadyVectorField2D {
    std::vector<std::vector<std::vector<Eigen::Vector2f>>> field; // first t, then y dimension then x dimension
    Eigen::Vector2f spatialDomainMinBoundary;
    Eigen::Vector2f spatialDomainMaxBoundary;
    Eigen::Vector2f spatialGridInterval;
    float tmin, tmax;
    int timeSteps = -1;
    Eigen::Vector2f getVector(int x, int y, int t) const
    {
        return field[t][y][x];
    }
    Eigen::Vector2f getVector(float x, float y, float t) const
    {
        const float inverse_grid_interval_t = (float)(timeSteps) / (tmax - tmin);

        const float inverse_grid_interval_x = 1.0f / (float)spatialGridInterval(0);
        const float inverse_grid_interval_y = 1.0f / (float)spatialGridInterval(1);
        float floatIndicesX = (x - spatialDomainMinBoundary(0)) * inverse_grid_interval_x;
        float floatIndicesY = (y - spatialDomainMinBoundary(1)) * inverse_grid_interval_y;
        float floatIndicesT = (t - tmin) * inverse_grid_interval_t;

        // Trilinear interpolation
        int x0 = static_cast<int>(std::floor(floatIndicesX));
        int x1 = std::min(x0 + 1, static_cast<int>(field[0][0].size() - 1));
        int y0 = static_cast<int>(std::floor(floatIndicesY));
        int y1 = std::min(y0 + 1, static_cast<int>(field[0].size() - 1));
        int t0 = static_cast<int>(std::floor(floatIndicesT));
        int t1 = std::min(t0 + 1, static_cast<int>(field.size() - 1));

        float tx = floatIndicesX - x0;
        float ty = floatIndicesY - y0;
        float tt = floatIndicesT - t0;

        Eigen::Vector2f c000 = field[t0][y0][x0];
        Eigen::Vector2f c100 = field[t0][y0][x1];
        Eigen::Vector2f c010 = field[t0][y1][x0];
        Eigen::Vector2f c110 = field[t0][y1][x1];
        Eigen::Vector2f c001 = field[t1][y0][x0];
        Eigen::Vector2f c101 = field[t1][y0][x1];
        Eigen::Vector2f c011 = field[t1][y1][x0];
        Eigen::Vector2f c111 = field[t1][y1][x1];

        Eigen::Vector2f c00 = c000 * (1 - tx) + c100 * tx;
        Eigen::Vector2f c10 = c010 * (1 - tx) + c110 * tx;
        Eigen::Vector2f c01 = c001 * (1 - tx) + c101 * tx;
        Eigen::Vector2f c11 = c011 * (1 - tx) + c111 * tx;

        Eigen::Vector2f c0 = c00 * (1 - ty) + c10 * ty;
        Eigen::Vector2f c1 = c01 * (1 - ty) + c11 * ty;

        return c0 * (1 - tt) + c1 * tt;
    }

    inline Eigen::Vector2f getVector(Eigen::Vector2f pos, float t) const
    {
        return getVector(pos(0), pos(1), t);
    }
};

class VastistasVelocityGenerator {
public:
    VastistasVelocityGenerator(int Xdim, int Ydim, Eigen::Vector2f minBondary, Eigen::Vector2f maxBondary, float rc, float n);

    // generate a deformed steady velocity slice  following the paper: Vortex Boundary Identification using Convolutional Neural Network
    velocityFieldData generateSteady(float sx, float sy, float theta, int Si) const noexcept;

    // generate a deformed Unsteady velocity slice  following the paper: Robust Reference Frame Extraction from Unsteady 2D Vector
    // Fields with Convolutional Neural Networks

    // tx,ty is the critial point,   c = (cx, cy) describes the vortical
    // motion and d = (dx, dy)denotes the in - flow and out - flow
    velocityFieldData generateSteadyV2(float cx, float cy, float dx, float dy, float tx, float ty) const noexcept;

    inline auto NormalizedVastistasV0(const float r) const noexcept
    {
        const auto v0_r = r / (2 * M_PI * (rc * rc) * std::pow(std::pow(r / rc, 2 * n) + 1, 1 / n));
        return v0_r / (r);
    }

    inline auto getPosition(int xid, int yid) const noexcept
    {
        float x = xmin + xid * gridIntervalX;
        float y = ymin + yid * gridIntervalY;
        return Eigen::Vector2f { x, y };
    }

private:
    int mgridDim_x;
    int mgridDim_y;
    float rc, n;
    float gridIntervalX, gridIntervalY;
    float ymin, ymax;
    float xmin, xmax;
    Eigen::Matrix2f SiMatices_[3];
};

class KillingAbcField {
public:
    KillingAbcField(int n, std::function<Eigen::Vector3f(float)> func, float tmin, float tmax)
        : tmin(tmin)
        , tmax(tmax)
        , func_(func)
        , n(n)
    {
        dt = (tmax - tmin) / (float)(n - 1);
        assert(n > 1);
        assert(dt > 0.0);
        centerPos = { 0.0, 0.0 };
    }
    ~KillingAbcField();
    float tmin;
    float tmax;
    float dt;
    int n;
    Eigen::Vector2f centerPos;
    std::function<Eigen::Vector3f(float)> func_ = nullptr;
    // give a curve of a(t),b(t),c(t), killing vector(x,t) = a(t)+b(t)+ [0,-c(t);c(t),0]*(x-o), where o is the center of the plane.
    inline Eigen::Vector2f getKillingVector(const Eigen::Vector2f& queryPos, float t)
    {
        if (this->func_)
            [[likely]] {
            Eigen::Vector3f abc = this->func_(t);

            Eigen::Vector2f uv = { abc(0), abc(1) };
            auto ra = queryPos - centerPos;

            const auto c = abc(2);
            //[0,-c(t);c(t),0]*(ra)
            Eigen::Vector2f c_componnet = { ra(1) * -c, ra(0) * c };
            return uv + c_componnet;
        }
        return {};
    }
};

#endif