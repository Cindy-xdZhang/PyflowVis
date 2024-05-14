
#pragma once
#ifndef VASTISTASVELOCITYGENERATOR_H
#define VASTISTASVELOCITYGENERATOR_H
#include "cereal/cereal.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <corecrt_math_defines.h>
#include <string>
#include <vector>
// using IndexType = int;

template <typename T>
struct vec2d {
    T x;
    T y;
    inline vec2d<T> operator+(const vec2d<T>& rhs) const
    {
        return { this->x + rhs.x, this->y + rhs.y };
    }
    inline vec2d<T>& operator+=(const vec2d<T>& rhs)
    {
        this->x += rhs.x;
        this->y += rhs.y;
        return *this;
    }
    inline vec2d<T> operator-(const vec2d<T>& rhs) const
    {
        return { this->x - rhs.x, this->y - rhs.y };
    }
    inline vec2d<T>& operator-=(const vec2d<T>& rhs)
    {
        this->x -= rhs.x;
        this->y -= rhs.y;
        return *this;
    }
    inline vec2d<T> operator*(T scalar) const
    {
        return { this->x * scalar, this->y * scalar };
    }
    inline T operator*(const vec2d<T>& rhs) const
    {
        return this->x * rhs.x + this->y * rhs.y;
    }
    inline T norm() const
    {
        const auto r = sqrtf(this->x * this->x + this->y * this->y);
        return r;
    }

    template <class Archive>
    void serialize(Archive& ar)
    {
        ar(
            CEREAL_NVP(x),
            CEREAL_NVP(y));
    }
};

template <typename T>
using Matrix22 = std::array<vec2d<T>, 2>; // fist row, second row.

using velocityFieldData = std::vector<std::vector<vec2d<float>>>;

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
    std::vector<std::vector<vec2d<float>>> field; // first y dimension then x dimension
    vec2d<float> spatialDomainMinBoundary;
    vec2d<float> spatialDomainMaxBoundary;
    vec2d<float> spatialGridInterval;
    vec2d<float> getVector(int x, int y) const
    {
        return field[y][x];
    }
    // paramter x,y are the physical positon in the vector field domian.
    vec2d<float> getVector(float x, float y) const
    {
        const float inverse_grid_interval_x = 1.0f / (float)spatialGridInterval.x;
        const float inverse_grid_interval_y = 1.0f / (float)spatialGridInterval.y;
        float floatIndicesX = (x - spatialDomainMinBoundary.x) * inverse_grid_interval_x;
        float floatIndicesY = (y - spatialDomainMinBoundary.y) * inverse_grid_interval_y;
        return bilinear_interpolate(field, floatIndicesX, floatIndicesY);
    }
};

struct UnSteadyVectorField2D {
    std::vector<std::vector<std::vector<vec2d<float>>>> field; // first t, then y dimension then x dimension
    vec2d<float> spatialDomainMinBoundary;
    vec2d<float> spatialDomainMaxBoundary;
    vec2d<float> spatialGridInterval;
    float tmin, tmax;
    int timeSteps = -1;
    vec2d<float> getVector(int x, int y, int t) const
    {
        return field[t][y][x];
    }
    vec2d<float> getVector(float x, float y, float t) const
    {
        const float inverse_grid_interval_t = (float)(timeSteps) / (tmax - tmin);

        const float inverse_grid_interval_x = 1.0f / (float)spatialGridInterval.x;
        const float inverse_grid_interval_y = 1.0f / (float)spatialGridInterval.y;
        float floatIndicesX = (x - spatialDomainMinBoundary.x) * inverse_grid_interval_x;
        float floatIndicesY = (y - spatialDomainMinBoundary.y) * inverse_grid_interval_y;
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

        vec2d<float> c000 = field[t0][y0][x0];
        vec2d<float> c100 = field[t0][y0][x1];
        vec2d<float> c010 = field[t0][y1][x0];
        vec2d<float> c110 = field[t0][y1][x1];
        vec2d<float> c001 = field[t1][y0][x0];
        vec2d<float> c101 = field[t1][y0][x1];
        vec2d<float> c011 = field[t1][y1][x0];
        vec2d<float> c111 = field[t1][y1][x1];

        vec2d<float> c00 = c000 * (1 - tx) + c100 * tx;
        vec2d<float> c10 = c010 * (1 - tx) + c110 * tx;
        vec2d<float> c01 = c001 * (1 - tx) + c101 * tx;
        vec2d<float> c11 = c011 * (1 - tx) + c111 * tx;

        vec2d<float> c0 = c00 * (1 - ty) + c10 * ty;
        vec2d<float> c1 = c01 * (1 - ty) + c11 * ty;

        return c0 * (1 - tt) + c1 * tt;
    }

    inline auto getVector(vec2d<float> pos, float t) const
    {
        return getVector(pos.x, pos.y, t);
    }
};

class VastistasVelocityGenerator {
public:
    VastistasVelocityGenerator(int Xdim, int Ydim, vec2d<float> minBondary, vec2d<float> maxBondary, float rc, float n);

    // generate a deformed steady velocity slice  following the paper: Vortex Boundary Identification using Convolutional Neural Network
    velocityFieldData generateSteady(float sx, float sy, float theta, int Si) const noexcept;

    // generate a deformed Unsteady velocity slice  following the paper: Robust Reference Frame Extraction from Unsteady 2D Vector
    // Fields with Convolutional Neural Networks
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
        return vec2d { x, y };
    }

private:
    int mgridDim_x;
    int mgridDim_y;
    float rc, n;
    float gridIntervalX, gridIntervalY;
    float ymin, ymax;
    float xmin, xmax;
    Matrix22<float> SiMatices_[3];
};

#endif