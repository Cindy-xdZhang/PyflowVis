
#pragma once
#ifndef VASTISTASVELOCITYGENERATOR_H
#define VASTISTASVELOCITYGENERATOR_H
#include "cereal/cereal.hpp"
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

class VastistasVelocityGenerator {
public:
    VastistasVelocityGenerator(int Xdim, int Ydim, vec2d<float> minBondary, vec2d<float> maxBondary, float rc, float n);
    // generate a deformed velocity slice
    velocityFieldData generate(float sx, float sy, float theta, int Si) const noexcept;
    velocityFieldData generate() const noexcept;

    inline auto NormalizedVastistasV0(vec2d<float> xy) const noexcept
    {
        float xpos = xy.x;
        float ypos = xy.y;
        const auto r = sqrtf(xpos * xpos + ypos * ypos);
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