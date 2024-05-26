
#include "VastistasVelocityGenerator.h"
#include "cereal/archives/binary.hpp"
#include "cereal/archives/json.hpp"
#include "cereal/types/vector.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <random>
#include <vector>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <execution>
#include <filesystem>
#include <fstream>
#include <thread>

// define execute policy
#if !defined(ENABLE_CPP_PARALLELISM) || defined(_DEBUG)
auto policy = std::execution::seq;
#else
auto policy = std::execution::par_unseq;
#endif
std::mt19937 rng(static_cast<unsigned int>(std::time(0)));
using namespace std;

// Function to save the 2D vector as a PNG image
void saveAsPNG(const std::vector<std::vector<Eigen::Vector3f>>& data, const std::string& filename)
{
    int width = data[0].size();
    int height = data.size();

    // Create an array to hold the image data
    std::vector<unsigned char> image_data(width * height * 3); // 3 channels (RGB)

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            auto value = data[y][x];

            image_data[3 * (y * width + x) + 0] = static_cast<unsigned char>(value(0) * 255.0f); // Convert to 0-255
            image_data[3 * (y * width + x) + 1] = static_cast<unsigned char>(value(1) * 255.0f); // Convert to 0-255
            image_data[3 * (y * width + x) + 2] = static_cast<unsigned char>(value(2) * 255.0f); // Convert to 0-255pixel_value; // B
        }
    }

    // Save the image
    stbi_write_png(filename.c_str(), width, height, 3, image_data.data(), width * 3);
}

// Function to generate a 2D vector of random noise
std::vector<std::vector<float>> randomNoiseTexture(int width, int height)
{
    std::vector<std::vector<float>> texture(height, std::vector<float>(width));
    std::random_device rd; // Seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            texture[y][x] = static_cast<float>(dis(gen));
        }
    }

    return texture;
}
// Function to flatten a 2D vector to a 1D vector
std::vector<float> flatten2D(const std::vector<std::vector<Eigen::Vector2f>>& x2D)
{
    const size_t ydim = x2D.size();
    assert(ydim > 0);
    const size_t xdim = x2D[0].size();
    std::vector<float> result;
    result.resize(xdim * ydim * 2);
    for (size_t i = 0; i < ydim; i++)
        for (size_t j = 0; j < xdim; j++) {
            result[2 * (i * xdim + j)] = x2D[i][j](0);
            result[2 * (i * xdim + j) + 1] = x2D[i][j](1);
        }

    return result;
}

std::pair<float, float> computeMinMax(const std::vector<float>& values)
{
    if (values.empty()) {
        throw std::invalid_argument("The input vector is empty.");
    }

    float minVal = std::numeric_limits<float>::infinity();
    float maxVal = -std::numeric_limits<float>::infinity();

    for (float val : values) {
        if (val < minVal) {
            minVal = val;
        }
        if (val > maxVal) {
            maxVal = val;
        }
    }

    return { minVal, maxVal };
}
std::pair<float, float> computeMinMax(const std::vector<std::vector<float>>& values)
{
    if (values.empty() || values[0].empty()) {
        throw std::invalid_argument("The input vector is empty.");
    }

    float minVal = std::numeric_limits<float>::infinity();
    float maxVal = -std::numeric_limits<float>::infinity();

    for (const auto& row : values) {
        for (float val : row) {
            if (val < minVal) {
                minVal = val;
            }
            if (val > maxVal) {
                maxVal = val;
            }
        }
    }

    return { minVal, maxVal };
}
// The result image size is the same as the input texture.
std::vector<std::vector<Eigen::Vector3f>> LICAlgorithm(
    const std::vector<std::vector<float>>& texture,
    const SteadyVectorField2D& vecfield,
    const int licImageSizeX,
    const int licImageSizeY,
    float stepSize,
    int MaxIntegrationSteps,
    bool curlColorBlend = true)
{
    const int YTexdim = texture.size();
    const int XTexdim = texture[0].size();
    const int Ydim = vecfield.field.size();
    const int Xdim = vecfield.field[0].size();
    assert(YTexdim == Ydim && Xdim == XTexdim);
    std::vector<std::vector<Eigen::Vector3f>> output_texture(licImageSizeY, std::vector<Eigen::Vector3f>(licImageSizeX, { 0.0f, 0.0f, 0.0f }));

    const auto& vecfieldData = vecfield.field;
    const Eigen::Vector2f domainRange = vecfield.spatialDomainMaxBoundary - vecfield.spatialDomainMinBoundary;
    const float inverse_grid_interval_x = 1.0f / (float)vecfield.spatialGridInterval(0);
    const float inverse_grid_interval_y = 1.0f / (float)vecfield.spatialGridInterval(1);

    std::vector<std::vector<float>> curl(Ydim, std::vector<float>(Xdim, 0.0f));
    float minCurl;
    float maxCurl;
    if (curlColorBlend) {
        // Calculate curl (vorticity) of the vector field
        for (int y = 1; y < Ydim - 1; ++y) {
            for (int x = 1; x < Xdim - 1; ++x) {
                auto dv_dx = (vecfieldData[y][x + 1] - vecfieldData[y][x - 1]) * 0.5f * inverse_grid_interval_x;
                auto du_dy = (vecfieldData[y + 1][x] - vecfieldData[y - 1][x]) * 0.5f * inverse_grid_interval_y;
                curl[y][x] = dv_dx(1) - du_dy(0);
            }
        }

        // Normalize curl values for color mapping
        auto minMaxCurl = computeMinMax(curl);
        minCurl = minMaxCurl.first;
        maxCurl = minMaxCurl.second;
        if (!(maxCurl > minCurl && maxCurl > 0 && minCurl < 0)) {
            curlColorBlend = false;
        }
    }

    for (int y = 0; y < licImageSizeY; ++y) {
        for (int x = 0; x < licImageSizeX; ++x) {
            float accum_value = 0.0f;
            int accum_count = 0;

            // map position from texture image grid coordinate to vector field
            float ratio_x = (float)((float)x / (float)licImageSizeX);
            float ratio_y = (float)((float)y / (float)licImageSizeY);

            // Trace forward
            // physicalPositionInVectorfield
            Eigen::Vector2f pos = { ratio_x * domainRange(0) + vecfield.spatialDomainMinBoundary(0),
                ratio_y * domainRange(1) + vecfield.spatialDomainMinBoundary(1) };

            for (int i = 0; i < MaxIntegrationSteps; ++i) {
                float floatIndicesX = (pos(0) - vecfield.spatialDomainMinBoundary(0)) * inverse_grid_interval_x;
                float floatIndicesY = (pos(1) - vecfield.spatialDomainMinBoundary(1)) * inverse_grid_interval_y;

                if (!(0 <= floatIndicesX && floatIndicesX < Xdim && 0 <= floatIndicesY && floatIndicesY < Ydim)) {
                    break; // Stop if we move outside the texture bounds
                }
                accum_value += bilinear_interpolate(texture, floatIndicesX, floatIndicesY);
                accum_count += 1;
                Eigen::Vector2f vec = bilinear_interpolate(vecfieldData, floatIndicesX, floatIndicesY);
                pos += vec * stepSize;
            }

            // Trace backward
            pos = { ratio_x * domainRange(0) + vecfield.spatialDomainMinBoundary(0),
                ratio_y * domainRange(1) + vecfield.spatialDomainMinBoundary(1) };

            for (int i = 0; i < MaxIntegrationSteps; ++i) {
                float floatIndicesX = (pos(0) - vecfield.spatialDomainMinBoundary(0)) * inverse_grid_interval_x;
                float floatIndicesY = (pos(1) - vecfield.spatialDomainMinBoundary(1)) * inverse_grid_interval_y;
                if (!(0 <= floatIndicesX && floatIndicesX < Xdim && 0 <= floatIndicesY && floatIndicesY < Ydim)) {
                    break; // Stop if we move outside the texture bounds
                }
                accum_value += bilinear_interpolate(texture, floatIndicesX, floatIndicesY);
                accum_count += 1;
                Eigen::Vector2f vec = bilinear_interpolate(vecfieldData, floatIndicesX, floatIndicesY);
                pos -= vec * stepSize;
            }

            // Compute the average value along the path
            if (accum_count > 0) {
                auto licValue = accum_value / accum_count;
                if (curlColorBlend) {
                    // Compute the normalized curl value
                    float curlValue = curl[static_cast<int>(ratio_y * Ydim)][static_cast<int>(ratio_x * Xdim)];
                    float normalizedCurl = (curlValue - minCurl) / (maxCurl - minCurl);
                    Eigen::Vector3f curlColor = { normalizedCurl, 0.1f, 1.0f - normalizedCurl };

                    float whiteish = licValue;
                    whiteish = std::min(std::max(0.0f, (whiteish - 0.4f) * (1.5f / 0.4f)), 1.0f);
                    // output_texture[y][x] = mix(curlColor, Eigen::Vector3f(licValue, licValue, licValue), 1.0 - whiteish);
                    output_texture[y][x] = Eigen::Vector3f(licValue, licValue, licValue) * (1.0f - whiteish) + curlColor * whiteish;

                } else {
                    output_texture[y][x] = { licValue, licValue, licValue };
                }
            }
        }
    }

    return output_texture;
}

std::vector<std::vector<std::vector<Eigen::Vector3f>>>
LICAlgorithm_UnsteadyField(const std::vector<std::vector<float>>& texture,
    const UnSteadyVectorField2D& vecfield,
    const int licImageSizeX,
    const int licImageSizeY,
    float stepSize,
    int MaxIntegrationSteps)
{
    std::vector<int> timeIndex;
    timeIndex.resize(vecfield.timeSteps);
    std::iota(timeIndex.begin(), timeIndex.end(), 0);
    std::vector<std::vector<std::vector<Eigen::Vector3f>>> resultData;
    resultData.resize(vecfield.timeSteps);

    std::transform(std::execution::par_unseq, timeIndex.begin(), timeIndex.end(), resultData.begin(), [&](int time) {
        std::cout << "parallel lic rendering.. timeIndex size: " << time << std::endl;
        auto slice = vecfield.getVectorfieldSliceAtTime(time);
        auto licPic = LICAlgorithm(texture, slice, licImageSizeX, licImageSizeY, stepSize, MaxIntegrationSteps);
        return std::move(licPic);
    });
    return resultData;
}

std::vector<std::pair<float, int>> generateNParamters(int n)
{
    std::vector<std::pair<float, int>> parameters = {
        { 1.0, 2 },
        /*   { 1.0, 2 },
           { 1.0, 3 },
           { 1.0, 10 },
           { 2.0, 1 },
           { 2.0, 2 },
           { 2.0, 2 },
           { 2.0, 3 },
           { 2.0, 10 },*/
    };

    std::uniform_real_distribution<float> dist_T(0.01, 2.0);
    std::uniform_int_distribution<int> dist_int(1, 6);

    // for (int i = 0; i < n; ++i) {
    //     float first = dist_T(rng);
    //     int second = dist_int(rng);
    //     parameters.emplace_back(first, second);
    // }

    return parameters;
}

// todo: transform input field with respect to observer motion(v*=pullback(v-u))
template <typename T>
void referenceFrameTransformation(const SteadyVectorField2D& input_field, const int timesteps)
{

    auto integratePathlineOneStep = [](T x, T y, T t, T dt, T& x_new, T& y_new) {
        // RK4 integration step
        Eigen::Vector2f odeStepStartPoint;
        odeStepStartPoint << x, y;

        const T h = dt;

        // coefficients
        constexpr T a21 = 0.5;
        constexpr T a31 = 0.;
        constexpr T a32 = 0.5;
        constexpr T a41 = 0.;
        constexpr T a42 = 0.;
        constexpr T a43 = 1.;

        constexpr T c2 = 0.5;
        constexpr T c3 = 0.5;
        constexpr T c4 = 1.;

        constexpr T b1 = 1. / 6.;
        constexpr T b2 = 1. / 3.;
        constexpr T b3 = b2;
        constexpr T b4 = b1;

        // 4 stages of 2 equations (i.e., 2 dimensions of the manifold and the tangent vector space)

        // stage 1
        Eigen::Vector2f k1 = fieldFunction_u(odeStepStartPoint, t);

        // stage 2
        Eigen::Vector2f stagePoint = odeStepStartPoint + k1 * a21 * h;
        Eigen::Vector2f k2 = fieldFunction_u(stagePoint, t + c2 * h);

        // stage 3
        stagePoint = odeStepStartPoint + (a31 * k1 + a32 * k2) * h;
        Eigen::Vector2f k3 = fieldFunction_u(stagePoint, t + c3 * h);

        // stage 4
        stagePoint = odeStepStartPoint + (a41 * k1 + a42 * k2 + a43 * k3) * h;
        Eigen::Vector2f k4 = fieldFunction_u(stagePoint, t + c4 * h);

        Eigen::Vector2f result_p = odeStepStartPoint + h * (k1 * b1 + k2 * b2 + k3 * b3 + k4 * b4);

        x_new = result_p(0);
        y_new = result_p(1);
    };

    //// integrationStep integrates starting from x, y, t, to target_t arriving at point x_new, y_new.
    //// the push_forward the matrix that does the transformation that a vector would undergo when going one step dt starting at t
    //// the push_forward is computed by draging a frame (2 vectors) along the curve.
    // auto integrationStep = [&integratePathlineStep, dx, dy](T x, T y, T t, T dt, T& x_new, T& y_new, Matrix22& push_forward) {
    //     Eigen::Vector2f p_north;
    //     Eigen::Vector2f p_south;
    //     Eigen::Vector2f p_east;
    //     Eigen::Vector2f p_west;

    //    integratePathlineStep(x, y, t, dt, x_new, y_new);
    //    integratePathlineStep(x, y + 0.5 * dy, t, dt, p_north(0), p_north(1));
    //    integratePathlineStep(x, y - 0.5 * dy, t, dt, p_south(0), p_south(1));
    //    integratePathlineStep(x + 0.5 * dx, y, t, dt, p_east(0), p_east(1));
    //    integratePathlineStep(x - 0.5 * dx, y, t, dt, p_west(0), p_west(1));

    //    push_forward.col(0) = (p_east - p_west) * (1. / dx); // TODO: check the divide by dx
    //    push_forward.col(1) = (p_north - p_south) * (1. / dy); // TODO: check the divide by dy
    //};

    //// integrate starting from x, y, t, to target_t arriving at point x_new, y_new.
    //// push_forward is the matrix that does the transformation that a vector would undergo when going from t to target_t.
    //// internally it uses the composition of multiple integration steps.
    //// if the estimation of the push_forward of one step becomes unreliable a shorter stepsize can be chosen.
    // auto integrateFrame = [&integrationStep](T x, T y, T t, T target_t, int numberOfSteps, T& x_new, T& y_new, Matrix22& push_forward) {
    //     T stepsize = (target_t - t) / numberOfSteps;

    //    push_forward(0, 0) = 1;
    //    push_forward(1, 1) = 1;
    //    push_forward(0, 1) = 0;
    //    push_forward(1, 0) = 0;

    //    T x_running = x;
    //    T y_running = y;

    //    for (int i = 0; i < numberOfSteps; i++) {
    //        T t_running = t + i * stepsize;
    //        Matrix22 push_forward_step;
    //        integrationStep(x_running, y_running, t_running, stepsize, x_running, y_running, push_forward_step);
    //        push_forward *= push_forward_step; // TODO: check there is no times dt
    //    }

    //    x_new = x_running;
    //    y_new = y_running;
    //};
}

// function killingABCtransformation transform a SteadyVectorField2D& inputField to an unsteady field by observing it with respect to KillingAbcField& observerfield
// @note: the result field has same spatial information(grid size, domain size) as the steady inputField and has same time (domain size, timesteps)  as the killing observerfield.
// inputField could be steady or unsteady field
// if  inputField  is unsteady field, the observerfield should have the same time domain as the inputField.
template <typename T = float, class inputFieldType>
UnSteadyVectorField2D killingABCtransformation(const KillingAbcField& observerfield, const Eigen::Vector2f StartPosition, const inputFieldType& inputField)
{
    const auto tmin = observerfield.tmin;
    const auto tmax = observerfield.tmax;
    const float dt = observerfield.dt;
    const int timestep = observerfield.timeSteps;

    auto integratePathlineOneStep_RK4 = [&observerfield](T x, T y, T t, T dt) -> Eigen::Vector2f {
        // RK4 integration step
        Eigen::Vector2f odeStepStartPoint = { x, y };

        const T h = dt;

        // coefficients
        constexpr T a21 = 0.5;
        constexpr T a31 = 0.;
        constexpr T a32 = 0.5;
        constexpr T a41 = 0.;
        constexpr T a42 = 0.;
        constexpr T a43 = 1.;

        constexpr T c2 = 0.5;
        constexpr T c3 = 0.5;
        constexpr T c4 = 1.;

        constexpr T b1 = 1. / 6.;
        constexpr T b2 = 1. / 3.;
        constexpr T b3 = b2;
        constexpr T b4 = b1;

        // 4 stages of 2 equations (i.e., 2 dimensions of the manifold and the tangent vector space)

        // stage 1
        Eigen::Vector2f k1 = observerfield.getKillingVector(odeStepStartPoint, t);

        // stage 2
        Eigen::Vector2f stagePoint = odeStepStartPoint + k1 * a21 * h;
        Eigen::Vector2f k2 = observerfield.getKillingVector(stagePoint, t + c2 * h);

        // stage 3
        stagePoint = odeStepStartPoint + (a31 * k1 + a32 * k2) * h;
        Eigen::Vector2f k3 = observerfield.getKillingVector(stagePoint, t + c3 * h);

        // stage 4
        stagePoint = odeStepStartPoint + (a41 * k1 + a42 * k2 + a43 * k3) * h;
        Eigen::Vector2f k4 = observerfield.getKillingVector(stagePoint, t + c4 * h);

        Eigen::Vector2f result_p = odeStepStartPoint + h * (k1 * b1 + k2 * b2 + k3 * b3 + k4 * b4);

        return result_p;
    };

    // integration of worldline
    auto runIntegration = [&](const Eigen::Vector2f& startPoint, /*const double observationTime,*/ const double targetIntegrationTime,
                              std::vector<Eigen::Vector2f>& pathVelocitys, std::vector<Eigen::Vector3f>& pathPositions) -> bool {
        const float startTime = 0.0;
        const int maxIterationCount = 2000;
        const float spaceConversionRatio = 1.0;

        bool integrationOutOfDomainBounds = false;
        bool outOfIntegrationTimeBounds = false;
        int iterationCount = 0;
        float integrationTimeStepSize = dt;

        if (targetIntegrationTime < startTime) {
            // we integrate back in time
            integrationTimeStepSize *= -1.0;
        }
        const auto minDomainBounds = inputField.spatialDomainMinBoundary;
        const auto maxDomainBounds = inputField.spatialDomainMaxBoundary;
        // This function checks if the current point is in the domain
        std::function<bool(const Eigen::Vector2f& point)> checkIfOutOfDomain = [&minDomainBounds, &maxDomainBounds](const Eigen::Vector2f& point) {
            if (point(0) <= minDomainBounds(0))
                return true;
            if (point(0) >= maxDomainBounds(0))
                return true;
            if (point(1) <= minDomainBounds(1))
                return true;
            if (point(1) >= maxDomainBounds(1))
                return true;
            return false;
        };

        Eigen::Vector2f currentPoint = startPoint;

        integrationOutOfDomainBounds = checkIfOutOfDomain(currentPoint);

        // do integration
        auto currentTime = startTime;

        // push init_velocity  &start point
        Eigen::Vector3f pointAndTime = { currentPoint(0), currentPoint(1), currentTime };
        pathPositions.emplace_back(pointAndTime);
        auto init_velocity = observerfield.getKillingVector(currentPoint, currentTime);
        pathVelocitys.emplace_back(init_velocity);

        // integrate until either
        // - we reached the max iteration count
        // - we reached the upper limit of the time domain
        // - we ran out of spatial domain
        while ((!integrationOutOfDomainBounds) && (!outOfIntegrationTimeBounds) && (pathPositions.size() < maxIterationCount)) {

            // advance to a new point in the chart
            Eigen::Vector2f newPoint = integratePathlineOneStep_RK4(currentPoint(0), currentPoint(1), currentTime, dt);
            integrationOutOfDomainBounds = checkIfOutOfDomain(newPoint);
            if (!integrationOutOfDomainBounds) {
                auto newTime = currentTime + integrationTimeStepSize;
                // check if currentTime is out of the time domain -> we are done
                if ((targetIntegrationTime > startTime) && (newTime >= targetIntegrationTime)) {
                    outOfIntegrationTimeBounds = true;
                } else if ((targetIntegrationTime < startTime) && (newTime <= targetIntegrationTime)) {
                    outOfIntegrationTimeBounds = true;
                } else {
                    // add  current point to the result list and set currentPoint to newPoint -> everything fine -> continue with the while loop
                    Eigen::Vector3f new_pointAndTime = { newPoint(0), newPoint(1), newTime };
                    pathPositions.emplace_back(new_pointAndTime);
                    auto velocity = observerfield.getKillingVector(newPoint, newTime);
                    pathVelocitys.emplace_back(velocity);
                    currentPoint = newPoint;
                    currentTime = newTime;
                    iterationCount++;
                }
            }
        }
        bool suc = pathPositions.size() > 1 && pathVelocitys.size() == pathPositions.size();
        return suc;
    };

    std::vector<Eigen::Vector2f> pathVelocitys;
    std::vector<Eigen::Vector3f> pathPositions;
    bool suc = runIntegration(StartPosition, tmax, pathVelocitys, pathPositions);
    assert(suc);

    int validPathSize = pathPositions.size();
    // if (validPathSize == timestep) {
    //     // this happen when the observer is moving with valid domian
    //} else {
    //    // this happen when the observerfield has large velocity, observer move out of the domain before the end of the time domain
    //    // fill Steady points When OutOfBoundary to make pathline has the same step  as time steps?
    //    // extendPath =true;
    //}

    auto NoramlizeSpinTensor = [](Eigen::Matrix3f& input) {
        Eigen::Vector3f unitAngular;
        unitAngular << input(2, 1), input(0, 2), input(1, 0);
        unitAngular.normalize();
        input << 0, -unitAngular(2), unitAngular(1),
            unitAngular(2), 0, -unitAngular(0),
            -unitAngular(1), unitAngular(0), 0;
        return;
    };
    std::vector<Eigen::Matrix3f> rotationMatrices;
    rotationMatrices.resize(timestep);
    rotationMatrices[0] = Eigen::Matrix3f::Identity();

    std::vector<Eigen::Matrix4f> transformationMatrices;
    transformationMatrices.resize(timestep);
    transformationMatrices[0] = Eigen::Matrix4f::Identity();
    const auto observerStartPoint = pathPositions.at(0);

    // force start point saved in worldline.
    // assert(observerStartPoint == StartPosition);

    for (size_t i = 1; i < validPathSize; i++) {
        const float t = observerfield.tmin + i * observerfield.dt;
        const Eigen::Vector3f abc = observerfield.func_(t);
        const auto c_ = std::abs(abc(2));
        const auto theta = dt * c_;
        Eigen::Matrix3f Spintensor;
        Spintensor(0, 0) = 0.0;
        Spintensor(1, 0) = c_;
        Spintensor(2, 0) = 0;

        Spintensor(0, 1) = -c_;
        Spintensor(1, 1) = 0.0;
        Spintensor(2, 1) = 0.0;

        Spintensor(0, 2) = 0;
        Spintensor(1, 2) = 0.0;
        Spintensor(2, 2) = 0.0;
        NoramlizeSpinTensor(Spintensor);
        Eigen::Matrix3f Spi_2;
        Spi_2 = Spintensor * Spintensor;
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);
        Eigen::Matrix3f I = Eigen::Matrix<float, 3, 3>::Identity();
        Eigen::Matrix3f stepInstanenousRotation = I + sinTheta * Spintensor + (1 - cosTheta) * Spi_2;
        // get rotate c of t
        rotationMatrices[i] = stepInstanenousRotation * rotationMatrices[i - 1];
        const auto& stepRotation = rotationMatrices[i];
        // compute observer's relative transformation as M=T(position)*integral of [ R(matrix_exponential(spinTensor))] * T(-startPoint)
        // then observer bring transformation  M-1=T(startPoint)*integral of [ R(matrix_exponential(spinTensor))]^T * T(-position)

        auto tP1 = pathPositions.at(i);

        // eigen << fill data in row major regardless of storage order

        Eigen::Matrix4f inv_translationPostR;
        inv_translationPostR << 1, 0, 0, -tP1(0), // first row
            0, 1, 0, -tP1(1), // second row
            0, 0, 1, 0,
            0.0, 0, 0, 1;

        Eigen::Matrix4f inv_translationPreR;
        inv_translationPreR << 1, 0, 0, observerStartPoint(0), // first row
            0, 1, 0, observerStartPoint(1), // second row
            0, 0, 1, 0,
            0.0, 0, 0, 1;
        Eigen::Matrix4f inv_rotation = Eigen::Matrix4f::Zero();
        inv_rotation(0, 0) = stepRotation(0, 0);
        inv_rotation(0, 1) = stepRotation(0, 1);
        inv_rotation(0, 2) = stepRotation(0, 2);
        inv_rotation(1, 0) = stepRotation(1, 0);
        inv_rotation(1, 1) = stepRotation(1, 1);
        inv_rotation(1, 2) = stepRotation(1, 2);
        inv_rotation(2, 0) = stepRotation(2, 0);
        inv_rotation(2, 1) = stepRotation(2, 1);
        inv_rotation(2, 2) = stepRotation(2, 2);
        inv_rotation(3, 3) = 1.0;

        // Eigen::Matrix4f  Transformation = translationPostR * (rotationMatrices[i] * translationPreR);
        //  combine translation and rotation into final transformation
        Eigen::Matrix4f InvserseTransformation = inv_translationPreR * (inv_rotation.transpose() * inv_translationPostR);
        transformationMatrices[i] = InvserseTransformation;
    }
    const auto lastTransformation = transformationMatrices[validPathSize - 1];
    for (size_t i = validPathSize; i < timestep; i++) {
        transformationMatrices[i] = lastTransformation;
        pathVelocitys.emplace_back(0.0f, 0.0f);
    }

    UnSteadyVectorField2D resultField;
    resultField.spatialDomainMaxBoundary = inputField.spatialDomainMaxBoundary;
    resultField.spatialDomainMinBoundary = inputField.spatialDomainMinBoundary;
    resultField.spatialGridInterval = inputField.spatialGridInterval;
    resultField.tmin = tmin;
    resultField.tmax = tmax;
    resultField.timeSteps = timestep;
    resultField.field.resize(timestep);

    if constexpr (std::is_same_v<inputFieldType, UnSteadyVectorField2D>) {
        assert(inputField.field.size() == timestep);
        for (size_t i = 0; i < timestep; i++) {
            // time slice i
            resultField.field[i].resize(inputField.field[i].size());
            const Eigen::Vector2f u = pathVelocitys[i];
            for (size_t j = 0; j < inputField.field[i].size(); j++) {
                // y slice
                resultField.field[i][j].resize(inputField.field[i][j].size());
                for (size_t k = 0; k < inputField.field[i][j].size(); k++) {
                    Eigen::Vector2f v = inputField.field[i][j][k];

                    Eigen::Vector4f v4;
                    v4 << v(0) - u(0), v(1) - u(1), 0, 0;
                    Eigen::Vector4f v4_new = transformationMatrices[i] * v4;
                    resultField.field[i][j][k] = { v4_new(0), v4_new(1) };
                }
            }
        }
    } else {

        for (size_t i = 0; i < timestep; i++) {
            // time slice i
            resultField.field[i].resize(inputField.field.size());
            const Eigen::Vector2f u = pathVelocitys[i];
            for (size_t j = 0; j < inputField.field.size(); j++) {
                // y slice
                resultField.field[i][j].resize(inputField.field[j].size());
                for (size_t k = 0; k < inputField.field[j].size(); k++) {
                    Eigen::Vector2f v = inputField.field[j][k];

                    Eigen::Vector4f v4;
                    v4 << v(0) - u(0), v(1) - u(1), 0, 0;
                    Eigen::Vector4f v4_new = transformationMatrices[i] * v4;
                    resultField.field[i][j][k] = { v4_new(0), v4_new(1) };
                }
            }
        }
    }

    return resultField;
}

void generateUnsteadyField()
{
    const int Xdim = 64, Ydim = 64;
    const int LicImageSize = 64;
    Eigen::Vector2f domainMinBoundary = { -2.0, -2.0 };
    Eigen::Vector2f domainMaxBoundary = { 2.0, 2.0 };
    const float stepSize = 0.01;
    const int maxIteratioOneDirection = 128;
    int numVelocityFields = 1; // num of fields per n, rc parameter setting

    Eigen::Vector2f gridInterval = {
        (domainMaxBoundary(0) - domainMinBoundary(0)) / (Xdim - 1),
        (domainMaxBoundary(1) - domainMinBoundary(1)) / (Ydim - 1)
    };

    const std::vector<std::pair<float, int>> paramters = generateNParamters(1);
    const auto licNoisetexture = randomNoiseTexture(Xdim, Ydim);

    uniform_real_distribution<float> genTheta(-1.0, 1.0);
    uniform_real_distribution<float> genSx(-2.0, 2.0);

    string root_folder = "../data/unsteady/" + to_string(Xdim) + "_" + to_string(Xdim) + "/";
    if (!filesystem::exists(root_folder)) {
        filesystem::create_directories(root_folder);
    }

    for_each(policy, paramters.begin(), paramters.cend(), [&](const std::pair<float, int>& params) {
        const float rc = params.first;
        const float n = params.second;
        const string task_name = "velocity_field_rc_" + to_string(rc) + "n_" + to_string(n);
        printf("generate data for %s\n", task_name.c_str());

        VastistasVelocityGenerator generator(Xdim, Ydim, domainMinBoundary, domainMaxBoundary, rc, n);

        for (size_t sample = 0; sample < numVelocityFields; sample++) {

            const auto theta = genTheta(rng);
            const auto tx = 1 - 0.45 * genSx(rng);
            const auto ty = 1 - 0.45 * genSx(rng);

            const auto cx = 1;
            const auto cy = 1;
            const auto dx = 0;
            const auto dy = 0;
            // generate steady field data
            auto resData = generator.generateSteadyV2(cx, cy, dx, dy, tx, ty);
            const string tag_name = "velocity_field_" + std::to_string(sample) + "rc_" + to_string(rc) + "n_" + to_string(n) + "_sample_" + to_string(sample);

            string metaFilename = root_folder + tag_name + "meta.json";
            string velocityFilename = root_folder + tag_name + "velocity.bin";
            string licFilename = root_folder + tag_name + "lic.png";

            // save meta info:
            std::ofstream out(metaFilename);
            if (!out.good()) {
                printf("couldn't open file: %s", metaFilename.c_str());
                return;
            }
            std::ofstream outBin(velocityFilename, std::ios::binary);
            if (!outBin.good()) {
                printf("couldn't open file: %s", velocityFilename.c_str());
                return;
            }

            cereal::BinaryOutputArchive archive_Binary(outBin);
            const std::vector<float> rawData = flatten2D(resData);
            auto [minV, maxV] = computeMinMax(rawData);

            {

                cereal::JSONOutputArchive archive_o(out);
                archive_o(CEREAL_NVP(Xdim));
                archive_o(CEREAL_NVP(Ydim));
                std::vector<float> domainMininumBoundary = { domainMinBoundary(0), domainMinBoundary(1) };
                std::vector<float> domainMaxinumBoundary = { domainMaxBoundary(0), domainMaxBoundary(1) };
                archive_o(CEREAL_NVP(domainMininumBoundary));
                archive_o(CEREAL_NVP(domainMaxinumBoundary));

                archive_o(CEREAL_NVP(n));
                archive_o(CEREAL_NVP(rc));

                archive_o(CEREAL_NVP(theta));
                archive_o(CEREAL_NVP(tx));
                archive_o(CEREAL_NVP(ty));
                archive_o(CEREAL_NVP(cx));
                archive_o(CEREAL_NVP(cy));
                archive_o(CEREAL_NVP(dx));
                archive_o(CEREAL_NVP(dy));
                archive_o(CEREAL_NVP(minV));
                archive_o(CEREAL_NVP(maxV));
            }

            // do not manually close file before creal deconstructor, as cereal will preprend a ]/} to finish json class/array
            out.close();

            // write raw data

            // ar(make_size_tag(static_cast<size_type=uint64>(vector.size()))); // number of elements
            // ar(binary_data(vector.data(), vector.size() * sizeof(T)));
            //  when using other library to read, need to discard the first uint64 (8bytes.)
            archive_Binary(rawData);

            outBin.close();

            SteadyVectorField2D outField {
                resData,
                domainMinBoundary,
                domainMaxBoundary,
                gridInterval
            };
            auto outputTexture = LICAlgorithm(licNoisetexture, outField, LicImageSize, LicImageSize, stepSize, maxIteratioOneDirection);

            saveAsPNG(outputTexture, licFilename);
        }
    });
}

void generateSteadyField()
{
    using namespace std;
    const int Xdim = 64, Ydim = 64;
    const int LicImageSize = 512;
    Eigen::Vector2f domainMinBoundary = { -2.0, -2.0 };
    Eigen::Vector2f domainMaxBoundary = { 2.0, 2.0 };
    const float stepSize = 0.01;
    const int maxIteratioOneDirection = 256;
    int numVelocityFields = 1; // num of fields per n, rc parameter setting

    Eigen::Vector2f gridInterval = {
        (domainMaxBoundary(0) - domainMinBoundary(0)) / (Xdim - 1),
        (domainMaxBoundary(1) - domainMinBoundary(1)) / (Ydim - 1)
    };

    const std::vector<std::pair<float, int>> paramters = generateNParamters(1);
    const auto licNoisetexture = randomNoiseTexture(Xdim, Ydim);

    uniform_real_distribution<float> genTheta(-1.0, 1.0);
    uniform_real_distribution<float> genSx(-2.0, 2.0);

    string root_folder = "../data/unsteady/" + to_string(Xdim) + "_" + to_string(Xdim) + "/";
    if (!filesystem::exists(root_folder)) {
        filesystem::create_directories(root_folder);
    }

    for_each(policy, paramters.begin(), paramters.cend(), [&](const std::pair<float, int>& params) {
        const float rc = params.first;
        const float n = params.second;
        const string task_name = "velocity_field_rc_" + to_string(rc) + "n_" + to_string(n);
        printf("generate data for %s\n", task_name.c_str());

        VastistasVelocityGenerator generator(Xdim, Ydim, domainMinBoundary, domainMaxBoundary, rc, n);

        for (size_t sample = 0; sample < numVelocityFields; sample++) {

            const auto theta = genTheta(rng);
            const auto sx = 1 - 0.5 * genSx(rng);
            const auto sy = 1 - 0.5 * genSx(rng);
            for (size_t j = 0; j < 3; j++) { // j denotes selection of si matix

                auto resData = generator.generateSteady(sx, sy, theta, j);
                const string tag_name = "velocity_field_ " + std::to_string(sample) + "rc_" + to_string(rc) + "n_" + to_string(n) + "S[" + to_string(j) + "]_";

                string metaFilename = root_folder + tag_name + "meta.json";
                string velocityFilename = root_folder + tag_name + "velocity.bin";
                string licFilename = root_folder + tag_name + "lic.png";

                // save meta info:
                std::ofstream out(metaFilename);
                if (!out.good()) {
                    printf("couldn't open file: %s", metaFilename.c_str());
                    return;
                }
                std::ofstream outBin(velocityFilename, std::ios::binary);
                if (!outBin.good()) {
                    printf("couldn't open file: %s", velocityFilename.c_str());
                    return;
                }

                cereal::BinaryOutputArchive archive_Binary(outBin);
                const std::vector<float> rawData = flatten2D(resData);
                auto [minV, maxV] = computeMinMax(rawData);

                {

                    cereal::JSONOutputArchive archive_o(out);
                    archive_o(CEREAL_NVP(Xdim));
                    archive_o(CEREAL_NVP(Ydim));
                    std::vector<float> domainMininumBoundary = { domainMinBoundary(0), domainMinBoundary(1) };
                    std::vector<float> domainMaxinumBoundary = { domainMaxBoundary(0), domainMaxBoundary(1) };
                    archive_o(CEREAL_NVP(domainMininumBoundary));
                    archive_o(CEREAL_NVP(domainMaxinumBoundary));

                    archive_o(CEREAL_NVP(n));
                    archive_o(CEREAL_NVP(rc));

                    archive_o(CEREAL_NVP(theta));
                    archive_o(CEREAL_NVP(sx));
                    archive_o(CEREAL_NVP(sy));
                    archive_o(CEREAL_NVP(minV));
                    archive_o(CEREAL_NVP(maxV));
                }

                // do not manually close file before creal deconstructor, as cereal will preprend a ]/} to finish json class/array
                out.close();

                // write raw data

                // ar(make_size_tag(static_cast<size_type=uint64>(vector.size()))); // number of elements
                // ar(binary_data(vector.data(), vector.size() * sizeof(T)));
                //  when using other library to read, need to discard the first uint64 (8bytes.)
                archive_Binary(rawData);

                outBin.close();

                SteadyVectorField2D outField {
                    resData,
                    domainMinBoundary,
                    domainMaxBoundary,
                    gridInterval
                };
                auto outputTexture = LICAlgorithm(licNoisetexture, outField, LicImageSize, LicImageSize, stepSize, maxIteratioOneDirection);

                saveAsPNG(outputTexture, licFilename);

            } // for j
        }
    });
}

struct KillingComponentFunctionFactory {

    static std::function<Eigen::Vector3f(float)> constantTranslation(int direction, float scale)
    {
        return [=](float t) {
            if (direction == 0) {
                return Eigen::Vector3f(scale, 0, 0);
            } else
                return Eigen::Vector3f(0, scale, 0);
        };
    }

    static std::function<Eigen::Vector3f(float)> constantRotation(float speed)
    {
        return [=](float t) {
            return Eigen::Vector3f(0, 0, speed);
        };
    }
};

void testKillingTransformation2()
{

    const int Xdim = 64, Ydim = 64;
    const int LicImageSize = 128;
    Eigen::Vector2f domainMinBoundary = { -2.0, -2.0 };
    Eigen::Vector2f domainMaxBoundary = { 2.0, 2.0 };
    const int unsteadyFieldTimeStep = 32;

    const float stepSize = 0.01;
    const int maxLICIteratioOneDirection = 256;
    int numVelocityFields = 1; // num of fields per n, rc parameter setting
    std::string root_folder = "../data/unsteady/" + to_string(Xdim) + "_" + to_string(Xdim) + "/";

    Eigen::Vector2f gridInterval = {
        (domainMaxBoundary(0) - domainMinBoundary(0)) / (Xdim - 1),
        (domainMaxBoundary(1) - domainMinBoundary(1)) / (Ydim - 1)
    };
    const auto licNoisetexture = randomNoiseTexture(Xdim, Ydim);

    const std::vector<std::pair<float, int>> paramters = generateNParamters(1);
    float rc = 1.0;
    float n = 2;
    std::uniform_real_distribution<float> genSx(-2.0, 2.0);
    const auto tx = 1 - 0.45 * genSx(rng);
    const auto ty = 1 - 0.45 * genSx(rng);

    const auto cx = 1;
    const auto cy = 1;
    const auto dx = 0;
    const auto dy = 0;
    VastistasVelocityGenerator generator(Xdim, Ydim, domainMinBoundary, domainMaxBoundary, rc, n);
    auto resData = generator.generateSteadyV2(cx, cy, dx, dy, tx, ty);

    SteadyVectorField2D steadyField {
        resData,
        domainMinBoundary,
        domainMaxBoundary,
        gridInterval
    };
    auto outputSteadyTexture = LICAlgorithm(licNoisetexture, steadyField, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
    string tag_name = "steady_beforeTransformation";
    string licFilename0 = root_folder + tag_name + "lic.png";

    auto func_const_trans = KillingComponentFunctionFactory::constantTranslation(0, 0.1);

    std::function<Eigen::Vector3f(float)> func_decreasespeedTranslate = [=](float t) {
        return Eigen::Vector3f(0.1 * (1 - t / 2 * M_PI), 0, 0) * 0.1;
    };

    KillingAbcField observerfield(
        func_decreasespeedTranslate, unsteadyFieldTimeStep, 0.0f, 2 * M_PI);
    Eigen::Vector2f StartPosition = { 0.0, 0.0 };
    auto unsteady_field = killingABCtransformation(observerfield, StartPosition, steadyField);

    saveAsPNG(outputSteadyTexture, licFilename0);
    auto outputTextures = LICAlgorithm_UnsteadyField(licNoisetexture, unsteady_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);

    for (size_t i = 0; i < outputTextures.size(); i++) {
        string tag_name = "killing_transformation_" + std::to_string(i);
        string licFilename = root_folder + tag_name + "lic.png";
        saveAsPNG(outputTextures[i], licFilename);
    }
}

void testKillingTransformation()
{
    const float tmin = 0.0;
    const float tmax = 2.0;
    const int Xdim = 64, Ydim = 64;
    const int LicImageSize = 128;
    Eigen::Vector2f domainMinBoundary = { -2.0, -2.0 };
    Eigen::Vector2f domainMaxBoundary = { 2.0, 2.0 };
    const int unsteadyFieldTimeStep = 7;
    const float stepSize = 0.01;
    const int maxLICIteratioOneDirection = 256;
    int numVelocityFields = 1; // num of fields per n, rc parameter setting
    std::string root_folder = "../data/unsteady/" + to_string(Xdim) + "_" + to_string(Xdim) + "/";

    // Create an instance of AnalyticalFlowCreator
    Eigen::Vector2i grid_size(Xdim, Ydim);
    int time_steps = unsteadyFieldTimeStep;

    AnalyticalFlowCreator flowCreator(grid_size, time_steps, domainMinBoundary, domainMaxBoundary);

    // Define a lambda function for rotating flow
    auto rotatingFourCenter = [](Eigen::Vector2f p, double t) {
        double x = p(0);
        double y = p(1);
        double al_t = 1.0, scale = 1.0, maxVelocity = 1.0;

        double u = exp(-y * y - x * x) * (al_t * y * exp(y * y + x * x) - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * y * y * y + (12.0 * scale * (cos(al_t * t) * cos(al_t * t)) - 6.0 * scale) * x * y * y + (6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * x + 6.0 * scale * cos(al_t * t) * sin(al_t * t)) * y + (3.0 * scale - 6.0 * scale * (cos(al_t * t) * cos(al_t * t))) * x);
        double v = -exp(-y * y - x * x) * (al_t * x * exp(y * y + x * x) - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * y * y + ((12.0 * scale * (cos(al_t * t) * cos(al_t * t)) - 6.0 * scale) * x * x - 6.0 * scale * (cos(al_t * t) * cos(al_t * t)) + 3.0 * scale) * y + 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * x * x - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x);

        double vecU = maxVelocity * u;
        double vecV = maxVelocity * v;

        Eigen::Vector2f components;
        components << vecU, vecV;
        return components;
    };
    // Create the flow field
    UnSteadyVectorField2D InputflowField = flowCreator.createFlowField(rotatingFourCenter);

    auto func_const_trans = KillingComponentFunctionFactory::constantRotation(1);

    KillingAbcField observerfield(
        func_const_trans, unsteadyFieldTimeStep, tmin, tmax);
    Eigen::Vector2f StartPosition = { 0.0, 0.0 };

    const auto licNoisetexture = randomNoiseTexture(Xdim, Ydim);

    auto unsteady_field = killingABCtransformation(observerfield, StartPosition, InputflowField);
    auto resample_observerfield = observerfield.resample2UnsteadyField(grid_size, domainMinBoundary, domainMaxBoundary);
    // auto outputObserverFieldLic = LICAlgorithm_UnsteadyField(licNoisetexture, resample_observerfield, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
    //  auto outputTextures0 = LICAlgorithm_UnsteadyField(licNoisetexture, InputflowField, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);

    auto outputTexturesObservedField = LICAlgorithm_UnsteadyField(licNoisetexture, unsteady_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);

    for (size_t i = 0; i < unsteadyFieldTimeStep; i++) {
        string tag_name0 = "inputfield_" + std::to_string(i);
        string licFilename0 = root_folder + tag_name0 + "lic.png";
        // saveAsPNG(outputTextures0[i], licFilename0);

        string tag_name1 = "obfield_" + std::to_string(i);
        string licFilename1 = root_folder + tag_name1 + "lic.png";
        // saveAsPNG(outputObserverFieldLic[i], licFilename1);

        string tag_name = "killing_transformation_Observed" + std::to_string(i);
        string licFilename = root_folder + tag_name + "lic.png";
        saveAsPNG(outputTexturesObservedField[i], licFilename);
    }
}