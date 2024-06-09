
#include "flowGenerator.h"
#include "VastistasVelocityGenerator.h"
#include "cereal/archives/binary.hpp"
#include "cereal/archives/json.hpp"
#include "cereal/types/vector.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "flowGenerator.h"
#include <execution>
#include <filesystem>
#include <fstream>
#include <thread>

// #define DISABLE_CPP_PARALLELISM
//  define execute policy
#if defined(DISABLE_CPP_PARALLELISM) || defined(_DEBUG)
auto policy = std::execution::seq;
#else
auto policy = std::execution::par_unseq;
#endif

std::mt19937 rng(static_cast<unsigned int>(std::time(0)));
using namespace std;

const double tmin = 0.0;
const double tmax = 2.0 * M_PI;
const int Xdim = 64, Ydim = 64;
const int LicImageSize = 64;
Eigen::Vector2d domainMinBoundary = { -2.0, -2.0 };
Eigen::Vector2d domainMaxBoundary = { 2.0, 2.0 };
const int unsteadyFieldTimeStep = 32;
const int LicSaveFrequency = 4; // every 2 time steps save one

// Function to save the 2D vector as a PNG image
void saveAsPNG(const std::vector<std::vector<Eigen::Vector3d>>& data, const std::string& filename)
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
std::vector<std::vector<double>> randomNoiseTexture(int width, int height)
{
    std::vector<std::vector<double>> texture(height, std::vector<double>(width));
    std::random_device rd; // Seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            texture[y][x] = static_cast<double>(dis(gen));
        }
    }

    return texture;
}
// Function to flatten a 2D vector to a 1D vector
std::vector<float> flatten2DAs1Dfloat(const std::vector<std::vector<Eigen::Vector2d>>& x2D)
{
    const size_t ydim = x2D.size();
    assert(ydim > 0);
    const size_t xdim = x2D[0].size();
    std::vector<float> result;
    result.resize(xdim * ydim * 2);
    for (size_t i = 0; i < ydim; i++)
        for (size_t j = 0; j < xdim; j++) {
            result[2 * (i * xdim + j)] = static_cast<float>(x2D[i][j](0));
            result[2 * (i * xdim + j) + 1] = static_cast<float>(x2D[i][j](1));
        }

    return result;
}
std::vector<float> flatten3DAs1Dfloat(const std::vector<std::vector<std::vector<Eigen::Vector2d>>>& x3D)
{
    const size_t tdim = x3D.size();
    assert(tdim > 0);
    const size_t ydim = x3D[0].size();
    assert(ydim > 0);
    const size_t xdim = x3D[0][0].size();
    std::vector<float> result;
    result.resize(xdim * ydim * tdim * 2);
    for (size_t t = 0; t < tdim; t++)
        for (size_t i = 0; i < ydim; i++)
            for (size_t j = 0; j < xdim; j++) {
                result[2 * ((t * ydim + i) * xdim + j)] = static_cast<float>(x3D[t][i][j](0));
                result[2 * ((t * ydim + i) * xdim + j) + 1] = static_cast<float>(x3D[t][i][j](1));
            }

    return result;
}
template <class T>
std::pair<double, double> computeMinMax(const std::vector<T>& values)
{
    if (values.empty()) {
        throw std::invalid_argument("The input vector is empty.");
    }

    T minVal = std::numeric_limits<T>::infinity();
    T maxVal = -std::numeric_limits<T>::infinity();

    for (T val : values) {
        if (val < minVal) {
            minVal = val;
        }
        if (val > maxVal) {
            maxVal = val;
        }
    }

    return { minVal, maxVal };
}
template <class T>
std::pair<double, double> computeMinMax(const std::vector<std::vector<T>>& values)
{
    if (values.empty() || values[0].empty()) {
        throw std::invalid_argument("The input vector is empty.");
    }

    T minVal = std::numeric_limits<T>::infinity();
    T maxVal = -std::numeric_limits<T>::infinity();

    for (const auto& row : values) {
        for (T val : row) {
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
std::vector<std::vector<Eigen::Vector3d>> LICAlgorithm(
    const std::vector<std::vector<double>>& texture,
    const SteadyVectorField2D& vecfield,
    const int licImageSizeX,
    const int licImageSizeY,
    double stepSize,
    int MaxIntegrationSteps,
    bool curlColorBlend = true)
{
    const int YTexdim = texture.size();
    const int XTexdim = texture[0].size();
    const int Ydim = vecfield.field.size();
    const int Xdim = vecfield.field[0].size();
    assert(YTexdim == Ydim && Xdim == XTexdim);
    std::vector<std::vector<Eigen::Vector3d>> output_texture(licImageSizeY, std::vector<Eigen::Vector3d>(licImageSizeX, { 0.0f, 0.0f, 0.0f }));

    const auto& vecfieldData = vecfield.field;
    const Eigen::Vector2d domainRange = vecfield.spatialDomainMaxBoundary - vecfield.spatialDomainMinBoundary;
    const double inverse_grid_interval_x = 1.0f / (double)vecfield.spatialGridInterval(0);
    const double inverse_grid_interval_y = 1.0f / (double)vecfield.spatialGridInterval(1);

    std::vector<std::vector<double>> curl(Ydim, std::vector<double>(Xdim, 0.0f));
    double minCurl;
    double maxCurl;
    if (curlColorBlend) {
        // Calculate curl (vorticity) of the vector field
        for (int y = 1; y < Ydim - 1; ++y) {
            for (int x = 1; x < Xdim - 1; ++x) {
                Eigen::Vector2d dv_dx = (vecfieldData[y][x + 1] - vecfieldData[y][x - 1]) * 0.5f * inverse_grid_interval_x;
                Eigen::Vector2d du_dy = (vecfieldData[y + 1][x] - vecfieldData[y - 1][x]) * 0.5f * inverse_grid_interval_y;
                double curl_ = dv_dx(1) - du_dy(0);
                curl[y][x] = curl_;
            }
        }

        // Normalize curl values for color mapping
        auto minMaxCurl = computeMinMax(curl);
        minCurl = minMaxCurl.first;
        maxCurl = minMaxCurl.second;
        if (!(maxCurl > minCurl && maxCurl > 0 && minCurl < 0) || std::abs(maxCurl) < 1e-7 || std::abs(minCurl) < 1e-7) {
            curlColorBlend = false;
        }
    }

    for (int y = 0; y < licImageSizeY; ++y) {
        for (int x = 0; x < licImageSizeX; ++x) {
            double accum_value = 0.0f;
            int accum_count = 0;

            // map position from texture image grid coordinate to vector field
            double ratio_x = (double)((double)x / (double)licImageSizeX);
            double ratio_y = (double)((double)y / (double)licImageSizeY);

            // Trace forward
            // physicalPositionInVectorfield
            Eigen::Vector2d pos = { ratio_x * domainRange(0) + vecfield.spatialDomainMinBoundary(0),
                ratio_y * domainRange(1) + vecfield.spatialDomainMinBoundary(1) };

            for (int i = 0; i < MaxIntegrationSteps; ++i) {
                double floatIndicesX = (pos(0) - vecfield.spatialDomainMinBoundary(0)) * inverse_grid_interval_x;
                double floatIndicesY = (pos(1) - vecfield.spatialDomainMinBoundary(1)) * inverse_grid_interval_y;

                if (!(0 <= floatIndicesX && floatIndicesX < Xdim && 0 <= floatIndicesY && floatIndicesY < Ydim)) {
                    break; // Stop if we move outside the texture bounds
                }
                accum_value += bilinear_interpolate(texture, floatIndicesX, floatIndicesY);
                accum_count += 1;
                Eigen::Vector2d vec = bilinear_interpolate(vecfieldData, floatIndicesX, floatIndicesY);
                pos += vec * stepSize;
            }

            // Trace backward
            pos = { ratio_x * domainRange(0) + vecfield.spatialDomainMinBoundary(0),
                ratio_y * domainRange(1) + vecfield.spatialDomainMinBoundary(1) };

            for (int i = 0; i < MaxIntegrationSteps; ++i) {
                double floatIndicesX = (pos(0) - vecfield.spatialDomainMinBoundary(0)) * inverse_grid_interval_x;
                double floatIndicesY = (pos(1) - vecfield.spatialDomainMinBoundary(1)) * inverse_grid_interval_y;
                if (!(0 <= floatIndicesX && floatIndicesX < Xdim && 0 <= floatIndicesY && floatIndicesY < Ydim)) {
                    break; // Stop if we move outside the texture bounds
                }
                accum_value += bilinear_interpolate(texture, floatIndicesX, floatIndicesY);
                accum_count += 1;
                Eigen::Vector2d vec = bilinear_interpolate(vecfieldData, floatIndicesX, floatIndicesY);
                pos -= vec * stepSize;
            }

            // Compute the average value along the path
            if (accum_count > 0) {
                auto licValue = accum_value / accum_count;
                if (curlColorBlend) {
                    // Compute the normalized curl value
                    auto curlValue = curl[static_cast<int>(ratio_y * Ydim)][static_cast<int>(ratio_x * Xdim)];
                    double normalizedCurl = (curlValue - minCurl) / (maxCurl - minCurl);
                    Eigen::Vector3d curlColor = { normalizedCurl, 0.1, 1.0 - normalizedCurl };

                    auto whiteish = licValue;
                    whiteish = std::min(std::max(0.0, (whiteish - 0.4) * (1.5 / 0.4)), 1.0);
                    // output_texture[y][x] = mix(curlColor, Eigen::Vector3d(licValue, licValue, licValue), 1.0 - whiteish);
                    output_texture[y][x] = Eigen::Vector3d(licValue, licValue, licValue) * (1.0 - whiteish) + curlColor * whiteish;

                } else {
                    output_texture[y][x] = { licValue, licValue, licValue };
                }
            }
        }
    }

    return output_texture;
}

std::vector<std::vector<std::vector<Eigen::Vector3d>>>
LICAlgorithm_UnsteadyField(const std::vector<std::vector<double>>& texture,
    const UnSteadyVectorField2D& vecfield,
    const int licImageSizeX,
    const int licImageSizeY,
    double stepSize,
    int MaxIntegrationSteps, bool curlColorBlend = true)
{
    std::vector<int> timeIndex;
    timeIndex.resize(vecfield.timeSteps);
    std::iota(timeIndex.begin(), timeIndex.end(), 0);
    std::vector<std::vector<std::vector<Eigen::Vector3d>>> resultData;
    resultData.resize(vecfield.timeSteps);

    std::transform(policy, timeIndex.begin(), timeIndex.end(), resultData.begin(), [&](int time) {
        std::cout << "parallel lic rendering.. timeIndex size: " << time << std::endl;
        auto slice = vecfield.getVectorfieldSliceAtTime(time);
        auto licPic = LICAlgorithm(texture, slice, licImageSizeX, licImageSizeY, stepSize, MaxIntegrationSteps, curlColorBlend);
        return std::move(licPic);
    });
    return resultData;
}
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const
    {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

std::vector<std::pair<double, double>> generateNParamters(int n)
{
    std::vector<std::pair<double, double>> parameters = {
        { 1.0, 2.0 },
        { 1.0, 3.0 },
        { 1.0, 5.0 },
        { 2.0, 1.0 },
        { 2.0, 2.0 },
        { 2.0, 3.0 },
        { 2.0, 5.0 },
    };

    std::unordered_set<std::pair<double, double>, pair_hash> unique_params(parameters.begin(), parameters.end());

    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<double> dist_rc(1.87, 0.37); // mean = 1.87, stddev = 0.37
    std::normal_distribution<double> dist_n(1.96, 0.61); // mean = 1.96, stddev = 0.61

    while (parameters.size() < n) {
        double rc = dist_rc(rng);
        double n = static_cast<double>(dist_n(rng));
        std::pair<double, double> new_pair = { rc, n };

        if (unique_params.find(new_pair) == unique_params.end()) {
            parameters.emplace_back(new_pair);
            unique_params.insert(new_pair);
        }
    }

    return parameters;
}

// todo: transform input field with respect to observer motion(v*=pullback(v-u))
template <typename T>
void referenceFrameTransformation(const SteadyVectorField2D& input_field, const int timesteps)
{

    auto integratePathlineOneStep = [](T x, T y, T t, T dt, T& x_new, T& y_new) {
        // RK4 integration step
        Eigen::Vector2d odeStepStartPoint;
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
        Eigen::Vector2d k1 = fieldFunction_u(odeStepStartPoint, t);

        // stage 2
        Eigen::Vector2d stagePoint = odeStepStartPoint + k1 * a21 * h;
        Eigen::Vector2d k2 = fieldFunction_u(stagePoint, t + c2 * h);

        // stage 3
        stagePoint = odeStepStartPoint + (a31 * k1 + a32 * k2) * h;
        Eigen::Vector2d k3 = fieldFunction_u(stagePoint, t + c3 * h);

        // stage 4
        stagePoint = odeStepStartPoint + (a41 * k1 + a42 * k2 + a43 * k3) * h;
        Eigen::Vector2d k4 = fieldFunction_u(stagePoint, t + c4 * h);

        Eigen::Vector2d result_p = odeStepStartPoint + h * (k1 * b1 + k2 * b2 + k3 * b3 + k4 * b4);

        x_new = result_p(0);
        y_new = result_p(1);
    };

    //// integrationStep integrates starting from x, y, t, to target_t arriving at point x_new, y_new.
    //// the push_forward the matrix that does the transformation that a vector would undergo when going one step dt starting at t
    //// the push_forward is computed by draging a frame (2 vectors) along the curve.
    // auto integrationStep = [&integratePathlineStep, dx, dy](T x, T y, T t, T dt, T& x_new, T& y_new, Matrix22& push_forward) {
    //     Eigen::Vector2d p_north;
    //     Eigen::Vector2d p_south;
    //     Eigen::Vector2d p_east;
    //     Eigen::Vector2d p_west;

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
template <typename T = double, class InputFieldTYPE>
UnSteadyVectorField2D killingABCtransformation(const KillingAbcField& observerfield, const Eigen::Vector2d StartPosition, const InputFieldTYPE& inputField)
{
    const auto tmin = observerfield.tmin;
    const auto tmax = observerfield.tmax;
    const auto dt = observerfield.dt;
    const int timestep = observerfield.timeSteps;

    auto integratePathlineOneStep_RK4 = [&observerfield](T x, T y, T t, T dt) -> Eigen::Vector2d {
        // RK4 integration step
        Eigen::Vector2d odeStepStartPoint = { x, y };

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
        Eigen::Vector2d k1 = observerfield.getKillingVector(odeStepStartPoint, t);

        // stage 2
        Eigen::Vector2d stagePoint = odeStepStartPoint + k1 * a21 * h;
        Eigen::Vector2d k2 = observerfield.getKillingVector(stagePoint, t + c2 * h);

        // stage 3
        stagePoint = odeStepStartPoint + (a31 * k1 + a32 * k2) * h;
        Eigen::Vector2d k3 = observerfield.getKillingVector(stagePoint, t + c3 * h);

        // stage 4
        stagePoint = odeStepStartPoint + (a41 * k1 + a42 * k2 + a43 * k3) * h;
        Eigen::Vector2d k4 = observerfield.getKillingVector(stagePoint, t + c4 * h);

        Eigen::Vector2d result_p = odeStepStartPoint + h * (k1 * b1 + k2 * b2 + k3 * b3 + k4 * b4);

        return result_p;
    };

    // integration of worldline
    auto runIntegration = [&](const Eigen::Vector2d& startPoint, /*const double observationTime,*/ const double targetIntegrationTime,
                              std::vector<Eigen::Vector2d>& pathVelocitys, std::vector<Eigen::Vector3d>& pathPositions) -> bool {
        const double startTime = 0.0;
        const int maxIterationCount = 2000;
        const double spaceConversionRatio = 1.0;

        bool integrationOutOfDomainBounds = false;
        bool outOfIntegrationTimeBounds = false;
        int iterationCount = 0;
        double integrationTimeStepSize = dt;

        if (targetIntegrationTime < startTime) {
            // we integrate back in time
            integrationTimeStepSize *= -1.0;
        }
        const auto minDomainBounds = inputField.getSpatialMinBoundary();
        const auto maxDomainBounds = inputField.getSpatialMaxBoundary();
        // This function checks if the current point is in the domain
        std::function<bool(const Eigen::Vector2d& point)> checkIfOutOfDomain = [&minDomainBounds, &maxDomainBounds](const Eigen::Vector2d& point) {
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

        Eigen::Vector2d currentPoint = startPoint;

        integrationOutOfDomainBounds = checkIfOutOfDomain(currentPoint);

        // do integration
        auto currentTime = startTime;

        // push init_velocity  &start point
        Eigen::Vector3d pointAndTime = { currentPoint(0), currentPoint(1), currentTime };
        pathPositions.emplace_back(pointAndTime);
        auto init_velocity = observerfield.getKillingVector(currentPoint, currentTime);
        pathVelocitys.emplace_back(init_velocity);

        // integrate until either
        // - we reached the max iteration count
        // - we reached the upper limit of the time domain
        // - we ran out of spatial domain
        while ((!integrationOutOfDomainBounds) && (!outOfIntegrationTimeBounds) && (pathPositions.size() < maxIterationCount)) {

            // advance to a new point in the chart
            Eigen::Vector2d newPoint = integratePathlineOneStep_RK4(currentPoint(0), currentPoint(1), currentTime, dt);
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
                    Eigen::Vector3d new_pointAndTime = { newPoint(0), newPoint(1), newTime };
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

    std::vector<Eigen::Vector2d> pathVelocitys;
    std::vector<Eigen::Vector3d> pathPositions;
    bool suc = runIntegration(StartPosition, tmax, pathVelocitys, pathPositions);
    assert(suc);

    int validPathSize = pathPositions.size();

    auto NoramlizeSpinTensor = [](Eigen::Matrix3d& input) {
        Eigen::Vector3d unitAngular;
        unitAngular << input(2, 1), input(0, 2), input(1, 0);
        unitAngular.normalize();
        input << 0, -unitAngular(2), unitAngular(1),
            unitAngular(2), 0, -unitAngular(0),
            -unitAngular(1), unitAngular(0), 0;
        return;
    };
    std::vector<Eigen::Matrix3d> observerRotationMatrices;
    observerRotationMatrices.resize(timestep);
    observerRotationMatrices[0] = Eigen::Matrix3d::Identity();

    std::vector<Eigen::Matrix4d> observertransformationMatrices;
    observertransformationMatrices.resize(timestep);
    observertransformationMatrices[0] = Eigen::Matrix4d::Identity();
    const auto observerStartPoint = pathPositions.at(0);

    for (size_t i = 1; i < validPathSize; i++) {
        const double t = observerfield.tmin + i * observerfield.dt;
        const Eigen::Vector3d abc = observerfield.func_(t);
        const auto c_ = abc(2);
        // this abs is important, otherwise flip sign of c_ will cause the spin tensor and rotation angle theta to be flipped simultaneously,
        // two flip sign cancel out the result stepInstanenousRotation never  change.
        //
        // theta is just scalar measure of how many degree the observer rotate with out direction. the rotation angle encoded in Spintensor
        const auto theta = dt * std::abs(c_);
        Eigen::Matrix3d Spintensor;
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
        Eigen::Matrix3d Spi_2;
        Spi_2 = Spintensor * Spintensor;
        double sinTheta = sin(theta);
        double cosTheta = cos(theta);
        Eigen::Matrix3d I = Eigen::Matrix<double, 3, 3>::Identity();
        Eigen::Matrix3d stepInstanenousRotation = I + sinTheta * Spintensor + (1 - cosTheta) * Spi_2;
        // get the rotation matrix of observer, which is the Q(t)^T of frame transformation x*=Q(t)x+c(t)
        observerRotationMatrices[i] = stepInstanenousRotation * observerRotationMatrices[i - 1];
        const auto& stepRotation = observerRotationMatrices[i];
        // compute observer's relative transformation as M=T(position)*integral of [ R(matrix_exponential(spinTensor))] * T(-startPoint)
        // then observer bring transformation  M-1=T(startPoint)*integral of [ R(matrix_exponential(spinTensor))]^T * T(-position)

        auto tP1 = pathPositions.at(i);

        // eigen << fill data in row major regardless of storage order

        Eigen::Matrix4d inv_translationPostR;
        inv_translationPostR << 1, 0, 0, -tP1(0), // first rowm
            0, 1, 0, -tP1(1), // second row
            0, 0, 1, 0,
            0.0, 0, 0, 1;

        Eigen::Matrix4d inv_translationPreR;
        inv_translationPreR << 1, 0, 0, observerStartPoint(0), // first row
            0, 1, 0, observerStartPoint(1), // second row
            0, 0, 1, 0,
            0.0, 0, 0, 1;

        // this Q_t  is exactly the frame transformation x*=Q(t)x+c(t)
        Eigen::Matrix4d Q_t_transpose = Eigen::Matrix4d::Zero();
        Q_t_transpose(0, 0) = stepRotation(0, 0);
        Q_t_transpose(0, 1) = stepRotation(0, 1);
        Q_t_transpose(0, 2) = stepRotation(0, 2);
        Q_t_transpose(1, 0) = stepRotation(1, 0);
        Q_t_transpose(1, 1) = stepRotation(1, 1);
        Q_t_transpose(1, 2) = stepRotation(1, 2);
        Q_t_transpose(2, 0) = stepRotation(2, 0);
        Q_t_transpose(2, 1) = stepRotation(2, 1);
        Q_t_transpose(2, 2) = stepRotation(2, 2);
        Q_t_transpose(3, 3) = 1.0;

        // Eigen::Matrix4f  ObserverTransformation = translationPostR * (Q[i] * translationPreR);
        //  combine translation and rotation into final transformation
        Eigen::Matrix4d InvserseTransformation = inv_translationPreR * (Q_t_transpose.transpose() * inv_translationPostR);

        observertransformationMatrices[i] = InvserseTransformation;
    }
    const auto lastPushforward = observerRotationMatrices[validPathSize - 1];
    const auto lastTransformation = observertransformationMatrices[validPathSize - 1];
    for (size_t i = validPathSize; i < timestep; i++) {
        observertransformationMatrices[i] = lastTransformation;
        observerRotationMatrices[i] = lastPushforward;
        // pathVelocitys.emplace_back(0.0f, 0.0f);
        pathPositions.emplace_back(pathPositions.back());
    }

    UnSteadyVectorField2D resultField;
    resultField.spatialDomainMaxBoundary = inputField.getSpatialMaxBoundary();
    resultField.spatialDomainMinBoundary = inputField.getSpatialMinBoundary();
    resultField.spatialGridInterval = inputField.spatialGridInterval;
    resultField.XdimYdim = inputField.XdimYdim;
    resultField.tmin = tmin;
    resultField.tmax = tmax;
    resultField.timeSteps = timestep;
    resultField.field.resize(timestep);

    // if inputField has analytical expression v(x,t) then result field u  has transformatd analytical expression u(x,y)=   pushforward* v(x*,t) =Q(t)^T *v(x*,t)
    if (inputField.analyticalFlowfunc_) {
        resultField.analyticalFlowfunc_ = [inputField, dt, observerfield, observerRotationMatrices, pathPositions](const Eigen::Vector2d& pos, double t) -> Eigen::Vector2d {
            double tmin = observerfield.tmin;
            const double floatingTimeStep = (t - tmin) / dt;
            const int timestep_floor = std::clamp((int)std::floor(floatingTimeStep), 0, observerfield.timeSteps - 1);
            const int timestep_ceil = std::clamp((int)std::floor(floatingTimeStep) + 1, 0, observerfield.timeSteps - 1);
            const double ratio = floatingTimeStep - timestep_floor;

            // const auto transformationMat = observertransformationMatrices[timestep_floor] * (1 - ratio) + observertransformationMatrices[timestep_ceil] * ratio;
            const auto tmp = pathPositions[timestep_floor] * (1 - ratio) + pathPositions[timestep_ceil] * ratio;

            const Eigen ::Matrix3d Q_transpose = observerRotationMatrices[timestep_floor] * (1 - ratio) + observerRotationMatrices[timestep_ceil] * ratio;
            const Eigen ::Matrix3d Q_t = Q_transpose.transpose();
            const Eigen ::Vector3d pos_3d = { pos.x(), pos.y(), 1.0 };
            const Eigen ::Vector3d position_t = { tmp.x(), tmp.y(), 1.0 };
            // frame transformation is F(x):x*=Q(t)x+c(t)  or x*=T(Os) *Q*T(-Pt)*x
            // =>F(x):x* = Q(t)*(x-pt)+Os= Qx-Q*pt+Os -> c=-Q*pt+Os
            const Eigen ::Vector3d Os = { pathPositions[0].x(), pathPositions[0].y(), 1.0 };
            auto c_t = Os - Q_t * position_t;
            // => F^(-1)(x)= Q^T (x-c)= Q^T *( x+Q*pt-Os)
            Eigen ::Vector3d F_inverse_x = Q_transpose * (pos_3d - c_t);
            F_inverse_x /= F_inverse_x(2);
            // get 2d position from 4d x_star
            Eigen::Vector2d F_inverse_x_2d = { F_inverse_x(0), F_inverse_x(1) };
            Eigen::Vector2d v = inputField.getVectorAnalytical(F_inverse_x_2d, t);
            auto u = observerfield.getVector(F_inverse_x(0), F_inverse_x(1), t);
            Eigen::Vector3d vminusu_lab_frame;
            vminusu_lab_frame << v(0) - u(0), v(1) - u(1), 0;

            // then w*(x,t)=Q(t)* w( F^-1(x),t )= Q(t)*w( Q^T *( x +Q*pt-Os),t )
            const auto vminus_observer_frame = Q_t * vminusu_lab_frame;
            Eigen::Vector2d retValue = { vminus_observer_frame(0), vminus_observer_frame(1) };
            return retValue;
        };

        resultField.resampleFromAnalyticalExpression();

    } else { // if no analytical expression
        if constexpr (std::is_same_v<InputFieldTYPE, UnSteadyVectorField2D>) {
            for (size_t i = 0; i < timestep; i++) {
                resultField.field[i].resize(inputField.field[i].size());
                const double physical_t_this_slice = tmin + i * dt; // time slice i
                for (size_t j = 0; j < inputField.field[i].size(); j++) {
                    // y slice
                    resultField.field[i][j].resize(inputField.field[i][j].size());
                    for (size_t k = 0; k < inputField.field[i][j].size(); k++) {
                        Eigen::Vector4d pos = { k * inputField.spatialGridInterval(0) + inputField.spatialDomainMinBoundary(0),
                            j * inputField.spatialGridInterval(1) + inputField.spatialDomainMinBoundary(1), 0.0f, 1.0f }; // get position of this grid point
                        const Eigen ::Matrix3d Q_transpose = observerRotationMatrices[i];
                        const Eigen ::Matrix3d Q_t = Q_transpose.transpose();
                        const Eigen ::Vector3d pos_3d = { pos.x(), pos.y(), 1.0 };
                        const Eigen ::Vector3d position_t = { pathPositions[i].x(), pathPositions[i].y(), 1.0 };
                        // frame transformation is F(x):x*=Q(t)x+c(t)  or x*=T(Os) *Q*T(-Pt)*x
                        // =>F(x):x* = Q(t)*(x-pt)+Os= Qx-Q*pt+Os -> c=-Q*pt+Os
                        const Eigen ::Vector3d Os = { pathPositions[0].x(), pathPositions[0].y(), 1.0 };
                        auto c_t = Os - Q_t * position_t;
                        // => F^(-1)(x)= Q^T (x-c)= Q^T *( x+Q*pt-Os)
                        Eigen ::Vector3d F_inverse_x = Q_transpose * (pos_3d - c_t);
                        F_inverse_x /= F_inverse_x(2);

                        Eigen::Vector2d F_inverse_x_2d = { F_inverse_x(0), F_inverse_x(1) };

                        Eigen::Vector2d v = inputField.getVector(F_inverse_x_2d, physical_t_this_slice);
                        auto u = observerfield.getVector(F_inverse_x(0), F_inverse_x(1), physical_t_this_slice);

                        Eigen::Vector3d vminus_lab_frame;
                        vminus_lab_frame << v(0) - u(0), v(1) - u(1), 0;

                        // (v-u)*(x,t)=  push forward(  (v-u)(F^-1(x),t) )= Q(t)^T *(v-u)(F^-1(x),t)
                        Eigen::Vector3d vminus_observer_frame = Q_t * vminus_lab_frame;
                        resultField.field[i][j][k] = { vminus_observer_frame(0), vminus_observer_frame(1) };
                    }
                }
            }
        } else {
            for (size_t i = 0; i < timestep; i++) {
                // time slice i
                resultField.field[i].resize(inputField.field.size());
                const double physical_t_this_slice = tmin + i * dt;
                for (size_t j = 0; j < inputField.field.size(); j++) {
                    // y slice
                    resultField.field[i][j].resize(inputField.field[j].size());
                    for (size_t k = 0; k < inputField.field[j].size(); k++) {

                        // get position of this grid point
                        Eigen::Vector4d pos = { k * inputField.spatialGridInterval(0) + inputField.spatialDomainMinBoundary(0),
                            j * inputField.spatialGridInterval(1) + inputField.spatialDomainMinBoundary(1), 0.0f, 1.0f };

                        const auto tmp = pathPositions[i];
                        const Eigen ::Matrix3d Q_transpose = observerRotationMatrices[i];
                        const Eigen ::Matrix3d Q_t = Q_transpose.transpose();
                        const Eigen ::Vector3d pos_3d = { pos.x(), pos.y(), 1.0 };
                        const Eigen ::Vector3d position_t = { tmp.x(), tmp.y(), 1.0 };
                        // frame transformation is F(x):x*=Q(t)x+c(t)  or x*=T(Os) *Q*T(-Pt)*x
                        // =>F(x):x* = Q(t)*(x-pt)+Os= Qx-Q*pt+Os -> c=-Q*pt+Os
                        const Eigen ::Vector3d Os = { pathPositions[0].x(), pathPositions[0].y(), 1.0 };
                        auto c_t = Os - Q_t * position_t;
                        // => F^(-1)(x)= Q^T (x-c)= Q^T *( x+Q*pt-Os)
                        Eigen ::Vector3d F_inverse_x = Q_transpose * (pos_3d - c_t);
                        F_inverse_x /= F_inverse_x(2);

                        Eigen::Vector2d F_inverse_x_2d = { F_inverse_x(0), F_inverse_x(1) };

                        Eigen::Vector2d v = inputField.getVector(F_inverse_x_2d, physical_t_this_slice);
                        auto u = observerfield.getVector(F_inverse_x(0), F_inverse_x(1), physical_t_this_slice);

                        Eigen::Vector3d vminus_lab_frame;
                        vminus_lab_frame << v(0) - u(0), v(1) - u(1), 0;

                        Eigen::Vector3d vminus_observer_frame = Q_t * vminus_lab_frame;
                        resultField.field[i][j][k] = { vminus_observer_frame(0), vminus_observer_frame(1) };
                    }
                }
            }
        }
    }

    return resultField;
}
//
// void generateSteadyField()
//{
//
//    const double stepSize = 0.01;
//    const int maxIteratioOneDirection = 256;
//    int numVelocityFields = 1; // num of fields per n, rc parameter setting
//
//    Eigen::Vector2d gridInterval = {
//        (domainMaxBoundary(0) - domainMinBoundary(0)) / (Xdim - 1),
//        (domainMaxBoundary(1) - domainMinBoundary(1)) / (Ydim - 1)
//    };
//
//    const std::vector<std::pair<double, double>> paramters = generateNParamters(1);
//    const auto licNoisetexture = randomNoiseTexture(Xdim, Ydim);
//
//    uniform_real_distribution<double> genTheta(-1.0, 1.0);
//    uniform_real_distribution<double> genSx(-2.0, 2.0);
//
//    string root_folder = "../data/unsteady/" + to_string(Xdim) + "_" + to_string(Xdim) + "/";
//    if (!filesystem::exists(root_folder)) {
//        filesystem::create_directories(root_folder);
//    }
//
//    for_each(policy, paramters.begin(), paramters.cend(), [&](const std::pair<double, double>& params) {
//        const double rc = params.first;
//        const double n = params.second;
//        const string task_name = "velocity_field_rc_" + to_string(rc) + "n_" + to_string(n);
//        printf("generate data for %s\n", task_name.c_str());
//
//        VastistasVelocityGenerator generator(Xdim, Ydim, domainMinBoundary, domainMaxBoundary, rc, n);
//
//        for (size_t sample = 0; sample < numVelocityFields; sample++) {
//
//            const auto theta = genTheta(rng);
//            const auto sx = 1 - 0.5 * genSx(rng);
//            const auto sy = 1 - 0.5 * genSx(rng);
//            for (size_t j = 0; j < 3; j++) { // j denotes selection of si matix
//
//                auto resData = generator.generateSteady(sx, sy, theta, j);
//                const string tag_name = "velocity_field_ " + std::to_string(sample) + "rc_" + to_string(rc) + "n_" + to_string(n) + "S[" + to_string(j) + "]_";
//
//                string metaFilename = root_folder + tag_name + "meta.json";
//                string velocityFilename = root_folder + tag_name + "velocity.bin";
//                string licFilename = root_folder + tag_name + "lic.png";
//
//                // save meta info:
//                std::ofstream out(metaFilename);
//                if (!out.good()) {
//                    printf("couldn't open file: %s", metaFilename.c_str());
//                    return;
//                }
//                std::ofstream outBin(velocityFilename, std::ios::binary);
//                if (!outBin.good()) {
//                    printf("couldn't open file: %s", velocityFilename.c_str());
//                    return;
//                }
//
//                cereal::BinaryOutputArchive archive_Binary(outBin);
//                const auto rawData = flatten2DAs1Dfloat(resData);
//                auto [minV, maxV] = computeMinMax(rawData);
//
//                {
//
//                    cereal::JSONOutputArchive archive_o(out);
//                    archive_o(CEREAL_NVP(Xdim));
//                    archive_o(CEREAL_NVP(Ydim));
//                    std::vector<double> domainMininumBoundary = { domainMinBoundary(0), domainMinBoundary(1) };
//                    std::vector<double> domainMaxinumBoundary = { domainMaxBoundary(0), domainMaxBoundary(1) };
//                    archive_o(CEREAL_NVP(domainMininumBoundary));
//                    archive_o(CEREAL_NVP(domainMaxinumBoundary));
//
//                    archive_o(CEREAL_NVP(n));
//                    archive_o(CEREAL_NVP(rc));
//
//                    archive_o(CEREAL_NVP(theta));
//                    archive_o(CEREAL_NVP(sx));
//                    archive_o(CEREAL_NVP(sy));
//                    archive_o(CEREAL_NVP(minV));
//                    archive_o(CEREAL_NVP(maxV));
//                }
//
//                // do not manually close file before creal deconstructor, as cereal will preprend a ]/} to finish json class/array
//                out.close();
//
//                // write raw data
//
//                // ar(make_size_tag(static_cast<size_type=uint64>(vector.size()))); // number of elements
//                // ar(binary_data(vector.data(), vector.size() * sizeof(T)));
//                //  when using other library to read, need to discard the first uint64 (8bytes.)
//                archive_Binary(rawData);
//
//                outBin.close();
//
//                SteadyVectorField2D outField {
//                    resData,
//                    domainMinBoundary,
//                    domainMaxBoundary,
//                    gridInterval
//                };
//                auto outputTexture = LICAlgorithm(licNoisetexture, outField, LicImageSize, LicImageSize, stepSize, maxIteratioOneDirection);
//
//                saveAsPNG(outputTexture, licFilename);
//
//            } // for j
//        }
//    });
//}

// void testKillingTransformASteadyField()
//{
//
//     const int unsteadyFieldTimeStep = 32;
//
//     const double stepSize = 0.01;
//     const int maxLICIteratioOneDirection = 256;
//     int numVelocityFields = 1; // num of fields per n, rc parameter setting
//     std::string root_folder = "../data/unsteady/" + to_string(Xdim) + "_" + to_string(Xdim) + "/";
//
//     Eigen::Vector2d gridInterval = {
//         (domainMaxBoundary(0) - domainMinBoundary(0)) / (Xdim - 1),
//         (domainMaxBoundary(1) - domainMinBoundary(1)) / (Ydim - 1)
//     };
//     const auto licNoisetexture = randomNoiseTexture(Xdim, Ydim);
//
//     double rc = 1.0;
//     double n = 2;
//     std::uniform_real_distribution<double> genSx(-2.0, 2.0);
//     const auto tx = 1 - 0.45 * genSx(rng);
//     const auto ty = 1 - 0.45 * genSx(rng);
//
//     const auto cx = 1;
//     const auto cy = 1;
//     const auto dx = 0;
//     const auto dy = 0;
//     VastistasVelocityGenerator generator(Xdim, Ydim, domainMinBoundary, domainMaxBoundary, rc, n);
//     auto resData = generator.generateSteadyV2(cx, cy, dx, dy, tx, ty);
//
//     SteadyVectorField2D steadyField {
//         resData,
//         domainMinBoundary,
//         domainMaxBoundary,
//         gridInterval
//     };
//     auto outputSteadyTexture = LICAlgorithm(licNoisetexture, steadyField, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
//     string tag_name = "steady_beforeTransformation";
//     string licFilename0 = root_folder + tag_name + "lic.png";
//
//     auto func_const_trans = KillingComponentFunctionFactory::constantTranslation(0, 0.1);
//
//     std::function<Eigen::Vector3d(double)> func_decreasespeedTranslate = [=](double t) {
//         return Eigen::Vector3d(0.1 * (1 - t / 2 * M_PI), 0, 0) * 0.1;
//     };
//
//     KillingAbcField observerfield(
//         func_decreasespeedTranslate, unsteadyFieldTimeStep, 0.0f, 2 * M_PI);
//     Eigen::Vector2d StartPosition = { 0.0, 0.0 };
//     auto unsteady_field = killingABCtransformation(observerfield, StartPosition, steadyField);
//
//     saveAsPNG(outputSteadyTexture, licFilename0);
//     auto outputTextures = LICAlgorithm_UnsteadyField(licNoisetexture, unsteady_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
//
//     for (size_t i = 0; i < outputTextures.size(); i++) {
//         string tag_name = "killing_transformation_" + std::to_string(i);
//         string licFilename = root_folder + tag_name + "lic.png";
//         saveAsPNG(outputTextures[i], licFilename);
//     }
// }
////
void testKillingTransformationForRFC()
{
    const int LicImageSize = 128;
    const int unsteadyFieldTimeStep = 32;
    const double stepSize = 0.01;
    const int maxLICIteratioOneDirection = 256;
    int numVelocityFields = 1; // num of fields per n, rc parameter setting
    std::string root_folder = "../data/unsteady/" + to_string(Xdim) + "_" + to_string(Xdim) + "/" + "test/";
    if (!filesystem::exists(root_folder)) {
        filesystem::create_directories(root_folder);
    }
    // Create an instance of AnalyticalFlowCreator
    Eigen::Vector2i grid_size(Xdim, Ydim);
    int time_steps = unsteadyFieldTimeStep;

    AnalyticalFlowCreator flowCreator(grid_size, time_steps, domainMinBoundary, domainMaxBoundary);

    // Define a lambda function for rotating flow
    auto rotatingFourCenter = [](Eigen::Vector2d p, double t) {
        const double x = p(0);
        const double y = p(1);
        const double al_t = 1.0;
        const double scale = 8.0;
        const double maxVelocity = 1.0;

        double u = exp(-y * y - x * x) * (al_t * y * exp(y * y + x * x) - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * y * y * y + (12.0 * scale * (cos(al_t * t) * cos(al_t * t)) - 6.0 * scale) * x * y * y + (6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * x + 6.0 * scale * cos(al_t * t) * sin(al_t * t)) * y + (3.0 * scale - 6.0 * scale * (cos(al_t * t) * cos(al_t * t))) * x);
        double v = -exp(-y * y - x * x) * (al_t * x * exp(y * y + x * x) - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * y * y + ((12.0 * scale * (cos(al_t * t) * cos(al_t * t)) - 6.0 * scale) * x * x - 6.0 * scale * (cos(al_t * t) * cos(al_t * t)) + 3.0 * scale) * y + 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * x * x - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x);

        double vecU = maxVelocity * u;
        double vecV = maxVelocity * v;

        Eigen::Vector2d components;
        components << vecU, vecV;
        return components;
    };
    // Create the flow field
    UnSteadyVectorField2D InputflowField = flowCreator.createFlowField(rotatingFourCenter);

    auto func_const_trans = KillingComponentFunctionFactory::constantRotation(-1.0);

    KillingAbcField observerfield(
        func_const_trans, unsteadyFieldTimeStep, tmin, tmax);
    Eigen::Vector2d StartPosition = { 0.0, 0.0 };

    const auto licNoisetexture = randomNoiseTexture(Xdim, Ydim);

    auto unsteady_field = killingABCtransformation(observerfield, StartPosition, InputflowField);
    auto resample_observerfield = observerfield.resample2UnsteadyField(grid_size, domainMinBoundary, domainMaxBoundary);
    // auto outputObserverFieldLic = LICAlgorithm_UnsteadyField(licNoisetexture, resample_observerfield, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
    auto inputTextures = LICAlgorithm_UnsteadyField(licNoisetexture, InputflowField, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);

    auto outputTexturesObservedField = LICAlgorithm_UnsteadyField(licNoisetexture, unsteady_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);

    for (size_t i = 0; i < unsteadyFieldTimeStep; i++) {
        string tag_name0 = "inputField_" + std::to_string(i);
        string licFilename0 = root_folder + tag_name0 + "lic.png";
        saveAsPNG(inputTextures[i], licFilename0);

        string tag_name1 = "observerField_" + std::to_string(i);
        string licFilename1 = root_folder + tag_name1 + "lic.png";
        // saveAsPNG(outputObserverFieldLic[i], licFilename1);

        string tag_name = "killing_transformed_" + std::to_string(i);
        string licFilename = root_folder + tag_name + "lic.png";
        saveAsPNG(outputTexturesObservedField[i], licFilename);
    }
}

void addSegmentationVisualization(std::vector<std::vector<Eigen::Vector3d>>& inputLicImage, const SteadyVectorField2D& vectorField, const Eigen::Vector3d& meta_n_rc_si, const Eigen::Vector2d& domainMax, const Eigen::Vector2d& domainMIn, const Eigen::Vector2d& txy)
{
    // if si=0 then no vortex
    if (meta_n_rc_si.z() == 0.0) {
        return;
    }
    const auto rc = meta_n_rc_si(1);
    auto judgeVortex = [rc, txy](const Eigen::Vector2d& pos) -> bool {
        Eigen::Vector2d center = { txy(0), txy(1) };
        auto radius = (pos - center).norm();
        return radius < rc;
    };
    const int licImageSizeY = inputLicImage.size();
    const int licImageSizeX = inputLicImage[0].size();
    const auto domainRange = domainMax - domainMIn;
    const auto dx = domainRange(0) / licImageSizeX;
    const auto dy = domainRange(1) / licImageSizeY;
    const auto maxDistanceAnyPoint2gridPoints = sqrt((0.5 * dx) * (0.5 * dx) + (0.5 * dy) * (0.5 * dy));

    for (size_t i = 0; i < licImageSizeX; i++) {
        for (size_t j = 0; j < licImageSizeY; j++) {
            // map position from texture image grid coordinate to vector field
            double ratio_x = (double)((double)i / (double)licImageSizeX);
            double ratio_y = (double)((double)j / (double)licImageSizeY);

            // Trace forward
            // physicalPositionInVectorfield
            Eigen::Vector2d pos = { ratio_x * domainRange(0) + domainMIn(0),
                ratio_y * domainRange(1) + domainMIn(1) };

            if (judgeVortex(pos)) {
                auto preColor = inputLicImage[j][i];
                // red for critial point(coreline)
                auto velocity = vectorField.getVector(pos.x(), pos.y());
                // !note: because txy is random core line center, then critical point==txy(velocity ==0.0) might not  lie on any grid points
                if (velocity.norm() < 1e-7 || (pos - txy).norm() < maxDistanceAnyPoint2gridPoints)
                    [[unlikely]] {
                    inputLicImage[j][i] = 0.3 * preColor + 0.7 * Eigen::Vector3d(1.0, 0.0, 0.0);
                } else {
                    // yellow for vortex region
                    inputLicImage[j][i] = 0.3 * preColor + 0.7 * Eigen::Vector3d(1.0, 1.0, 0.0);
                }
            }
        }
    }
}

// number of result traing data = Nparamters * samplePerParameters * observerPerSetting
void generateUnsteadyField(int Nparamters, int samplePerParameters, int observerPerSetting)
{

    const double stepSize = 0.01;
    const int maxLICIteratioOneDirection = 200;
    int numVelocityFields = samplePerParameters; // num of fields per n, rc parameter setting
    std::string root_folder = "../data/unsteady/" + to_string(Xdim) + "_" + to_string(Xdim) + "/";
    if (!filesystem::exists(root_folder)) {
        filesystem::create_directories(root_folder);
    }

    Eigen::Vector2d gridInterval = {
        (domainMaxBoundary(0) - domainMinBoundary(0)) / (Xdim - 1),
        (domainMaxBoundary(1) - domainMinBoundary(1)) / (Ydim - 1)
    };

    const auto paramters = generateNParamters(Nparamters);
    const auto licNoisetexture = randomNoiseTexture(Xdim, Ydim);

    std::normal_distribution<double> genTheta(0.0, 1.0);
    std::normal_distribution<double> genSx(0.0, 2.0);

    // this generate coreline's point(critial point) for vortex region
    std::normal_distribution<double> genTx(0.0, 0.59);

    // Distribution for selecting type
    std::uniform_int_distribution<int> dist_Observer_type(0, (int)ObserverType::NumTypes);
    std::uniform_int_distribution<int> dist_int(0, 4); // we prefer more si =1,2->generate 0,1,2,3,4->ceil(divide by two)

    for_each(policy, paramters.begin(), paramters.cend(), [&](const std::pair<double, double>& params) {
        const double rc = params.first;
        const double n = params.second;
        std::string str_Rc = std::to_string(rc);
        str_Rc.erase(str_Rc.find_last_not_of('0') + 1, std::string::npos);
        str_Rc.erase(str_Rc.find_last_not_of('.') + 1, std::string::npos);
        std::string str_n = std::to_string(n);
        str_n.erase(str_n.find_last_not_of('0') + 1, std::string::npos);
        str_n.erase(str_n.find_last_not_of('.') + 1, std::string::npos);

        VastistasVelocityGenerator generator(Xdim, Ydim, domainMinBoundary, domainMaxBoundary, rc, n);

        printf("generate %d sample for rc=%f , n=%f \n", numVelocityFields, rc, n);

        const string Major_task_foldername = "velocity_rc_" + str_Rc + "n_" + str_n + "/";
        const string Major_task_Licfoldername = Major_task_foldername + "/LIC/";
        std::string task_folder = root_folder + Major_task_foldername;
        if (!filesystem::exists(task_folder)) {
            filesystem::create_directories(task_folder);
        }
        std::string task_licfolder = root_folder + Major_task_Licfoldername;
        if (!filesystem::exists(task_licfolder)) {
            filesystem::create_directories(task_licfolder);
        }
        // create Root meta json file, save plane information here instead of every sample's meta file
        string taskFolder_rootMetaFilename = task_folder + "meta.json";
        // save root meta info:
        std::ofstream root_jsonOut(taskFolder_rootMetaFilename);
        if (!root_jsonOut.good()) {
            printf("couldn't open file: %s", taskFolder_rootMetaFilename.c_str());
            return;
        } else {

            cereal::JSONOutputArchive archive_o(root_jsonOut);
            archive_o(CEREAL_NVP(Xdim));
            archive_o(CEREAL_NVP(Ydim));
            archive_o(CEREAL_NVP(unsteadyFieldTimeStep));
            archive_o(CEREAL_NVP(domainMinBoundary));
            archive_o(CEREAL_NVP(domainMaxBoundary));
            archive_o(CEREAL_NVP(tmin));
            archive_o(CEREAL_NVP(tmax));
        }

        for (size_t sample = 0; sample < numVelocityFields; sample++) {

            // generate steady field with vortex
            const auto theta = genTheta(rng);
            const auto sx = 0.5f * genSx(rng);
            const auto sy = 0.5f * genSx(rng);
            auto tx = genTx(rng);
            auto ty = genTx(rng);
            // clamp tx ty to 0.5*domian
            tx = std::clamp(tx, 0.5 * domainMinBoundary.x(), 0.5 * domainMaxBoundary.x());
            ty = std::clamp(ty, 0.5 * domainMinBoundary.y(), 0.5 * domainMaxBoundary.y());
            Eigen::Vector2d txy = { tx, ty };

            int Si = std::clamp((int)std::ceil(dist_int(rng) / 2), 0, 2);
            Eigen::Vector3d n_rc_si = { n, rc, (double)Si };
            const SteadyVectorField2D steadyField = generator.generateSteadyField(tx, ty, sx, sy, theta, Si);
            const auto& resData = steadyField.field;
            for (size_t observerIndex = 0; observerIndex < observerPerSetting; observerIndex++) {

                // Randomly select a type
                const int type = dist_Observer_type(rng);

                const string sample_tag_name = "rc_" + str_Rc + "_n_" + str_n + "_sample_" + to_string(sample) + "Si_" + to_string(Si) + "observer_" + to_string(observerIndex) + "type_" + to_string(type);

                // printf("generating sample %s \n", sample_tag_name.c_str());
                //  create folder for every n rc parameter setting.

                string metaFilename = task_folder + sample_tag_name + "meta.json";
                string velocityFilename = task_folder + sample_tag_name + ".bin";

                // save meta info:
                std::ofstream jsonOut(metaFilename);
                if (!jsonOut.good()) {
                    printf("couldn't open file: %s", metaFilename.c_str());
                    return;
                }
                std::ofstream outBin(velocityFilename, std::ios::binary);
                if (!outBin.good()) {
                    printf("couldn't open file: %s", velocityFilename.c_str());
                    return;
                }

                cereal::BinaryOutputArchive archive_Binary(outBin);
                const std::vector<float> rawData = flatten2DAs1Dfloat(resData);
                auto [minV, maxV] = computeMinMax(rawData);

                auto func = KillingComponentFunctionFactory::randomObserver(type);

                auto inv_func = KillingComponentFunctionFactory::getInverseObserver(func);

                KillingAbcField observerfieldDeform(
                    func, unsteadyFieldTimeStep, 0.0f, 2 * M_PI);

                KillingAbcField observerfield(
                    inv_func, unsteadyFieldTimeStep, 0.0f, 2 * M_PI);

                {
                    cereal::JSONOutputArchive archive_o(jsonOut);
                    /*     archive_o(CEREAL_NVP(Xdim));
                         archive_o(CEREAL_NVP(Ydim));
                         archive_o(CEREAL_NVP(domainMinBoundary));
                         archive_o(CEREAL_NVP(domainMaxBoundary));*/

                    Eigen::Vector3d deform = { theta, sx, sy };
                    archive_o(cereal::make_nvp("n_rc_Si", n_rc_si));
                    archive_o(cereal::make_nvp("deform_theta_sx_sy", deform));
                    archive_o(cereal::make_nvp("txy", txy));
                    // archive_o(CEREAL_NVP(theta));
                    // archive_o(CEREAL_NVP(sx));
                    // archive_o(CEREAL_NVP(sy));
                    archive_o(CEREAL_NVP(minV));
                    archive_o(CEREAL_NVP(maxV));
                    // meta for observer field
                    archive_o(cereal::make_nvp("Observer Type", type));
                    archive_o(CEREAL_NVP(observerfieldDeform));
                }

                // do not manually close file before creal deconstructor, as cereal will preprend a ]/} to finish json class/array
                jsonOut.close();

                Eigen::Vector2d StartPosition = { 0.0, 0.0 };
                auto unsteady_field = killingABCtransformation(observerfieldDeform, StartPosition, steadyField);
                // reconstruct unsteady field from observer field
                auto reconstruct_field = killingABCtransformation(observerfield, StartPosition, unsteady_field);
                // #if _DEBUG
                // //validate reconstruction result
                for (size_t rec = 1; rec < reconstruct_field.field.size() - 1; rec++) {
                    auto reconstruct_slice = reconstruct_field.field[rec];
                    // compute reconstruct slice difference with steady field
                    double diffSum = 0.0;
                    for (size_t y = 1; y < Ydim - 1; y++)
                        for (size_t x = 1; x < Xdim - 1; x++) {
                            auto diff = reconstruct_slice[y][x] - resData[y][x];
                            diffSum += diff.norm();
                        }
                    // has debug, major reson for reconstruction failure is velocity too big make observer transformation query value from region out of boundary
                    if (diffSum > (Xdim - 2) * (Ydim - 2) * 0.001) {
                        printf("\n\n");
                        printf("reconstruct field not equal to steady field at step %u,check observe type %d\n", (unsigned int)rec, type);
                        printf("\n\n");
                    }
                }
                // #endif

                // rendering LIC
                if (false) {
                    auto outputSteadyTexture = LICAlgorithm(licNoisetexture, steadyField, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
                    // add segmentation visualization for steady lic
                    addSegmentationVisualization(outputSteadyTexture, steadyField, n_rc_si, domainMaxBoundary, domainMinBoundary, txy);
                    string steadyField_name = "steady_beforeTransformation_";
                    string licFilename0 = task_licfolder + sample_tag_name + steadyField_name + "lic.png";
                    saveAsPNG(outputSteadyTexture, licFilename0);

                    auto outputTextures = LICAlgorithm_UnsteadyField(licNoisetexture, unsteady_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection, false);
                    auto outputTexturesReconstruct = LICAlgorithm_UnsteadyField(licNoisetexture, reconstruct_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection, false);

                    for (size_t i = 0; i < outputTextures.size(); i += LicSaveFrequency) {
                        string tag_name = sample_tag_name + "killing_deformed_" + std::to_string(i);
                        string licFilename = task_licfolder + tag_name + "lic.png";
                        saveAsPNG(outputTextures[i], licFilename);

                        string tag_name_rec = sample_tag_name + "reconstruct_" + std::to_string(i);
                        string licFilename_rec = task_licfolder + tag_name_rec + "lic.png";
                        saveAsPNG(outputTexturesReconstruct[i], licFilename_rec);
                    }
                    auto rawUnsteadyFieldData = flatten3DAs1Dfloat(unsteady_field.field);
                    // write raw data
                    // ar(make_size_tag(static_cast<size_type=uint64>(vector.size()))); // number of elements
                    // ar(binary_data(vector.data(), vector.size() * sizeof(T)));
                    //  when using other library to read, need to discard the first uint64 (8bytes.)
                    archive_Binary(rawUnsteadyFieldData);
                    outBin.close();
                }
            }
        } // for sample
    });
}
