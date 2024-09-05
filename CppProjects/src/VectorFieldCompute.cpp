
#include "VectorFieldCompute.h"
#include "VastistasVelocityGenerator.h"
#include <Eigen/Dense>
#include <array>
#include <execution>
#include <fstream>
#include <random>

//  #define DISABLE_CPP_PARALLELISM
//   define execute policy
namespace {
#if defined(DISABLE_CPP_PARALLELISM) || defined(_DEBUG)
auto policy = std::execution::seq;
#else
auto policy = std::execution::par_unseq;
#endif

std::mt19937 rng(static_cast<unsigned int>(std::time(0)));

}
using namespace std;
#include "stablized_texture_512png.cpp"

// Function to generate a 2D vector of random noise
std::vector<std::vector<double>> randomNoiseTexture(int width, int height)
{
    std::vector<std::vector<double>> texture(height, std::vector<double>(width));
    std::random_device rd; // Seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    printf("randomNoiseTexture:\n");
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            texture[y][x] = static_cast<double>(dis(gen));
            printf("%f, ", texture[y][x]);
        }
    }

    return texture;
}

std::vector<std::vector<Eigen::Vector3d>> LICAlgorithm(
    const SteadyVectorField2D& vecfield,
    const int licImageSizeX,
    const int licImageSizeY,
    double stepSize,
    int MaxIntegrationSteps,
    VORTEX_CRITERION criterionlColorBlend)
{

    const int Ydim = vecfield.field.size();
    const int Xdim = vecfield.field[0].size();

    std::vector<std::vector<Eigen::Vector3d>> output_texture(licImageSizeY, std::vector<Eigen::Vector3d>(licImageSizeX, { 0.0f, 0.0f, 0.0f }));

    const auto& vecfieldData = vecfield.field;
    const Eigen::Vector2d domainRange = vecfield.spatialDomainMaxBoundary - vecfield.spatialDomainMinBoundary;
    const double inverse_grid_interval_x = 1.0f / (double)vecfield.spatialGridInterval(0);
    const double inverse_grid_interval_y = 1.0f / (double)vecfield.spatialGridInterval(1);

    double minCurl;
    double maxCurl;
    std::vector<std::vector<double>> curl;
    if (criterionlColorBlend != VORTEX_CRITERION::NONE) {
        curl = computeTargetCrtierion(vecfieldData, Xdim, Ydim, vecfield.spatialGridInterval(0), vecfield.spatialGridInterval(1), criterionlColorBlend);
        // Normalize curl values for color mapping
        auto minMaxCurl = computeMinMax(curl);
        minCurl = minMaxCurl.first;
        maxCurl = minMaxCurl.second;
        if (!(maxCurl > minCurl) || (std::abs(maxCurl) < 1e-7 && std::abs(minCurl) < 1e-7)) {
            printf("Warning: criterion  %u  has scalar field are too small to be used for color mapping. Switching to NONE coloring.\n", (unsigned int)criterionlColorBlend);
            criterionlColorBlend = VORTEX_CRITERION::NONE;
        }
    }

    constexpr auto& noiseTexture = stablizedTexture::noiseTexture64;
    constexpr int TexDim = noiseTexture.size();
    const double FloatIdx_field2textureMultipilierX = ((double)TexDim - 1.0) / (double)(Xdim - 1.0);
    const double FloatIdx_field2textureMultipilierY = ((double)TexDim - 1.0) / (double)(Ydim - 1.0);

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
                double floatIndicesXInField = (pos(0) - vecfield.spatialDomainMinBoundary(0)) * inverse_grid_interval_x;
                double floatIndicesYInField = (pos(1) - vecfield.spatialDomainMinBoundary(1)) * inverse_grid_interval_y;

                if (!(0 <= floatIndicesXInField && floatIndicesXInField < Xdim && 0 <= floatIndicesYInField && floatIndicesYInField < Ydim)) {
                    break; // Stop if we move outside the texture bounds
                }

                accum_value += bilinear_interpolate<double, TexDim>(noiseTexture, floatIndicesXInField * FloatIdx_field2textureMultipilierX, floatIndicesYInField * FloatIdx_field2textureMultipilierY);
                accum_count += 1;
                Eigen::Vector2d vec = bilinear_interpolate(vecfieldData, floatIndicesXInField, floatIndicesYInField);
                pos += vec * stepSize;
            }

            // Trace backward
            pos = { ratio_x * domainRange(0) + vecfield.spatialDomainMinBoundary(0),
                ratio_y * domainRange(1) + vecfield.spatialDomainMinBoundary(1) };

            for (int i = 0; i < MaxIntegrationSteps; ++i) {
                double floatIndicesXInField = (pos(0) - vecfield.spatialDomainMinBoundary(0)) * inverse_grid_interval_x;
                double floatIndicesYInField = (pos(1) - vecfield.spatialDomainMinBoundary(1)) * inverse_grid_interval_y;
                if (!(0 <= floatIndicesXInField && floatIndicesXInField < Xdim && 0 <= floatIndicesYInField && floatIndicesYInField < Ydim)) {
                    break; // Stop if we move outside the texture bounds
                }
                accum_value += bilinear_interpolate<double, TexDim>(noiseTexture, floatIndicesXInField * FloatIdx_field2textureMultipilierX, floatIndicesYInField * FloatIdx_field2textureMultipilierY);
                accum_count += 1;
                Eigen::Vector2d vec = bilinear_interpolate(vecfieldData, floatIndicesXInField, floatIndicesYInField);
                pos -= vec * stepSize;
            }

            // Compute the average value along the path
            if (accum_count > 0) {
                auto licValue = accum_value / accum_count;
                if (criterionlColorBlend != VORTEX_CRITERION::NONE) {
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
LICAlgorithm_UnsteadyField(
    const UnSteadyVectorField2D& vecfield,
    const int licImageSizeX,
    const int licImageSizeY,
    double stepSize,
    int MaxIntegrationSteps, VORTEX_CRITERION curlColorBlend)
{
    std::vector<int> timeIndex;
    timeIndex.resize(vecfield.timeSteps);
    std::iota(timeIndex.begin(), timeIndex.end(), 0);
    std::vector<std::vector<std::vector<Eigen::Vector3d>>> resultData;
    resultData.resize(vecfield.timeSteps);
#if defined(DISABLE_CPP_PARALLELISM) || defined(_DEBUG)
    auto policy = std::execution::seq;
#else
    auto policy = std::execution::par_unseq;
#endif
    std::transform(policy, timeIndex.begin(), timeIndex.end(), resultData.begin(), [&](int time) {
        // std::cout << "parallel lic rendering.. timeIndex size: " << time << std::endl;
        auto slice = vecfield.getVectorfieldSliceAtTime(time);
        auto licPic = LICAlgorithm(slice, licImageSizeX, licImageSizeY, stepSize, MaxIntegrationSteps, curlColorBlend);
        return std::move(licPic);
    });
    return resultData;
}

bool PathhlineIntegrationRK4(const Eigen::Vector2d& StartPosition, const IUnsteadField2D& inputField, const double tstart, const double targetIntegrationTime, const double dt_, std::vector<Eigen::Vector2d>& pathVelocitys, std::vector<Eigen::Vector3d>& pathPositions)
{
    auto integratePathlineOneStep_RK4 = [](const IUnsteadField2D& observerfield, double x, double y, double t, double dt) -> Eigen::Vector2d {
        // RK4 integration step
        Eigen::Vector2d odeStepStartPoint = { x, y };

        const double h = dt;

        // coefficients
        constexpr double a21 = 0.5;
        constexpr double a31 = 0.;
        constexpr double a32 = 0.5;
        constexpr double a41 = 0.;
        constexpr double a42 = 0.;
        constexpr double a43 = 1.;

        constexpr double c2 = 0.5;
        constexpr double c3 = 0.5;
        constexpr double c4 = 1.;
        constexpr double b1 = 1. / 6.;
        constexpr double b2 = 1. / 3.;
        constexpr double b3 = b2;
        constexpr double b4 = b1;

        // 4 stages of 2 equations (i.e., 2 dimensions of the manifold and the tangent vector space)

        // stage 1
        Eigen::Vector2d k1 = observerfield.getVector(odeStepStartPoint, t);

        // stage 2
        Eigen::Vector2d stagePoint = odeStepStartPoint + k1 * a21 * h;
        Eigen::Vector2d k2 = observerfield.getVector(stagePoint, t + c2 * h);

        // stage 3
        stagePoint = odeStepStartPoint + (a31 * k1 + a32 * k2) * h;
        Eigen::Vector2d k3 = observerfield.getVector(stagePoint, t + c3 * h);

        // stage 4
        stagePoint = odeStepStartPoint + (a41 * k1 + a42 * k2 + a43 * k3) * h;
        Eigen::Vector2d k4 = observerfield.getVector(stagePoint, t + c4 * h);

        Eigen::Vector2d result_p = odeStepStartPoint + h * (k1 * b1 + k2 * b2 + k3 * b3 + k4 * b4);

        return result_p;
    };

    const double startTime = tstart;
    const int maxIterationCount = 5000;
    const double spaceConversionRatio = 1.0;

    bool integrationOutOfDomainBounds = false;
    bool outOfIntegrationTimeBounds = false;
    int iterationCount = 0;
    // dt
    double dt = dt_;
    double integrationTimeStepSize = dt;
    if (targetIntegrationTime < startTime) {
        // we integrate back in time
        integrationTimeStepSize *= -1.0;
    }

    const Eigen::Vector2d minDomainBounds = inputField.getSpatialMinBoundary();
    const Eigen::Vector2d maxDomainBounds = inputField.getSpatialMaxBoundary();
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

    Eigen::Vector2d currentPoint = StartPosition;

    integrationOutOfDomainBounds = checkIfOutOfDomain(currentPoint);

    // do integration
    auto currentTime = startTime;

    // push init_velocity  &start point
    Eigen::Vector3d pointAndTime = { currentPoint(0), currentPoint(1), currentTime };
    pathPositions.emplace_back(pointAndTime);
    auto init_velocity = inputField.getVector(currentPoint, currentTime);
    pathVelocitys.emplace_back(init_velocity);

    // integrate until either
    // - we reached the max iteration count
    // - we reached the upper limit of the time domain
    // - we ran out of spatial domain
    while ((!integrationOutOfDomainBounds) && (!outOfIntegrationTimeBounds) && (pathPositions.size() < maxIterationCount)) {

        // advance to a new point in the chart
        Eigen::Vector2d newPoint = integratePathlineOneStep_RK4(inputField, currentPoint(0), currentPoint(1), currentTime, dt);
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
                auto velocity = inputField.getVector(newPoint, newTime);
                pathVelocitys.emplace_back(velocity);
                currentPoint = newPoint;
                currentTime = newTime;
                iterationCount++;
            }
        }
    }
    bool suc = pathPositions.size() > 1 && pathVelocitys.size() == pathPositions.size();
    return suc;
}

bool PathhlineIntegrationRK4v2(const Eigen::Vector2d& StartPosition, const IUnsteadField2D& inputField, const double tstart, const double targetIntegrationTime, const double dt_, std::vector<Eigen::Vector3d>& pathPositions)
{
    auto integratePathlineOneStep_RK4 = [](const IUnsteadField2D& observerfield, double x, double y, double t, double dt) -> Eigen::Vector2d {
        // RK4 integration step
        Eigen::Vector2d odeStepStartPoint = { x, y };

        const double h = dt;

        // coefficients
        constexpr double a21 = 0.5;
        constexpr double a31 = 0.;
        constexpr double a32 = 0.5;
        constexpr double a41 = 0.;
        constexpr double a42 = 0.;
        constexpr double a43 = 1.;

        constexpr double c2 = 0.5;
        constexpr double c3 = 0.5;
        constexpr double c4 = 1.;
        constexpr double b1 = 1. / 6.;
        constexpr double b2 = 1. / 3.;
        constexpr double b3 = b2;
        constexpr double b4 = b1;

        // 4 stages of 2 equations (i.e., 2 dimensions of the manifold and the tangent vector space)

        // stage 1
        Eigen::Vector2d k1 = observerfield.getVector(odeStepStartPoint, t);

        // stage 2
        Eigen::Vector2d stagePoint = odeStepStartPoint + k1 * a21 * h;
        Eigen::Vector2d k2 = observerfield.getVector(stagePoint, t + c2 * h);

        // stage 3
        stagePoint = odeStepStartPoint + (a31 * k1 + a32 * k2) * h;
        Eigen::Vector2d k3 = observerfield.getVector(stagePoint, t + c3 * h);

        // stage 4
        stagePoint = odeStepStartPoint + (a41 * k1 + a42 * k2 + a43 * k3) * h;
        Eigen::Vector2d k4 = observerfield.getVector(stagePoint, t + c4 * h);

        Eigen::Vector2d result_p = odeStepStartPoint + h * (k1 * b1 + k2 * b2 + k3 * b3 + k4 * b4);

        return result_p;
    };

    const double startTime = tstart;
    const int maxIterationCount = 5000;
    const double spaceConversionRatio = 1.0;

    pathPositions.reserve(1024);

    bool integrationOutOfDomainBounds = false;
    bool outOfIntegrationTimeBounds = false;
    int iterationCount = 0;
    double integrationTimeStepSize = dt_;
    if (targetIntegrationTime < startTime) {
        // we integrate back in time
        integrationTimeStepSize *= -1.0;
    }

    const Eigen::Vector2d minDomainBounds = inputField.getSpatialMinBoundary();
    const Eigen::Vector2d maxDomainBounds = inputField.getSpatialMaxBoundary();
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

    Eigen::Vector2d currentPoint = StartPosition;

    integrationOutOfDomainBounds = checkIfOutOfDomain(currentPoint);

    // do integration
    auto currentTime = startTime;

    // push init_velocity  &start point
    Eigen::Vector3d pointAndTime = { currentPoint(0), currentPoint(1), currentTime };
    pathPositions.emplace_back(pointAndTime);
    auto init_velocity = inputField.getVector(currentPoint, currentTime);

    // integrate until either
    // - we reached the max iteration count
    // - we reached the upper limit of the time domain
    // - we ran out of spatial domain
    while ((!integrationOutOfDomainBounds) && (!outOfIntegrationTimeBounds) && (pathPositions.size() < maxIterationCount)) {

        // advance to a new point in the chart
        Eigen::Vector2d newPoint = integratePathlineOneStep_RK4(inputField, currentPoint(0), currentPoint(1), currentTime, integrationTimeStepSize);
        integrationOutOfDomainBounds = checkIfOutOfDomain(newPoint);
        if (!integrationOutOfDomainBounds) {
            auto newTime = currentTime + integrationTimeStepSize;
            // check if currentTime is out of the time domain -> we are done
            if ((targetIntegrationTime > startTime) && (newTime >= targetIntegrationTime)) {
                outOfIntegrationTimeBounds = true;
            } else if ((targetIntegrationTime < startTime) && (newTime <= targetIntegrationTime)) {
                outOfIntegrationTimeBounds = true;
            } else {
                if (std::isnan(newPoint(0)) || std::isnan(newPoint(1))) {
                    // Handle the case where newPoint(0) is NaN
                    printf("nan.");
                    break;
                }

                // add  current point to the result list and set currentPoint to newPoint -> everything fine -> continue with the while loop
                Eigen::Vector3d new_pointAndTime = { newPoint(0), newPoint(1), newTime };
                pathPositions.emplace_back(new_pointAndTime);
                auto velocity = inputField.getVector(newPoint, newTime);

                currentPoint = newPoint;
                currentTime = newTime;
                iterationCount++;
            }
        }
    }
    bool suc = pathPositions.size() > 1;
    return suc;
}
