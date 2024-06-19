
#include "VectorFieldCompute.h"
#include "VastistasVelocityGenerator.h"
#include <Eigen/Dense>
#include <execution>
#include <random>
//  #define DISABLE_CPP_PARALLELISM
//   define execute policy
#if defined(DISABLE_CPP_PARALLELISM) || defined(_DEBUG)
auto policy = std::execution::seq;
#else
auto policy = std::execution::par_unseq;
#endif
namespace {

std::mt19937 rng(static_cast<unsigned int>(std::time(0)));
}

using namespace std;

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

// The result image size is the same as the input texture.
std::vector<std::vector<Eigen::Vector3d>> LICAlgorithm(
    const std::vector<std::vector<double>>& texture,
    const SteadyVectorField2D& vecfield,
    const int licImageSizeX,
    const int licImageSizeY,
    double stepSize,
    int MaxIntegrationSteps,
    VORTEX_CRITERION criterionlColorBlend)
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
LICAlgorithm_UnsteadyField(const std::vector<std::vector<double>>& texture,
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
        auto licPic = LICAlgorithm(texture, slice, licImageSizeX, licImageSizeY, stepSize, MaxIntegrationSteps, curlColorBlend);
        return std::move(licPic);
    });
    return resultData;
}
