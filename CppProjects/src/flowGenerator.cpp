
#include "VastistasVelocityGenerator.h"
#include "cereal/archives/binary.hpp"
#include "cereal/archives/json.hpp"
#include "cereal/types/vector.hpp"
#include <iostream>
#include <random>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include <algorithm>
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

// Function to save the 2D vector as a PNG image
void saveAsPNG(const std::vector<std::vector<float>>& data, const std::string& filename)
{
    int width = data[0].size();
    int height = data.size();

    // Create an array to hold the image data
    std::vector<unsigned char> image_data(width * height * 3); // 3 channels (RGB)

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float value = data[y][x];
            unsigned char pixel_value = static_cast<unsigned char>(value * 255.0f); // Convert to 0-255

            image_data[3 * (y * width + x) + 0] = pixel_value; // R
            image_data[3 * (y * width + x) + 1] = pixel_value; // G
            image_data[3 * (y * width + x) + 2] = pixel_value; // B
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
std::vector<float> flatten2D(const std::vector<std::vector<vec2d<float>>>& x2D)
{
    const int ydim = x2D.size();
    assert(ydim > 0);
    const int xdim = x2D[0].size();
    std::vector<float> result;
    result.resize(xdim * ydim * 2);
    for (size_t i = 0; i < ydim; i++)
        for (size_t j = 0; j < xdim; j++) {
            result[i * xdim + j] = x2D[i][j].x;
            result[i * xdim + j + 1] = x2D[i][j].y;
        }

    return result;
}

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

struct VectorField2D {
    std::vector<std::vector<vec2d<float>>> field;
    vec2d<float> domainMinBoundary;
    vec2d<float> gridInterval;
};

std::vector<std::vector<float>> LICAlgorithm(
    const std::vector<std::vector<float>>& texture,
    const VectorField2D& vecfield,
    float stepSize,
    int MaxIntegrationSteps)
{
    int Ydim = texture.size();
    int Xdim = texture[0].size();
    std::vector<std::vector<float>> output_texture(Ydim, std::vector<float>(Xdim, 0.0f));

    const auto& vecfieldData = vecfield.field;

    for (int y = 0; y < Ydim; ++y) {
        for (int x = 0; x < Xdim; ++x) {
            float accum_value = 0.0f;
            int accum_count = 0;

            // Trace forward
            vec2d<float> pos = { x * vecfield.gridInterval.x + vecfield.domainMinBoundary.x,
                y * vecfield.gridInterval.y + vecfield.domainMinBoundary.y };

            for (int i = 0; i < MaxIntegrationSteps; ++i) {
                float floatIndicesX = (pos.x - vecfield.domainMinBoundary.x) / vecfield.gridInterval.x;
                float floatIndicesY = (pos.y - vecfield.domainMinBoundary.y) / vecfield.gridInterval.y;

                if (!(0 <= floatIndicesX && floatIndicesX < Xdim && 0 <= floatIndicesY && floatIndicesY < Ydim)) {
                    break; // Stop if we move outside the texture bounds
                }
                accum_value += bilinear_interpolate(texture, floatIndicesX, floatIndicesY);
                accum_count += 1;
                vec2d<float> vec = bilinear_interpolate(vecfieldData, floatIndicesX, floatIndicesY);
                pos += vec * stepSize;
            }

            // Trace backward
            pos = { x * vecfield.gridInterval.x + vecfield.domainMinBoundary.x,
                y * vecfield.gridInterval.y + vecfield.domainMinBoundary.y };

            for (int i = 0; i < MaxIntegrationSteps; ++i) {
                float floatIndicesX = (pos.x - vecfield.domainMinBoundary.x) / vecfield.gridInterval.x;
                float floatIndicesY = (pos.y - vecfield.domainMinBoundary.y) / vecfield.gridInterval.y;
                if (!(0 <= floatIndicesX && floatIndicesX < Xdim && 0 <= floatIndicesY && floatIndicesY < Ydim)) {
                    break; // Stop if we move outside the texture bounds
                }
                accum_value += bilinear_interpolate(texture, floatIndicesX, floatIndicesY);
                accum_count += 1;
                vec2d<float> vec = bilinear_interpolate(vecfieldData, floatIndicesX, floatIndicesY);
                pos -= vec * stepSize;
            }

            // Compute the average value along the path
            if (accum_count > 0) {
                output_texture[y][x] = accum_value / accum_count;
            }
        }
    }

    return output_texture;
}

std::vector<std::pair<float, int>> generateNParamters(int n)
{
    std::vector<std::pair<float, int>> parameters = {
        { 1.0, 2 },
        { 1.0, 2 },
        { 1.0, 3 },
        { 1.0, 10 },
        { 2.0, 1 },
        { 2.0, 2 },
        { 2.0, 2 },
        { 2.0, 3 },
        { 2.0, 10 },
    };

    std::uniform_real_distribution<float> dist_double(0.01, 2.0);
    std::uniform_int_distribution<int> dist_int(1, 6);

    for (int i = 0; i < n; ++i) {
        float first = dist_double(rng);
        int second = dist_int(rng);
        parameters.emplace_back(first, second);
    }

    return parameters;
}

int main()
{

    using namespace std;
    const int Xdim = 128, Ydim = 128;
    vec2d<float> domainMinBoundary = { -2.0, -2.0 };
    vec2d<float> domainMaxBoundary = { 2.0, 2.0 };
    const float stepSize = 0.01;
    const int maxIteratioOneDirection = 256;
    int numVelocityFields = 3; // num of fields per n, rc parameter setting

    vec2d<float> gridInterval = {
        (domainMaxBoundary.x - domainMinBoundary.x) / (Xdim - 1),
        (domainMaxBoundary.y - domainMinBoundary.y) / (Ydim - 1)
    };

    const std::vector<std::pair<float, int>> paramters = generateNParamters(1);
    const auto licNoisetexture = randomNoiseTexture(Xdim, Ydim);

    uniform_real_distribution<float> genTheta(-1.0, 1.0);
    uniform_real_distribution<float> genSx(-2.0, 2.0);

    string root_folder = "../data/" + to_string(Xdim) + "_" + to_string(Xdim) + "/";
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

                auto resData = generator.generate(sx, sy, theta, j);
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
                {

                    cereal::JSONOutputArchive archive_o(out);
                    archive_o(CEREAL_NVP(Xdim));
                    archive_o(CEREAL_NVP(Ydim));
                    archive_o(CEREAL_NVP(domainMinBoundary));
                    archive_o(CEREAL_NVP(domainMaxBoundary));

                    archive_o(CEREAL_NVP(n));
                    archive_o(CEREAL_NVP(rc));

                    archive_o(CEREAL_NVP(theta));
                    archive_o(CEREAL_NVP(sx));
                    archive_o(CEREAL_NVP(sy));
                }

                // do not manually close file before creal deconstructor, as cereal will preprend a ]/} to finish json class/array
                out.close();

                std::ofstream outBin(velocityFilename, std::ios::binary);
                if (!outBin.good()) {
                    printf("couldn't open file: %s", velocityFilename.c_str());
                    return;
                }
                // write raw data

                cereal::BinaryOutputArchive archive_Binary(outBin);

                const std::vector<float> rawData = flatten2D(resData);
                archive_Binary(rawData);

                outBin.close();

                VectorField2D outField {
                    resData,
                    domainMinBoundary,
                    gridInterval
                };
                auto outputTexture = LICAlgorithm(licNoisetexture, outField, stepSize, maxIteratioOneDirection);

                saveAsPNG(outputTexture, licFilename);

            } // for j
        }
    });

    return 0;
}