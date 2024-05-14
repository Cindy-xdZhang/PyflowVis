
#include "VastistasVelocityGenerator.h"
#include "cereal/archives/binary.hpp"
#include "cereal/archives/json.hpp"
#include "cereal/types/vector.hpp"
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
    const size_t ydim = x2D.size();
    assert(ydim > 0);
    const size_t xdim = x2D[0].size();
    std::vector<float> result;
    result.resize(xdim * ydim * 2);
    for (size_t i = 0; i < ydim; i++)
        for (size_t j = 0; j < xdim; j++) {
            result[2 * (i * xdim + j)] = x2D[i][j].x;
            result[2 * (i * xdim + j) + 1] = x2D[i][j].y;
        }

    return result;
}

// The result image size is the same as the input texture.
std::vector<std::vector<float>> LICAlgorithm(
    const std::vector<std::vector<float>>& texture,
    const SteadyVectorField2D& vecfield,
    const int licImageSizeX,
    const int licImageSizeY,
    float stepSize,
    int MaxIntegrationSteps)
{
    const int YTexdim = texture.size();
    const int XTexdim = texture[0].size();
    const int Ydim = vecfield.field.size();
    const int Xdim = vecfield.field[0].size();
    assert(YTexdim == Ydim && Xdim == XTexdim);
    std::vector<std::vector<float>> output_texture(licImageSizeY, std::vector<float>(licImageSizeX, 0.0f));

    const auto& vecfieldData = vecfield.field;
    const vec2d<float> domainRange = vecfield.spatialDomainMaxBoundary - vecfield.spatialDomainMinBoundary;
    const float inverse_grid_interval_x = 1.0f / (float)vecfield.spatialGridInterval.x;
    const float inverse_grid_interval_y = 1.0f / (float)vecfield.spatialGridInterval.y;
    for (int y = 0; y < licImageSizeY; ++y) {
        for (int x = 0; x < licImageSizeX; ++x) {
            float accum_value = 0.0f;
            int accum_count = 0;

            // map position from texture image grid coordinate to vector field
            float ratio_x = (float)((float)x / (float)licImageSizeX);
            float ratio_y = (float)((float)y / (float)licImageSizeY);

            // Trace forward
            // physicalPositionInVectorfield
            vec2d<float> pos = { ratio_x * domainRange.x + vecfield.spatialDomainMinBoundary.x,
                ratio_y * domainRange.y + vecfield.spatialDomainMinBoundary.y };

            for (int i = 0; i < MaxIntegrationSteps; ++i) {
                float floatIndicesX = (pos.x - vecfield.spatialDomainMinBoundary.x) * inverse_grid_interval_x;
                float floatIndicesY = (pos.y - vecfield.spatialDomainMinBoundary.y) * inverse_grid_interval_y;

                if (!(0 <= floatIndicesX && floatIndicesX < Xdim && 0 <= floatIndicesY && floatIndicesY < Ydim)) {
                    break; // Stop if we move outside the texture bounds
                }
                accum_value += bilinear_interpolate(texture, floatIndicesX, floatIndicesY);
                accum_count += 1;
                vec2d<float> vec = bilinear_interpolate(vecfieldData, floatIndicesX, floatIndicesY);
                pos += vec * stepSize;
            }

            // Trace backward
            pos = { ratio_x * domainRange.x + vecfield.spatialDomainMinBoundary.x,
                ratio_y * domainRange.y + vecfield.spatialDomainMinBoundary.y };

            for (int i = 0; i < MaxIntegrationSteps; ++i) {
                float floatIndicesX = (pos.x - vecfield.spatialDomainMinBoundary.x) * inverse_grid_interval_x;
                float floatIndicesY = (pos.y - vecfield.spatialDomainMinBoundary.y) * inverse_grid_interval_y;
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

// todo: transform input field with respect to observer motion(v*=pullback(v-u))
template <typename T>
void referenceFrameTransformation(const SteadyVectorField2D& input_field, const int timesteps)
{

    auto integratePathlineOneStep = [](T x, T y, T t, T dt, T& x_new, T& y_new) {
        // RK4 integration step
        vec2d<float> odeStepStartPoint;
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
        vec2d<float> k1 = fieldFunction_u(odeStepStartPoint, t);

        // stage 2
        vec2d<float> stagePoint = odeStepStartPoint + k1 * a21 * h;
        vec2d<float> k2 = fieldFunction_u(stagePoint, t + c2 * h);

        // stage 3
        stagePoint = odeStepStartPoint + (a31 * k1 + a32 * k2) * h;
        vec2d<float> k3 = fieldFunction_u(stagePoint, t + c3 * h);

        // stage 4
        stagePoint = odeStepStartPoint + (a41 * k1 + a42 * k2 + a43 * k3) * h;
        vec2d<float> k4 = fieldFunction_u(stagePoint, t + c4 * h);

        vec2d<float> result_p = odeStepStartPoint + h * (k1 * b1 + k2 * b2 + k3 * b3 + k4 * b4);

        x_new = result_p(0);
        y_new = result_p(1);
    };

    //// integrationStep integrates starting from x, y, t, to target_t arriving at point x_new, y_new.
    //// the push_forward the matrix that does the transformation that a vector would undergo when going one step dt starting at t
    //// the push_forward is computed by draging a frame (2 vectors) along the curve.
    // auto integrationStep = [&integratePathlineStep, dx, dy](T x, T y, T t, T dt, T& x_new, T& y_new, Matrix22& push_forward) {
    //     vec2d<float> p_north;
    //     vec2d<float> p_south;
    //     vec2d<float> p_east;
    //     vec2d<float> p_west;

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

void generateUnsteadyField()
{
    using namespace std;
    const int Xdim = 64, Ydim = 64;
    const int LicImageSize = 512;
    vec2d<float> domainMinBoundary = { -2.0, -2.0 };
    vec2d<float> domainMaxBoundary = { 2.0, 2.0 };
    const float stepSize = 0.01;
    const int maxIteratioOneDirection = 128;
    int numVelocityFields = 1; // num of fields per n, rc parameter setting

    vec2d<float> gridInterval = {
        (domainMaxBoundary.x - domainMinBoundary.x) / (Xdim - 1),
        (domainMaxBoundary.y - domainMinBoundary.y) / (Ydim - 1)
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
                archive_o(CEREAL_NVP(domainMinBoundary));
                archive_o(CEREAL_NVP(domainMaxBoundary));

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
    vec2d<float> domainMinBoundary = { -2.0, -2.0 };
    vec2d<float> domainMaxBoundary = { 2.0, 2.0 };
    const float stepSize = 0.01;
    const int maxIteratioOneDirection = 256;
    int numVelocityFields = 1; // num of fields per n, rc parameter setting

    vec2d<float> gridInterval = {
        (domainMaxBoundary.x - domainMinBoundary.x) / (Xdim - 1),
        (domainMaxBoundary.y - domainMinBoundary.y) / (Ydim - 1)
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
                    archive_o(CEREAL_NVP(domainMinBoundary));
                    archive_o(CEREAL_NVP(domainMaxBoundary));

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

int main()
{

    generateUnsteadyField();

    return 0;
}