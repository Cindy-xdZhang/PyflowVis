
#include "flowGenerator.h"
#include "VastistasVelocityGenerator.h"
#include "VectorFieldCompute.h"
#include "cereal/archives/binary.hpp"
#include "cereal/archives/json.hpp"
#include "cereal/types/array.hpp"
#include "cereal/types/vector.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "observerGenerator.h"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include <execution>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <magic_enum/magic_enum.hpp>

#define RENDERING_LIC_SAVE_DATA
#define VALIDATE_RECONSTRUCTION_RESULT

using namespace std;
namespace {
constexpr double tmin = 0.0;
constexpr double tmax = M_PI * 0.25;
constexpr int Xdim = 16, Ydim = 16;
constexpr int unsteadyFieldTimeStep = 5;
constexpr int LicImageSize = 64;
Eigen::Vector2d domainMinBoundary = { -2.0, -2.0 };
Eigen::Vector2d domainMaxBoundary = { 2.0, 2.0 };
constexpr int LicSaveFrequency = 1; // every 2 time steps save one
const double stepSize = 0.012;
const int maxLICIteratioOneDirection = 256;
#if defined(DISABLE_CPP_PARALLELISM) || defined(_DEBUG)
auto policy = std::execution::seq;
#else
auto policy = std::execution::par_unseq;
#endif
}
inline std::string trimNumString(const std::string& numString)
{
    std::string str = numString;
    str.erase(str.find_last_not_of('0') + 1, std::string::npos);
    str.erase(str.find_last_not_of('.') + 1, std::string::npos);
    return str;
}
std::vector<std::vector<double>> loadPngFile(const std::string& filename, int& width, int& height)
{
    int n;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &n, 1); // Load image in grayscale
    if (!data) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return {};
    }

    std::vector<std::vector<double>> texture(height, std::vector<double>(width));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            texture[y][x] = data[y * width + x] / 255.0; // Normalize to [0, 1]
        }
    }

    stbi_image_free(data); // Free the image memory

    return texture;
}

void ConvertNoiseTextureImage2Text(const std::string& infilename, const std::string& outFile, int width, int height)
{
    auto texture = loadPngFile(infilename, width, height);

    std::ofstream out(outFile);
    if (!out.is_open()) {
        std::cerr << "Failed to open file for writing: " << outFile << std::endl;
        return;
    }
    constexpr int precision = 6;
    out << "constexpr std::array<std::array<double, 64>, 64> noiseTexture = {\n";
    for (const auto& row : texture) {
        std::string beginer_string = "    std::array<double," + std::to_string(width) + ">{";
        out << beginer_string;
        for (size_t x = 0; x < row.size(); ++x) {
            out << std::fixed << std::setprecision(precision) << row[x];
            if (x < row.size() - 1)
                out << ", ";
        }
        out << " },\n";
    }
    out << "};\n";

    out.close();
}

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

auto NoramlizeSpinTensor(Eigen::Matrix3d& input)
{
    Eigen::Vector3d unitAngular;
    unitAngular << input(2, 1), input(0, 2), input(1, 0);
    unitAngular.normalize();
    input << 0, -unitAngular(2), unitAngular(1),
        unitAngular(2), 0, -unitAngular(0),
        -unitAngular(1), unitAngular(0), 0;
    return;
};

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const
    {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};
std::vector<std::pair<double, double>> presetRCNParameters = {
    { 0.25, 2.0 },
    { 1.0, 2.0 },
    { 1.0, 3.0 },
    { 1.0, 5.0 },
    { 2.0, 1.0 },
    { 2.0, 2.0 },
    { 2.0, 3.0 },
    { 2.0, 10.0 },
};

std::vector<std::pair<double, double>> generateNParamters(int n, std::string mode)
{
    static std::unordered_set<std::pair<double, double>, pair_hash> unique_params;

    std::vector<std::pair<double, double>> parameters;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<double> dist_rc(1.87, 0.37); // mean = 1.87, stddev = 0.34
    std::normal_distribution<double> dist_n(1.96, 0.61); // mean = 1.96, stddev = 0.61

    int i = 0;
    while (parameters.size() < n) {

        if (i < presetRCNParameters.size() && mode == "train") {
            std::pair<double, double> preset_pair = presetRCNParameters.at(i++);
            if (unique_params.find(preset_pair) == unique_params.end()) {
                parameters.emplace_back(preset_pair);
                unique_params.insert(preset_pair);
            }
        } else {

            double rc = dist_rc(rng);
            double n = static_cast<double>(dist_n(rng));
            std::pair<double, double> new_pair = { rc, n };

            if (unique_params.find(new_pair) == unique_params.end()) {
                parameters.emplace_back(new_pair);
                unique_params.insert(new_pair);
            }
        }
    }

    return parameters;
}
std::pair<Eigen::Vector3d, Eigen::Vector3d> generateRandomABCVectors()
{
    // Random device and generator
    std::random_device rd;
    std::mt19937 gen(rd());
    // range of velocity and acc is -0.3-0.3, -0.01-0.01(from paper "robust reference frame...")
    std::uniform_real_distribution<double> dist(-0.3, 0.3);
    std::uniform_real_distribution<double> dist_acc(-0.01, 0.01);
    std::uniform_int_distribution<int> dist_int(0, 5);
    // Generate two random Eigen::Vector3d
    auto option = dist_int(gen);
    if (option == 0) {
        Eigen::Vector3d vec1(dist(gen), dist(gen), dist(gen));
        Eigen::Vector3d vec2(dist_acc(gen), dist_acc(gen), dist_acc(gen));
        return std::make_pair(vec1, vec2);
    } else if (option == 1) {
        Eigen::Vector3d vec1(dist(gen), dist(gen), 0);
        Eigen::Vector3d vec2(dist_acc(gen), dist_acc(gen), 0);
        return std::make_pair(vec1, vec2);
    } else if (option == 2) {
        Eigen::Vector3d vec1(dist(gen), dist(gen), 0);
        Eigen::Vector3d vec2(0, 0, 0);
        return std::make_pair(vec1, vec2);
    } else if (option == 3) {
        Eigen::Vector3d vec1(0, 0, dist(gen));
        Eigen::Vector3d vec2(0, 0, 0);
        return std::make_pair(vec1, vec2);
    } else {
        Eigen::Vector3d vec1(0, 0, dist(gen));
        Eigen::Vector3d vec2(0, 0, dist_acc(gen));
        return std::make_pair(vec1, vec2);
    }
}

// Function to convert a 2x2 rotation matrix to an angle
double matrix2angle(const Eigen::Matrix2d& rotationMat)
{
    // Ensure the matrix is orthogonal and its determinant is 1
    assert(rotationMat.determinant() > 0.999 && rotationMat.determinant() < 1.001);
    // Calculate the angle theta
    double theta = std::atan2(rotationMat(1, 0), rotationMat(0, 0));

    return theta;
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

    observerfield.spatialDomainMaxBoundary = inputField.getSpatialMaxBoundary();
    observerfield.spatialDomainMinBoundary = inputField.getSpatialMinBoundary();

    std::vector<Eigen::Vector2d> pathVelocitys;
    std::vector<Eigen::Vector3d> pathPositions;
    bool suc = PathhlineIntegrationRK4(StartPosition, observerfield, tmin, tmax, dt, pathVelocitys, pathPositions);
    assert(suc);

    int validPathSize = pathPositions.size();

    std::vector<Eigen::Matrix3d> observerRotationMatrices;
    observerRotationMatrices.resize(timestep);
    observerRotationMatrices[0] = Eigen::Matrix3d::Identity();

    std::vector<Eigen::Matrix4d> observertransformationMatrices;
    observertransformationMatrices.resize(timestep);
    observertransformationMatrices[0] = Eigen::Matrix4d::Identity();
    const auto observerStartPoint = pathPositions.at(0);

    for (size_t i = 1; i < validPathSize; i++) {
        const double t = observerfield.tmin + i * observerfield.dt;
        const Eigen::Vector3d abc = observerfield.killingABCfunc_(t);
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

    if (inputField.analyticalFlowfunc_) {
        const Eigen ::Vector2d Os = { pathPositions[0].x(), pathPositions[0].y() };
        resultField.Q_t.resize(timestep);
        resultField.c_t.resize(timestep);
        for (size_t i = 0; i < timestep; i++) {
            //  frame transformation is F(x):x*=Q(t)x+c(t)  or x*=T(Os) *Q*T(-Pt)*x
            //  =>F(x):x* = Q(t)*(x-pt)+Os= Qx-Q*pt+Os -> c=-Q*pt+Os  // => F^(-1)(x)= Q^T (x-c)= Q^T *( x+Q*pt-Os)
            resultField.Q_t[i] = observerRotationMatrices[i].transpose().block<2, 2>(0, 0);
            auto& Q_t = resultField.Q_t[i];
            const Eigen ::Vector2d position_t = { pathPositions[i].x(), pathPositions[i].y() };
            Eigen ::Vector2d c_t = Os - Q_t * position_t;
            resultField.c_t[i] = c_t;
        }

        // resultField.analyticalFlowfunc_ = [inputField, observerfield, resultField, dt, observerRotationMatrices, pathPositions](const Eigen::Vector2d& pos, double t) -> Eigen::Vector2d {
        //     double tmin = observerfield.tmin;
        //     const double floatingTimeStep = (t - tmin) / dt;
        //     const int timestep_floor = std::clamp((int)std::floor(floatingTimeStep), 0, observerfield.timeSteps - 1);
        //     const int timestep_ceil = std::clamp((int)std::floor(floatingTimeStep) + 1, 0, observerfield.timeSteps - 1);
        //     const double ratio = floatingTimeStep - timestep_floor;
        //    const Eigen::Matrix2d Q_t = resultField.Q_t[timestep_floor] * (1 - ratio) + resultField.Q_t[timestep_ceil] * ratio;
        //    auto Q_transpose = Q_t.transpose();
        //    auto c_t = resultField.c_t[timestep_floor] * (1 - ratio) + resultField.c_t[timestep_ceil] * ratio;
        //    // => F^(-1)(x)= Q^T (x-c)= Q^T *( x+Q*pt-Os)
        //    Eigen ::Vector2d F_inverse_x_2d = Q_transpose * (pos - c_t);
        //    auto v = inputField.getVectorAnalytical(F_inverse_x_2d, t);
        //    auto u = observerfield.getVector(F_inverse_x_2d(0), F_inverse_x_2d(1), t);
        //    Eigen::Vector2d vminusu_lab_frame;
        //    vminusu_lab_frame << v(0) - u(0), v(1) - u(1);
        //    // then w*(x,t)=Q(t)* w( F^-1(x),t )= Q(t)*w( Q^T *( x +Q*pt-Os),t )
        //    return Q_t * vminusu_lab_frame;
        //};
        resultField.analyticalFlowfunc_ = [inputField, observerfield, resultField, dt](const Eigen::Vector2d& pos, double t) -> Eigen::Vector2d {
            double tmin = observerfield.tmin;
            const double floatingTimeStep = (t - tmin) / dt;
            const int timestep_floor = std::clamp((int)std::floor(floatingTimeStep), 0, observerfield.timeSteps - 1);
            const int timestep_ceil = std::clamp((int)std::floor(floatingTimeStep) + 1, 0, observerfield.timeSteps - 1);
            const double ratio = floatingTimeStep - timestep_floor;

            const Eigen::Matrix2d Q_t = resultField.Q_t[timestep_floor] * (1 - ratio) + resultField.Q_t[timestep_ceil] * ratio;
            auto Q_transpose = Q_t.transpose();
            auto c_t = resultField.c_t[timestep_floor] * (1 - ratio) + resultField.c_t[timestep_ceil] * ratio;
            // => F^(-1)(x)= Q^T (x-c)= Q^T *( x+Q*pt-Os)
            Eigen ::Vector2d F_inverse_x_2d = Q_transpose * (pos - c_t);
            auto v = inputField.getVectorAnalytical(F_inverse_x_2d, t);
            auto u = observerfield.getVector(F_inverse_x_2d(0), F_inverse_x_2d(1), t);
            const Eigen::Vector3d abc = observerfield.killingABCfunc_(t);
            const auto c_ = abc(2);
            Eigen::Matrix2d Spintensor;
            Spintensor(0, 0) = 0.0;
            Spintensor(1, 0) = -c_;
            Spintensor(0, 1) = c_;
            Spintensor(1, 1) = 0.0;
            Eigen::Matrix2d Q_dot = Q_t * Spintensor;
            Eigen ::Vector2d translationTdot = { -abc.x(), -abc.y() };

            Eigen::Vector2d res = Q_t * v + Q_dot * F_inverse_x_2d + translationTdot;
            return res;
        };
        resultField.resampleFromAnalyticalExpression();
    } else {
        printf("error...");
    }
    return resultField;
}

// const Eigen::Vector3d& abc, const Eigen::Vector3d& abc_dot represents the xdot,ydot,theta_dot, xdotdot,ydotdot,theta_dotdot of paper "Roboust reference frame..."
UnSteadyVectorField2D Tobias_ObserverTransformation(const SteadyVectorField2D& inputField, const Eigen::Vector3d& abc, const Eigen::Vector3d& abc_dot, const double tmin, const double tmax, const int timestep)
{
    const auto dt = (tmax - tmin) / ((double)(timestep)-1.0);
    UnSteadyVectorField2D resultField;
    resultField.spatialDomainMaxBoundary = inputField.getSpatialMaxBoundary();
    resultField.spatialDomainMinBoundary = inputField.getSpatialMinBoundary();
    resultField.spatialGridInterval = inputField.spatialGridInterval;
    resultField.XdimYdim = inputField.XdimYdim;
    resultField.tmin = tmin;
    resultField.tmax = tmax;
    resultField.timeSteps = timestep;
    resultField.field.resize(timestep);

    // Q(0)=I ->theta(0)=0; translation(0)=0;
    std::vector<Eigen::Vector2d> Velocities(timestep);
    std::vector<double> AngularVelocities(timestep);
    resultField.Q_t.resize(timestep);
    resultField.c_t.resize(timestep);
    resultField.Q_t[0] = Eigen::Matrix2d::Identity();
    resultField.c_t[0] = Eigen::Vector2d::Zero();

    // rotation
    double theta = 0;
    double angularVelocity = abc(2);
    AngularVelocities[0] = { angularVelocity };

    // translation
    Eigen ::Vector2d translation_c_t = { 0, 0 };
    Eigen ::Vector2d translation_cdot = { abc(0), abc(1) };
    Velocities[0] = translation_cdot;
    Eigen ::Vector2d translation_cdotdot = { abc_dot(0), abc_dot(1) };

    for (size_t i = 1; i < timestep; i++) {
        theta = theta + dt * angularVelocity;
        angularVelocity = angularVelocity + dt * abc_dot(2);
        AngularVelocities[i] = angularVelocity;

        // translation
        translation_c_t = translation_c_t + dt * translation_cdot;
        translation_cdot = translation_cdot + dt * translation_cdotdot;
        Velocities[i] = translation_cdot;

        Eigen::Matrix2d rotQ;
        rotQ << cos(theta), -sin(theta),
            sin(theta), cos(theta);
        resultField.Q_t[i] = rotQ;
        resultField.c_t[i] = translation_c_t;
    }

    if (inputField.analyticalFlowfunc_) {

        resultField.analyticalFlowfunc_ = [inputField, tmin, resultField, dt, Velocities, AngularVelocities](const Eigen::Vector2d& pos, double t) -> Eigen::Vector2d {
            const double floatingTimeStep = (t - tmin) / dt;
            const int timestep_floor = std::clamp((int)std::floor(floatingTimeStep), 0, resultField.timeSteps - 1);
            const int timestep_ceil = std::clamp((int)std::floor(floatingTimeStep) + 1, 0, resultField.timeSteps - 1);
            const double ratio = floatingTimeStep - timestep_floor;

            const Eigen::Matrix2d Q_t = resultField.Q_t[timestep_floor] * (1 - ratio) + resultField.Q_t[timestep_ceil] * ratio;
            auto Q_transpose = Q_t.transpose();
            auto c_t = resultField.c_t[timestep_floor] * (1 - ratio) + resultField.c_t[timestep_ceil] * ratio;
            // => F^(-1)(x)= Q^T (x-c)= Q^T *( x+Q*pt-Os)
            Eigen ::Vector2d F_inverse_x_2d = Q_transpose * (pos - c_t);

            auto v = inputField.getVectorAnalytical(F_inverse_x_2d, t);
            auto Velocity = Velocities[timestep_floor] * (1 - ratio) + Velocities[timestep_ceil] * ratio;
            auto AngularVelocity = AngularVelocities[timestep_floor] * (1 - ratio) + AngularVelocities[timestep_ceil] * ratio;

            Eigen::Matrix2d Spintensor;
            Spintensor(0, 0) = 0.0;
            Spintensor(1, 0) = -AngularVelocity;
            Spintensor(0, 1) = AngularVelocity;
            Spintensor(1, 1) = 0.0;
            Eigen::Matrix2d Q_dot = Q_t * Spintensor;
            Eigen ::Vector2d translationTdot = Velocity;
            Eigen::Vector2d res = Q_t * v + Q_dot * F_inverse_x_2d + translationTdot;
            return res;
        };
        resultField.resampleFromAnalyticalExpression();
    } else {
        printf("error...");
    }
    return resultField;
}
// this function is similar to but 0
UnSteadyVectorField2D Tobias_reconstructUnsteadyField(const UnSteadyVectorField2D& inputField, const Eigen::Vector3d& abc, const Eigen::Vector3d& abc_dot)
{
    const auto tmax = inputField.tmax;
    const auto tmin = inputField.tmin;
    const double dt = (inputField.tmax - inputField.tmin) / ((double)inputField.timeSteps - 1.0);
    if (inputField.analyticalFlowfunc_) {

        UnSteadyVectorField2D outputField;
        outputField.spatialDomainMaxBoundary = inputField.getSpatialMaxBoundary();
        outputField.spatialDomainMinBoundary = inputField.getSpatialMinBoundary();
        outputField.spatialGridInterval = inputField.spatialGridInterval;
        outputField.XdimYdim = inputField.XdimYdim;
        outputField.tmin = tmin;
        outputField.tmax = tmax;
        outputField.timeSteps = inputField.timeSteps;
        outputField.field.resize(inputField.timeSteps);

        // Q(0)=I ->theta(0)=0; translation(0)=0;
        std::vector<Eigen::Vector2d> translation_c_t_list(inputField.timeSteps);
        std::vector<Eigen::Vector2d> Velocities(inputField.timeSteps);
        std::vector<double> AngularVelocities(inputField.timeSteps);
        std::vector<Eigen::Matrix2d> Q_t_list(inputField.timeSteps);
        translation_c_t_list[0] = { 0, 0 };
        // rotation
        double theta = 0;
        double angularVelocity = abc(2);
        AngularVelocities[0] = angularVelocity;
        Q_t_list[0] = Eigen::Matrix2d::Identity();

        // translation
        Eigen ::Vector2d translation_c_t = { 0, 0 };
        Eigen ::Vector2d translation_cdot = { abc(0), abc(1) };
        Velocities[0] = translation_cdot;
        Eigen ::Vector2d translation_cdotdot = { abc_dot(0), abc_dot(1) };
        for (size_t i = 1; i < inputField.timeSteps; i++) {
            theta = theta + dt * angularVelocity;
            angularVelocity = angularVelocity + dt * abc_dot(2);
            AngularVelocities[i] = angularVelocity;
            Eigen::Matrix2d rotQ;
            rotQ << cos(theta), -sin(theta),
                sin(theta), cos(theta);
            Q_t_list[i] = rotQ;

            // translation
            translation_c_t = translation_c_t + dt * translation_cdot;
            translation_cdot = translation_cdot + dt * translation_cdotdot;
            Velocities[i] = translation_cdot;
            translation_c_t_list[i] = translation_c_t;
        }

        outputField.analyticalFlowfunc_ = [=](const Eigen::Vector2d& posX, double t) -> Eigen::Vector2d {
            const double floatingTimeStep = (t - tmin) / dt;
            const int timestep_floor = std::clamp((int)std::floor(floatingTimeStep), 0, inputField.timeSteps - 1);
            const int timestep_ceil = std::clamp((int)std::floor(floatingTimeStep) + 1, 0, inputField.timeSteps - 1);
            const double ratio = floatingTimeStep - timestep_floor;

            const auto Q_t = Q_t_list[timestep_floor] * (1 - ratio) + Q_t_list[timestep_ceil] * ratio;
            const auto c_t = translation_c_t_list[timestep_floor] * (1 - ratio) + translation_c_t_list[timestep_ceil] * ratio;
            const auto Velocity = Velocities[timestep_floor] * (1 - ratio) + Velocities[timestep_ceil] * ratio;
            const auto AngularVelocity = AngularVelocities[timestep_floor] * (1 - ratio) + AngularVelocities[timestep_ceil] * ratio;

            const auto Q_transpose = Q_t.transpose();

            Eigen::Vector2d xStar = Q_t * posX + c_t;
            Eigen::Vector2d v_star_xstar = inputField.getVectorAnalytical({ xStar(0), xStar(1) }, t);

            // compute Qdot=Q(t)*Omega(t) where Omega(t) is the anti-symmetric matrix of the angular velocity vector
            Eigen::Matrix2d Spintensor;
            Spintensor(0, 0) = 0.0;
            Spintensor(1, 0) = -AngularVelocity;
            Spintensor(0, 1) = AngularVelocity;
            Spintensor(1, 1) = 0.0;

            Eigen::Matrix2d Q_dot = Q_t * Spintensor;
            Eigen ::Vector2d translationTdot = Velocity;
            Eigen::Vector2d v_at_pos = Q_transpose * (v_star_xstar - Q_dot * posX - translationTdot);
            return v_at_pos;
        };
        outputField.resampleFromAnalyticalExpression();
        return outputField;
    } else {
        printf("reconstructUnsteadyField only support analyticalFlowfunc_");
        return {};
    }
}
void addSegmentationVisualization(std::vector<std::vector<Eigen::Vector3d>>& inputLicImage, const SteadyVectorField2D& vectorField, const Eigen::Vector3d& meta_n_rc_si, const Eigen::Vector2d& domainMax, const Eigen::Vector2d& domainMIn, const Eigen::Vector2d& translation_t, const Eigen::Matrix2d& deformMat)
{
    // if si=0 then no vortex
    if (meta_n_rc_si.z() == 0.0) {
        return;
    }
    const auto rc = meta_n_rc_si(1);
    const auto deformInverse = deformMat.inverse();

    auto judgeVortex = [rc, translation_t, deformInverse](const Eigen::Vector2d& pos) -> bool {
        auto originalPos = deformInverse * (pos - translation_t);
        auto dx = rc - originalPos.norm();
        return dx > 0;
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
                if (velocity.norm() < 1e-7)
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
void addSegmentationVisualization(std::vector<std::vector<std::vector<Eigen::Vector3d>>>& inputLicImages, const UnSteadyVectorField2D& vectorField, const Eigen::Vector3d& meta_n_rc_si, const Eigen::Vector2d& domainMax, const Eigen::Vector2d& domainMIn, const Eigen::Vector2d& txy, const Eigen::Matrix2d& deformMat)
{
    //// if si=0 then no vortex
    // if (meta_n_rc_si.z() == 0.0) {
    //     return;
    // }
    // const auto rc = meta_n_rc_si(1);
    // const auto deformInverse = deformMat.inverse();

    // auto judgeVortex = [rc, translation_t = txy, deformInverse](const Eigen::Vector2d& pos) -> bool {
    //     auto originalPos = deformInverse * (pos - translation_t);
    //     auto dx = rc - originalPos.norm();
    //     return dx > 0;
    // };
    // const int licImageSizeY = inputLicImages[0].size();
    // const int licImageSizeX = inputLicImages[0][0].size();
    // const auto domainRange = domainMax - domainMIn;
    // const auto dx = domainRange(0) / licImageSizeX;
    // const auto dy = domainRange(1) / licImageSizeY;
    // const auto maxDistanceAnyPoint2gridPoints = sqrt((0.5 * dx) * (0.5 * dx) + (0.5 * dy) * (0.5 * dy));

    // assert(vectorField.Q_t.size() == vectorField.timeSteps);
    // assert(vectorField.c_t.size() == vectorField.timeSteps);
    // double dt = (vectorField.tmax - vectorField.tmin) / double(vectorField.timeSteps - 1);
    // for (size_t t = 0; t < vectorField.timeSteps; t++) {
    //     auto& inputLicImage = inputLicImages[t];
    //     auto Q_inverse = vectorField.Q_t[t].transpose();
    //     auto c_t = vectorField.c_t[t];
    //     double time = vectorField.tmin + t * dt;
    //     for (size_t i = 0; i < licImageSizeX; i++) {
    //         for (size_t j = 0; j < licImageSizeY; j++) {
    //             // map position from texture image grid coordinate to vector field
    //             double ratio_x = (double)((double)i / (double)licImageSizeX);
    //             double ratio_y = (double)((double)j / (double)licImageSizeY);

    //            Eigen::Vector3d posInObserverFrame = { ratio_x * domainRange(0) + domainMIn(0),
    //                ratio_y * domainRange(1) + domainMIn(1), 1.0 };

    //            Eigen ::Vector3d F_inverse_x = Q_inverse * (posInObserverFrame - c_t);
    //            F_inverse_x /= F_inverse_x(2);
    //            Eigen::Vector2d posInOriginalFrame = { F_inverse_x(0), F_inverse_x(1) };

    //            if (judgeVortex(posInOriginalFrame)) {
    //                auto preColor = inputLicImage[j][i];
    //                // red for critial point(coreline)
    //                auto velocity = vectorField.getVector(posInObserverFrame.x(), posInObserverFrame.y(), time);
    //                // !note: because txy is random core line center, then critical point==txy(velocity ==0.0) might not  lie on any grid points
    //                if (velocity.norm() < 1e-7)
    //                    [[unlikely]] {
    //                    inputLicImage[j][i] = 0.3 * preColor + 0.7 * Eigen::Vector3d(1.0, 0.0, 0.0);
    //                } else {
    //                    // yellow for vortex region
    //                    inputLicImage[j][i] = 0.3 * preColor + 0.7 * Eigen::Vector3d(1.0, 1.0, 0.0);
    //                }
    //            }
    //        }
    //    }
    //}
}

// this function is similar to but 0different from killingABCtransformation, this function assume inputField is some unsteady field get deformed by an observer represented by
//  predictKillingABCfunc, reconstructKillingDeformedUnsteadyField will remove the transformation imposed by predictKillingABCfunc.
UnSteadyVectorField2D reconstructKillingDeformedUnsteadyField(std::function<Eigen::Vector3d(double)> predictKillingABCfunc, const UnSteadyVectorField2D& inputField)
{
    const auto tmax = inputField.tmax;
    const auto tmin = inputField.tmin;
    const double dt = (inputField.tmax - inputField.tmin) / ((double)inputField.timeSteps - 1.0);
    if (inputField.analyticalFlowfunc_) {

        UnSteadyVectorField2D outputField;
        outputField.spatialDomainMaxBoundary = inputField.getSpatialMaxBoundary();
        outputField.spatialDomainMinBoundary = inputField.getSpatialMinBoundary();
        outputField.spatialGridInterval = inputField.spatialGridInterval;
        outputField.XdimYdim = inputField.XdimYdim;
        outputField.tmin = tmin;
        outputField.tmax = tmax;
        outputField.timeSteps = inputField.timeSteps;
        outputField.field.resize(inputField.timeSteps);

        outputField.analyticalFlowfunc_ = [=](const Eigen::Vector2d& posX, double t) -> Eigen::Vector2d {
            auto abc_t = predictKillingABCfunc(t);

            const double floatingTimeStep = (t - tmin) / dt;
            const int timestep_floor = std::clamp((int)std::floor(floatingTimeStep), 0, inputField.timeSteps - 1);
            const int timestep_ceil = std::clamp((int)std::floor(floatingTimeStep) + 1, 0, inputField.timeSteps - 1);
            const double ratio = floatingTimeStep - timestep_floor;

            const auto Q_t = inputField.Q_t[timestep_floor] * (1 - ratio) + inputField.Q_t[timestep_ceil] * ratio;
            const auto Q_transpose = Q_t.transpose();
            const auto c_t = inputField.c_t[timestep_floor] * (1 - ratio) + inputField.c_t[timestep_ceil] * ratio;

            Eigen::Vector2d xStar = Q_t * posX + c_t;
            Eigen::Vector2d v_star_xstar = inputField.getVectorAnalytical({ xStar(0), xStar(1) }, t);

            // compute Qdot=Q(t)*Omega(t) where Omega(t) is the anti-symmetric matrix of the angular velocity vector
            Eigen::Matrix2d Spintensor;
            Spintensor(0, 0) = 0.0;
            Spintensor(1, 0) = -abc_t(2);
            Spintensor(0, 1) = abc_t(2);
            Spintensor(1, 1) = 0.0;

            Eigen::Matrix2d Q_dot = Q_t * Spintensor;
            Eigen ::Vector2d translationTdot = { -abc_t.x(), -abc_t.y() };
            Eigen::Vector2d v_at_pos = Q_transpose * (v_star_xstar - Q_dot * posX - translationTdot);
            return v_at_pos;
        };
        outputField.resampleFromAnalyticalExpression();
        return outputField;
    } else {
        printf("reconstructUnsteadyField only support analyticalFlowfunc_");
        return {};
    }
}

// number of result traing data = Nparamters * samplePerParameters * observerPerSetting;dataSetSplitTag should be "train"/"test"/"validation"
void generateUnsteadyField(int Nparamters, int samplePerParameters, int observerPerSetting, const std::string in_root_fodler, const std::string dataSetSplitTag)
{

    // check datasplittag is "train"/"test"/"validation"
    if (dataSetSplitTag != "train" && dataSetSplitTag != "test" && dataSetSplitTag != "validation") {
        printf("dataSetSplitTag should be \"train\"/\"test\"/\"validation\"");
        return;
    }

    int numVelocityFields = samplePerParameters; // num of fields per n, rc parameter setting
    std::string root_folder = in_root_fodler + "/X" + to_string(Xdim) + "_Y" + to_string(Ydim) + "_T" + to_string(unsteadyFieldTimeStep) + "_no_mixture/" + dataSetSplitTag + "/";
    if (!filesystem::exists(root_folder)) {
        filesystem::create_directories(root_folder);
    }

    Eigen::Vector2d gridInterval = {
        (domainMaxBoundary(0) - domainMinBoundary(0)) / (Xdim - 1),
        (domainMaxBoundary(1) - domainMinBoundary(1)) / (Ydim - 1)
    };

    const auto paramters = generateNParamters(Nparamters, dataSetSplitTag);

    std::normal_distribution<double> genTheta(0.0, 0.50); // rotation angle's distribution
    // normal distribution from supplementary material of Vortex Boundary Identification Paper
    std::normal_distribution<double> genSx(0, 3.59);
    std::normal_distribution<double> genSy(0, 2.24);
    std::normal_distribution<double> genTx(0.0, 1.34);
    std::normal_distribution<double> genTy(0.0, 1.27);
    double minMagintude = INFINITY;
    double maxMagintude = -INFINITY;

    for_each(policy, paramters.begin(), paramters.cend(), [&](const std::pair<double, double>& params) {
        const double rc = params.first;
        const double n = params.second;
        std::string str_Rc = trimNumString(std::to_string(rc));
        std::string str_n = trimNumString(std::to_string(n));

        VastistasVelocityGenerator generator(Xdim, Ydim, domainMinBoundary, domainMaxBoundary, rc, n);
        int totalSamples = numVelocityFields * observerPerSetting;
        printf("generate %d sample for rc=%f , n=%f \n", totalSamples, rc, n);

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

        std::mt19937 rngSample(static_cast<unsigned int>(std::time(nullptr)));
        for (size_t sample = 0; sample < numVelocityFields; sample++) {
            // the type of this sample(divergence,cw vortex, cc2 vortex)
            auto Si = static_cast<VastisVortexType>(sample % 3);

            const auto theta = genTheta(rngSample);
            auto sx = genSx(rngSample);
            auto sy = genSy(rngSample);
            while (sx * sy == 0.0) {
                sx = genSx(rngSample);
                sy = genSy(rngSample);
            }

            auto tx = genTx(rngSample);
            auto ty = genTy(rngSample);
            // clamp tx ty to 0.5*domian
            tx = std::clamp(tx, 0.3 * domainMinBoundary.x(), 0.3 * domainMaxBoundary.x());
            ty = std::clamp(ty, 0.3 * domainMinBoundary.y(), 0.3 * domainMaxBoundary.y());
            Eigen::Vector2d txy = { tx, ty };

            Eigen::Vector3d n_rc_si = { n, rc, (double)Si };
            const SteadyVectorField2D steadyField = generator.generateSteadyField_VortexBoundaryVIS2020(tx, ty, sx, sy, theta, Si);

            const auto& steadyFieldResData = steadyField.field;
            for (size_t observerIndex = 0; observerIndex < observerPerSetting; observerIndex++) {
                printf(".");

                const int taskSampleId = sample * observerPerSetting + observerIndex;

                const string vortexTypeName = string { magic_enum::enum_name<VastisVortexType>(Si) };
                const string sample_tag_name
                    = "sample_" + to_string(taskSampleId) + vortexTypeName;
                string metaFilename = task_folder + sample_tag_name + "meta.json";
                string velocityFilename = task_folder + sample_tag_name + ".bin";
                const std::vector<float> rawSteadyData = flatten2DAs1Dfloat(steadyFieldResData);
                const auto& observerParameters = generateRandomABCVectors();
                const auto& abc = observerParameters.first;
                const auto& abc_dot = observerParameters.second;
                /* auto func = KillingComponentFunctionFactory::arbitrayObserver(abc, abc_dot);
               KillingAbcField observerfieldDeform(  func, unsteadyFieldTimeStep, tmin, tmax);*/
                auto unsteady_field = Tobias_ObserverTransformation(steadyField, abc, abc_dot, tmin, tmax, unsteadyFieldTimeStep);
                // reconstruct unsteady field from observer field
                auto reconstruct_field = Tobias_reconstructUnsteadyField(unsteady_field, abc, abc_dot);
#ifdef VALIDATE_RECONSTRUCTION_RESULT

                // //validate reconstruction result
                for (size_t rec = 0; rec < unsteadyFieldTimeStep; rec++) {
                    const auto& reconstruct_slice = reconstruct_field.field[rec];
                    // compute reconstruct slice difference with steady field
                    double diffSum = 0.0;
                    for (size_t y = 1; y < Ydim - 1; y++)
                        for (size_t x = 1; x < Xdim - 1; x++) {
                            auto diff = reconstruct_slice[y][x] - steadyFieldResData[y][x];
                            diffSum += diff.norm();
                        }
                    double tolerance = (Xdim - 2) * (Ydim - 2) * 0.0001;
                    // has debug, major reason for reconstruction failure is velocity too big make observer transformation query value from region out of boundary
                    if (diffSum > tolerance) {
                        printf("\n\n");
                        printf("\n reconstruct field not equal to steady field at step %u\n", (unsigned int)rec);
                        printf("\n\n");
                    }
                }
#endif
#ifdef RENDERING_LIC_SAVE_DATA

                {
                    auto outputSteadyTexture = LICAlgorithm(steadyField, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
                    // add segmentation visualization for steady lic
                    Eigen::Matrix2d deformMat = Eigen::Matrix2d::Identity();
                    deformMat(0, 0) = sx * cos(theta);
                    deformMat(0, 1) = -sy * sin(theta);
                    deformMat(1, 0) = sx * sin(theta);
                    deformMat(1, 1) = sy * cos(theta);
                    addSegmentationVisualization(outputSteadyTexture, steadyField, n_rc_si, domainMaxBoundary, domainMinBoundary, txy, deformMat);
                    string steadyField_name = "steady_beforeTransformation_";
                    string licFilename0 = task_licfolder + sample_tag_name + steadyField_name + "lic.png";
                    saveAsPNG(outputSteadyTexture, licFilename0);

                    auto outputTextures = LICAlgorithm_UnsteadyField(unsteady_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
                    auto outputTexturesReconstruct = LICAlgorithm_UnsteadyField(reconstruct_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
                    addSegmentationVisualization(outputTextures, unsteady_field, n_rc_si, domainMaxBoundary, domainMinBoundary, txy, deformMat);
                    for (size_t i = 0; i < outputTextures.size(); i += LicSaveFrequency) {
                        string tag_name = sample_tag_name + "killing_deformed_" + std::to_string(i);
                        string licFilename = task_licfolder + tag_name + "lic.png";

                        saveAsPNG(outputTextures[i], licFilename);

                        string tag_name_rec = sample_tag_name + "reconstruct_" + std::to_string(i);
                        string licFilename_rec = task_licfolder + tag_name_rec + "lic.png";
                        saveAsPNG(outputTexturesReconstruct[i], licFilename_rec);
                    }
                    auto rawUnsteadyFieldData = flatten3DAs1Dfloat(unsteady_field.field);
                    auto [minV, maxV] = computeMinMax(rawUnsteadyFieldData);
                    if (minV < minMagintude) {
                        minMagintude = minV;
                    }
                    if (maxV > maxMagintude) {
                        maxMagintude = maxV;
                    }
                    // save meta info:
                    std::ofstream jsonOut(metaFilename);
                    if (!jsonOut.good()) {
                        printf("couldn't open file: %s", metaFilename.c_str());
                        return;
                    }
                    {
                        cereal::JSONOutputArchive archive_o(jsonOut);

                        Eigen::Vector3d deform = { theta, sx, sy };
                        archive_o(cereal::make_nvp("n_rc_Si", n_rc_si));
                        archive_o(cereal::make_nvp("txy", txy));
                        archive_o(cereal::make_nvp("deform_theta_sx_sy", deform));

                        archive_o(cereal::make_nvp("observer_abc", abc));
                        archive_o(cereal::make_nvp("observer_abc_dot", abc_dot));

                        std::vector<double> thetaObserver;
                        thetaObserver.reserve(unsteadyFieldTimeStep);
                        std::vector<std::array<double, 2>> c_t;
                        c_t.reserve(unsteadyFieldTimeStep);
                        thetaObserver.reserve(unsteadyFieldTimeStep);
                        for (size_t i = 0; i < unsteadyFieldTimeStep; i++) {

                            Eigen::Matrix2d Q_t2d_t = unsteady_field.Q_t[i];
                            const double theta_rot = matrix2angle(Q_t2d_t);
                            thetaObserver.push_back(theta_rot);
                            c_t.push_back({ unsteady_field.c_t[i].x(), unsteady_field.c_t[i].y() });
                        }
                        // theta is exponnetial to killing c(angular velocity)
                        archive_o(cereal::make_nvp("theta(t)", thetaObserver));
                        // archive_o(cereal::make_nvp("c(t)", c_t));
                    }
                    // do not manually close file before creal deconstructor, as cereal will preprend a ]/} to finish json class/array
                    jsonOut.close();
                    std::ofstream outBin(velocityFilename, std::ios::binary);
                    if (!outBin.good()) {
                        printf("couldn't open file: %s", velocityFilename.c_str());
                        return;
                    }

                    cereal::BinaryOutputArchive archive_Binary(outBin);
                    // write raw data
                    // ar(make_size_tag(static_cast<size_type=uint64>(vector.size()))); // number of elements
                    // ar(binary_data(vector.data(), vector.size() * sizeof(T)));
                    //  when using other library to read, need to discard the first uint64 (8bytes.)
                    archive_Binary(rawUnsteadyFieldData);
                    outBin.close();
                }
#endif
            } // for (size_t observerIndex = 0..)

        } // for sample
    });

    // create Root meta json file, save plane information here instead of every sample's meta file
    string taskFolder_rootMetaFilename = root_folder + "meta.json";
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
        // save min and max
        archive_o(cereal::make_nvp("minV", minMagintude));
        archive_o(cereal::make_nvp("maxV", maxMagintude));
    }
}

void testCriterion()
{
    const int LicImageSize = 256;
    const int unsteadyFieldTimeStep = 32;
    const double stepSize = 0.01;
    const int maxLICIteratioOneDirection = 256;
    int numVelocityFields = 1; // num of fields per n, rc parameter setting
    const int LicSaveFrequency = 2;
    std::string root_folder = "../data/test_criterion/";
    if (!filesystem::exists(root_folder)) {
        filesystem::create_directories(root_folder);
    }
    // Create an instance of AnalyticalFlowCreator
    Eigen::Vector2i grid_size(Xdim, Ydim);
    int time_steps = unsteadyFieldTimeStep;

    AnalyticalFlowCreator flowCreator(grid_size, time_steps, domainMinBoundary, domainMaxBoundary);

    // Create the flow field
    const auto rfc = flowCreator.createRFC();
    const auto breads = flowCreator.createBeadsFlow();
    const auto beadsNC = flowCreator.createBeadsFlowNoContraction();
    const UnSteadyVectorField2D grye = flowCreator.createUnsteadyGyre();

    // add segmentation visualization for lic
    std::vector<std::pair<VORTEX_CRITERION, std::string>> enumCriterion = {
        { VORTEX_CRITERION::Q_CRITERION, "Q_CRITERION" },
        { VORTEX_CRITERION::CURL, "CURL" },
        { VORTEX_CRITERION::IVD_CRITERION, "IVD_CRITERION" },
        { VORTEX_CRITERION::LAMBDA2_CRITERION, "LAMBDA2_CRITERION" },
        { VORTEX_CRITERION::DELTA_CRITERION, "DELTA_CRITERION" },
        { VORTEX_CRITERION::SUJUDI_HAIMES_CRITERION, "SUJUDI_HAIMES" },
        { VORTEX_CRITERION::SOBEL_EDGE_DETECTION, "SOBEL_EDGE" }
    };

    for (size_t i = 0; i < enumCriterion.size(); i++) {
        auto criterion = enumCriterion[i].first;
        auto name_criterion = enumCriterion[i].second;

        auto outputTexturesRfc = LICAlgorithm_UnsteadyField(rfc, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection, criterion);
        auto outputTexturesBeads = LICAlgorithm_UnsteadyField(breads, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection, criterion);
        auto outputTexturesBeadsNc = LICAlgorithm_UnsteadyField(beadsNC, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection, criterion);
        auto outputTexturesGype = LICAlgorithm_UnsteadyField(grye, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection, criterion);

        for (size_t i = 0; i < unsteadyFieldTimeStep; i += LicSaveFrequency) {
            string tag_name0 = "rfc_" + name_criterion + std::to_string(i);
            string licFilename0 = root_folder + tag_name0 + "lic.png";
            saveAsPNG(outputTexturesRfc[i], licFilename0);

            string tag_name = "beads_" + name_criterion + std::to_string(i);
            string licFilename = root_folder + tag_name + "lic.png";
            saveAsPNG(outputTexturesBeads[i], licFilename);

            tag_name = "beadsNc_" + name_criterion + std::to_string(i);
            licFilename = root_folder + tag_name + "lic.png";
            saveAsPNG(outputTexturesBeadsNc[i], licFilename);

            tag_name = "gyre_" + name_criterion + std::to_string(i);
            licFilename = root_folder + tag_name + "lic.png";
            saveAsPNG(outputTexturesGype[i], licFilename);
        }
    }
}

//
// void testKillingTransformationForRFC()
//{
//    const int LicImageSize = 128;
//    const int unsteadyFieldTimeStep = 32;
//    const double stepSize = 0.01;
//    const int maxLICIteratioOneDirection = 256;
//    int numVelocityFields = 1; // num of fields per n, rc parameter setting
//    std::string root_folder = "../data/unsteady/" + to_string(Xdim) + "_" + to_string(Xdim) + "/" + "test/";
//    if (!filesystem::exists(root_folder)) {
//        filesystem::create_directories(root_folder);
//    }
//    // Create an instance of AnalyticalFlowCreator
//    Eigen::Vector2i grid_size(Xdim, Ydim);
//    int time_steps = unsteadyFieldTimeStep;
//
//    AnalyticalFlowCreator flowCreator(grid_size, time_steps, domainMinBoundary, domainMaxBoundary);
//
//    // Create the flow field
//    UnSteadyVectorField2D InputflowField = flowCreator.createRFC();
//
//    auto func_const_trans = KillingComponentFunctionFactory::constantRotation(-1.0);
//
//    KillingAbcField observerfield(
//        func_const_trans, unsteadyFieldTimeStep, tmin, tmax);
//    Eigen::Vector2d StartPosition = { 0.0, 0.0 };
//
//    auto unsteady_field = killingABCtransformation(observerfield, StartPosition, InputflowField);
//    auto resample_observerfield = observerfield.resample2UnsteadyField(grid_size, domainMinBoundary, domainMaxBoundary);
//    // auto outputObserverFieldLic = LICAlgorithm_UnsteadyField(licNoisetexture, resample_observerfield, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
//    auto inputTextures = LICAlgorithm_UnsteadyField(InputflowField, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection, VORTEX_CRITERION::NONE);
//
//    auto outputTexturesObservedField = LICAlgorithm_UnsteadyField(unsteady_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection, VORTEX_CRITERION::NONE);
//
//    for (size_t i = 0; i < unsteadyFieldTimeStep; i++) {
//        string tag_name0 = "inputField_" + std::to_string(i);
//        string licFilename0 = root_folder + tag_name0 + "lic.png";
//        saveAsPNG(inputTextures[i], licFilename0);
//
//        string tag_name1 = "observerField_" + std::to_string(i);
//        string licFilename1 = root_folder + tag_name1 + "lic.png";
//        // saveAsPNG(outputObserverFieldLic[i], licFilename1);
//
//        string tag_name = "killing_transformed_" + std::to_string(i);
//        string licFilename = root_folder + tag_name + "lic.png";
//        saveAsPNG(outputTexturesObservedField[i], licFilename);
//    }
//}

// reproduce paper : Vortex Boundary Identification using Convolutional Neural Network
void GenerateSteadyVortexBoundary(int Nparamters, int samplePerParameters, const std::string in_root_fodler, const std::string dataSetSplitTag)
{

    // check datasplittag is "train"/"test"/"validation"
    if (dataSetSplitTag != "train" && dataSetSplitTag != "test" && dataSetSplitTag != "validation") {
        printf("dataSetSplitTag should be \"train\"/\"test\"/\"validation\"");
        return;
    }
    int numVelocityFields = samplePerParameters; // num of fields per n, rc parameter setting
    std::string root_folder = in_root_fodler + "/X" + to_string(Xdim) + "_Y" + to_string(Ydim) + "_T" + to_string(unsteadyFieldTimeStep) + "Steady/" + dataSetSplitTag + "/";
    if (!filesystem::exists(root_folder)) {
        filesystem::create_directories(root_folder);
    }
    Eigen::Vector2d gridInterval = {
        (domainMaxBoundary(0) - domainMinBoundary(0)) / (Xdim - 1),
        (domainMaxBoundary(1) - domainMinBoundary(1)) / (Ydim - 1)
    };
    const auto paramters = generateNParamters(Nparamters, dataSetSplitTag);
    std::normal_distribution<double> genTheta(0.0, 0.50); // rotation angle's distribution
    // normal distribution from supplementary material of Vortex Boundary Identification Paper
    std::normal_distribution<double> genSx(0, 3.59);
    std::normal_distribution<double> genSy(0, 2.24);
    std::normal_distribution<double> genTx(0.0, 1.34);
    std::normal_distribution<double> genTy(0.0, 1.27);
    double minMagintude = INFINITY;
    double maxMagintude = -INFINITY;

    for_each(policy, paramters.begin(), paramters.cend(), [&](const std::pair<double, double>& params) {
        const double rc = params.first;
        const double n = params.second;
        std::string str_Rc = trimNumString(std::to_string(rc));
        std::string str_n = trimNumString(std::to_string(n));
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

        std::mt19937 rngSample(static_cast<unsigned int>(std::time(nullptr)));
        for (size_t sample = 0; sample < numVelocityFields; sample++) {
            // the type of this sample(divergence,cw vortex, cc2 vortex)
            auto Si = static_cast<VastisVortexType>(sample % 3);

            const auto theta = genTheta(rngSample);
            auto sx = genSx(rngSample);
            auto sy = genSy(rngSample);
            while (sx * sy == 0.0) {
                sx = genSx(rngSample);
                sy = genSy(rngSample);
            }

            auto tx = genTx(rngSample);
            auto ty = genTy(rngSample);
            // clamp tx ty to 0.5*domian
            tx = std::clamp(tx, 0.3 * domainMinBoundary.x(), 0.3 * domainMaxBoundary.x());
            ty = std::clamp(ty, 0.3 * domainMinBoundary.y(), 0.3 * domainMaxBoundary.y());
            Eigen::Vector2d txy = { tx, ty };

            Eigen::Vector3d n_rc_si = { n, rc, (double)Si };
            const SteadyVectorField2D steadyField = generator.generateSteadyField_VortexBoundaryVIS2020(tx, ty, sx, sy, theta, Si);
            const std::vector<std::vector<Eigen::Vector2d>>& steadyFieldResData = steadyField.field;

            printf(".");
            const int taskSampleId = sample;
            const string vortexTypeName = string { magic_enum::enum_name<VastisVortexType>(Si) };
            const string sample_tag_name
                = "sample_" + to_string(taskSampleId) + vortexTypeName;

#ifdef RENDERING_LIC_SAVE_DATA

            {
                auto outputSteadyTexture = LICAlgorithm(steadyField, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
                // add segmentation visualization for steady lic
                Eigen::Matrix2d deformMat = Eigen::Matrix2d::Identity();
                deformMat(0, 0) = sx * cos(theta);
                deformMat(0, 1) = -sy * sin(theta);
                deformMat(1, 0) = sx * sin(theta);
                deformMat(1, 1) = sy * cos(theta);
                // addSegmentationVisualization(outputSteadyTexture, steadyField, n_rc_si, domainMaxBoundary, domainMinBoundary, txy, deformMat);
                string steadyField_name = "steady_";
                string licFilename0 = task_licfolder + sample_tag_name + steadyField_name + "lic.png";
                saveAsPNG(outputSteadyTexture, licFilename0);

                auto rawSteadyFieldData = flatten2DAs1Dfloat(steadyField.field);
                auto [minV, maxV] = computeMinMax(rawSteadyFieldData);
                if (minV < minMagintude) {
                    minMagintude = minV;
                }
                if (maxV > maxMagintude) {
                    maxMagintude = maxV;
                }

                string metaFilename = task_folder + sample_tag_name + "meta.json";
                string velocityFilename = task_folder + sample_tag_name + ".bin";
                // save meta info:
                std::ofstream jsonOut(metaFilename);
                if (!jsonOut.good()) {
                    printf("couldn't open file: %s", metaFilename.c_str());
                    return;
                }
                {
                    cereal::JSONOutputArchive archive_o(jsonOut);

                    Eigen::Vector3d deform = { theta, sx, sy };
                    archive_o(cereal::make_nvp("n_rc_Si", n_rc_si));
                    archive_o(cereal::make_nvp("txy", txy));
                    archive_o(cereal::make_nvp("deform_theta_sx_sy", deform));
                }
                // do not manually close file before creal deconstructor, as cereal will preprend a ]/} to finish json class/array
                jsonOut.close();

                std::ofstream outBin(velocityFilename, std::ios::binary);
                if (!outBin.good()) {
                    printf("couldn't open file: %s", velocityFilename.c_str());
                    return;
                }
                cereal::BinaryOutputArchive archive_Binary(outBin);
                archive_Binary(rawSteadyFieldData);
                outBin.close();
            }
#endif

        } // for sample
    });

    // create Root meta json file, save plane information here instead of every sample's meta file
    string taskFolder_rootMetaFilename = root_folder + "meta.json";
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
        // save min and max
        archive_o(cereal::make_nvp("minV", minMagintude));
        archive_o(cereal::make_nvp("maxV", maxMagintude));
    }
}

// number of result traing data = Nparamters * samplePerParameters * observerPerSetting;dataSetSplitTag should be "train"/"test"/"validation"
void generateUnsteadyFieldPathline(int Nparamters, int samplePerParameters, int observerPerSetting, const std::string in_root_fodler, const std::string dataSetSplitTag)
{

    // check datasplittag is "train"/"test"/"validation"
    if (dataSetSplitTag != "train" && dataSetSplitTag != "test" && dataSetSplitTag != "validation") {
        printf("dataSetSplitTag should be \"train\"/\"test\"/\"validation\"");
        return;
    }

    int numVelocityFields = samplePerParameters; // num of fields per n, rc parameter setting
    std::string root_folder = in_root_fodler + "/X" + to_string(Xdim) + "_Y" + to_string(Ydim) + "_T" + to_string(unsteadyFieldTimeStep) + "_no_mixture/" + dataSetSplitTag + "/";
    if (!filesystem::exists(root_folder)) {
        filesystem::create_directories(root_folder);
    }

    Eigen::Vector2d gridInterval = {
        (domainMaxBoundary(0) - domainMinBoundary(0)) / (Xdim - 1),
        (domainMaxBoundary(1) - domainMinBoundary(1)) / (Ydim - 1)
    };

    const auto paramters = generateNParamters(Nparamters, dataSetSplitTag);

    std::normal_distribution<double> genTheta(0.0, 0.50); // rotation angle's distribution
    // normal distribution from supplementary material of Vortex Boundary Identification Paper
    std::normal_distribution<double> genSx(0, 3.59);
    std::normal_distribution<double> genSy(0, 2.24);
    std::normal_distribution<double> genTx(0.0, 1.34);
    std::normal_distribution<double> genTy(0.0, 1.27);
    double minMagintude = INFINITY;
    double maxMagintude = -INFINITY;

    for_each(policy, paramters.begin(), paramters.cend(), [&](const std::pair<double, double>& params) {
        const double rc = params.first;
        const double n = params.second;
        std::string str_Rc = trimNumString(std::to_string(rc));
        std::string str_n = trimNumString(std::to_string(n));

        VastistasVelocityGenerator generator(Xdim, Ydim, domainMinBoundary, domainMaxBoundary, rc, n);
        int totalSamples = numVelocityFields * observerPerSetting;
        printf("generate %d sample for rc=%f , n=%f \n", totalSamples, rc, n);

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

        std::mt19937 rngSample(static_cast<unsigned int>(std::time(nullptr)));
        for (size_t sample = 0; sample < numVelocityFields; sample++) {
            // the type of this sample(divergence,cw vortex, cc2 vortex)
            auto Si = static_cast<VastisVortexType>(sample % 3);

            const auto theta = genTheta(rngSample);
            auto sx = genSx(rngSample);
            auto sy = genSy(rngSample);
            while (sx * sy == 0.0) {
                sx = genSx(rngSample);
                sy = genSy(rngSample);
            }

            auto tx = genTx(rngSample);
            auto ty = genTy(rngSample);
            // clamp tx ty to 0.5*domian
            tx = std::clamp(tx, 0.3 * domainMinBoundary.x(), 0.3 * domainMaxBoundary.x());
            ty = std::clamp(ty, 0.3 * domainMinBoundary.y(), 0.3 * domainMaxBoundary.y());
            Eigen::Vector2d txy = { tx, ty };

            Eigen::Vector3d n_rc_si = { n, rc, (double)Si };
            const SteadyVectorField2D steadyField = generator.generateSteadyField_VortexBoundaryVIS2020(tx, ty, sx, sy, theta, Si);
            const auto& steadyFieldResData = steadyField.field;
            for (size_t observerIndex = 0; observerIndex < observerPerSetting; observerIndex++) {
                printf(".");

                const int taskSampleId = sample * observerPerSetting + observerIndex;

                const string vortexTypeName = string { magic_enum::enum_name<VastisVortexType>(Si) };
                const string sample_tag_name
                    = "sample_" + to_string(taskSampleId) + vortexTypeName;
                string metaFilename = task_folder + sample_tag_name + "meta.json";
                string velocityFilename = task_folder + sample_tag_name + ".bin";
                const std::vector<float> rawSteadyData = flatten2DAs1Dfloat(steadyFieldResData);
                const auto& observerParameters = generateRandomABCVectors();
                const auto& abc = observerParameters.first;
                const auto& abc_dot = observerParameters.second;
                /* auto func = KillingComponentFunctionFactory::arbitrayObserver(abc, abc_dot);
               KillingAbcField observerfieldDeform(  func, unsteadyFieldTimeStep, tmin, tmax);*/
                UnSteadyVectorField2D unsteady_field = Tobias_ObserverTransformation(steadyField, abc, abc_dot, tmin, tmax, unsteadyFieldTimeStep);
                Eigen::Vector2d txy0 = { tx, ty };
                std::vector<Eigen::Vector2d> pathVelocitys;
                std::vector<Eigen::Vector3d> pathPositions;
                auto suc = PathhlineIntegrationRK4(txy0, unsteady_field, 0, tmax, tmax / 100.0, pathVelocitys, pathPositions);
                assert(suc);

            } // for (size_t observerIndex = 0..)

        } // for sample
    });

    // create Root meta json file, save plane information here instead of every sample's meta file
    string taskFolder_rootMetaFilename = root_folder + "meta.json";
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
        // save min and max
        archive_o(cereal::make_nvp("minV", minMagintude));
        archive_o(cereal::make_nvp("maxV", maxMagintude));
    }
}

// reproduce paper : Robust Reference Frame Extraction from Unsteady 2D Vector
Fields with Convolutional Neural Networks void generateUnsteadyFieldMixture(int Nfield, int observerPerField, const std::string in_root_fodler, const std::string dataSetSplitTag)
{

    // check datasplittag is "train"/"test"/"validation"
    if (dataSetSplitTag != "train" && dataSetSplitTag != "test" && dataSetSplitTag != "validation") {
        printf("dataSetSplitTag should be \"train\"/\"test\"/\"validation\"");
        return;
    }

    int numVelocityFields = Nfield; // num of fields per n, rc parameter setting
    std::string root_folder = in_root_fodler + "/X" + to_string(Xdim) + "_Y" + to_string(Ydim) + "_T" + to_string(unsteadyFieldTimeStep) + "_mixture/" + dataSetSplitTag + "/";
    if (!filesystem::exists(root_folder)) {
        filesystem::create_directories(root_folder);
    }

    Eigen::Vector2d gridInterval = {
        (domainMaxBoundary(0) - domainMinBoundary(0)) / (Xdim - 1),
        (domainMaxBoundary(1) - domainMinBoundary(1)) / (Ydim - 1)
    };

    // std::random_device rd;
    // std::mt19937 rng(rd());
    // std::normal_distribution<double> genTheta(0.0, 0.50); // rotation angle's distribution
    //// normal distribution from supplementary material of Vortex Boundary Identification Paper
    // std::normal_distribution<double> genSx(0, 3.59);
    // std::normal_distribution<double> genSy(0, 2.24);
    // std::normal_distribution<double> genTx(0.0, 1.34);
    // std::normal_distribution<double> genTy(0.0, 1.27);
    // std::uniform_int_distribution<int> dist_int(0, 2);
    double minMagintude = INFINITY;
    double maxMagintude = -INFINITY;

    const string Major_task_foldername = "velocity_mix/";
    const string Major_task_Licfoldername = Major_task_foldername + "/LIC/";

    std::string task_folder = root_folder + Major_task_foldername;
    if (!filesystem::exists(task_folder)) {
        filesystem::create_directories(task_folder);
    }
    std::string task_licfolder = root_folder + Major_task_Licfoldername;
    if (!filesystem::exists(task_licfolder)) {
        filesystem::create_directories(task_licfolder);
    }

    std::vector<int> threadRange(Nfield);
    std::generate(threadRange.begin(), threadRange.end(), [n = 0]() mutable { return n++; });
    for_each(policy, threadRange.begin(), threadRange.end(), [&](const int threadID) {
        int totalSamplesThisThread = observerPerField;

        VastistasVelocityGenerator generator(Xdim, Ydim, domainMinBoundary, domainMaxBoundary, 0.0, 0.0);
        auto steadyMixture = generator.generateSteadyFieldMixture(3);

        for (size_t observerIndex = 0; observerIndex < observerPerField; observerIndex++) {
            printf(".");

            const int taskSampleId = threadID * totalSamplesThisThread + observerIndex;

            const string sample_tag_name
                = "sample_" + to_string(taskSampleId);
            string metaFilename = task_folder + sample_tag_name + "meta.json";
            string velocityFilename = task_folder + sample_tag_name + ".bin";
            const std::vector<float> rawSteadyData = flatten2DAs1Dfloat(steadyMixture.field);
            const auto& observerParameters = generateRandomABCVectors();
            const auto& abc = observerParameters.first;
            const auto& abc_dot = observerParameters.second;
            /* auto func = KillingComponentFunctionFactory::arbitrayObserver(abc, abc_dot);
           KillingAbcField observerfieldDeform(  func, unsteadyFieldTimeStep, tmin, tmax);*/
            auto unsteady_field = Tobias_ObserverTransformation(steadyMixture, abc, abc_dot, tmin, tmax, unsteadyFieldTimeStep);
            // reconstruct unsteady field from observer field
            auto reconstruct_field = Tobias_reconstructUnsteadyField(unsteady_field, abc, abc_dot);
#ifdef VALIDATE_RECONSTRUCTION_RESULT

            // //validate reconstruction result
            for (size_t rec = 0; rec < unsteadyFieldTimeStep; rec++) {
                const auto& reconstruct_slice = reconstruct_field.field[rec];
                // compute reconstruct slice difference with steady field
                double diffSum = 0.0;
                for (size_t y = 1; y < Ydim - 1; y++)
                    for (size_t x = 1; x < Xdim - 1; x++) {
                        auto diff = reconstruct_slice[y][x] - steadyMixture.field[y][x];
                        diffSum += diff.norm();
                    }
                double tolerance = (Xdim - 2) * (Ydim - 2) * 0.0001;
                // has debug, major reason for reconstruction failure is velocity too big make observer transformation query value from region out of boundary
                if (diffSum > tolerance) {
                    printf("\n\n");
                    printf("\n reconstruct field not equal to steady field at step %u\n", (unsigned int)rec);
                    printf("\n\n");
                }
            }
#endif
#ifdef RENDERING_LIC_SAVE_DATA

            {
                auto outputSteadyTexture = LICAlgorithm(steadyMixture, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);

                string steadyField_name = "steady_";
                string licFilename0 = task_licfolder + sample_tag_name + steadyField_name + "lic.png";
                saveAsPNG(outputSteadyTexture, licFilename0);

                auto outputTextures = LICAlgorithm_UnsteadyField(unsteady_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
                auto outputTexturesReconstruct = LICAlgorithm_UnsteadyField(reconstruct_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);

                for (size_t i = 0; i < outputTextures.size(); i += LicSaveFrequency) {
                    string tag_name = sample_tag_name + "deformed_" + std::to_string(i);
                    string licFilename = task_licfolder + tag_name + "lic.png";

                    saveAsPNG(outputTextures[i], licFilename);

                    string tag_name_rec = sample_tag_name + "reconstruct_" + std::to_string(i);
                    string licFilename_rec = task_licfolder + tag_name_rec + "lic.png";
                    saveAsPNG(outputTexturesReconstruct[i], licFilename_rec);
                }
                auto rawUnsteadyFieldData = flatten3DAs1Dfloat(unsteady_field.field);
                auto [minV, maxV] = computeMinMax(rawUnsteadyFieldData);
                if (minV < minMagintude) {
                    minMagintude = minV;
                }
                if (maxV > maxMagintude) {
                    maxMagintude = maxV;
                }
                // save meta info:
                std::ofstream jsonOut(metaFilename);
                if (!jsonOut.good()) {
                    printf("couldn't open file: %s", metaFilename.c_str());
                    return;
                }
                {
                    cereal::JSONOutputArchive archive_o(jsonOut);

                    archive_o(cereal::make_nvp("observer_abc", abc));
                    archive_o(cereal::make_nvp("observer_abc_dot", abc_dot));
                }
                // do not manually close file before creal deconstructor, as cereal will preprend a ]/} to finish json class/array
                jsonOut.close();
                std::ofstream outBin(velocityFilename, std::ios::binary);
                if (!outBin.good()) {
                    printf("couldn't open file: %s", velocityFilename.c_str());
                    return;
                }

                cereal::BinaryOutputArchive archive_Binary(outBin);
                archive_Binary(rawUnsteadyFieldData);
                outBin.close();
            }
#endif
        } // for (size_t observerIndex = 0..)
    });

    // create Root meta json file, save plane information here instead of every sample's meta file
    string taskFolder_rootMetaFilename = root_folder + "meta.json";
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
        // save min and max
        archive_o(cereal::make_nvp("minV", minMagintude));
        archive_o(cereal::make_nvp("maxV", maxMagintude));
    }
}
