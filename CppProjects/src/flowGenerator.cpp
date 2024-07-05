
#include "flowGenerator.h"
#include "VastistasVelocityGenerator.h"
#include "VectorFieldCompute.h"
#include "cereal/archives/binary.hpp"
#include "cereal/archives/json.hpp"
#include "cereal/types/array.hpp"
#include "cereal/types/vector.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include <execution>
#include <filesystem>
#include <fstream>
#include <thread>
#define RENDERING_LIC_SAVE_DATA

std::mt19937 rng(static_cast<unsigned int>(std::time(0)));
using namespace std;

constexpr double tmin = 0.0;
constexpr double tmax = M_PI * 0.5;
constexpr int Xdim = 64, Ydim = 64;
constexpr int LicImageSize = 256;
Eigen::Vector2d domainMinBoundary = { -2.0, -2.0 };
Eigen::Vector2d domainMaxBoundary = { 2.0, 2.0 };
constexpr int unsteadyFieldTimeStep = 8;
constexpr int LicSaveFrequency = 2; // every 2 time steps save one
const double stepSize = 0.01;
const int maxLICIteratioOneDirection = 256;

std::string trimNumString(const std::string& numString)
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

#include <iomanip>
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

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const
    {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};
std::vector<std::pair<double, double>> presetNRCParameters = {
    { 0.25, 2.0 },
    { 1.0, 2.0 },
    { 1.0, 3.0 },
    { 1.0, 5.0 },
    { 2.0, 1.0 },
    { 2.0, 2.0 },
    { 2.0, 3.0 },
    { 2.0, 5.0 },
};

std::vector<std::pair<double, double>> generateNParamters(int n, std::string mode)
{
    static std::unordered_set<std::pair<double, double>, pair_hash> unique_params;

    std::vector<std::pair<double, double>> parameters;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<double> dist_rc(1.87, 0.37); // mean = 1.87, stddev = 0.37
    std::normal_distribution<double> dist_n(1.96, 0.61); // mean = 1.96, stddev = 0.61

    int i = 0;
    while (parameters.size() < n) {

        if (i < presetNRCParameters.size() && mode == "train") {
            std::pair<double, double> preset_pair = presetNRCParameters.at(i++);
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

    // if inputField has analytical expression v(x,t) then result field u  has transformatd analytical expression u(x,y)=   pushforward* v(x*,t) =Q(t)^T *v(x*,t)
    if (inputField.analyticalFlowfunc_) {
        const Eigen ::Vector3d Os = { pathPositions[0].x(), pathPositions[0].y(), 1.0 };
        resultField.Q_t.resize(timestep);
        resultField.c_t.resize(timestep);
        for (size_t i = 0; i < timestep; i++) {
            //  frame transformation is F(x):x*=Q(t)x+c(t)  or x*=T(Os) *Q*T(-Pt)*x
            //  =>F(x):x* = Q(t)*(x-pt)+Os= Qx-Q*pt+Os -> c=-Q*pt+Os  // => F^(-1)(x)= Q^T (x-c)= Q^T *( x+Q*pt-Os)
            resultField.Q_t[i] = observerRotationMatrices[i].transpose();
            auto& Q_t = resultField.Q_t[i];
            const Eigen ::Vector3d position_t = { pathPositions[i].x(), pathPositions[i].y(), 1.0 };
            Eigen ::Vector3d c_t = Os - Q_t * position_t;
            resultField.c_t[i] = c_t;
        }

        resultField.analyticalFlowfunc_ = [inputField, observerfield, resultField, dt, observerRotationMatrices, pathPositions](const Eigen::Vector2d& pos, double t) -> Eigen::Vector2d {
            double tmin = observerfield.tmin;
            const double floatingTimeStep = (t - tmin) / dt;
            const int timestep_floor = std::clamp((int)std::floor(floatingTimeStep), 0, observerfield.timeSteps - 1);
            const int timestep_ceil = std::clamp((int)std::floor(floatingTimeStep) + 1, 0, observerfield.timeSteps - 1);
            const double ratio = floatingTimeStep - timestep_floor;

            const Eigen ::Matrix3d Q_t = resultField.Q_t[timestep_floor] * (1 - ratio) + resultField.Q_t[timestep_ceil] * ratio;
            const Eigen ::Matrix3d Q_transpose = observerRotationMatrices[timestep_floor] * (1 - ratio) + observerRotationMatrices[timestep_ceil] * ratio;
            const Eigen ::Vector3d c_t = resultField.c_t[timestep_floor] * (1 - ratio) + resultField.c_t[timestep_ceil] * ratio;
            const Eigen ::Vector3d pos_3d = { pos.x(), pos.y(), 1.0 };

            // => F^(-1)(x)= Q^T (x-c)= Q^T *( x+Q*pt-Os)
            Eigen ::Vector3d F_inverse_x = Q_transpose * (pos_3d - c_t);
            F_inverse_x /= F_inverse_x(2);

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
    }

    return resultField;
}
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

    // Create the flow field
    UnSteadyVectorField2D InputflowField = flowCreator.createRFC();

    auto func_const_trans = KillingComponentFunctionFactory::constantRotation(-1.0);

    KillingAbcField observerfield(
        func_const_trans, unsteadyFieldTimeStep, tmin, tmax);
    Eigen::Vector2d StartPosition = { 0.0, 0.0 };

    auto unsteady_field = killingABCtransformation(observerfield, StartPosition, InputflowField);
    auto resample_observerfield = observerfield.resample2UnsteadyField(grid_size, domainMinBoundary, domainMaxBoundary);
    // auto outputObserverFieldLic = LICAlgorithm_UnsteadyField(licNoisetexture, resample_observerfield, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
    auto inputTextures = LICAlgorithm_UnsteadyField(InputflowField, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection, VORTEX_CRITERION::NONE);

    auto outputTexturesObservedField = LICAlgorithm_UnsteadyField(unsteady_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection, VORTEX_CRITERION::NONE);

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
    // if si=0 then no vortex
    if (meta_n_rc_si.z() == 0.0) {
        return;
    }
    const auto rc = meta_n_rc_si(1);
    const auto deformInverse = deformMat.inverse();

    auto judgeVortex = [rc, translation_t = txy, deformInverse](const Eigen::Vector2d& pos) -> bool {
        auto originalPos = deformInverse * (pos - translation_t);
        auto dx = rc - originalPos.norm();
        return dx > 0;
    };
    const int licImageSizeY = inputLicImages[0].size();
    const int licImageSizeX = inputLicImages[0][0].size();
    const auto domainRange = domainMax - domainMIn;
    const auto dx = domainRange(0) / licImageSizeX;
    const auto dy = domainRange(1) / licImageSizeY;
    const auto maxDistanceAnyPoint2gridPoints = sqrt((0.5 * dx) * (0.5 * dx) + (0.5 * dy) * (0.5 * dy));

    assert(vectorField.Q_t.size() == vectorField.timeSteps);
    assert(vectorField.c_t.size() == vectorField.timeSteps);
    double dt = (vectorField.tmax - vectorField.tmin) / double(vectorField.timeSteps - 1);
    for (size_t t = 0; t < vectorField.timeSteps; t++) {
        auto& inputLicImage = inputLicImages[t];
        auto Q_inverse = vectorField.Q_t[t].transpose();
        auto c_t = vectorField.c_t[t];
        double time = vectorField.tmin + t * dt;
        for (size_t i = 0; i < licImageSizeX; i++) {
            for (size_t j = 0; j < licImageSizeY; j++) {
                // map position from texture image grid coordinate to vector field
                double ratio_x = (double)((double)i / (double)licImageSizeX);
                double ratio_y = (double)((double)j / (double)licImageSizeY);

                Eigen::Vector3d posInObserverFrame = { ratio_x * domainRange(0) + domainMIn(0),
                    ratio_y * domainRange(1) + domainMIn(1), 1.0 };

                Eigen ::Vector3d F_inverse_x = Q_inverse * (posInObserverFrame - c_t);
                F_inverse_x /= F_inverse_x(2);
                Eigen::Vector2d posInOriginalFrame = { F_inverse_x(0), F_inverse_x(1) };

                if (judgeVortex(posInOriginalFrame)) {
                    auto preColor = inputLicImage[j][i];
                    // red for critial point(coreline)
                    auto velocity = vectorField.getVector(posInObserverFrame.x(), posInObserverFrame.y(), time);
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
}

// number of result traing data = Nparamters * samplePerParameters * observerPerSetting;dataSetSplitTag should be "train"/"test"/"validation"
void generateUnsteadyField(int Nparamters, int samplePerParameters, int observerPerSetting, std::string dataSetSplitTag)
{
#if defined(DISABLE_CPP_PARALLELISM) || defined(_DEBUG)
    auto policy = std::execution::seq;
#else
    auto policy = std::execution::par_unseq;
#endif
    // check datasplittag is "train"/"test"/"validation"
    if (dataSetSplitTag != "train" && dataSetSplitTag != "test" && dataSetSplitTag != "validation") {
        printf("dataSetSplitTag should be \"train\"/\"test\"/\"validation\"");
        return;
    }

    int numVelocityFields = samplePerParameters; // num of fields per n, rc parameter setting
    std::string root_folder = "../data/debugX" + to_string(Xdim) + "_Y" + to_string(Ydim) + "_T" + to_string(unsteadyFieldTimeStep) + "_no_mixture/" + dataSetSplitTag + "/";
    if (!filesystem::exists(root_folder)) {
        filesystem::create_directories(root_folder);
    }

    Eigen::Vector2d gridInterval = {
        (domainMaxBoundary(0) - domainMinBoundary(0)) / (Xdim - 1),
        (domainMaxBoundary(1) - domainMinBoundary(1)) / (Ydim - 1)
    };

    const auto paramters = generateNParamters(Nparamters, dataSetSplitTag);
    // const auto licNoisetexture = randomNoiseTexture(Xdim, Ydim);

    std::normal_distribution<double> genTheta(0.0, 0.50); // rotation angle's distribution
    std::normal_distribution<double> genSx(1.0, 0.667); // scaling factor's distribution

    // this generate coreline's point(critical point) for vortex region
    std::normal_distribution<double> genTx(0.0, 0.59);

    // Distribution for selecting type
    std::uniform_int_distribution<int> dist_Observer_type(0, (int)(magic_enum::enum_count<ObserverType>() - 1));
    std::uniform_int_distribution<int> dist_int(0, 4); // we prefer more si =1,2->generate 0,1,2,3,4->ceil(divide by two)

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

            std::mt19937 rngSample(static_cast<unsigned int>(std::time(nullptr)));
            // generate steady field with vortex
            int Si = std::clamp((int)std::ceil(dist_int(rngSample) / 2), 0, 2);
            const auto thetaSteady = genTheta(rngSample);
            const auto sx = genSx(rngSample);
            const auto sy = genSx(rngSample);
            auto tx = genTx(rngSample);
            auto ty = genTx(rngSample);
            // clamp tx ty to 0.5*domian
            tx = std::clamp(tx, 0.5 * domainMinBoundary.x(), 0.5 * domainMaxBoundary.x());
            ty = std::clamp(ty, 0.5 * domainMinBoundary.y(), 0.5 * domainMaxBoundary.y());
            Eigen::Vector2d txy = { tx, ty };

            Eigen::Vector3d n_rc_si = { n, rc, (double)Si };
            const SteadyVectorField2D steadyField = generator.generateSteadyField_VortexBoundaryVIS2020(tx, ty, sx, sy, thetaSteady, Si);
            const auto& steadyFieldResData = steadyField.field;
            for (size_t observerIndex = 0; observerIndex < observerPerSetting; observerIndex++) {
                printf(".");
                // Randomly select a type
                const int type
                    = dist_Observer_type(rng);
                // type = ConstTranslationRotation;

                const auto observer_name
                    = std::string { magic_enum::enum_name((ObserverType)type) };

                const int taskSampleId = sample * observerPerSetting + observerIndex;
                const string sample_tag_name = "rc_" + str_Rc + "_n_" + str_n + "_sample_" + to_string(taskSampleId) + "Si_" + to_string(Si) + "observer_" + observer_name;

                // printf("generating sample %s \n", sample_tag_name.c_str());
                //  create folder for every n rc parameter setting.

                string metaFilename = task_folder + sample_tag_name + "meta.json";
                string velocityFilename = task_folder + sample_tag_name + ".bin";

                const std::vector<float> rawSteadyData = flatten2DAs1Dfloat(steadyFieldResData);

                auto func = KillingComponentFunctionFactory::randomObserver(type);

                auto inv_func = KillingComponentFunctionFactory::getInverseObserver(func);

                KillingAbcField observerfieldDeform(
                    func, unsteadyFieldTimeStep, tmin, tmax);

                KillingAbcField observerfield(
                    inv_func, unsteadyFieldTimeStep, tmin, tmax);

                Eigen::Vector2d StartPosition = { 0.0, 0.0 };
                auto unsteady_field = killingABCtransformation(observerfieldDeform, StartPosition, steadyField);
                // reconstruct unsteady field from observer field
                auto reconstruct_field = killingABCtransformation(observerfield, StartPosition, unsteady_field);

                // validate reference transformation
                auto& Q_t_deforming = unsteady_field.Q_t;
                auto& c_t_deforming = unsteady_field.c_t;

                auto& Q_t_rec = reconstruct_field.Q_t;
                auto& c_t_rec = reconstruct_field.c_t;
                for (size_t rec = 0; rec < reconstruct_field.field.size() - 1; rec++) {
                    Eigen ::Matrix3d shouldbeI = (Q_t_deforming[rec] * Q_t_rec[rec]);

                    // if deforming transformation is x*=Q_tx+c1(t) then observer transformation is x*=R_t*x+c2(t)
                    // if rec success it should satisfy R_t=Q_t^(-1) and c2(t)=-Q_t^(-1)*c1(t)
                    auto c2t_ideal = -Q_t_rec[rec] * c_t_deforming[rec];

                    Eigen::Vector3d shouldbe0 = c2t_ideal - c_t_rec[rec];
                    double Qerror = (shouldbeI - Eigen::Matrix3d::Identity()).norm();
                    double c_error = shouldbe0.norm();
                    if (Qerror > 1e-7 || c_error > 1e-7) {
                        printf("\n\n");
                        printf("\n observer transformation not equal to inverse observer transformation at step %u,check observe type %s\n", (unsigned int)rec, observer_name.c_str());
                        printf("\n\n");
                    }
                }

                // //validate reconstruction result
                for (size_t rec = 0; rec < reconstruct_field.field.size() - 1; rec++) {
                    auto reconstruct_slice = reconstruct_field.field[rec];
                    // compute reconstruct slice difference with steady field
                    double diffSum = 0.0;
                    for (size_t y = 1; y < Ydim - 1; y++)
                        for (size_t x = 1; x < Xdim - 1; x++) {
                            auto diff = reconstruct_slice[y][x] - steadyFieldResData[y][x];
                            diffSum += diff.norm();
                        }
                    double tolerance = (Xdim - 2) * (Ydim - 2) * 0.01;
                    tolerance = (rec == 0) ? 1e-7 : tolerance;
                    // has debug, major reason for reconstruction failure is velocity too big make observer transformation query value from region out of boundary
                    if (diffSum > tolerance) {
                        printf("\n\n");
                        printf("\n reconstruct field not equal to steady field at step %u,check observe type %s\n", (unsigned int)rec, observer_name.c_str());
                        printf("\n\n");
                    }
                }

#ifdef RENDERING_LIC_SAVE_DATA

                // rendering LIC

                {
                    auto outputSteadyTexture = LICAlgorithm(steadyField, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
                    // add segmentation visualization for steady lic
                    Eigen::Matrix2d deformMat = Eigen::Matrix2d::Identity();
                    deformMat(0, 0) = sx * cos(thetaSteady);
                    deformMat(0, 1) = -sy * sin(thetaSteady);
                    deformMat(1, 0) = sx * sin(thetaSteady);
                    deformMat(1, 1) = sy * cos(thetaSteady);
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
                    // save meta info:
                    std::ofstream jsonOut(metaFilename);
                    if (!jsonOut.good()) {
                        printf("couldn't open file: %s", metaFilename.c_str());
                        return;
                    }
                    {
                        cereal::JSONOutputArchive archive_o(jsonOut);
                        /*     archive_o(CEREAL_NVP(Xdim));
                             archive_o(CEREAL_NVP(Ydim));
                             archive_o(CEREAL_NVP(domainMinBoundary));
                             archive_o(CEREAL_NVP(domainMaxBoundary));*/

                        Eigen::Vector3d deform = { thetaSteady, sx, sy };
                        archive_o(cereal::make_nvp("n_rc_Si", n_rc_si));
                        archive_o(cereal::make_nvp("deform_theta_sx_sy", deform));
                        archive_o(cereal::make_nvp("txy", txy));
                        archive_o(CEREAL_NVP(minV));
                        archive_o(CEREAL_NVP(maxV));
                        // meta for observer field
                        archive_o(cereal::make_nvp("Observer Type", observer_name));
                        // archive_o(CEREAL_NVP(observerfieldDeform));

                        std::vector<double> thetaObserver;
                        thetaObserver.reserve(unsteadyFieldTimeStep);
                        std::vector<std::array<double, 2>> c_t;
                        c_t.reserve(unsteadyFieldTimeStep);
                        thetaObserver.reserve(unsteadyFieldTimeStep);
                        for (size_t i = 0; i < unsteadyFieldTimeStep; i++) {
                            // matix3d unsteady_field.Q_t[i] to matrix2d
                            Eigen::Matrix2d Q_t2d_t = unsteady_field.Q_t[i].block<2, 2>(0, 0);
                            const double theta_rot = matrix2angle(Q_t2d_t);
                            thetaObserver.push_back(theta_rot);
                            c_t.push_back({ unsteady_field.c_t[i].x(), unsteady_field.c_t[i].y() });
                        }

                        archive_o(cereal::make_nvp("theta(t)", thetaObserver));
                        archive_o(cereal::make_nvp("c(t)", c_t));
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