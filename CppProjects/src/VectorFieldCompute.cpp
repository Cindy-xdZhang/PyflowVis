
#include "VectorFieldCompute.h"
#include "VastistasVelocityGenerator.h"
#include <Eigen/Dense>
#include <array>
#include <execution>
#include <fstream>
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

constexpr std::array<std::array<double, 32>, 32> noiseTexture = {
    std::array<double, 32> { 0.641471, 0.410781, 0.509297, 0.441120, 0.756802, 0.558924, 0.679415, 0.364784, 0.621610, 0.450447, 0.675637, 0.439173, 0.083099, 0.566237, 0.492459, 0.568064, 0.353738, 0.601859, 0.563558, 0.383641, 0.436737, 0.452217, 0.720313, 0.603411, 0.685168, 0.639921, 0.372336, 0.494764, 0.318765, 0.714130, 0.641209, 0.451945 },
    std::array<double, 32> { 0.559974, 0.345678, 0.340302, 0.755750, 0.731689, 0.506737, 0.334172, 0.389358, 0.547454, 0.639594, 0.681638, 0.798440, 0.519575, 0.814209, 0.384501, 0.670665, 0.681296, 0.334693, 0.838371, 0.328618, 0.894764, 0.402459, 0.428121, 0.720687, 0.611789, 0.613392, 0.768594, 0.475050, 0.755605, 0.676025, 0.400134, 0.429713 },
    std::array<double, 32> { 0.410781, 0.309631, 0.671442, 0.173756, 0.498534, 0.117193, 0.124403, 0.121163, 0.176852, 0.345678, 0.728346, 0.439350, 0.289035, 0.486872, 0.489764, 0.724729, 0.439188, 0.604208, 0.630342, 0.165986, 0.441886, 0.491037, 0.103710, 0.471420, 0.498376, 0.306899, 0.468192, 0.325921, 0.350478, 0.671206, 0.349662, 0.097849 },
    std::array<double, 32> { 0.652202, 0.649129, 0.637687, 0.617193, 0.666025, 0.683662, 0.661262, 0.401593, 0.752260, 0.642788, 0.448854, 0.650904, 0.422260, 0.608466, 0.636656, 0.297495, 0.461051, 0.666025, 0.422618, 0.410222, 0.267486, 0.498168, 0.336115, 0.639921, 0.836575, 0.345784, 0.196953, 0.481754, 0.471010, 0.193068, 0.345678, 0.498168 },
    std::array<double, 32> { 0.315773, 0.639072, 0.199135, 0.182216, 0.460737, 0.103719, 0.421869, 0.693633, 0.624345, 0.118703, 0.422618, 0.458145, 0.430154, 0.259188, 0.505912, 0.420167, 0.373297, 0.731689, 0.604208, 0.636839, 0.345678, 0.093207, 0.398543, 0.309017, 0.441120, 0.743145, 0.643391, 0.363558, 0.682948, 0.783099, 0.643832, 0.703902 },
    std::array<double, 32> { 0.432352, 0.444462, 0.778073, 0.340881, 0.392371, 0.490268, 0.444212, 0.309017, 0.819152, 0.462551, 0.462745, 0.181627, 0.443904, 0.117638, 0.345678, 0.654861, 0.153306, 0.159406, 0.143300, 0.108850, 0.151937, 0.106308, 0.109297, 0.198867, 0.197495, 0.055336, 0.244522, 0.636839, 0.845519, 0.180611, 0.809689, 0.064793 },
    std::array<double, 32> { 0.497204, 0.445324, 0.433445, 0.436115, 0.499962, 0.414539, 0.313498, 0.659193, 0.743145, 0.677530, 0.351937, 0.643300, 0.699848, 0.645519, 0.651057, 0.309017, 0.439173, 0.697204, 0.439173, 0.602634, 0.689821, 0.663558, 0.413498, 0.799689, 0.763558, 0.743145, 0.868632, 0.899698, 0.298477, 0.406737, 0.699849, 0.761629 },
    std::array<double, 32> { 0.299114, 0.200451, 0.240444, 0.300055, 0.655531, 0.355506, 0.344534, 0.589000, 0.447000, 0.355439, 0.355581, 0.355598, 0.453419, 0.453140, 0.660484, 0.666570, 0.364581, 0.453634, 0.355538, 0.633338, 0.444945, 0.455502, 0.632528, 0.655520, 0.614551, 0.365513, 0.345568, 0.6710475, 0.35654, 0.656476, 0.355004, 0.654429 },
    std::array<double, 32> { 0.336115, 0.539440, 0.599925, 0.592239, 0.590268, 0.371662, 0.586827, 0.528860, 0.581627, 0.783421, 0.575342, 0.530342, 0.409297, 0.499811, 0.425417, 0.498630, 0.495163, 0.492546, 0.596873, 0.521227, 0.592239, 0.598630, 0.551057, 0.699135, 0.692239, 0.651098, 0.699849, 0.699999, 0.694522, 0.692546, 0.699925, 0.697204 },
    std::array<double, 32> { 0.993712, 0.824567, 0.892239, 0.887688, 0.897495, 0.781627, 0.794522, 0.481627, 0.499689, 0.499811, 0.437892, 0.499135, 0.995925, 0.959689, 0.409634, 0.439173, 0.499925, 0.699811, 0.675632, 0.643219, 0.698761, 0.672345, 0.101234, 0.456789, 0.423456, 0.789012, 0.345678, 0.101234, 0.567890, 0.234567, 0.590123, 0.456789 },
    std::array<double, 32> { 0.412345, 0.678901, 0.345678, 0.701234, 0.267890, 0.234567, 0.390123, 0.456789, 0.412345, 0.678901, 0.345678, 0.345678, 0.301234, 0.367890, 0.334567, 0.690123, 0.456789, 0.412345, 0.678901, 0.345678, 0.401234, 0.467890, 0.434567, 0.490123, 0.456789, 0.412345, 0.678901, 0.345678, 0.401234, 0.467890, 0.434567, 0.490123 },
    std::array<double, 32> { 0.456789, 0.412345, 0.678901, 0.345678, 0.401234, 0.467890, 0.434567, 0.690123, 0.456789, 0.412345, 0.678901, 0.345678, 0.345678, 0.401234, 0.467890, 0.434567, 0.499962, 0.315462, 0.421562, 0.415620, 0.315162, 0.421562, 0.490123, 0.456789, 0.412345, 0.678901, 0.345678, 0.401234, 0.467890, 0.434567, 0.490123, 0.456789 },
    std::array<double, 32> { 0.412345, 0.678901, 0.345678, 0.401234, 0.467890, 0.434567, 0.090123, 0.456789, 0.412345, 0.678901, 0.345678, 0.345678, 0.401234, 0.467890, 0.434567, 0.490123, 0.456789, 0.412345, 0.678901, 0.345678, 0.401234, 0.467890, 0.434567, 0.490123, 0.456789, 0.412345, 0.678901, 0.345678, 0.401234, 0.467890, 0.434567, 0.490123 },
    std::array<double, 32> { 0.441471, 0.409297, 0.441120, 0.056802, 0.758924, 0.479415, 0.364784, 0.621610, 0.450447, 0.275637, 0.439173, 0.345678, 0.266237, 0.492459, 0.357506, 0.568064, 0.353738, 0.601859, 0.563558, 0.383641, 0.436737, 0.452217, 0.720313, 0.603411, 0.585168, 0.639921, 0.372336, 0.494764, 0.318765, 0.714130, 0.441209, 0.451945 },
    std::array<double, 32> { 0.655542, 0.299962, 0.125962, 0.429962, 0.411962, 0.059962, 0.459962, 0.212462, 0.429962, 0.499962, 0.299962, 0.499962, 0.619962, 0.645678, 0.234567, 0.349962, 0.399962, 0.345678, 0.397962, 0.699962, 0.690362, 0.699962, 0.699624, 0.711622, 0.699962, 0.299962, 0.299962, 0.390001, 0.600001, 0.633212, 0.222252, 0.334536 },
    std::array<double, 32> { 0.699462, 0.699962, 0.299962, 0.299962, 0.799962, 0.156962, 0.246562, 0.899962, 0.829962, 0.899962, 0.199962, 0.399962, 0.649962, 0.779962, 0.654321, 0.045678, 0.010521, 0.049962, 0.519962, 0.599962, 0.500962, 0.598962, 0.512962, 0.329962, 0.359962, 0.549962, 0.544962, 0.509624, 0.599962, 0.599962, 0.599962, 0.530500 },
    std::array<double, 32> { 0.699962, 0.783421, 0.651098, 0.524567, 0.537892, 0.509634, 0.575632, 0.543219, 0.598761, 0.672345, 0.501234, 0.556789, 0.523456, 0.189012, 0.345678, 0.501234, 0.567890, 0.534567, 0.590123, 0.556789, 0.512345, 0.678901, 0.345678, 0.501234, 0.567890, 0.534567, 0.590123, 0.390001, 0.533212, 0.522252, 0.394536, 0.567890 },
    std::array<double, 32> { 0.599962, 0.599962, 0.599962, 0.576543, 0.510987, 0.543210, 0.587654, 0.321098, 0.765432, 0.509876, 0.654321, 0.598765, 0.532109, 0.576543, 0.510987, 0.654321, 0.598765, 0.532109, 0.576543, 0.510987, 0.654321, 0.598765, 0.532109, 0.576543, 0.510987, 0.654321, 0.598765, 0.532109, 0.576543, 0.599962, 0.500000, 0.547890 },
    std::array<double, 32> { 0.599962, 0.599962, 0.599962, 0.345678, 0.598534, 0.317193, 0.501234, 0.567890, 0.534567, 0.590123, 0.556789, 0.512345, 0.678901, 0.345678, 0.501234, 0.567890, 0.534567, 0.590123, 0.556789, 0.512345, 0.678901, 0.345678, 0.501234, 0.567890, 0.534567, 0.590123, 0.556789, 0.512345, 0.678901, 0.345678, 0.501234, 0.567890 },
    std::array<double, 32> { 0.789012, 0.599962, 0.345678, 0.501234, 0.567890, 0.399962, 0.534567, 0.590123, 0.556789, 0.512345, 0.678901, 0.145678, 0.201234, 0.167890, 0.500567, 0.590123, 0.556789, 0.512345, 0.678901, 0.345678, 0.501234, 0.567890, 0.534567, 0.590123, 0.556789, 0.512345, 0.678901, 0.345678, 0.501234, 0.567890, 0.534567, 0.567890 },
    std::array<double, 32> { 0.529962, 0.519962, 0.592962, 0.543219, 0.598761, 0.599962, 0.672345, 0.501234, 0.556789, 0.523456, 0.789012, 0.345678, 0.501234, 0.567890, 0.534567, 0.590123, 0.556789, 0.512345, 0.678901, 0.345678, 0.501234, 0.567890, 0.534567, 0.590123, 0.556789, 0.512345, 0.678901, 0.345678, 0.301234, 0.367890, 0.334567, 0.670600 },
    std::array<double, 32> { 0.399962, 0.376543, 0.310987, 0.343210, 0.309962, 0.699962, 0.387654, 0.321098, 0.765432, 0.509876, 0.654321, 0.598765, 0.532109, 0.376543, 0.310987, 0.654321, 0.598765, 0.532109, 0.376543, 0.310987, 0.654321, 0.598765, 0.532109, 0.376543, 0.310987, 0.654321, 0.598765, 0.532109, 0.376543, 0.310987, 0.654321, 0.523365 },
    std::array<double, 32> { 0.599962, 0.399962, 0.783421, 0.651098, 0.324567, 0.537892, 0.509634, 0.375632, 0.343219, 0.398761, 0.672345, 0.301234, 0.556789, 0.523456, 0.789012, 0.345678, 0.301234, 0.367890, 0.334567, 0.390123, 0.556789, 0.678901, 0.345678, 0.301234, 0.367890, 0.234567, 0.290123, 0.556789, 0.512345, 0.678901, 0.345678, 0.622652 },
    std::array<double, 32> { 0.299962, 0.594962, 0.599962, 0.209962, 0.249962, 0.504962, 0.244442, 0.100962, 0.559962, 0.329962, 0.229962, 0.599962, 0.599962, 0.229962, 0.520962, 0.555962, 0.333962, 0.222962, 0.544962, 0.297962, 0.889962, 0.422325, 0.245562, 0.429962, 0.899962, 0.829962, 0.333962, 0.444962, 0.855962, 0.478901, 0.851114, 0.470665 },
    std::array<double, 32> { 0.499962, 0.810987, 0.843210, 0.387654, 0.321098, 0.765432, 0.409876, 0.654321, 0.498765, 0.432109, 0.376543, 0.310987, 0.654321, 0.498765, 0.432109, 0.376543, 0.310987, 0.654321, 0.498765, 0.432109, 0.376543, 0.310987, 0.654321, 0.498765, 0.432109, 0.876543, 0.810987, 0.654321, 0.498765, 0.432109, 0.876543, 0.470665 },
    std::array<double, 32> { 0.334567, 0.390123, 0.456789, 0.412345, 0.678901, 0.345678, 0.301234, 0.367890, 0.834567, 0.890123, 0.456789, 0.412345, 0.678901, 0.345678, 0.801234, 0.867890, 0.631774, 0.624248, 0.975857, 0.819500, 0.169168, 0.881598, 0.412345, 0.678901, 0.345678, 0.501234, 0.567890, 0.534567, 0.390123, 0.456789, 0.412345, 0.470665 },
    std::array<double, 32> { 0.699962, 0.789012, 0.345678, 0.301234, 0.367890, 0.134567, 0.190123, 0.456789, 0.412345, 0.678901, 0.345678, 0.101234, 0.167890, 0.134567, 0.190123, 0.456789, 0.412345, 0.678901, 0.345678, 0.301234, 0.367890, 0.634567, 0.390123, 0.456789, 0.412345, 0.678901, 0.345678, 0.301234, 0.367890, 0.334567, 0.390123, 0.670665 },
    std::array<double, 32> { 0.236623, 0.613926, 0.418028, 0.104975, 0.191739, 0.892855, 0.465252, 0.384898, 0.619587, 0.690716, 0.414940, 0.050378, 0.402000, 0.422000, 0.399962, 0.783099, 0.376543, 0.499962, 0.499962, 0.499962, 0.412345, 0.631774, 0.624248, 0.975857, 0.819500, 0.369168, 0.881598, 0.393962, 0.345678, 0.301234, 0.666534, 0.364444 },
    std::array<double, 32> { 0.399962, 0.789012, 0.345678, 0.301234, 0.367890, 0.334567, 0.390123, 0.456789, 0.412345, 0.078901, 0.345678, 0.301234, 0.367890, 0.334567, 0.390123, 0.456789, 0.412345, 0.678901, 0.345678, 0.301234, 0.367890, 0.334567, 0.390123, 0.456789, 0.412345, 0.678901, 0.345678, 0.501234, 0.567890, 0.534567, 0.590123, 0.570665 },
    std::array<double, 32> { 0.557506, 0.599962, 0.511898, 0.624810, 0.380884, 0.583955, 0.591445, 0.597204, 0.570296, 0.418630, 0.419962, 0.429962, 0.401898, 0.380884, 0.402000, 0.422000, 0.400962, 0.783099, 0.076543, 0.599962, 0.499962, 0.699962, 0.412345, 0.567890, 0.456789, 0.499962, 0.401898, 0.380884, 0.511898, 0.624810, 0.480884, 0.170665 },
    std::array<double, 32> { 0.599574, 0.540302, 0.755750, 0.731689, 0.406737, 0.534172, 0.589358, 0.447454, 0.639594, 0.681638, 0.798440, 0.419575, 0.414209, 0.384501, 0.670665, 0.681296, 0.534693, 0.438371, 0.528618, 0.794764, 0.402459, 0.428121, 0.720687, 0.511789, 0.613392, 0.768594, 0.475050, 0.755605, 0.576025, 0.400134, 0.429713, 0.480884 },
    std::array<double, 32> { 0.410781, 0.509631, 0.571442, 0.573756, 0.498534, 0.517193, 0.524403, 0.521163, 0.576852, 0.528346, 0.439350, 0.789035, 0.486872, 0.489764, 0.724729, 0.439188, 0.504208, 0.530342, 0.065986, 0.441886, 0.491037, 0.503710, 0.971420, 0.498376, 0.306899, 0.468192, 0.525921, 0.350478, 0.571206, 0.549662, 0.597849, 0.423444 }
};

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

std::vector<std::vector<double>> loadNoiseTexture(const std::string& filename, int width, int height)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return {};
    }

    std::vector<std::vector<double>> texture(height, std::vector<double>(width));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double value;
            file >> value;
            texture[y][x] = value;
        }
    }

    file.close();

    return texture;
}

// The result image size is the same as the input texture.
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
    constexpr int TexDim = 32;
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
