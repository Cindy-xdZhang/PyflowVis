#include <Eigen/Dense>
#include <VectorFieldCompute.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
namespace py = pybind11;

py::array_t<double> licRenderingPybindCPP(
    py::array_t<float> vector_field_np,
    int Xdim,
    int Ydim,
    double xMin,
    double xMax,
    double yMin,
    double yMax,
    int licImageSize,
    double stepSize,
    int MaxIntegrationSteps)
{

    const auto licNoisetexture = randomNoiseTexture(Xdim, Ydim);

    // Convert numpy array to SteadyVectorField2D for vecfield
    py::buffer_info vector_field_info = vector_field_np.request();
    int YdimData = vector_field_info.shape[0];
    int XdimData = vector_field_info.shape[1];

    if (YdimData == Ydim && XdimData == Xdim && vector_field_info.shape[2] == 2) {
        float* vector_field_ptr = static_cast<float*>(vector_field_info.ptr);

        SteadyVectorField2D vecfield;
        vecfield.field.resize(Ydim, std::vector<Eigen::Vector2d>(Xdim));
        for (int i = 0; i < Ydim; ++i) {
            for (int j = 0; j < Xdim; ++j) {
                vecfield.field[i][j] = Eigen::Vector2d(vector_field_ptr[(i * Xdim + j) * 2],
                    vector_field_ptr[(i * Xdim + j) * 2 + 1]);
            }
        }
        vecfield.spatialDomainMinBoundary = Eigen::Vector2d(xMin, yMin);
        vecfield.spatialDomainMaxBoundary = Eigen::Vector2d(xMax, yMax);
        vecfield.spatialGridInterval = Eigen::Vector2d((xMax - xMin) / (Xdim - 1), (yMax - yMin) / (Ydim - 1));

        // Call LICAlgorithm
        std::vector<std::vector<Eigen::Vector3d>> result = LICAlgorithm(licNoisetexture, vecfield, licImageSize, licImageSize, stepSize, MaxIntegrationSteps, static_cast<VORTEX_CRITERION>(0));

        // Convert result to numpy array
        py::array_t<double> result_np({ licImageSize, licImageSize, 3 }); // 3 for R, G, B
        py::buffer_info result_info = result_np.request();
        double* result_ptr = static_cast<double*>(result_info.ptr);
        for (int y = 0; y < licImageSize; ++y) {
            for (int x = 0; x < licImageSize; ++x) {
                for (int c = 0; c < 3; ++c) {
                    result_ptr[(y * licImageSize + x) * 3 + c] = result[y][x](c);
                }
            }
        }

        return result_np;

    } else {
        std::cout << "Input vector field has wrong dimensions." << std::endl;
        return py::array_t<double>({ 0, 0, 0 });
    }
}

PYBIND11_MODULE(CppLicRenderingModule, m)
{
    // module metadata
    m.doc() = "pybind11 lic rending module";
    m.attr("__version__") = "0.0.1";
    m.def("licRenderingPybindCPP", &licRenderingPybindCPP, "Render a steady vector field using LIC.\
     @params:\
    py::array_t<float> vector_field_np->feed with numpy 3d tensor(xdim,ydim,vectorComponnet=2),\
    int Xdim,\
    int Ydim,\
    double xMin,\
    double xMax,\
    double yMin,\
    double yMax,\
    int licImageSize,\
    double stepSize,\
    int MaxIntegrationSteps\
        ");
}