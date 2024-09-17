#include<VectorFieldCompute.h>

std::vector<std::vector<std::vector<Eigen::Vector3d>>> LICAlgorithm_UnsteadyField(const UnSteadyVectorField2D& vecfield, const int licImageSizeX, const int licImageSizeY, double stepSize, int MaxIntegrationSteps, VORTEX_CRITERION curlColorBlend = VORTEX_CRITERION::NONE);

