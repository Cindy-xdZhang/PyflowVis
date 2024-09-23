#include"ParalleLICrendering.h"
#include <execution>
#include <ctime>
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
