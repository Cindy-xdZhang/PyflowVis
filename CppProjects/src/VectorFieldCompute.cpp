#include "commonUtils.h"
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
			//printf("Warning: criterion  %u  has scalar field are too small to be used for color mapping. Switching to NONE coloring.\n", (unsigned int)criterionlColorBlend);
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
					Eigen::Vector3d curlColor = mapValueToColor(normalizedCurl);

					auto whiteish = licValue;
					whiteish = std::min(std::max(0.0, (whiteish - 0.4) * (1.5 / 0.4)), 1.0);
					// output_texture[y][x] = mix(curlColor, Eigen::Vector3d(licValue, licValue, licValue), 1.0 - whiteish);
					output_texture[y][x] = Eigen::Vector3d(licValue, licValue, licValue) * (1.0 - whiteish) + curlColor * whiteish;

				}
				else {
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
			}
			else if ((targetIntegrationTime < startTime) && (newTime <= targetIntegrationTime)) {
				outOfIntegrationTimeBounds = true;
			}
			else {
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
	const int maxIterationCount = 1000;
	const double spaceConversionRatio = 1.0;
	pathPositions.clear();
	pathPositions.reserve(128);

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
			}
			else if ((targetIntegrationTime < startTime) && (newTime <= targetIntegrationTime)) {
				outOfIntegrationTimeBounds = true;
			}
			else {
				if (std::isnan(newPoint(0)) || std::isnan(newPoint(1))) {
					// Handle the case where newPoint(0) is NaN
					printf("get nan during PathhlineIntegrationRK4v2.");
					break;
				}

				// add  current point to the result list and set currentPoint to newPoint -> everything fine -> continue with the while loop
				Eigen::Vector3d new_pointAndTime = { newPoint(0), newPoint(1), newTime };
				pathPositions.emplace_back(new_pointAndTime);

				currentPoint = newPoint;
				currentTime = newTime;
				iterationCount++;
			}
		}
	}
	bool suc = pathPositions.size() > 3;
	return suc;
}



std::vector<std::vector<PathlinePointInfo>> PathlineIntegrationInfoCollect2D(const UnSteadyVectorField2D& inputField, int KLines, const double pathline_dt_m, const Eigen::Matrix2d& deformMat, const Eigen::Vector3d& n_rc_si, const Eigen::Vector2d& txy, const int outputPathlineLength)
{
	constexpr int maximumLength = 50; // pathline_dt=1/5 dt, thus total have 9*5=45 steps.
	auto maxBound = inputField.getSpatialMaxBoundary();
	auto minBound = inputField.getSpatialMinBoundary();
	auto domainRange = maxBound - minBound;
	auto tmax = inputField.tmax;
	auto tmin = inputField.tmin;
	auto Xdim = inputField.XdimYdim.x();
	auto Ydim = inputField.XdimYdim.y();

	auto generateSeedings = [](double xmin, double xmax, double ymin, double ymax, int K) -> std::vector<Eigen::Vector2d> {
		assert(xmax - xmin > 0);
		assert(ymax - ymin > 0);
		std::vector<Eigen::Vector2d> seedings;
		seedings.reserve(K);

		// Define distributions for x and y coordinates within the rectangle
		std::uniform_real_distribution<> disX(xmin, xmax);
		std::uniform_real_distribution<> disY(ymin, ymax);

		for (int i = 0; i < K; ++i) {
			// Generate random (x, y) within the defined rectangle
			double x = disX(rng);
			double y = disY(rng);
			seedings.emplace_back(x, y);
		}

		return seedings;
		};

	const double dt = (inputField.tmax - inputField.tmin) / (double)(inputField.timeSteps - 1);
	const double pathline_dt = dt / pathline_dt_m;

	// std::vector<std::vector<std::vector<double>>> curlFields(inputField.timeSteps);
	std::vector<std::vector<std::vector<double>>> ivdFields(inputField.timeSteps);
	// precompute vorticity and ivd field
	for (size_t t = 0; t < inputField.timeSteps; t++) {

		// auto curlField = ComputeCurl(inputField.field[t], Xdim, Ydim, inputField.spatialGridInterval(0), inputField.spatialGridInterval(1));
		auto ivdField = ComputeIVD(inputField.field[t], Xdim, Ydim, inputField.spatialGridInterval(0), inputField.spatialGridInterval(1));
		// curlFields[t] = curlField;
		ivdFields[t] = ivdField;
	}

	// generate random window D as a local cluster in the physical domain near grid point P; or 4 clusters divide the whole domain
	//
	// generate  K(7?9?) random samples inside D
	// K pathlines, padding to equal length, stepsize makes to 1/m *dt->in total generate K*M points per cluster
	// every path line information will store position,time, vorticity,IVD.
	// the segmentation of steady vasts is an objective quantity, and should not change no matter what observer is applied.
	std::vector<std::vector<PathlinePointInfo>> clusterPathlines;
	constexpr int NClusters = 4;
	std::array<Eigen::Vector2d, 4> domainCenters{
		Eigen::Vector2d(0.25 * domainRange.x() + minBound.x(), 0.25 * domainRange.y() + minBound.y()), //-1,-1
		Eigen::Vector2d(0.25 * domainRange.x() + minBound.x(), 0.75 * domainRange.y() + minBound.y()), // 1,1
		Eigen::Vector2d(0.75 * domainRange.x() + minBound.x(), 0.25 * domainRange.y() + minBound.y()), // 1,-1
		Eigen::Vector2d(0.75 * domainRange.x() + minBound.x(), 0.75 * domainRange.y() + minBound.y()) //-1,1
	};

	clusterPathlines.resize(NClusters * KLines);
	for (int i = 0; i < NClusters; i++) {
		// the domain d is 1/5*range,(1/10 in one direction).
		Eigen::Vector2d domainCenter = domainCenters[i];

		auto Domain_minx = std::max(domainCenter.x() - 0.25 * domainRange.x(), minBound.x());
		auto Domain_maxx = std::min(domainCenter.x() + 0.25 * domainRange.x(), maxBound.x());
		auto Domain_miny = std::max(domainCenter.y() - 0.25 * domainRange.y(), minBound.y());
		auto Domain_maxy = std::min(domainCenter.y() + 0.25 * domainRange.y(), maxBound.y());
		auto seedings = generateSeedings(Domain_minx, Domain_maxx, Domain_miny, Domain_maxy, KLines);
		for (size_t k = 0; k < KLines; k++) {
			int thisPathlineGlobalId = i * KLines + k;
			clusterPathlines[thisPathlineGlobalId].reserve(maximumLength * 4);

			std::vector<Eigen::Vector3d> pathlinePositions;
			auto suc = PathhlineIntegrationRK4v2(seedings[k], inputField, 0, tmax, pathline_dt, pathlinePositions);
			while (suc == false) {
				seedings[k] = generateSeedings(Domain_minx, Domain_maxx, Domain_miny, Domain_maxy, 1)[0];
				suc = PathhlineIntegrationRK4v2(seedings[k], inputField, 0, tmax, pathline_dt, pathlinePositions);
			}

			const auto startPoint = seedings[k];
			for (int step = 0; step < pathlinePositions.size(); step++) {
				auto px = pathlinePositions[step].x();
				auto py = pathlinePositions[step].y();
				auto time = pathlinePositions[step].z();
				double floatIndexX = (px - minBound.x()) / inputField.spatialGridInterval(0);
				double floatIndexY = (py - minBound.y()) / inputField.spatialGridInterval(1);
				double floatIndexT = (time - inputField.tmin) / dt;

				auto ivd = trilinear_interpolate(ivdFields, floatIndexX, floatIndexY, floatIndexT);
				auto distance = sqrt((px - startPoint.x()) * (px - startPoint.x()) + (py - startPoint.y()) * (py - startPoint.y()));

				std::vector<double> pathlinePointAndInfo = { px, py, time, ivd, distance };

				clusterPathlines[thisPathlineGlobalId].emplace_back(pathlinePointAndInfo);
			}
		}
	}

	// auto rc = meta_n_rc_si.y();
	// auto si = meta_n_rc_si.z();
	// const auto deformInverse = deformMat.inverse();
	// auto judgeVortex = [si, rc, txy, deformInverse](const Eigen::Vector2d& pos) -> double {
	//     if (si == 1.0 || si == 2.0) {
	//         auto originalPos = deformInverse * (pos - txy);
	//         auto dx = rc - originalPos.norm();
	//         return dx > 0 ? 1.0 : 0.0;
	//     }
	//     return 0.0;
	// };
	std::vector PadValue = { -100.0, -100.0, -100.0 * 3.1415926, -100.0, -100.0 };
	// the first point is the distance to it self(always zero), use it as the segmentation label for this pathline.
	for (auto& pathline : clusterPathlines) {
		/*     Eigen::Vector2d pos = { pathline.at(0).at(0), pathline.at(0).at(1) };
			 pathline.at(0).at(4) = judgeVortex(pos);*/
		pathline.resize(outputPathlineLength, PadValue);
	}

	// ----------------------------------------------------------
	// verify code to make sure every pathline has same time stamps
	// ----------------------------------------------------------
	for (size_t step = 0; step < outputPathlineLength; step++) {
		auto time = clusterPathlines[0].at(step).at(2);
		if (time == -100.0 * 3.1415926) {
			for (size_t l = 1; l < clusterPathlines.size(); l++) {
				if (clusterPathlines[l].at(step).at(2) != -100.0 * 3.1415926) {
					time = clusterPathlines[l].at(step).at(2);
				}
			}
		}

		for (size_t l = 1; l < clusterPathlines.size(); l++) {
			if (clusterPathlines[l].at(step).at(2) != time && clusterPathlines[l].at(step).at(2) != -100.0 * 3.1415926) {
				printf("error: get not equal timestamps of pathlines.");
				assert(false);
			};
		}
	}

	return clusterPathlines;
}

