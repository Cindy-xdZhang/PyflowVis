#include "commonUtils.h"
#include "VectorFieldCompute.h"
#include "VastistasVelocityGenerator.h"
#include <Eigen/Dense>
#include <array>
#include <fstream>
#include <random>


using namespace std;
#include "stablized_texture_512png.cpp"


//give the analytical curl field from vectorfield, discrete curl field  as XDim X Ydim  vector.
std::vector<std::vector<double>> ComputeCurlAnalytical(const IUnsteadField2D& vectorfield, const int Xdim, const int Ydim, const double time)
{
	std::vector<std::vector<double>> curl(Ydim, std::vector<double>(Xdim, 0.0f));
	const auto domainMin = vectorfield.spatialDomainMinBoundary;
	const auto domainRange = vectorfield.spatialDomainMaxBoundary - domainMin;
	//discrete of curl field
	const auto spatialGridIntervalX = domainRange.x() / (double)(Xdim - 1);
	const auto spatialGridIntervalY = domainRange.y() / (double)(Ydim - 1);

	constexpr double delta_ratio = 0.00001;
	const Eigen::Vector2d   DeltaX = { delta_ratio * spatialGridIntervalX , 0.0 };
	const Eigen::Vector2d   DeltaY = { 0.0, delta_ratio * spatialGridIntervalY };
	const double inverse_DeltaX = 1.0f / (delta_ratio * spatialGridIntervalX);
	const double inverse_DeltaY = 1.0f / (delta_ratio * spatialGridIntervalY);

	// Calculate curl (vorticity) of the vector field
	for (int y = 0; y < Ydim; ++y) {
		for (int x = 0; x < Xdim; ++x) {
			double posX = domainMin.x() + x * spatialGridIntervalX;
			double posY = domainMin.y() + y * spatialGridIntervalY;

			Eigen::Vector2d  curl_pos = { posX,posY };
			Eigen::Vector2d v_xplus_delta = vectorfield.getVectorAnalytical(curl_pos + DeltaX, time);
			Eigen::Vector2d v_xminux_delta = vectorfield.getVectorAnalytical(curl_pos - DeltaX, time);
			Eigen::Vector2d dv_dx = (v_xplus_delta - v_xminux_delta) * 0.5f * (delta_ratio * inverse_DeltaX);

			Eigen::Vector2d v_yplus_delta = vectorfield.getVectorAnalytical(curl_pos + DeltaY, time);
			Eigen::Vector2d v_yminux_delta = vectorfield.getVectorAnalytical(curl_pos - DeltaY, time);

			Eigen::Vector2d du_dy = (v_yplus_delta - v_yminux_delta) * 0.5f * (delta_ratio * inverse_DeltaY);
			double curl_ = dv_dx(1) - du_dy(0);
			curl[y][x] = curl_;
		}
	}
	return curl;
}

std::vector<std::vector<double>> ComputeCurl(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim, double SpatialGridIntervalX, double SpatialGridIntervalY)
{
	std::vector<std::vector<double>> curl(Ydim, std::vector<double>(Xdim, 0.0f));
	const double inverse_grid_interval_x = 1.0f / (double)SpatialGridIntervalX;
	const double inverse_grid_interval_y = 1.0f / (double)SpatialGridIntervalY;
	// Calculate curl (vorticity) of the vector field
	for (int y = 1; y < Ydim - 1; ++y) {
		for (int x = 1; x < Xdim - 1; ++x) {
			Eigen::Vector2d dv_dx = (vecfieldData[y][x + 1] - vecfieldData[y][x - 1]) * 0.5f * inverse_grid_interval_x;
			Eigen::Vector2d du_dy = (vecfieldData[y + 1][x] - vecfieldData[y - 1][x]) * 0.5f * inverse_grid_interval_y;
			double curl_ = dv_dx(1) - du_dy(0);
			curl[y][x] = curl_;
		}
	}
	return curl;
}

std::vector<std::vector<double>> ComputeQCriterion(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim, double SpatialGridIntervalX, double SpatialGridIntervalY)
{
	std::vector<std::vector<double>> Q(Ydim, std::vector<double>(Xdim, 0.0));
	const double inverse_grid_interval_x = 1.0 / SpatialGridIntervalX;
	const double inverse_grid_interval_y = 1.0 / SpatialGridIntervalY;

	for (int y = 1; y < Ydim - 1; ++y) {
		for (int x = 1; x < Xdim - 1; ++x) {
			Eigen::Vector2d du_dx = (vecfieldData[y][x + 1] - vecfieldData[y][x - 1]) * 0.5 * inverse_grid_interval_x;
			Eigen::Vector2d dv_dy = (vecfieldData[y + 1][x] - vecfieldData[y - 1][x]) * 0.5 * inverse_grid_interval_y;
			Eigen::Matrix2d gradient;
			gradient << du_dx(0), du_dx(1),
				dv_dy(0), dv_dy(1);

			Eigen::Matrix2d S = 0.5 * (gradient + gradient.transpose());
			Eigen::Matrix2d Omega = 0.5 * (gradient - gradient.transpose());

			double Q_value = 0.5 * (Omega.squaredNorm() - S.squaredNorm());
			Q[y][x] = Q_value;
		}
	}
	return Q;
}

std::vector<std::vector<double>> ComputeDeltaCriterion(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim, double SpatialGridIntervalX, double SpatialGridIntervalY)
{
	std::vector<std::vector<double>> delta(Ydim, std::vector<double>(Xdim, 0.0));
	const double inverse_grid_interval_x = 1.0 / SpatialGridIntervalX;
	const double inverse_grid_interval_y = 1.0 / SpatialGridIntervalY;

	for (int y = 1; y < Ydim - 1; ++y) {
		for (int x = 1; x < Xdim - 1; ++x) {
			Eigen::Vector2d dv_dx = (vecfieldData[y][x + 1] - vecfieldData[y][x - 1]) * 0.5 * inverse_grid_interval_x;
			Eigen::Vector2d du_dy = (vecfieldData[y + 1][x] - vecfieldData[y - 1][x]) * 0.5 * inverse_grid_interval_y;
			Eigen::Matrix2d Jacobian;
			Jacobian << dv_dx(0), dv_dx(1),
				du_dy(0), du_dy(1);
			auto J2 = Jacobian * Jacobian;
			auto Q = -0.5 * J2.trace();
			auto R = Jacobian.determinant();
			double detlaVal = std::pow(Q / 3.0, 3.0) + std::pow(R / 2.0, 2.0);
			delta[y][x] = detlaVal;
		}
	}
	return delta;
}

std::vector<std::vector<double>> ComputeLambda2Criterion(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim, double SpatialGridIntervalX, double SpatialGridIntervalY)
{
	std::vector<std::vector<double>> lambda2(Ydim, std::vector<double>(Xdim, 0.0));
	const double inverse_grid_interval_x = 1.0 / SpatialGridIntervalX;
	const double inverse_grid_interval_y = 1.0 / SpatialGridIntervalY;

	for (int y = 1; y < Ydim - 1; ++y) {
		for (int x = 1; x < Xdim - 1; ++x) {
			Eigen::Vector2d du_dx = (vecfieldData[y][x + 1] - vecfieldData[y][x - 1]) * 0.5 * inverse_grid_interval_x;
			Eigen::Vector2d dv_dy = (vecfieldData[y + 1][x] - vecfieldData[y - 1][x]) * 0.5 * inverse_grid_interval_y;
			Eigen::Matrix2d gradient;
			gradient << du_dx(0), du_dx(1),
				dv_dy(0), dv_dy(1);

			Eigen::Matrix2d S = 0.5 * (gradient + gradient.transpose());
			Eigen::Matrix2d Omega = 0.5 * (gradient - gradient.transpose());

			Eigen::Matrix2d S2_ADD_OMEGA2 = S * S + Omega * Omega;
			Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(S2_ADD_OMEGA2);
			Eigen::Vector2d eigenvalues = solver.eigenvalues();
			lambda2[y][x] = eigenvalues(1); // The second largest eigenvalue
		}
	}
	return lambda2;
}

std::vector<std::vector<double>> ComputeIVD(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim, double SpatialGridIntervalX, double SpatialGridIntervalY)
{
	std::vector<std::vector<double>> IVD(Ydim, std::vector<double>(Xdim, 0.0));
	const double inverse_grid_interval_x = 1.0 / SpatialGridIntervalX;
	const double inverse_grid_interval_y = 1.0 / SpatialGridIntervalY;
	auto curlField = ComputeCurl(vecfieldData, Xdim, Ydim, SpatialGridIntervalX, SpatialGridIntervalY);
	double averageCurl = 0.0;
	for (const auto& row : curlField) {
		double sumRow = 0.0;
		for (auto val : row) {
			sumRow += val;
		}
		averageCurl += sumRow;
	}
	averageCurl /= (Xdim - 2) * (Ydim - 2);

	for (int y = 1; y < Ydim - 1; ++y) {
		for (int x = 1; x < Xdim - 1; ++x) {
			Eigen::Vector2d dv_dx = (vecfieldData[y][x + 1] - vecfieldData[y][x - 1]) * 0.5 * inverse_grid_interval_x;
			Eigen::Vector2d du_dy = (vecfieldData[y + 1][x] - vecfieldData[y - 1][x]) * 0.5 * inverse_grid_interval_y;
			double vorticity = dv_dx(1) - du_dy(0);

			IVD[y][x] = std::abs(vorticity - averageCurl);
		}
	}
	return IVD;
}

std::vector<std::vector<Eigen::Matrix2d>> ComputeNablaU(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim, double SpatialGridIntervalX, double SpatialGridIntervalY)
{
	std::vector<std::vector<Eigen::Matrix2d>> NablaU(Ydim, std::vector<Eigen::Matrix2d>(Xdim));
	const double inverse_grid_interval_x = 1.0 / SpatialGridIntervalX;
	const double inverse_grid_interval_y = 1.0 / SpatialGridIntervalY;

	for (int y = 0; y < Ydim; ++y) {
		for (int x = 0; x < Xdim; ++x) {
			int xPlus1 = std::min(x + 1, Xdim - 1);
			int xMinus1 = std::max(x - 1, 0);

			int yPlus1 = std::min(y + 1, Ydim - 1);
			int yMinus1 = std::max(y - 1, 0);

			Eigen::Vector2d du_dx = (vecfieldData[y][xPlus1] - vecfieldData[y][xMinus1]) * 0.5 * inverse_grid_interval_x;
			Eigen::Vector2d dv_dy = (vecfieldData[yPlus1][x] - vecfieldData[yMinus1][x]) * 0.5 * inverse_grid_interval_y;
			Eigen::Matrix2d gradient;
			gradient << du_dx(0), du_dx(1),
				dv_dy(0), dv_dy(1);

			NablaU[y][x] = gradient;
		}
	}

	return NablaU;
}

std::vector<std::vector<double>> ComputeSujudiHaimes(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim, double SpatialGridIntervalX, double SpatialGridIntervalY)
{
	std::vector<std::vector<double>> sujudiHaimes(Ydim, std::vector<double>(Xdim, 0.0));
	const double inverse_grid_interval_x = 1.0 / SpatialGridIntervalX;
	const double inverse_grid_interval_y = 1.0 / SpatialGridIntervalY;

	for (int y = 1; y < Ydim - 1; ++y) {
		for (int x = 1; x < Xdim - 1; ++x) {
			Eigen::Vector2d dv_dx = (vecfieldData[y][x + 1] - vecfieldData[y][x - 1]) * 0.5 * inverse_grid_interval_x;
			Eigen::Vector2d du_dy = (vecfieldData[y + 1][x] - vecfieldData[y - 1][x]) * 0.5 * inverse_grid_interval_y;
			Eigen::Matrix2d gradient;
			gradient << dv_dx(0), dv_dx(1),
				du_dy(0), du_dy(1);

			auto JV = gradient * vecfieldData[y][x];
			auto V = vecfieldData[y][x];
			// check JV and v is paralell?
			bool paralllel = JV.dot(V) == JV.norm() * V.norm();
			sujudiHaimes[y][x] = paralllel ? 1.0 : 0.0;
		}
	}
	return sujudiHaimes;
}

auto computeTargetCrtierion(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim, double SpatialGridIntervalX, double SpatialGridIntervalY, VORTEX_CRITERION criterionENUM)
{
	switch (criterionENUM) {
	case VORTEX_CRITERION::Q_CRITERION:
		return ComputeQCriterion(vecfieldData, Xdim, Ydim, SpatialGridIntervalX, SpatialGridIntervalY);
	case VORTEX_CRITERION::LAMBDA2_CRITERION:
		return ComputeLambda2Criterion(vecfieldData, Xdim, Ydim, SpatialGridIntervalX, SpatialGridIntervalY);
	case VORTEX_CRITERION::IVD_CRITERION:
		return ComputeIVD(vecfieldData, Xdim, Ydim, SpatialGridIntervalX, SpatialGridIntervalY);
	case VORTEX_CRITERION::DELTA_CRITERION:
		return ComputeDeltaCriterion(vecfieldData, Xdim, Ydim, SpatialGridIntervalX, SpatialGridIntervalY);
	case VORTEX_CRITERION::SUJUDI_HAIMES_CRITERION:
		return ComputeSujudiHaimes(vecfieldData, Xdim, Ydim, SpatialGridIntervalX, SpatialGridIntervalY);
	case VORTEX_CRITERION::SOBEL_EDGE_DETECTION:
		return ComputeSobelEdge(vecfieldData, Xdim, Ydim);
	case VORTEX_CRITERION::VELICITY_MAGINITUDE:
		return ComputeVelocityMagniture(vecfieldData, Xdim, Ydim);
	case VORTEX_CRITERION::CURL:
	default:
		return ComputeCurl(vecfieldData, Xdim, Ydim, SpatialGridIntervalX, SpatialGridIntervalY);

	}
}

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

	auto integratePathlineOneStep_RK4_analytical = [](const IUnsteadField2D& observerfield, double x, double y, double t, double dt) -> Eigen::Vector2d {
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
		Eigen::Vector2d k1 = observerfield.getVectorAnalytical(odeStepStartPoint, t);

		// stage 2
		Eigen::Vector2d stagePoint = odeStepStartPoint + k1 * a21 * h;
		Eigen::Vector2d k2 = observerfield.getVectorAnalytical(stagePoint, t + c2 * h);

		// stage 3
		stagePoint = odeStepStartPoint + (a31 * k1 + a32 * k2) * h;
		Eigen::Vector2d k3 = observerfield.getVectorAnalytical(stagePoint, t + c3 * h);

		// stage 4
		stagePoint = odeStepStartPoint + (a41 * k1 + a42 * k2 + a43 * k3) * h;
		Eigen::Vector2d k4 = observerfield.getVectorAnalytical(stagePoint, t + c4 * h);

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



	std::function<Eigen::Vector2d(const IUnsteadField2D& observerfield, double x, double y, double t, double dt) > integratorRk4 = integratePathlineOneStep_RK4_analytical;
	if (inputField.analyticalFlowfunc_ == nullptr) [[unlikely]] {
		integratorRk4 = integratePathlineOneStep_RK4;
		}

		// push init_velocity  &start point
	Eigen::Vector3d pointAndTime = { currentPoint(0), currentPoint(1), currentTime };
	pathPositions.emplace_back(pointAndTime);

	// integrate until either
	// - we reached the max iteration count
	// - we reached the upper limit of the time domain
	// - we ran out of spatial domain
	while ((!integrationOutOfDomainBounds) && (!outOfIntegrationTimeBounds) && (pathPositions.size() < maxIterationCount)) {

		// advance to a new point in the chart
		Eigen::Vector2d newPoint = integratorRk4(inputField, currentPoint(0), currentPoint(1), currentTime, integrationTimeStepSize);
		integrationOutOfDomainBounds = checkIfOutOfDomain(newPoint);
		if (!integrationOutOfDomainBounds) {
			auto newTime = currentTime + integrationTimeStepSize;
			// check if currentTime is out of the time domain -> we are done
			if ((targetIntegrationTime > startTime) && (newTime > targetIntegrationTime)) {
				outOfIntegrationTimeBounds = true;
			}
			else if ((targetIntegrationTime < startTime) && (newTime < targetIntegrationTime)) {
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
	bool suc = pathPositions.size() > 2;
	return suc;
}



std::vector<std::vector<PathlinePointInfo>> PathlineIntegrationInfoCollect2D(const UnSteadyVectorField2D& inputField, int KLines, const int outputPathlineLength, PATHLINE_SEEDING_SAMPLING sampleMEthod)
{
	Eigen::Matrix2d tmp;
	Eigen::Vector3d ttmp;
	Eigen::Vector2d tttmp;
	return PathlineIntegrationInfoCollect2D(inputField, KLines, tmp, ttmp, tttmp, outputPathlineLength, sampleMEthod);
}

std::vector<std::vector<PathlinePointInfo>>
PathlineIntegrationInfoCollect2D(const UnSteadyVectorField2D& inputField, int KLines,
	const Eigen::Matrix2d& deformMat, const Eigen::Vector3d& n_rc_si, const Eigen::Vector2d& txy, const int outputPathlineLength, PATHLINE_SEEDING_SAMPLING sampleMEthod)
{
	constexpr int maximumLength = 50; // pathline_dt=1/5 dt, thus total have 9*5=45 steps.
	auto maxBound = inputField.getSpatialMaxBoundary();
	auto minBound = inputField.getSpatialMinBoundary();
	auto domainRange = maxBound - minBound;
	auto tmax = inputField.tmax;
	auto tmin = inputField.tmin;
	auto Xdim = inputField.XdimYdim.x();
	auto Ydim = inputField.XdimYdim.y();
	if (inputField.analyticalFlowfunc_ == nullptr)
	{
		printf("error: input field has no analytical expression.. currently not support.");
		assert(false);
	}

	const double dt = (inputField.tmax - inputField.tmin) / (double)(inputField.timeSteps - 1);
	const double pathline_dt = (inputField.tmax - inputField.tmin) / (double)(outputPathlineLength - 1);

	// std::vector<std::vector<std::vector<double>>> curlFields(inputField.timeSteps);
	std::vector<std::vector<std::vector<double>>> ivdFields(inputField.timeSteps);
	std::vector<std::vector<std::vector<double>>> velocityMagnitudeFields(inputField.timeSteps);
	std::vector<std::vector<std::vector<Eigen::Matrix2d>>> nablaUfields(inputField.timeSteps);

	// precompute vorticity and ivd field
	for (size_t t = 0; t < inputField.timeSteps; t++) {

		// auto curlField = ComputeCurl(inputField.field[t], Xdim, Ydim, inputField.spatialGridInterval(0), inputField.spatialGridInterval(1));
		auto ivdField = ComputeIVD(inputField.field[t], Xdim, Ydim, inputField.spatialGridInterval(0), inputField.spatialGridInterval(1));
		// curlFields[t] = curlField;
		ivdFields[t] = ivdField;
		nablaUfields[t] = ComputeNablaU(inputField.field[t], Xdim, Ydim, inputField.spatialGridInterval(0), inputField.spatialGridInterval(1));

		velocityMagnitudeFields[t] = ComputeVelocityMagniture(inputField.field[t], Xdim, Ydim);
	}


	std::vector<std::vector<PathlinePointInfo>> clusterPathlines;
	std::vector<Eigen::Vector2d> seedings;
	if (sampleMEthod == PATHLINE_SEEDING_SAMPLING::GRID_CROSS_SEEDING)
	{
		seedings = GroupSeeding::GridCrossSampling(KLines / 2, KLines / 2, minBound, maxBound);//intotal klines*klines pathlines
	}
	else
	{
		seedings = GroupSeeding::RecTangular4ClusterSampling(KLines * KLines * 0.25, minBound, maxBound);//intotal klines*klines pathlines
	}
	const auto TOTAL_LINES = seedings.size();
	clusterPathlines.resize(TOTAL_LINES);

	for (size_t k = 0; k < TOTAL_LINES; k++) {
		//integrate pathline
		clusterPathlines[k].reserve(maximumLength);
		std::vector<Eigen::Vector3d> pathlinePositions;
		auto suc = PathhlineIntegrationRK4v2(seedings[k], inputField, 0, tmax, pathline_dt, pathlinePositions);
		while (suc == false) {
			seedings[k] = GroupSeeding::JittorReSeeding(seedings[k], minBound, maxBound);
			suc = PathhlineIntegrationRK4v2(seedings[k], inputField, 0, tmax, pathline_dt, pathlinePositions);
		}

		//compute properties.
		const auto startPoint = seedings[k];
		for (int step = 0; step < pathlinePositions.size(); step++) {
			auto px = pathlinePositions[step].x();
			auto py = pathlinePositions[step].y();
			auto time = pathlinePositions[step].z();
			double floatIndexX = (px - minBound.x()) / inputField.spatialGridInterval(0);
			double floatIndexY = (py - minBound.y()) / inputField.spatialGridInterval(1);
			double floatIndexT = (time - inputField.tmin) / dt;

			auto ivd = trilinear_interpolate(ivdFields, floatIndexX, floatIndexY, floatIndexT);
			auto nablau = trilinear_interpolate(nablaUfields, floatIndexX, floatIndexY, floatIndexT);
			Eigen::Vector2d pos = { px,py };
			auto velocity = inputField.getVectorAnalytical(pos, time);
			auto distance = sqrt((px - startPoint.x()) * (px - startPoint.x()) + (py - startPoint.y()) * (py - startPoint.y()));

			std::vector<double> pathlinePointAndInfo = { px, py, time, ivd, distance,velocity(0),velocity(1), nablau(0,0),nablau(0,1) ,nablau(1,1) };


			checkVectorValues(pathlinePointAndInfo);
			clusterPathlines[k].emplace_back(pathlinePointAndInfo);
		}
	}


	const auto rc = n_rc_si((int)VastisParamRC_N::VastisParamRC);
	const auto si = n_rc_si.z();
	const auto deformInverse = deformMat.inverse();
	auto judgeVortex = [si, rc, txy, deformInverse](const Eigen::Vector2d& pos) -> double {
		if ((si == 1.0 || si == 2.0) && rc > 0) {
			auto originalPos = deformInverse * (pos - txy);
			auto dx = rc - originalPos.norm();
			return dx > 0 ? 1.0 : 0.0;
		}
		return 0.0;
		};
	constexpr double paddingValue = -1000.0;
	constexpr double paddingValue_Time = -1000 * 3.1415926;
	std::vector PadValue = { paddingValue , paddingValue , paddingValue_Time , paddingValue , paddingValue , paddingValue , paddingValue ,paddingValue, paddingValue };
	for (auto& pathline : clusterPathlines) {
		Eigen::Vector2d start_pos = { pathline.at(0).at(0), pathline.at(0).at(1) };
		// the first point is the distance to it self(always zero), use it as the segmentation label for this pathline.
		pathline.at(0).at((int)PATHLINE_POINT_INFO::DISTANCE_OR_LABEL) = judgeVortex(start_pos);
		pathline.resize(outputPathlineLength, PadValue);
	}

	// ----------------------------------------------------------
	// verify code to make sure every pathline has same time stamps
	// ----------------------------------------------------------
	for (size_t step = 0; step < outputPathlineLength; step++) {
		auto time = clusterPathlines[0].at(step).at((int)PATHLINE_POINT_INFO::TIME_T);
		if (time == paddingValue_Time) {
			for (size_t l = 1; l < clusterPathlines.size(); l++) {
				if (clusterPathlines[l].at(step).at((int)PATHLINE_POINT_INFO::TIME_T) != paddingValue_Time) {
					time = clusterPathlines[l].at(step).at((int)PATHLINE_POINT_INFO::TIME_T);
				}
			}
		}

		for (size_t l = 1; l < clusterPathlines.size(); l++) {
			if (clusterPathlines[l].at(step).at((int)PATHLINE_POINT_INFO::TIME_T) != time && clusterPathlines[l].at(step).at((int)PATHLINE_POINT_INFO::TIME_T) != paddingValue_Time) {
				printf("error: get not equal timestamps of path lines.");
				assert(false);
			};
		}
	}





	return clusterPathlines;
}

Eigen::Vector2d GroupSeeding::JittorReSeeding(const Eigen::Vector2d& preSeeding, Eigen::Vector2d domainMin, Eigen::Vector2d domainMax)
{
	static std::random_device rd;
	static  std::mt19937 rng(rd());
	const auto domainRange = domainMax - domainMin;
	//plane center vector:
	Eigen::Vector2d Center = domainMin + 0.5 * domainRange;
	Eigen::Vector2d  Direction = Center - preSeeding;
	std::uniform_real_distribution<> disX(0.00001, 0.5);
	double shift = disX(rng);

	Eigen::Vector2d   seeding = preSeeding + shift * Direction;

	return seeding;
}

std::vector<Eigen::Vector2d> GroupSeeding::generateSeedingsRec(double xmin, double xmax, double ymin, double ymax, int K)
{
	static std::random_device rd;
	static  std::mt19937 rng(rd());
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
}

std::vector<Eigen::Vector2d> GroupSeeding::RecTangular4ClusterSampling(int samplesPercluster, Eigen::Vector2d domainMin, Eigen::Vector2d domainMax)
{
	const auto domainRange = domainMax - domainMin;
	constexpr int NClusters = 4;
	std::array<Eigen::Vector2d, 4> domainCenters{
		Eigen::Vector2d(0.25 * domainRange.x() + domainMin.x(), 0.25 * domainRange.y() + domainMin.y()), //-1,-1
		Eigen::Vector2d(0.25 * domainRange.x() + domainMin.x(), 0.75 * domainRange.y() + domainMin.y()), // 1,1
		Eigen::Vector2d(0.75 * domainRange.x() + domainMin.x(), 0.25 * domainRange.y() + domainMin.y()), // 1,-1
		Eigen::Vector2d(0.75 * domainRange.x() + domainMin.x(), 0.75 * domainRange.y() + domainMin.y()) //-1,1
	};
	std::vector<Eigen::Vector2d> res;
	res.reserve(NClusters * samplesPercluster);
	for (int i = 0; i < NClusters; i++) {
		// the domain d is 1/5*range,(1/10 in one direction).
		Eigen::Vector2d domainCenter = domainCenters[i];

		auto Domain_minx = std::max(domainCenter.x() - 0.25 * domainRange.x(), domainMin.x());
		auto Domain_maxx = std::min(domainCenter.x() + 0.25 * domainRange.x(), domainMax.x());
		auto Domain_miny = std::max(domainCenter.y() - 0.25 * domainRange.y(), domainMin.y());
		auto Domain_maxy = std::min(domainCenter.y() + 0.25 * domainRange.y(), domainMax.y());
		std::vector<Eigen::Vector2d>  seedings = generateSeedingsRec(Domain_minx, Domain_maxx, Domain_miny, Domain_maxy, samplesPercluster);
		res.insert(res.end(), seedings.begin(), seedings.end());
	}
	return res;
}

std::vector<Eigen::Vector2d> GroupSeeding::generateSeedingsCross(double xCenter, double yCenter, double dx, double dy)
{
	std::vector<Eigen::Vector2d> seedings;
	seedings.emplace_back(xCenter - dx, yCenter);
	seedings.emplace_back(xCenter, yCenter - dy);
	seedings.emplace_back(xCenter, yCenter);
	seedings.emplace_back(xCenter + dx, yCenter);
	seedings.emplace_back(xCenter, yCenter + dy);
	return seedings;
}

std::vector<Eigen::Vector2d> GroupSeeding::GridCrossSampling(int gx, int gy, Eigen::Vector2d domainMin, Eigen::Vector2d domainMax)
{
	Eigen::Vector2d domainRange = domainMax - domainMin;
	//need to get rid of points that are too close to boundary
	Eigen::Vector2d SamplingDomainStart = domainMin + 0.1 * domainRange;
	Eigen::Vector2d SamplingDomainEnd = domainMin + 0.9 * domainRange;
	Eigen::Vector2d SamplingDomainRange = SamplingDomainEnd - SamplingDomainStart;

	const double sampleGrid_dx = SamplingDomainRange.x() / (gx - 1);
	const double sampleGrid_dy = SamplingDomainRange.y() / (gy - 1);

	const double sampleCross_dx = sampleGrid_dx / 3.0;
	const double sampleCross_dy = sampleGrid_dy / 3.0;
	std::vector<Eigen::Vector2d> res;
	res.reserve(gx * gy * 4);
	for (size_t i = 0; i < gy; i++)
		for (size_t j = 0; j < gx; j++)
		{
			const double centerPointX = SamplingDomainStart.x() + sampleGrid_dx * j;
			const double centerPointY = SamplingDomainStart.y() + sampleGrid_dy * i;
			std::vector<Eigen::Vector2d>  seedings = generateSeedingsCross(centerPointX, centerPointY, sampleCross_dx, sampleCross_dy);
			res.insert(res.end(), seedings.begin(), seedings.end());
		};

	return res;
}

const SteadyVectorField2D UnSteadyVectorField2D::getVectorfieldSliceAtTime(int t) const
{
	SteadyVectorField2D vecfield;
	vecfield.spatialDomainMinBoundary = spatialDomainMinBoundary;
	vecfield.spatialDomainMaxBoundary = spatialDomainMaxBoundary;
	vecfield.spatialGridInterval = spatialGridInterval;
	vecfield.XdimYdim = this->XdimYdim;

	if (t >= 0 && t < timeSteps) {
		if (this->analyticalFlowfunc_ != nullptr) {
			assert(this->tmax> this->tmin && this->timeSteps>1);
			const auto dt = (this->tmax - this->tmin) / (double)(this->timeSteps - 1);
			assert(dt>0.0);
			double physical_time = dt * t + this->tmin;
			vecfield.analyticalFlowfunc_ = [this, physical_time](const Eigen::Vector2d& pos, double use_less_time) {
				return this->analyticalFlowfunc_(pos, physical_time);
				};
			vecfield.resampleFromAnalyticalExpression();

		}
		else
			vecfield.field = field[static_cast<int>(t)];
	}
	else {
		assert(false);
	}

	return vecfield;
}
