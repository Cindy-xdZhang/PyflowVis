#include <array>
#include "transformation.h"
#include "flowGenerator.h"
#include "VectorFieldCompute.h"
#include "cereal/archives/binary.hpp"
#include "cereal/archives/json.hpp"
#include "cereal/types/array.hpp"
#include "cereal/types/tuple.hpp"
#include "cereal/types/vector.hpp"
#include <execution>
#include <filesystem>
#include <fstream>
#include <algorithm>

#include <magic_enum/magic_enum.hpp>

#define RENDERING_LIC_SAVE_DATA
#define VALIDATE_RECONSTRUCTION_RESULT
using namespace std;
// normal distribution from supplementary material of Vortex Boundary Identification Paper
std::normal_distribution<double> UnsteadyPathlneDataSetGenerator::genTheta(0.0, 0.50);
std::normal_distribution<double> UnsteadyPathlneDataSetGenerator::genSx(0.0, 3.59);
std::normal_distribution<double> UnsteadyPathlneDataSetGenerator::genSy(0.0, 2.24);
std::normal_distribution<double> UnsteadyPathlneDataSetGenerator::genTx(0.0, 1.34);
std::normal_distribution<double> UnsteadyPathlneDataSetGenerator::genTy(0.0, 1.27);
namespace {
	constexpr int Xdim = 9, Ydim = 9;
	constexpr double tmin = 0.0;
	constexpr double tmax = M_PI * 0.25;
	constexpr int unsteadyFieldTimeStep = 5;//dt of vector field = (pi*0.25)/(5-1)=pi/16
	constexpr int outputPathlineLength = 9;//dt of vector field = (pi*0.25)/(9-1)=pi/32

	Eigen::Vector2d domainMinBoundary = { -2.0, -2.0 };
	Eigen::Vector2d domainMaxBoundary = { 2.0, 2.0 };

	// lic parameters
	constexpr int LicImageSize = 128;
	constexpr int LicSaveFrequency = 3; // every 2 time steps save one
	const double stepSize = 0.012;
	const int maxLICIteratioOneDirection = 256;
#if defined(DISABLE_CPP_PARALLELISM) || defined(_DEBUG)
	auto policy
		= std::execution::seq;
#else
	auto policy
		= std::execution::par_unseq;
#endif
	// Create a random number generator
	std::random_device rd;
	std::mt19937 rng(rd());


}







std::pair<Eigen::Vector3d, Eigen::Vector3d> generateRandomABCVectors()
{
	// Random device and generator
	std::random_device rd;
	std::mt19937 gen(rd());
	// range of velocity and acc is -0.3-0.3, -0.01-0.01(from paper "robust reference frame...")
	std::uniform_real_distribution<double> dist(-0.3, 0.3);
	std::uniform_real_distribution<double> dist_acc(-0.005, 0.005); // robust paper in domain [-2,2] acc is range [-0.01,0.01], our domain is [0,2] thus multiply 0.5 of acc range
	std::uniform_int_distribution<int> dist_int(0, 7);
	// Generate two random Eigen::Vector3d
	auto option = dist_int(gen);
	if (option == 0) {
		Eigen::Vector3d vec1(dist(gen), dist(gen), dist(gen));
		Eigen::Vector3d vec2(dist_acc(gen), dist_acc(gen), dist_acc(gen));
		return std::make_pair(vec1, vec2);
	}
	else if (option == 1) {
		Eigen::Vector3d vec1(dist(gen), dist(gen), 0);
		Eigen::Vector3d vec2(dist_acc(gen), dist_acc(gen), 0);
		return std::make_pair(vec1, vec2);
	}
	else if (option == 2) {
		Eigen::Vector3d vec1(dist(gen), dist(gen), 0);
		Eigen::Vector3d vec2(0, 0, 0);
		return std::make_pair(vec1, vec2);
	}
	else if (option == 3) {
		Eigen::Vector3d vec1(0, 0, dist(gen));
		Eigen::Vector3d vec2(0, 0, 0);
		return std::make_pair(vec1, vec2);
	}
	else if (option == 4) {
		Eigen::Vector3d vec1(0, 0, dist(gen));
		Eigen::Vector3d vec2(0, 0, dist_acc(gen));
		return std::make_pair(vec1, vec2);
	}
	else if (option == 5) {
		Eigen::Vector3d vec1(dist(gen), 0, 0);
		Eigen::Vector3d vec2(dist_acc(gen), 0, 0);
		return std::make_pair(vec1, vec2);
	}
	else if (option == 6) {
		Eigen::Vector3d vec1(0, dist(gen), 0);
		Eigen::Vector3d vec2(0, dist_acc(gen), 0);
		return std::make_pair(vec1, vec2);
	}
	else {
		Eigen::Vector3d vec1(0, 0, 0);
		Eigen::Vector3d vec2(0, 0, 0);
		return std::make_pair(vec1, vec2);
	}
}



std::vector<std::vector<Eigen::Vector3d>> addSegmentationVisualization(const std::vector<std::vector<Eigen::Vector3d>>& inputLicImage, const SteadyVectorField2D& vectorField, const Eigen::Vector3d& meta_rc_n_si, const Eigen::Vector2d& txy, const Eigen::Matrix2d& deformMat)
{
	// if si=0 then no vortex
	if (meta_rc_n_si.z() == 0.0 || meta_rc_n_si.z() == 3.0) {
		return inputLicImage;
	}

	constexpr int randomPickKlinesToDraw = 24;
	const Eigen::Vector2d& domainMax = vectorField.spatialDomainMaxBoundary;
	const Eigen::Vector2d& domainMIn = vectorField.spatialDomainMinBoundary;
	const auto domainRange = domainMax - domainMIn;
	const int licImageSizeY = inputLicImage.size();
	const int licImageSizeX = inputLicImage[0].size();
	auto InputLicImage = inputLicImage;

	const auto rc = meta_rc_n_si((int)VastisParamRC_N::VastisParamRC);
	const auto deformInverse = deformMat.inverse();
	/*const auto dx = domainRange(0) / licImageSizeX;
	const auto dy = domainRange(1) / licImageSizeY;
	const auto maxDistanceAnyPoint2gridPoints = sqrt((0.5 * dx) * (0.5 * dx) + (0.5 * dy) * (0.5 * dy));*/

	auto judgeVortex = [rc, txy, deformInverse](const Eigen::Vector2d& pos) -> bool {
		auto originalPos = deformInverse * (pos - txy);
		auto dx = rc - originalPos.norm();
		return dx > 0;
		};

	for (size_t i = 0; i < licImageSizeX; i++) {
		for (size_t j = 0; j < licImageSizeY; j++) {
			// map position from texture image grid coordinate to vector field
			double ratio_x = (double)((double)i / (double)licImageSizeX);
			double ratio_y = (double)((double)j / (double)licImageSizeY);

			// physicalPositionInVectorfield
			Eigen::Vector2d pos = { ratio_x * domainRange(0) + domainMIn(0),
				ratio_y * domainRange(1) + domainMIn(1) };
			if (judgeVortex(pos)) {
				auto preColor = InputLicImage[j][i];
				InputLicImage[j][i] = 0.5 * preColor + 0.5 * Eigen::Vector3d(1.0, 1.0, 0.0);
			}
		}
	}
	return InputLicImage;
}


std::vector<std::vector<Eigen::Vector3d>> addPathlineVisualization(const std::vector<std::vector<Eigen::Vector3d>>& inputLicImage, const Eigen::Vector2d& domainMIn, const Eigen::Vector2d& domainMax, const double t_min, const double t_max,
	const std::vector<std::vector<std::vector<double>>>& InputPathlines)
{
	auto colorCodingPathline = [](double time) -> Eigen::Vector3d {
		// Clamp time to [0, 1] range
		time = std::max(0.0, std::min(1.0, time));
		// White color
		Eigen::Vector3d white(1.0, 1.0, 1.0);

		// Blue color
		Eigen::Vector3d blue(0.0, 0.0, 1.0);

		// Linear interpolation between white and blue based on time
		return (1.0 - time) * blue + time * white;
		};

	constexpr int randomPickKlinesToDraw = 96;
	const Eigen::Vector2d domainRange = domainMax - domainMIn;
	const int licImageSizeY = inputLicImage.size();
	const int licImageSizeX = inputLicImage[0].size();
	auto InputLicImage = inputLicImage;

	std::uniform_int_distribution<int> dist_int(0, InputPathlines.size() - 1);

	std::vector<Eigen::Vector2d> selectedStartingPoints;
	for (int i = 0; i < randomPickKlinesToDraw; i++) {
		int lineIdx;
		bool isFarEnough;
		int failureTime = 0;
		double distanceThreshold = 0.1;
		do {
			lineIdx = dist_int(rng);
			isFarEnough = true;
			const Eigen::Vector2d currentStartPoint(InputPathlines[lineIdx][0][0], InputPathlines[lineIdx][0][1]);
			if (failureTime >= 10)
			{
				distanceThreshold *= 0.5;
				failureTime = 0;
			}

			for (const auto& previousStartPoint : selectedStartingPoints) {
				if ((currentStartPoint - previousStartPoint).norm() < distanceThreshold) {
					isFarEnough = false;
					failureTime++;
					break;
				}
			}
		} while (!isFarEnough);


		selectedStartingPoints.push_back(Eigen::Vector2d(InputPathlines[lineIdx][0][0], InputPathlines[lineIdx][0][1]));
		auto unique_pathline = InputPathlines[lineIdx];

		// draw the pathline points to lic image
		for (const auto& point : unique_pathline) {
			Eigen::Vector2d pos(point[0], point[1]); // Convert  point to 2D by taking x and y
			double time = point[2];
			double normalized_x = (pos.x() - domainMIn.x()) / domainRange.x();
			double normalized_y = (pos.y() - domainMIn.y()) / domainRange.y();

			int gridX = static_cast<int>(normalized_x * licImageSizeX);
			int gridY = static_cast<int>(normalized_y * licImageSizeY);

			if (gridX >= 0 && gridX < licImageSizeX && gridY >= 0 && gridY < licImageSizeY) {
				double ratio_time = (time - t_min) / (t_max - t_min);
				Eigen::Vector3d color = colorCodingPathline(ratio_time); // Blue for pathlines
				auto preColor = InputLicImage[gridY][gridX];
				InputLicImage[gridY][gridX] = 0.2 * preColor + 0.8 * color;
			}
		}
	}
	return InputLicImage;
}


std::vector<std::vector<Eigen::Vector3d>>  addSegmentationVis4MixtureVortex(const std::vector<std::vector<Eigen::Vector3d>>& inputLicImage, const SteadyVectorField2D& vectorField, const std::vector<VastisParamter>& params)
{

	auto InputLicImage = inputLicImage;
	for (size_t i = 0; i < params.size(); i++) {
		const  auto& parm = params.at(i);
		const int si = std::get<(int)VASTIS_PARAM::VastisParamSI >(parm);
		if (si == (int)VastisVortexType::saddle)
		{
			return inputLicImage;
		}
		const Eigen::Vector2d rc_n = std::get<(int)VASTIS_PARAM::VastisParamRC_N>(parm);
		const Eigen::Vector2d txy = std::get<(int)VASTIS_PARAM::VastisParamTXY>(parm);
		const Eigen::Vector3d sxsytheta = std::get<(int)VASTIS_PARAM::VastisParamSXSYTHETA>(parm);
		const double theta = sxsytheta.z();
		const double sx = sxsytheta.x();
		const double sy = sxsytheta.y();
		Eigen::Vector3d meta_rc_n_si = { rc_n.x(), rc_n.y(), (double)si };
		Eigen::Matrix2d deformMat = Eigen::Matrix2d::Identity();
		deformMat(0, 0) = sx * cos(theta);
		deformMat(0, 1) = -sy * sin(theta);
		deformMat(1, 0) = sx * sin(theta);
		deformMat(1, 1) = sy * cos(theta);
		InputLicImage = addSegmentationVisualization(InputLicImage, vectorField, meta_rc_n_si, txy, deformMat);
	}
	return InputLicImage;

}



std::vector<std::vector<uint8_t>>  generateSegmentationBinaryMask(const Eigen::Vector3d& rc_n_si, const Eigen::Vector2d& txy, const Eigen::Matrix2d& deformMat, const int Xdim, const int Ydim, const Eigen::Vector2d& gridInterval, const Eigen::Vector2d& domainMin)
{
	auto rc = rc_n_si((int)VastisParamRC_N::VastisParamRC);
	auto si = rc_n_si.z();
	const auto deformInverse = deformMat.inverse();

	std::vector<std::vector<uint8_t>> segmentation(Ydim, std::vector<uint8_t>(Xdim, 0));
	if (si == 0.0 || si == 3.0) [[unlikely]] {
		return segmentation;
		}
		auto judgeVortex = [si, rc, txy, deformInverse](const Eigen::Vector2d& pos) -> uint8_t {
		auto originalPos = deformInverse * (pos - txy);
		auto dx = rc - originalPos.norm();
		return dx > 0 ? 1 : 0;
		};

	for (int gy = 0; gy < Ydim; gy++)
		for (int gx = 0; gx < Xdim; gx++) {
			Eigen::Vector2d pos = { domainMin.x() + gridInterval.x() * gx, domainMin.y() + gridInterval.y() * gy };
			segmentation[gy][gx] = judgeVortex(pos);
		}
	return segmentation;
}
std::vector<std::vector<uint8_t>>  generateSegmentationBinaryMaskMix(const std::vector<VastisParamter>& params, const int Xdim, const int Ydim, const Eigen::Vector2d& gridInterval, const Eigen::Vector2d& domainMin)
{


	std::vector<std::vector<uint8_t>> final_segmentation(Ydim, std::vector<uint8_t>(Xdim, 0));
	std::vector <std::vector<std::vector<uint8_t>>>segmentations(params.size(), std::vector<std::vector<uint8_t>>(Ydim, std::vector<uint8_t>(Xdim, 0)));

	for (size_t i = 0; i < params.size(); i++) {
		const  auto& parm = params.at(i);
		const int si = std::get<(int)VASTIS_PARAM::VastisParamSI >(parm);
		if (si == (int)VastisVortexType::saddle)
		{
			return final_segmentation;
		}
		const Eigen::Vector2d rc_n = std::get<(int)VASTIS_PARAM::VastisParamRC_N>(parm);
		const Eigen::Vector2d txy = std::get<(int)VASTIS_PARAM::VastisParamTXY>(parm);
		const Eigen::Vector3d sxsytheta = std::get<(int)VASTIS_PARAM::VastisParamSXSYTHETA>(parm);
		const double theta = sxsytheta.z();
		const double sx = sxsytheta.x();
		const double sy = sxsytheta.y();
		Eigen::Vector3d meta_rc_n_si = { rc_n.x(), rc_n.y(), (double)si };
		Eigen::Matrix2d deformMat = Eigen::Matrix2d::Identity();
		deformMat(0, 0) = sx * cos(theta);
		deformMat(0, 1) = -sy * sin(theta);
		deformMat(1, 0) = sx * sin(theta);
		deformMat(1, 1) = sy * cos(theta);
		auto segmentationLabelOneVortex = generateSegmentationBinaryMask(meta_rc_n_si, txy, deformMat, Xdim, Ydim, gridInterval, domainMin);
		segmentations[i] = std::move(segmentationLabelOneVortex);
	}
	//merge segmentations
	for (size_t i = 0; i < segmentations.size(); i++) {
		for (size_t y = 0; y < Ydim; y++) {
			for (size_t x = 0; x < Xdim; x++) {
				if (segmentations[i][y][x] == 1) {
					final_segmentation[y][x] = 1;
				}
			}
		}
	}

	return final_segmentation;
}







// number of result traing data = Nparamters * samplePerParameters * observerPerSetting;dataSetSplitTag should be "train"/"test"/"validation"
void generateUnsteadyFieldV0(int Nparamters, int samplePerParameters, int observerPerSetting, const std::string in_root_fodler, const std::string dataSetSplitTag)
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

	const auto rc_n_paramters = generateVastisRC_NParamters(Nparamters, dataSetSplitTag);

	std::normal_distribution<double> genTheta(0.0, 0.50); // rotation angle's distribution
	// normal distribution from supplementary material of Vortex Boundary Identification Paper
	std::normal_distribution<double> genSx(0, 3.59);
	std::normal_distribution<double> genSy(0, 2.24);
	std::normal_distribution<double> genTx(0.0, 1.34);
	std::normal_distribution<double> genTy(0.0, 1.27);
	double minMagintude = INFINITY;
	double maxMagintude = -INFINITY;

	for_each(policy, rc_n_paramters.begin(), rc_n_paramters.cend(), [&](const std::pair<double, double>& params) {
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

				const string vortexTypeName = string{ magic_enum::enum_name<VastisVortexType>(Si) };
				const string sample_tag_name
					= "sample_" + to_string(taskSampleId) + vortexTypeName;
				string metaFilename = task_folder + sample_tag_name + "meta.json";
				string velocityFilename = task_folder + sample_tag_name + ".bin";
				const std::vector<float> rawSteadyData = flatten2DVectorsAs1Dfloat(steadyFieldResData);
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
					// addSegmentationVisualization(outputSteadyTexture, steadyField, n_rc_si, domainMaxBoundary, domainMinBoundary, txy, deformMat);
					string steadyField_name = "steady_beforeTransformation_";
					string licFilename0 = task_licfolder + sample_tag_name + steadyField_name + "lic.png";
					saveAsPNG(outputSteadyTexture, licFilename0);

					auto outputTextures = LICAlgorithm_UnsteadyField(unsteady_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
					auto outputTexturesReconstruct = LICAlgorithm_UnsteadyField(reconstruct_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
					/*        addSegmentationVisualization(outputTextures, unsteady_field, n_rc_si, domainMaxBoundary, domainMinBoundary, txy, deformMat);*/
					for (size_t i = 0; i < outputTextures.size(); i += LicSaveFrequency) {
						string tag_name = sample_tag_name + "killing_deformed_" + std::to_string(i);
						string licFilename = task_licfolder + tag_name + "lic.png";

						saveAsPNG(outputTextures[i], licFilename);

						string tag_name_rec = sample_tag_name + "reconstruct_" + std::to_string(i);
						string licFilename_rec = task_licfolder + tag_name_rec + "lic.png";
						saveAsPNG(outputTexturesReconstruct[i], licFilename_rec);
					}
					auto rawUnsteadyFieldData = flatten3DVectorsAs1Dfloat(unsteady_field.field);
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
							std::array<double, 2>tmp_Ct = { unsteady_field.c_t[i].x(), unsteady_field.c_t[i].y() };
							c_t.push_back(tmp_Ct);
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
	}
	else {
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


namespace DBG_TEST {


	void testCriterion()
	{
		const int LicImageSize = 256;
		const int unsteadyFieldTimeStep = 32;
		const double stepSize = 0.01;
		const int maxLICIteratioOneDirection = 256;
		int numVelocityFields = 1; // num of fields per n, rc parameter setting
		const int LicSaveFrequency = 2;
		Eigen::Vector2d domainMinBoundary = { -2.0, -2.0 };
		Eigen::Vector2d domainMaxBoundary = { 2.0, 2.0 };
		std::string root_folder = "../data/test_criterion/";
		if (!filesystem::exists(root_folder)) {
			filesystem::create_directories(root_folder);
		}
		// Create an instance of AnalyticalFlowCreator
		Eigen::Vector2i grid_size(Xdim, Ydim);
		int time_steps = unsteadyFieldTimeStep;
		AnalyticalFlowCreator flowCreator(grid_size, time_steps, domainMinBoundary, domainMaxBoundary, tmin, tmax);
		auto classicalAnalyticalFields = flowCreator.generateAnalyticalTestSuite();


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

		for (const auto& [fieldName, unsteady_field] : classicalAnalyticalFields) {

			std::vector<std::vector<PathlinePointInfo>> ClusterPathlines = PathlineIntegrationInfoCollect2D(unsteady_field, 25, outputPathlineLength);
			const auto steadyZeroSlice = unsteady_field.getVectorfieldSliceAtTime(0);
			auto outputSteadyTexture = LICAlgorithm(steadyZeroSlice, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
			outputSteadyTexture = addPathlineVisualization(outputSteadyTexture, domainMinBoundary, domainMaxBoundary, tmin, tmax, ClusterPathlines);
			string pathlinePngName = fieldName + "_pathline.png";
			saveAsPNG(outputSteadyTexture, root_folder + pathlinePngName);

			for (size_t i = 0; i < enumCriterion.size(); i++) {
				auto criterion = enumCriterion[i].first;
				auto name_criterion = enumCriterion[i].second;
				auto outputTextures = LICAlgorithm_UnsteadyField(unsteady_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection, criterion);

				for (size_t i = 1; i < unsteadyFieldTimeStep; i += LicSaveFrequency) {
					string tag_name0 = fieldName + "_" + name_criterion + std::to_string(i);
					string licFilename0 = root_folder + tag_name0 + "lic.png";
					saveAsPNG(outputTextures[i], licFilename0);
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


	}
}


namespace REPRODUCE {

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
		const auto paramters = generateVastisRC_NParamters(Nparamters, dataSetSplitTag);
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

				Eigen::Vector3d rc_n_si = { rc, n, (double)Si };
				const SteadyVectorField2D steadyField = generator.generateSteadyField_VortexBoundaryVIS2020(tx, ty, sx, sy, theta, Si);
				const std::vector<std::vector<Eigen::Vector2d>>& steadyFieldResData = steadyField.field;

				printf(".");
				const int taskSampleId = sample;
				const string vortexTypeName = string{ magic_enum::enum_name<VastisVortexType>(Si) };
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

					string steadyField_name = "steady_";
					string licFilename0 = task_licfolder + sample_tag_name + steadyField_name + "lic.png";
					saveAsPNG(outputSteadyTexture, licFilename0);

					auto rawSteadyFieldData = flatten2DVectorsAs1Dfloat(steadyField.field);
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
						archive_o(cereal::make_nvp("rc_n_si", rc_n_si));
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
		}
		else {
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

	// reproduce paper : Robust Reference Frame Extraction from Unsteady 2D Vector Fields with Convolutional Neural Networks
	void generateUnsteadyFieldMixture(int Nfield, int observerPerField, const std::string in_root_fodler, const std::string dataSetSplitTag)
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
			auto steadyMixture = generator.generateSteadyFieldMixtureRobustPaper(3);

			for (size_t observerIndex = 0; observerIndex < observerPerField; observerIndex++) {
				printf(".");

				const int taskSampleId = threadID * totalSamplesThisThread + observerIndex;

				const string sample_tag_name
					= "sample_" + to_string(taskSampleId);
				string metaFilename = task_folder + sample_tag_name + "meta.json";
				string velocityFilename = task_folder + sample_tag_name + ".bin";
				const std::vector<float> rawSteadyData = flatten2DVectorsAs1Dfloat(steadyMixture.field);
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
					auto rawUnsteadyFieldData = flatten3DVectorsAs1Dfloat(unsteady_field.field);
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
		}
		else {
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
}


void generateUnsteadyFieldPathlineclassicalFields(int Nparamters, int observerPerSetting, const std::string in_root_fodler, const std::string dataSetSplitTag)
{

	// rotating zero field
	// beads
	// rfc
}



void DataSetGenBase::GenDataset(int Nparamters, int samplePerParameters, int observerPerSetting, const std::string& in_root_fodler, double rateVal_Train /*= 0.1*/, double rateVal_Test /*= 0.1*/) {
	this->domainRange = domainMaxBoundary - domainMinBoundary;
	this->gridInterval = {
		(domainMaxBoundary(0) - domainMinBoundary(0)) / (Xdim - 1),
		(domainMaxBoundary(1) - domainMinBoundary(1)) / (Ydim - 1)
	};
	string root_folder = in_root_fodler + "/X" + to_string(Xdim) + "_Y" + to_string(Ydim) + "_T" + to_string(unsteadyFieldTimeStep) + "/";
	std::vector<string>splits = { "train","validation","test" };
	std::vector<std::tuple<int, int, int>>SampleCounts = { {Nparamters,samplePerParameters,observerPerSetting},//train
		{(int)(Nparamters * rateVal_Train), samplePerParameters,observerPerSetting},//validation
		{(int)(Nparamters * rateVal_Train), samplePerParameters,observerPerSetting} }; //test};
	int idx = 0;
	for (const auto& split : splits) {
		this->split_root_folder = root_folder + split + "/";
		if (!filesystem::exists(split_root_folder)) {
			filesystem::create_directories(split_root_folder);
		}
		const auto samplesCountOneSplit = SampleCounts[idx++];
		const auto NRC_Samples = std::get<0>(samplesCountOneSplit);
		const auto SamplesPerNRC_param = std::get<1>(samplesCountOneSplit);
		const auto observerPerSettingPerSample = std::get<2>(samplesCountOneSplit);
		GenOneSplit(NRC_Samples, SamplesPerNRC_param, observerPerSettingPerSample, split);
	}

	// create Root meta json file, save plane information here instead of every sample's meta file
	string taskFolder_rootMetaFilename = root_folder + "meta.json";
	// save root meta info:
	std::ofstream root_jsonOut(taskFolder_rootMetaFilename);
	if (!root_jsonOut.good()) {
		printf("couldn't open file: %s", taskFolder_rootMetaFilename.c_str());
		return;
	}
	else {
		cereal::JSONOutputArchive archive_o(root_jsonOut);
		archive_o(CEREAL_NVP(Xdim));
		archive_o(CEREAL_NVP(Ydim));
		archive_o(CEREAL_NVP(unsteadyFieldTimeStep));
		archive_o(CEREAL_NVP(domainMinBoundary));
		archive_o(CEREAL_NVP(domainMaxBoundary));
		archive_o(CEREAL_NVP(tmin));
		archive_o(CEREAL_NVP(tmax));
		archive_o(CEREAL_NVP(outputPathlineLength));

		// save min and max
		archive_o(cereal::make_nvp("minV", this->minV));
		archive_o(cereal::make_nvp("maxV", this->maxV));
	}


}

void UnsteadyPathlneDataSetGenerator::GenOneSplit(int Nparamters, int samplePerNRCParameters, int observerPerSetting, const std::string& dataSetSplitTag) {



	double minMagintude = INFINITY;
	double maxMagintude = -INFINITY;
	auto genTxty = [&]() -> Eigen::Vector2d {
		auto tx = genTx(rng);
		auto ty = genTy(rng);
		// clamp tx, ty into valid domain
		tx = std::clamp(tx, domainMinBoundary.x() + 0.05 * domainRange(0), domainMinBoundary.x() + 0.95 * domainRange(0));
		ty = std::clamp(ty, domainMinBoundary.y() + 0.05 * domainRange(1), domainMinBoundary.y() + 0.95 * domainRange(1));
		return { tx, ty };
		};

	const auto rc_n_paramters = generateVastisRC_NParamters(Nparamters, dataSetSplitTag);


	for_each(policy, rc_n_paramters.begin(), rc_n_paramters.cend(), [&](const std::pair<double, double>& params) {
		const double rc = params.first;
		const double n = params.second;
		std::string str_Rc = trimNumString(std::to_string(rc));
		std::string str_n = trimNumString(std::to_string(n));

		VastistasVelocityGenerator generator(Xdim, Ydim, domainMinBoundary, domainMaxBoundary, rc, n);
		int totalSamples = samplePerNRCParameters * observerPerSetting;
		printf("generate %d sample for rc=%f , n=%f \n", totalSamples, rc, n);

		const string Major_task_foldername = "velocity_rc_" + str_Rc + "n_" + str_n + "/";
		const string Major_task_Licfoldername = Major_task_foldername + "/LIC/";
		std::string task_folder = this->split_root_folder + Major_task_foldername;
		if (!filesystem::exists(task_folder)) {
			filesystem::create_directories(task_folder);
		}
		std::string task_licfolder = this->split_root_folder + Major_task_Licfoldername;
		if (!filesystem::exists(task_licfolder)) {
			filesystem::create_directories(task_licfolder);
		}

		for (int sample = 0; sample < samplePerNRCParameters; sample++) {
			// the type of this sample(divergence,cw vortex, cc2 vortex)
			auto Si = static_cast<VastisVortexType>((sample + 1) % 3);
			if ((sample % 100 == 0) && sample > 0)
			{
				Si = VastisVortexType::zero_field;
			}
			const string vortexTypeName = string{ magic_enum::enum_name<VastisVortexType>(Si) };

			// Si = VastisVortexType::saddle;
			const auto theta = genTheta(rng);
			auto sx = genSx(rng);
			auto sy = genSy(rng);
			// make sure deformat is invertible &, radius not two wired
			while (std::abs(sx * sy) <= 0.01) {
				sx = genSx(rng);
				sy = genSy(rng);
			}
			Eigen::Vector2d txy = genTxty();
			Eigen::Matrix2d deformMat = Eigen::Matrix2d::Identity();
			deformMat(0, 0) = sx * std::cos(theta);
			deformMat(0, 1) = -sy * std::sin(theta);
			deformMat(1, 0) = sx * std::sin(theta);
			deformMat(1, 1) = sy * std::cos(theta);
			Eigen::Vector3d rc_n_si = { rc,n,  (double)Si };
			const SteadyVectorField2D steadyField = generator.generateSteadyField_VortexBoundaryVIS2020(txy.x(), txy.y(), sx, sy, theta, Si);
			const auto& steadyFieldResData = steadyField.field;
			const auto SteadyTexture = LICAlgorithm(steadyField, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection, VORTEX_CRITERION::VELICITY_MAGINITUDE);
			auto [thisFieldMinV, thisFieldMaxV] = computeMinMax(steadyFieldResData);

			for (size_t observerIndex = 0; observerIndex < observerPerSetting; observerIndex++) {
				printf(".");

				const int taskSampleId = sample * observerPerSetting + observerIndex;

				const string sample_tag_name
					= "sample_" + to_string(taskSampleId) + vortexTypeName;

				auto observerParameters = generateRandomABCVectors();
				auto& abc = observerParameters.first;
				auto& abc_dot = observerParameters.second;
				if (Si != VastisVortexType::zero_field) [[likely]] {
					//modify observer if it is too large for steady field
					while (abc.norm() > thisFieldMaxV * 2)
					{
						abc *= 0.5;
					}
					while (abc_dot.norm() > thisFieldMaxV * 0.1)
					{
						abc_dot *= 0.5;
					}
					}


				UnSteadyVectorField2D unsteady_field = Tobias_ObserverTransformation(steadyField, abc, abc_dot, tmin, tmax, unsteadyFieldTimeStep);
#ifdef VALIDATE_RECONSTRUCTION_RESULT
				// reconstruct unsteady field from observer field
				auto reconstruct_field = Tobias_reconstructUnsteadyField(unsteady_field, abc, abc_dot);
				// //validate reconstruction result
				for (int rec = 0; rec < unsteadyFieldTimeStep; rec++) {
					const auto& reconstruct_slice = reconstruct_field.field[rec];
					// compute reconstruct slice difference with steady field
					double diffSum = 0.0;
					for (int y = 1; y < Ydim - 1; y++)
						for (int x = 1; x < Xdim - 1; x++) {
							auto diff = reconstruct_slice[y][x] - steadyField.field[y][x];
							diffSum += diff.norm();
						}
					double tolerance = (Xdim - 2) * (Ydim - 2) * 0.00001;
					// has debug, major reason for reconstruction failure is velocity too big make observer transformation query value from region out of boundary
					if (diffSum > tolerance) {
						printf("\n\n");
						printf("\n reconstruct field not equal to steady field at step %u\n", (unsigned int)rec);
						printf("\n\n");
					}
				}
#endif



				// the first point is the distance to it self(always zero), use it as the segmentation label for this pathline.
				std::vector<std::vector<PathlinePointInfo>> ClusterPathlines = PathlineIntegrationInfoCollect2D(unsteady_field, 25, deformMat, rc_n_si, txy, outputPathlineLength);

				// visualize segmentation & pick random k path lines to vis
				{

					auto licSegTexture = addSegmentationVisualization(SteadyTexture, steadyField, rc_n_si, txy, deformMat);
					auto licSegTexturewithPathline = addPathlineVisualization(licSegTexture, domainMinBoundary, domainMaxBoundary, tmin, tmax, ClusterPathlines);
					string licFilename0 = task_licfolder + sample_tag_name + "_steady_lic.png";
					saveAsPNG(licSegTexturewithPathline, licFilename0);
					auto outputTextures = LICAlgorithm_UnsteadyField(unsteady_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection, VORTEX_CRITERION::VELICITY_MAGINITUDE);
					for (size_t i = 0; i < outputTextures.size(); i += LicSaveFrequency) {
						string tag_name = sample_tag_name + "deformed_" + std::to_string(i);
						string licFilename = task_licfolder + tag_name + "lic.png";
						saveAsPNG(outputTextures[i], licFilename);
					}
				}

				auto rawUnsteadyFieldData = flatten3DVectorsAs1Dfloat(unsteady_field.field);
				auto [minV, maxV] = computeMinMax(rawUnsteadyFieldData);
				if (minV < minMagintude) {
					minMagintude = minV;
				}
				if (maxV > maxMagintude) {
					maxMagintude = maxV;
				}

				string metaFilename = task_folder + sample_tag_name + "meta.json";

				// save meta info:
				std::ofstream jsonOut(metaFilename);
				if (!jsonOut.good()) [[unlikely]] {
					printf("couldn't open file: %s", metaFilename.c_str());
					return;
					}
				{
					cereal::JSONOutputArchive archive_o(jsonOut);
					Eigen::Vector3d deform_TheteaSxSy = { theta, sx, sy };
					archive_o(cereal::make_nvp("rc_n_si", rc_n_si));
					archive_o(cereal::make_nvp("txy", txy));
					archive_o(cereal::make_nvp("deform_theta_sx_sy", deform_TheteaSxSy));
					archive_o(cereal::make_nvp("observer abc", abc));
					archive_o(cereal::make_nvp("observer abc_dot", abc_dot));

				}
				// do not manually close file before creal deconstructor, as cereal will preprend a ]/} to finish json class/array
				jsonOut.close();

				// output binary of vector field, segmentation, pathlines
				string velocityFilename = task_folder + sample_tag_name + ".bin";
				string pathlineFilename = task_folder + sample_tag_name + "_pathline.bin";
				string segmentationFilename = task_folder + sample_tag_name + "_segmentation.bin";
				cerealBinaryOut(rawUnsteadyFieldData, velocityFilename);
				cerealBinaryOut(flatten3DvecAs1Dfloat(ClusterPathlines), pathlineFilename);
				if (Si == VastisVortexType::center_ccw || Si == VastisVortexType::center_cw) {
					auto seg = generateSegmentationBinaryMask(rc_n_si, txy, deformMat, Xdim, Ydim, gridInterval, domainMinBoundary);
					cerealBinaryOut(flatten2DvecAs1Dfloat(seg), segmentationFilename);
				}
			} // for (size_t observerIndex = 0..)

		} // for n,rc sample
		});
}


void UnsteadyPathlneDataSetGenerator::DeSerialize(const std::string& dest_folder, const VastisParamter& v_param, const Eigen::Vector3d& ObserverAbc, const Eigen::Vector3d& ObserverAbcDot, const std::string& SampleName)
{
	if (!filesystem::exists(dest_folder)) {
		filesystem::create_directories(dest_folder);
	}
	std::string task_licfolder = dest_folder + "/LIC/";
	if (!filesystem::exists(task_licfolder)) {
		filesystem::create_directories(task_licfolder);
	}

	const auto rc_n = std::get<(int)VASTIS_PARAM::VastisParamRC_N>(v_param);
	const auto sxsy_theta = std::get<(int)VASTIS_PARAM::VastisParamSXSYTHETA>(v_param);
	const auto txy = std::get<(int)VASTIS_PARAM::VastisParamTXY>(v_param);
	const auto si = std::get<(int)VASTIS_PARAM::VastisParamSI>(v_param);
	const double sx = sxsy_theta.x();
	const double sy = sxsy_theta.y();
	const double theta = sxsy_theta.z();
	Eigen::Matrix2d deformMat = Eigen::Matrix2d::Identity();
	deformMat(0, 0) = sx * std::cos(theta);
	deformMat(0, 1) = -sy * std::sin(theta);
	deformMat(1, 0) = sx * std::sin(theta);
	deformMat(1, 1) = sy * std::cos(theta);
	Eigen::Vector3d rc_n_si = { rc_n.x(), rc_n.y(), (double)si };
	const string vortexTypeName = string{ magic_enum::enum_name<VastisVortexType>(static_cast<VastisVortexType>(si)) };
	const string sample_tag_name = SampleName + vortexTypeName;

	VastistasVelocityGenerator generator(Xdim, Ydim, domainMinBoundary, domainMaxBoundary, rc_n(0), rc_n(1));
	auto steadyField = generator.generateSteadyField_VortexBoundaryVIS2020(txy, sxsy_theta, static_cast<VastisVortexType>(si));
	auto SteadyTexture = LICAlgorithm(steadyField, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection, VORTEX_CRITERION::VELICITY_MAGINITUDE);
	UnSteadyVectorField2D unsteady_field = Tobias_ObserverTransformation(steadyField, ObserverAbc, ObserverAbcDot, tmin, tmax, unsteadyFieldTimeStep);
	auto rawUnsteadyFieldData = flatten3DVectorsAs1Dfloat(unsteady_field.field);
	// the first point is the distance to it self(always zero), use it as the segmentation label for this pathline.

	std::vector<std::vector<PathlinePointInfo>> ClusterPathlines = PathlineIntegrationInfoCollect2D(unsteady_field, 25, deformMat, rc_n_si, txy, outputPathlineLength);
	{

		auto licSegTexture = addSegmentationVisualization(SteadyTexture, steadyField, rc_n_si, txy, deformMat);
		auto licSegTexturewithPathline = addPathlineVisualization(licSegTexture, domainMinBoundary, domainMaxBoundary, tmin, tmax, ClusterPathlines);
		string licFilename0 = task_licfolder + sample_tag_name + "_steady_lic.png";
		saveAsPNG(licSegTexturewithPathline, licFilename0);
		auto outputTextures = LICAlgorithm_UnsteadyField(unsteady_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection, VORTEX_CRITERION::VELICITY_MAGINITUDE);
		for (size_t i = 0; i < outputTextures.size(); i += LicSaveFrequency) {
			string tag_name = sample_tag_name + "deformed_" + std::to_string(i);
			string licFilename = task_licfolder + tag_name + "lic.png";
			saveAsPNG(outputTextures[i], licFilename);
		}
	}

	string metaFilename = dest_folder + sample_tag_name + "meta.json";
	// save meta info:
	std::ofstream jsonOut(metaFilename);
	if (!jsonOut.good()) [[unlikely]] {
		printf("couldn't open file: %s", metaFilename.c_str());
		return;
		}
	{
		cereal::JSONOutputArchive archive_o(jsonOut);
		Eigen::Vector3d deform_TheteaSxSy = { theta, sx, sy };
		archive_o(cereal::make_nvp("rc_n_si", rc_n_si));
		archive_o(cereal::make_nvp("txy", txy));
		archive_o(cereal::make_nvp("deform_theta_sx_sy", deform_TheteaSxSy));
		archive_o(cereal::make_nvp("observer abc", ObserverAbc));
		archive_o(cereal::make_nvp("observer abc_dot", ObserverAbcDot));
	}
	// do not manually close file before creal deconstructor, as cereal will preprend a ]/} to finish json class/array
	jsonOut.close();

	// output binary of vector field, segmentation, path lines
	string velocityFilename = dest_folder + sample_tag_name + ".bin";
	string pathlineFilename = dest_folder + sample_tag_name + "_pathline.bin";
	string segmentationFilename = dest_folder + sample_tag_name + "_segmentation.bin";
	cerealBinaryOut(rawUnsteadyFieldData, velocityFilename);
	cerealBinaryOut(flatten3DvecAs1Dfloat(ClusterPathlines), pathlineFilename);
	if (si == (int)VastisVortexType::center_ccw || si == (int)VastisVortexType::center_cw) {
		auto seg = generateSegmentationBinaryMask(rc_n_si, txy, deformMat, Xdim, Ydim, gridInterval, domainMinBoundary);
		cerealBinaryOut(flatten2DvecAs1Dfloat(seg), segmentationFilename);
	}


}

void UnsteadyPathlneDataSetGenerator::analyticalTestCasesGeneration(const std::string& dst_folder) {
	//analytical test field.
	Eigen::Vector2i grid_size(Xdim, Ydim);
	Eigen::Vector2d domainMinBoundary = { -2.0, -2.0 };
	Eigen::Vector2d domainMaxBoundary = { 2.0, 2.0 };
	int time_steps = 32;

	//taking global field paramters.
	AnalyticalFlowCreator flowCreator(grid_size, time_steps, domainMinBoundary, domainMaxBoundary, tmin, tmax);
	auto classicalAnalyticalFields = flowCreator.generateAnalyticalTestSuite();
	auto ouput_Folder = dst_folder + "/analytical/";
	if (!filesystem::exists(ouput_Folder)) {
		filesystem::create_directories(ouput_Folder);
	}
	for (const auto& [fieldName, unsteady_field] : classicalAnalyticalFields) {


		std::vector<std::vector<PathlinePointInfo>> ClusterPathlines = PathlineIntegrationInfoCollect2D(unsteady_field, 25, outputPathlineLength);
		const auto steadyZeroSlice = unsteady_field.getVectorfieldSliceAtTime(0);
		auto outputSteadyTexture = LICAlgorithm(steadyZeroSlice, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
		outputSteadyTexture = addPathlineVisualization(outputSteadyTexture, domainMinBoundary, domainMaxBoundary, tmin, tmax, ClusterPathlines);
		string pathlinePngName = fieldName + "_pathline.png";
		saveAsPNG(outputSteadyTexture, ouput_Folder + pathlinePngName);


		auto rawData = flatten3DVectorsAs1Dfloat(unsteady_field.field);
		auto binaryFile = ouput_Folder + fieldName + ".bin";
		cerealBinaryOut(rawData, binaryFile);
		//no  vortex segmentation label to save.
		// output binary of vector field, segmentation, pathlines
		string pathlineFilename = ouput_Folder + fieldName + "_pathline.bin";
		cerealBinaryOut(flatten3DvecAs1Dfloat(ClusterPathlines), pathlineFilename);

	}

	string root_metaFilename = ouput_Folder + "analyticalfields_meta.json";
	// save meta info:
	std::ofstream jsonOut(root_metaFilename);
	if (!jsonOut.good()) [[unlikely]] {
		printf("couldn't open file: %s", root_metaFilename.c_str());
		return;
		}
	{
		cereal::JSONOutputArchive archive_o(jsonOut);
		archive_o(CEREAL_NVP(Xdim));
		archive_o(CEREAL_NVP(Ydim));
		archive_o(CEREAL_NVP(time_steps));
		archive_o(CEREAL_NVP(domainMinBoundary));
		archive_o(CEREAL_NVP(domainMaxBoundary));
		archive_o(CEREAL_NVP(tmin));
		archive_o(CEREAL_NVP(tmax));
	}
	// do not manually close file before creal deconstructor, as cereal will preprend a ]/} to finish json class/array
	jsonOut.close();
}


void UnsteadyPathlneDataSetGenerator::classicalParametersDeserialization(const std::string& dst_folder) {
	//one VastisParamter is Eigen::Vector2d rc_n, Eigen::Vector2d tx_ty,Eigen::Vector3d sxsytheta, int si.
	const std::vector<std::tuple<Eigen::Vector2d, Eigen::Vector2d, Eigen::Vector3d, int>> Vastisparams = {
		std::make_tuple(Eigen::Vector2d(0.3, 2.0), Eigen::Vector2d(0.0, 0.0), Eigen::Vector3d(1.0, 1.0, 2.0), (int)VastisVortexType::zero_field),
		std::make_tuple(Eigen::Vector2d(0.57, 2.0), Eigen::Vector2d(0.0, 0.0), Eigen::Vector3d(1.0, 1.0, 2.0), 1),
		std::make_tuple(Eigen::Vector2d(0.99, 2.0), Eigen::Vector2d(0.0, 0.0), Eigen::Vector3d(1.0, 1.0, 2.0), 2),
		std::make_tuple(Eigen::Vector2d(1.2, 2.0), Eigen::Vector2d(0.0, 0.0), Eigen::Vector3d(1.0, 1.0, 2.0), 0),
	   std::make_tuple(Eigen::Vector2d(1.17, 3.0), Eigen::Vector2d(0.0, 0.0), Eigen::Vector3d(1.0, 1.0, 3.0), 1),
	   std::make_tuple(Eigen::Vector2d(1.187, 2.0), Eigen::Vector2d(0.0, 0.0), Eigen::Vector3d(1.15, 1.25, 3.0), 2),
	};
	const std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d>> ObserverParams = {
		//no acc 
		std::make_tuple(Eigen::Vector3d(0.0, 0.0,-0.1), Eigen::Vector3d(0.0, 0.0, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.0, 0.0,0.0), Eigen::Vector3d(0.0, 0.0, 0.0)),

		std::make_tuple(Eigen::Vector3d(-0.10, 0.1,0.0), Eigen::Vector3d(0.0, 0.0, 0.0)),

		std::make_tuple(Eigen::Vector3d(0.20, 0.0,0.0), Eigen::Vector3d(0.0, 0.0, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.01, 0.0,0.0), Eigen::Vector3d(0.0, 0.0, 0.0)),

		std::make_tuple(Eigen::Vector3d(0.0, 0.1,0.0), Eigen::Vector3d(0.0, 0.0, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.0, 0.2,0.0), Eigen::Vector3d(0.0, 0.0, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.0, 0.01,0.0), Eigen::Vector3d(0.0, 0.0, 0.0)),

		std::make_tuple(Eigen::Vector3d(0.1, 0.1,0.0), Eigen::Vector3d(0.0, 0.0, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.1, 0.2,0.0), Eigen::Vector3d(0.0, 0.0, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.07, 0.01,0.0), Eigen::Vector3d(0.0, 0.0, 0.0)),

		std::make_tuple(Eigen::Vector3d(0.1, 0.1,0.02), Eigen::Vector3d(0.0, 0.0, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.1, 0.2,0.001), Eigen::Vector3d(0.0, 0.0, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.07, 0.00,0.03), Eigen::Vector3d(0.0, 0.0, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.00, 0.00,0.1), Eigen::Vector3d(0.0, 0.0, 0.00)),

		//with acc 
		std::make_tuple(Eigen::Vector3d(0.0, 0.0,0.0), Eigen::Vector3d(0.0, 0.0, 0.01)),
		std::make_tuple(Eigen::Vector3d(0.0, 0.0,0.0), Eigen::Vector3d(0.0, 0.002, 0.00)),
		std::make_tuple(Eigen::Vector3d(0.10, 0.0,0.0), Eigen::Vector3d(0.0, 0.0, 0.001)),
		std::make_tuple(Eigen::Vector3d(0.20, 0.0,0.0), Eigen::Vector3d(0.0, 0.0, 0.002)),
		std::make_tuple(Eigen::Vector3d(0.01, 0.0,0.0), Eigen::Vector3d(0.0, 0.01, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.0, 0.1,0.0), Eigen::Vector3d(0.0, 0.0, 0.0002)),
		std::make_tuple(Eigen::Vector3d(0.0, 0.2,0.0), Eigen::Vector3d(0.001, 0.0, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.0, 0.01,0.0), Eigen::Vector3d(0.0, 0.001, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.1, 0.1,0.0), Eigen::Vector3d(0.0, 0.0, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.1, 0.2,0.0), Eigen::Vector3d(0.0, 0.0, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.07, 0.01,0.0), Eigen::Vector3d(0.0, 0.0, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.1, 0.1,0.02), Eigen::Vector3d(0.0, 0.0, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.1, 0.2,0.001), Eigen::Vector3d(0.0, 0.0, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.07, 0.00,0.03), Eigen::Vector3d(0.0, 0.0, 0.0)),
		std::make_tuple(Eigen::Vector3d(0.00, 0.00,0.0), Eigen::Vector3d(0.0, 0.0, 0.01)),
		std::make_tuple(Eigen::Vector3d(0.00, 0.00,0.1), Eigen::Vector3d(0.0, 0.0, 0.01)),
		std::make_tuple(Eigen::Vector3d(0.01, 0.22,0.0), Eigen::Vector3d(0.0, 0.0, 0.01)),

	};

	int sampleId = 0;
	for (const auto& param : Vastisparams) {
		Eigen::Vector2d rc_n;
		Eigen::Vector2d txy;
		Eigen::Vector3d sxsyTheta;
		int si;
		std::tie(rc_n, txy, sxsyTheta, si) = param;
		for (size_t j = 0; j < ObserverParams.size(); j++)
		{
			Eigen::Vector3d abc;
			Eigen::Vector3d abc_dot;
			std::tie(abc, abc_dot) = ObserverParams[j];

			auto sampleName = "sample_" + std::to_string(sampleId++);
			// Call DeSerialize for each set of parameters
			DeSerialize(dst_folder, param, abc, abc_dot, sampleName);
		}
	}




}



std::vector<VastisParamter> UnsteadyPathlneDataSetGenerator::generateRandomMixtureVastisParam(const int mix)
{

	constexpr double meanOfVortexRc = 1.87;
	constexpr double minimalDistanceOfTwoVortex = meanOfVortexRc + 0.01; //

	static std::uniform_int_distribution<> dis_vortexType(0, 4);
	std::vector<VastisParamter> vectorFieldMeta;

	auto minimaltxydistance = [&](const Eigen::Vector2d& txy) -> double {
		double minV = INFINITY;

		for (const auto& mixtureParam : vectorFieldMeta) {
			const auto txy0 = std::get<(int)VASTIS_PARAM::VastisParamTXY>(mixtureParam);
			double dis = (txy0 - txy).norm();
			if (dis < minV) {
				minV = dis;
			}
		}
		return minV;
		};

	auto genTxty = [&]() -> Eigen::Vector2d {
		auto tx = genTx(rng);
		auto ty = genTy(rng);
		// clamp tx, ty into valid domain
		tx = std::clamp(tx, domainMinBoundary.x() + 0.05 * domainRange(0), domainMinBoundary.x() + 0.95 * domainRange(0));
		ty = std::clamp(ty, domainMinBoundary.y() + 0.05 * domainRange(1), domainMinBoundary.y() + 0.95 * domainRange(1));

		// make sure not too close to   previous vortex cores.
		Eigen::Vector2d txy = { tx, ty };
		while (minimaltxydistance(txy) < minimalDistanceOfTwoVortex) {
			auto tx = genTx(rng);
			auto ty = genTy(rng);
			txy(0) = std::clamp(tx, domainMinBoundary.x() + 0.05 * domainRange(0), domainMinBoundary.x() + 0.95 * domainRange(0));
			txy(1) = std::clamp(ty, domainMinBoundary.y() + 0.05 * domainRange(1), domainMinBoundary.y() + 0.95 * domainRange(1));
		}
		return txy;
		};

	bool hasSaddleField = false;
	double radius_divde = 1 / (double)(mix + 0.01);
	for (int i = 0; i < mix; i++) {
		//	//generate rc,n
		const  std::pair<double, double> rc_n_tuple = generateVastisRC_NParamters(1, "not_train")[0];
		const auto rc_n = Eigen::Vector2d{ std::get<0>(rc_n_tuple) * radius_divde,std::get<1>(rc_n_tuple) };
		//	// make sure txty is in good range: that no vortex center are too close.
		Eigen::Vector2d txy = genTxty();
		Eigen::Vector3d sxsytheta = { 1.0, 1.0, genTheta(rng) };
		int vortexTypeSi = std::floor((dis_vortexType(rng) + 1) / 2);

		hasSaddleField = (vortexTypeSi == (int)(VastisVortexType::saddle)) || hasSaddleField;
		//we don't accept saddle field mix with vortex field, as we are not 100%  sure where will be the vortex boundary anymore after complicate mixture
		//we only assure when two vortex are far away enough, they stay as vortex after mixture.

		auto parmas = std::make_tuple(rc_n,
			txy,
			sxsytheta,
			vortexTypeSi);
		vectorFieldMeta.emplace_back(parmas);
	}

	if (hasSaddleField)
	{
		for (auto& parmas : vectorFieldMeta)
		{
			std::get<(int)VASTIS_PARAM::VastisParamSI>(parmas) = (int)(VastisVortexType::saddle);
		}
	}

	return vectorFieldMeta;

}



void UnsteadyPathlneDataSetGenerator::generateMixUnsteadyFieldPathline(const std::string& dest_folder, int Samples, int ObserversPerSample)
{
	domainRange = domainMaxBoundary - domainMinBoundary;
	gridInterval = {
		(domainMaxBoundary(0) - domainMinBoundary(0)) / (Xdim - 1),
		(domainMaxBoundary(1) - domainMinBoundary(1)) / (Ydim - 1)
	};
	double minMagintude = INFINITY;
	double maxMagintude = -INFINITY;
	const string Major_task_foldername = dest_folder + "/velocity_mix/";
	const string Major_task_Licfoldername = Major_task_foldername + "/LIC/";
	if (!filesystem::exists(Major_task_foldername)) {
		filesystem::create_directories(Major_task_foldername);
	};
	if (!filesystem::exists(Major_task_Licfoldername)) {
		filesystem::create_directories(Major_task_Licfoldername);
	}
	static std::uniform_int_distribution<int> dis_mixture(2, 4);
	std::vector<int> threadRange(Samples);
	std::generate(threadRange.begin(), threadRange.end(), [n = 0]() mutable { return n++; });

	// task0:generate random field
	printf("generate random  field with random observer..");
	for_each(policy, threadRange.begin(), threadRange.end(), [&](const int threadID) {
		VastistasVelocityGenerator generator(Xdim, Ydim, domainMinBoundary, domainMaxBoundary, 0, 0);
		printf("generate %d samples  \n", ObserversPerSample);

		const int mix = dis_mixture(rng);
		std::vector<VastisParamter> vectorFieldMeta = generateRandomMixtureVastisParam(mix);

		SteadyVectorField2D  steadyField = generator.generateSteadyMixtureOFVortexBoundaryPaper(vectorFieldMeta);
		const auto& steadyFieldResData = steadyField.field;



		std::vector<std::vector<Eigen::Vector3d>> outputSteadyTexture = LICAlgorithm(steadyField, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
		const auto steadyLicTexture = addSegmentationVis4MixtureVortex(outputSteadyTexture, steadyField, vectorFieldMeta);
		string steadyLicFilePath = Major_task_Licfoldername + to_string(threadID * ObserversPerSample) + "_" + to_string((threadID + 1) * ObserversPerSample - 1) + "steady_lic.png";

		bool IsSaddle = false;
		string vortexTypeName;
		for (const auto meta : vectorFieldMeta) {
			auto Si = static_cast<VastisVortexType>(std::get<(int)VASTIS_PARAM::VastisParamSI>(meta));
			vortexTypeName += string{ magic_enum::enum_name<VastisVortexType>(Si) };
			IsSaddle = Si == VastisVortexType::saddle;
		}

		for (size_t observerIndex = 0; observerIndex < ObserversPerSample; observerIndex++) {
			printf(".");
			const int taskSampleId = threadID * ObserversPerSample + observerIndex;
			const string sample_tag_name
				= "sample_" + to_string(taskSampleId) + vortexTypeName;

			const auto& observerParameters = generateRandomABCVectors();
			const auto& abc = observerParameters.first;
			const auto& abc_dot = observerParameters.second;

			UnSteadyVectorField2D unsteady_field = Tobias_ObserverTransformation(steadyField, abc, abc_dot, tmin, tmax, unsteadyFieldTimeStep);
			// the first point is the distance to it self(always zero), use it as the segmentation label for this pathline.
			std::vector<std::vector<PathlinePointInfo>> ClusterPathlines = PathlineIntegrationInfoCollect2D(unsteady_field, 25, outputPathlineLength);
			{

				auto licSegTexturewithPathline = addPathlineVisualization(steadyLicTexture, domainMinBoundary, domainMaxBoundary, tmin, tmax, ClusterPathlines);
				string licFilename0 = Major_task_Licfoldername + sample_tag_name + "_steady_lic.png";
				saveAsPNG(licSegTexturewithPathline, licFilename0);
				auto outputTextures = LICAlgorithm_UnsteadyField(unsteady_field, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection, VORTEX_CRITERION::VELICITY_MAGINITUDE);
				for (size_t i = 0; i < outputTextures.size(); i += LicSaveFrequency) {
					string deform_tag_name = sample_tag_name + "deformed_" + std::to_string(i);
					string licFilename = Major_task_Licfoldername + deform_tag_name + "lic.png";
					saveAsPNG(outputTextures[i], licFilename);
				}
			}

			auto rawUnsteadyFieldData = flatten3DVectorsAs1Dfloat(unsteady_field.field);
			auto [minV, maxV] = computeMinMax(rawUnsteadyFieldData);
			if (minV < minMagintude) {
				minMagintude = minV;
			}
			if (maxV > maxMagintude) {
				maxMagintude = maxV;
			}


#ifdef VALIDATE_RECONSTRUCTION_RESULT
			// reconstruct unsteady field from observer field
			auto reconstruct_field = Tobias_reconstructUnsteadyField(unsteady_field, abc, abc_dot);
			// //validate reconstruction result
			for (int rec = 0; rec < unsteadyFieldTimeStep; rec++) {
				const auto& reconstruct_slice = reconstruct_field.field[rec];
				// compute reconstruct slice difference with steady field
				double diffSum = 0.0;
				for (int y = 1; y < Ydim - 1; y++)
					for (int x = 1; x < Xdim - 1; x++) {
						auto diff = reconstruct_slice[y][x] - steadyField.field[y][x];
						diffSum += diff.norm();
					}
				double tolerance = (Xdim - 2) * (Ydim - 2) * 0.00001;
				// has debug, major reason for reconstruction failure is velocity too big make observer transformation query value from region out of boundary
				if (diffSum > tolerance) {
					printf("\n\n");
					printf("\n reconstruct field not equal to steady field at step %u\n", (unsigned int)rec);
					printf("\n\n");
				}
			}
#endif
			string metaFilename = Major_task_foldername + sample_tag_name + "meta.json";
			// save meta info:
			std::ofstream jsonOut(metaFilename);
			if (!jsonOut.good()) [[unlikely]] {
				printf("couldn't open file: %s", metaFilename.c_str());
				return;
				}
			{
				cereal::JSONOutputArchive archive_o(jsonOut);
				archive_o(CEREAL_NVP(vectorFieldMeta));

			}
			// do not manually close file before creal deconstructor, as cereal will preprend a ]/} to finish json class/array
			jsonOut.close();

			// output binary of vector field, segmentation, pathlines
			string velocityFilename = Major_task_foldername + sample_tag_name + ".bin";
			string pathlineFilename = Major_task_foldername + sample_tag_name + "_pathline.bin";
			string segmentationFilename = Major_task_foldername + sample_tag_name + "_segmentation.bin";
			cerealBinaryOut(rawUnsteadyFieldData, velocityFilename);
			cerealBinaryOut(flatten3DvecAs1Dfloat(ClusterPathlines), pathlineFilename);
			if (!IsSaddle) {
				auto seg = generateSegmentationBinaryMaskMix(vectorFieldMeta, Xdim, Ydim, gridInterval, domainMinBoundary);
				cerealBinaryOut(flatten2DvecAs1Dfloat(seg), segmentationFilename);
			}


		} // for (size_t observerIndex = 0..)
		});//for_each(policy, threadRange.begin(), threadRange.end(), [&](const int threadID) {

	// create Root meta json file, save plane information here instead of every sample's meta file
	string taskFolder_rootMetaFilename = Major_task_foldername + "meta.json";
	// save root meta info:
	std::ofstream root_jsonOut(taskFolder_rootMetaFilename);
	if (!root_jsonOut.good()) {
		printf("couldn't open file: %s", taskFolder_rootMetaFilename.c_str());
		return;
	}
	else {
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
