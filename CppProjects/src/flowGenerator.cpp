#include <array>
#include "commonUtils.h"
#include "transformation.h"
#include "flowGenerator.h"
#include "VastistasVelocityGenerator.h"
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
namespace {
	constexpr int Xdim = 32, Ydim = 32;
	constexpr int unsteadyFieldTimeStep = 7;
	constexpr double tmin = 0.0;
	constexpr double tmax = M_PI * 0.5;
	Eigen::Vector2d domainMinBoundary = { 0.0, 0.0 };
	Eigen::Vector2d domainMaxBoundary = { 2.0, 2.0 };
	constexpr int outputPathlineLength = 25;

	// lic parameters
	constexpr int LicImageSize = 128;
	constexpr int LicSaveFrequency = 1; // every 2 time steps save one
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


	std::vector<std::pair<double, double>> presetRCNParameters = {
		{ 0.5, 1.0 },
		{ 0.25, 2.0 },
		{ 0.8, 2.0 },
		{ 1.0, 2.0 },
		{ 1.0, 3.0 },
		{ 1.0, 5.0 },
		{ 1.5, 2.0 },
		{ 1.87, 10.0 },
		{ 2.0, 2.0 },
	};
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


}


std::vector<std::pair<double, double>> generateVastisRC_NParamters(int n, std::string mode)
{
	static std::unordered_set<std::pair<double, double>, pair_hash> unique_params;

	std::vector<std::pair<double, double>> parameters;

	std::random_device rd;

	std::normal_distribution<double> dist_rc(1.87 * 0.5, 0.37); // mean = 1.87, stddev = 0.34 our domain now change from [-2,2] ->[0,2], thus radius rc mulptile 0.5 otherwise its two large.
	std::normal_distribution<double> dist_n(1.96, 0.61); // mean = 1.96, stddev = 0.61

	int i = 0;
	while (parameters.size() < n) {

		if (i < presetRCNParameters.size() && mode == "train") {
			std::pair<double, double> preset_pair = presetRCNParameters.at(i++);
			if (unique_params.find(preset_pair) == unique_params.end()) {
				parameters.emplace_back(preset_pair);
				unique_params.insert(preset_pair);
			}
		}
		else {

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
	std::uniform_real_distribution<double> dist_acc(-0.005, 0.005); // robust paper in domain [-2,2] acc is range [-0.01,0.01], our domain is [0,2] thus multiply 0.5 of acc range
	std::uniform_int_distribution<int> dist_int(0, 5);
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



void addSegmentationVisualization(std::vector<std::vector<Eigen::Vector3d>>& inputLicImage, const SteadyVectorField2D& vectorField, const Eigen::Vector3d& meta_n_rc_si, const Eigen::Vector2d& domainMax, const Eigen::Vector2d& domainMIn, const Eigen::Vector2d& txy, const Eigen::Matrix2d& deformMat)
{
	// if si=0 then no vortex
	if (meta_n_rc_si.z() == 0.0) {
		return;
	}
	const auto rc = meta_n_rc_si(1);
	const auto deformInverse = deformMat.inverse();

	auto judgeVortex = [rc, txy, deformInverse](const Eigen::Vector2d& pos) -> bool {
		auto originalPos = deformInverse * (pos - txy);
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
				// auto velocity = vectorField.getVector(pos.x(), pos.y());
				// !note: because txy is random core line center, then critical point==txy(velocity ==0.0) might not  lie on any grid points
				/*     if (velocity.norm() < 1e-7)
						 [[unlikely]] {
						 inputLicImage[j][i] = 0.3 * preColor + 0.7 * Eigen::Vector3d(1.0, 0.0, 0.0);
					 } else {*/
					 // yellow for vortex region
				inputLicImage[j][i] = 0.4 * preColor + 0.6 * Eigen::Vector3d(1.0, 1.0, 0.0);
				//}
			}
		}
	}
}

void addSegmentationVisualization(std::vector<std::vector<Eigen::Vector3d>>& inputLicImage, const SteadyVectorField2D& vectorField, const Eigen::Vector2d& domainMax, const Eigen::Vector2d& domainMIn, const std::vector<VastisParamter>& params)
{

	for (size_t i = 0; i < params.size(); i++) {
		auto parm = params.at(i);
		Eigen::Vector2d n_rc = std::get<0>(parm);
		Eigen::Vector2d txy = std::get<1>(parm);
		Eigen::Vector3d sxsytheta = std::get<2>(parm);
		auto theta = sxsytheta.z();
		auto sx = sxsytheta.x();
		auto sy = sxsytheta.y();
		int si = std::get<3>(parm);
		Eigen::Vector3d meta_n_rc_si = { n_rc.x(), n_rc.y(), (double)si };
		Eigen::Matrix2d deformMat = Eigen::Matrix2d::Identity();
		deformMat(0, 0) = sx * cos(theta);
		deformMat(0, 1) = -sy * sin(theta);
		deformMat(1, 0) = sx * sin(theta);
		deformMat(1, 1) = sy * cos(theta);
		addSegmentationVisualization(inputLicImage, vectorField, meta_n_rc_si, domainMax, domainMIn, txy, deformMat);
	}
}

std::vector<std::vector<Eigen::Vector3d>> addPathlinesSegmentationVisualization(const std::vector<std::vector<Eigen::Vector3d>>& steadyInputLicImage, const SteadyVectorField2D& vectorField, const Eigen::Vector3d& meta_n_rc_si, const Eigen::Vector2d& txy, const Eigen::Matrix2d& deformMat,
	const std::vector<std::vector<std::vector<double>>>& InputPathlines)
{
	auto inputLicImage = steadyInputLicImage;
	constexpr int randomPickKlinesToDraw = 16;
	const Eigen::Vector2d& domainMax = vectorField.spatialDomainMaxBoundary;
	const Eigen::Vector2d& domainMIn = vectorField.spatialDomainMinBoundary;
	const auto domainRange = domainMax - domainMIn;
	const int licImageSizeY = inputLicImage.size();
	const int licImageSizeX = inputLicImage[0].size();

	// if si=0 then no vortex
	if (meta_n_rc_si.z() != 0.0) {

		const auto rc = meta_n_rc_si(1);
		const auto deformInverse = deformMat.inverse();

		auto judgeVortex = [rc, txy, deformInverse](const Eigen::Vector2d& pos) -> bool {
			auto originalPos = deformInverse * (pos - txy);
			auto dx = rc - originalPos.norm();
			return dx > 0;
			};
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

					inputLicImage[j][i] = 0.4 * preColor + 0.6 * Eigen::Vector3d(1.0, 1.0, 0.0);
				}
			}
		}
	}

	auto colorCoding = [](double time) -> Eigen::Vector3d {
		// Clamp time to [0, 1] range
		time = time / tmax;
		time = std::max(0.0, std::min(1.0, time));
		// White color
		Eigen::Vector3d white(1.0, 1.0, 1.0);

		// Blue color
		Eigen::Vector3d blue(0.0, 0.0, 1.0);

		// Linear interpolation between white and blue based on time
		return (1.0 - time) * blue + time * white;
		};

	std::uniform_int_distribution<int> dist_int(0, InputPathlines.size() - 1);
	std::vector<Eigen::Vector2d> selectedStartingPoints;
	for (int i = 0; i < randomPickKlinesToDraw; i++) {
		int lineIdx;
		bool isFarEnough;
		do {
			lineIdx = dist_int(rng);
			isFarEnough = true;
			const Eigen::Vector2d currentStartPoint(InputPathlines[lineIdx][0][0], InputPathlines[lineIdx][0][1]);
			for (const auto& previousStartPoint : selectedStartingPoints) {
				if ((currentStartPoint - previousStartPoint).norm() < 0.2) {
					isFarEnough = false;
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
				auto color = colorCoding(time); // Blue for pathlines
				auto preColor = inputLicImage[gridY][gridX];
				inputLicImage[gridY][gridX] = 0.2 * preColor + 0.8 * color;
			}
		}
	}
	return inputLicImage;
}

auto generateSegmentationBinaryMask(const Eigen::Vector3d& meta_n_rc_si, const Eigen::Vector2d& txy, const Eigen::Matrix2d& deformMat, const int Xdim, const int Ydim, const Eigen::Vector2d& gridInterval, const Eigen::Vector2d& domainMin)
{
	auto rc = meta_n_rc_si.y();
	auto si = meta_n_rc_si.z();
	const auto deformInverse = deformMat.inverse();
	if (si == 0.0) [[unlikely]] {
		std::vector<std::vector<uint8_t>> empty;
		return empty;
	}
	auto judgeVortex = [si, rc, txy, deformInverse](const Eigen::Vector2d& pos) -> uint8_t {
		auto originalPos = deformInverse * (pos - txy);
		auto dx = rc - originalPos.norm();
		return dx > 0 ? 1 : 0;
		};

	std::vector<std::vector<uint8_t>> segmentation(Ydim, std::vector<uint8_t>(Xdim, 0));
	for (size_t gy = 0; gy < Ydim; gy++)
		for (size_t gx = 0; gx < Xdim; gx++) {
			Eigen::Vector2d pos = { domainMin.x() + gridInterval.x() * gx, domainMin.y() + gridInterval.y() * gy };
			segmentation[gy][gx] = judgeVortex(pos);
		}
	return segmentation;
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

				Eigen::Vector3d n_rc_si = { n, rc, (double)Si };
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
					// addSegmentationVisualization(outputSteadyTexture, steadyField, n_rc_si, domainMaxBoundary, domainMinBoundary, txy, deformMat);
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

// since we are doing for lagrangian, T=5 is too small, now fix T=7.
auto PathlineIntegrationInfoCollect2D(const UnSteadyVectorField2D& inputField, /*int NClusters,*/ int KLines, const double pathline_dt_m, const Eigen::Matrix2d& deformMat, const Eigen::Vector3d& meta_n_rc_si, const Eigen::Vector2d& txy)
{
	constexpr int maximumLength = 50; // pathline_dt=1/5 dt, thus total have 9*5=45 steps.
	auto maxBound = inputField.getSpatialMaxBoundary();
	auto minBound = inputField.getSpatialMinBoundary();
	auto domainRange = maxBound - minBound;

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
		Eigen::Vector2d(0.25 * domainRange.x() + domainMinBoundary.x(), 0.25 * domainRange.y() + domainMinBoundary.y()), //-1,-1
		Eigen::Vector2d(0.25 * domainRange.x() + domainMinBoundary.x(), 0.75 * domainRange.y() + domainMinBoundary.y()), // 1,1
		Eigen::Vector2d(0.75 * domainRange.x() + domainMinBoundary.x(), 0.25 * domainRange.y() + domainMinBoundary.y()), // 1,-1
		Eigen::Vector2d(0.75 * domainRange.x() + domainMinBoundary.x(), 0.75 * domainRange.y() + domainMinBoundary.y()) //-1,1
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
				double floatIndexX = (px - domainMinBoundary.x()) / inputField.spatialGridInterval(0);
				double floatIndexY = (py - domainMinBoundary.y()) / inputField.spatialGridInterval(1);
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

				assert(false);
			};
		}
	}

	return clusterPathlines;
}



// generateUnsteadyFieldPathlineMix will have mixture steady field that has multiple(2-4) vortex & rotating zero field& static field deformed by zero (identity observer)
void generateUnsteadyFieldPathlineMix(int Nparamters, int observerPerSetting, const std::string in_root_fodler, const std::string dataSetSplitTag)
{

	// check datasplittag is "train"/"test"/"validation"
	if (dataSetSplitTag != "train" && dataSetSplitTag != "test" && dataSetSplitTag != "validation") {
		printf("dataSetSplitTag should be \"train\"/\"test\"/\"validation\"");
		return;
	}
	std::string root_folder = in_root_fodler + "/X" + to_string(Xdim) + "_Y" + to_string(Ydim) + "_T" + to_string(unsteadyFieldTimeStep) + "/" + dataSetSplitTag + "/";
	if (!filesystem::exists(root_folder)) {
		filesystem::create_directories(root_folder);
	}

	const Eigen::Vector2d domainRange = domainMaxBoundary - domainMinBoundary;
	Eigen::Vector2d gridInterval = {
		(domainMaxBoundary(0) - domainMinBoundary(0)) / (Xdim - 1),
		(domainMaxBoundary(1) - domainMinBoundary(1)) / (Ydim - 1)
	};

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

	static std::normal_distribution<double> genTheta(0.0, 0.50); // rotation angle's distribution
	// normal distribution from supplementary material of Vortex Boundary Identification Paper
	static std::normal_distribution<double> genSx(1, 3.59);
	static std::normal_distribution<double> genSy(1, 2.24);
	static std::normal_distribution<double> genTx(1.0, 1.34);
	static std::normal_distribution<double> genTy(1.0, 1.27);
	static std::uniform_int_distribution<> dis_mixture(1, 4);

	auto genTxty = [&]() -> Eigen::Vector2d {
		auto tx = genTx(rng);
		auto ty = genTy(rng);
		// clamp tx, ty into valid domain
		tx = std::clamp(tx, domainMinBoundary.x() + 0.05 * domainRange(0), domainMinBoundary.x() + 0.95 * domainRange(0));
		ty = std::clamp(ty, domainMinBoundary.y() + 0.05 * domainRange(1), domainMinBoundary.y() + 0.95 * domainRange(1));
		return { tx, ty };
		};
	std::vector<int> threadRange(Nparamters);
	std::generate(threadRange.begin(), threadRange.end(), [n = 0]() mutable { return n++; });

	// task0:generate random field
	printf("generate random  field with random observer..");
	for_each(policy, threadRange.begin(), threadRange.end(), [&](const int threadID) {
		// VastistasVelocityGenerator generator(Xdim, Ydim, domainMinBoundary, domainMaxBoundary, 0, 0);
		// int totalSamples = observerPerSetting;
		// printf("generate %d samples  \n", totalSamples);
		// string licFilename0 = task_licfolder + to_string(threadID * observerPerSetting) + "_" + to_string((threadID + 1) * observerPerSetting - 1) + "steady_lic.png";
		// std::vector<std::vector<Eigen::Vector3d>> outputSteadyTexture;
		//// the type of this sample(divergence,cw vortex, cc2 vortex)
		// auto Si = static_cast<VastisVortexType>(threadID % 3);
		// SteadyVectorField2D steadyField;
		// if (Si == VastisVortexType::saddle) {
		//     const auto theta = genTheta(rng);
		//     auto sx = genSx(rng);
		//     auto sy = genSy(rng);
		//     while (sx * sy == 0.0) {
		//         sx = genSx(rng);
		//         sy = genSy(rng);
		//     }
		//     Eigen::Vector2d txy = genTxty();
		//     steadyField = generator.generateSteadyField_VortexBoundaryVIS2020(txy.x(), txy.y(), sx, sy, theta, Si);
		//     const auto& steadyFieldResData = steadyField.field;
		//     outputSteadyTexture = LICAlgorithm(steadyField, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
		// } else {
		//     std::vector<VastisParamter> vectorFieldMeta;
		//     const int mix = dis_mixture(rng);
		//     steadyField = generator.generateSteadyFieldMixture(vectorFieldMeta, mix);
		//     const auto& steadyFieldResData = steadyField.field;
		//     outputSteadyTexture = LICAlgorithm(steadyField, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection);
		//     addSegmentationVisualization(outputSteadyTexture, steadyField, domainMaxBoundary, domainMinBoundary, vectorFieldMeta);
		// }

		// saveAsPNG(outputSteadyTexture, licFilename0);

		// for (size_t observerIndex = 0; observerIndex < observerPerSetting; observerIndex++) {
		//     printf(".");

		//    const int taskSampleId = sample * observerPerSetting + observerIndex;

		//    const string vortexTypeName = string { magic_enum::enum_name<VastisVortexType>(Si) };
		//    const string sample_tag_name
		//        = "sample_" + to_string(taskSampleId) + vortexTypeName;

		//    const auto& observerParameters = generateRandomABCVectors();
		//    const auto& abc = observerParameters.first;
		//    const auto& abc_dot = observerParameters.second;

		//    UnSteadyVectorField2D unsteady_field = Tobias_ObserverTransformation(steadyField, abc, abc_dot, tmin, tmax, unsteadyFieldTimeStep);
		//    // the first point is the distance to it self(always zero), use it as the segmentation label for this pathline.
		//    std::vector<std::vector<PathlinePointInfo>> ClusterPathlines = PathlineIntegrationInfoCollect2D(unsteady_field, 48, 4.0, deformMat, n_rc_si, txy);

		//    auto rawUnsteadyFieldData = flatten3DVectorsAs1Dfloat(unsteady_field.field);
		//    auto [minV, maxV] = computeMinMax(rawUnsteadyFieldData);
		//    if (minV < minMagintude) {
		//        minMagintude = minV;
		//    }
		//    if (maxV > maxMagintude) {
		//        maxMagintude = maxV;
		//    }

		//    string metaFilename = task_folder + sample_tag_name + "meta.json";

		//    // save meta info:
		//    std::ofstream jsonOut(metaFilename);
		//    if (!jsonOut.good()) [[unlikely]] {
		//        printf("couldn't open file: %s", metaFilename.c_str());
		//        return;
		//    }
		//    {
		//        cereal::JSONOutputArchive archive_o(jsonOut);
		//        Eigen::Vector3d deform = { theta, sx, sy };
		//        archive_o(cereal::make_nvp("n_rc_Si", n_rc_si));
		//        archive_o(cereal::make_nvp("txy", txy));
		//        archive_o(cereal::make_nvp("deform_theta_sx_sy", deform));

		//        /*archive_o(cereal::make_nvp("ClusterPathlines", ClusterPathlines));*/
		//    }
		//    // do not manually close file before creal deconstructor, as cereal will preprend a ]/} to finish json class/array
		//    jsonOut.close();

		//    // output binary of vector field, segmentation, pathlines
		//    string velocityFilename = task_folder + sample_tag_name + ".bin";
		//    string pathlineFilename = task_folder + sample_tag_name + "_pathline.bin";
		//    string segmentationFilename = task_folder + sample_tag_name + "_segmentation.bin";
		//    cerealBinaryOut(rawUnsteadyFieldData, velocityFilename);
		//    cerealBinaryOut(flatten3DAs1Dfloat(ClusterPathlines), pathlineFilename);
		//    if (Si != VastisVortexType::saddle) {
		//        auto seg = generateSegmentation(n_rc_si, txy, deformMat, Xdim, Ydim, gridInterval, domainMinBoundary);
		//        cerealBinaryOut(flatten2DAs1Dfloat(seg), segmentationFilename);
		//    }
		//} // for (size_t observerIndex = 0..)
		});

	// task1:generate random field with identity observer(steady field)
	printf("generate random  field with identity observer(steady field)..");

	// task3:generate zero field
	printf("generate zero field ..");

	// create Root meta json file, save plane information here instead of every sample's meta file
	// string taskFolder_rootMetaFilename = root_folder + "meta.json";
	//// save root meta info:
	// std::ofstream root_jsonOut(taskFolder_rootMetaFilename);
	// if (!root_jsonOut.good()) {
	//     printf("couldn't open file: %s", taskFolder_rootMetaFilename.c_str());
	//     return;
	// } else {
	//     cereal::JSONOutputArchive archive_o(root_jsonOut);
	//     archive_o(CEREAL_NVP(Xdim));
	//     archive_o(CEREAL_NVP(Ydim));
	//     archive_o(CEREAL_NVP(unsteadyFieldTimeStep));
	//     archive_o(CEREAL_NVP(domainMinBoundary));
	//     archive_o(CEREAL_NVP(domainMaxBoundary));
	//     archive_o(CEREAL_NVP(tmin));
	//     archive_o(CEREAL_NVP(tmax));
	//     // save min and max
	//     archive_o(cereal::make_nvp("minV", minMagintude));
	//     archive_o(cereal::make_nvp("maxV", maxMagintude));
	// }
}

void generateUnsteadyFieldPathlineclassicalFields(int Nparamters, int observerPerSetting, const std::string in_root_fodler, const std::string dataSetSplitTag)
{

	// rotating zero field
	// beads
	// rfc
}





void DataSetGenBase::GenDataset(int Nparamters, int samplePerParameters, int observerPerSetting, const std::string in_root_fodler, double rateVal_Train /*= 0.1*/, double rateVal_Test /*= 0.1*/) {
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

void UnsteadyPathlneDataSetGenerator::GenOneSplit(int Nparamters, int samplePerNRCParameters, int observerPerSetting, const std::string dataSetSplitTag) {

	// normal distribution from supplementary material of Vortex Boundary Identification Paper
	static std::normal_distribution<double> genTheta(0.0, 0.50); // rotation angle's distribution
	static std::normal_distribution<double> genSx(0, 3.59 * 0.25); // this 0.25 comes from our range [0-2] is 1/2 of tobias range [-2,2]
	static std::normal_distribution<double> genSy(0, 2.24 * 0.25);
	static std::normal_distribution<double> genTx(1.0, 1.34 * 0.25);
	static std::normal_distribution<double> genTy(1.0, 1.27 * 0.25);

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

		for (size_t sample = 0; sample < samplePerNRCParameters; sample++) {
			// the type of this sample(divergence,cw vortex, cc2 vortex)
			auto Si = static_cast<VastisVortexType>(sample % 3);
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
			Eigen::Vector3d n_rc_si = { n, rc, (double)Si };
			const SteadyVectorField2D steadyField = generator.generateSteadyField_VortexBoundaryVIS2020(txy.x(), txy.y(), sx, sy, theta, Si);
			const auto& steadyFieldResData = steadyField.field;

			const auto SteadyTexture = LICAlgorithm(steadyField, LicImageSize, LicImageSize, stepSize, maxLICIteratioOneDirection, VORTEX_CRITERION::VELICITY_MAGINITUDE);
			/*  if (Si == VastisVortexType::saddle) {
				  string licFilename0 = task_licfolder + to_string(sample * observerPerSetting) + "_" + to_string((sample + 1) * observerPerSetting - 1) + "saddle_lic.png";
				  saveAsPNG(SteadyTexture, licFilename0);
			  }*/

			for (size_t observerIndex = 0; observerIndex < observerPerSetting; observerIndex++) {
				printf(".");

				const int taskSampleId = sample * observerPerSetting + observerIndex;

				const string vortexTypeName = string{ magic_enum::enum_name<VastisVortexType>(Si) };
				const string sample_tag_name
					= "sample_" + to_string(taskSampleId) + vortexTypeName;

				const auto& observerParameters = generateRandomABCVectors();

				const auto& abc = observerParameters.first;
				const auto& abc_dot = observerParameters.second;

				/* auto func = KillingComponentFunctionFactory::arbitrayObserver(abc, abc_dot);
			   KillingAbcField observerfieldDeform(  func, unsteadyFieldTimeStep, tmin, tmax);*/
				UnSteadyVectorField2D unsteady_field = Tobias_ObserverTransformation(steadyField, abc, abc_dot, tmin, tmax, unsteadyFieldTimeStep);
				// the first point is the distance to it self(always zero), use it as the segmentation label for this pathline.
				std::vector<std::vector<PathlinePointInfo>> ClusterPathlines = PathlineIntegrationInfoCollect2D(unsteady_field, 25, 4.0, deformMat, n_rc_si, txy);

				// visualize segmentation & pick random k pathlines to vis
				{
					auto steadyLicWithVaringPathlines = addPathlinesSegmentationVisualization(SteadyTexture, steadyField, n_rc_si, txy, deformMat, ClusterPathlines);
					string licFilename0 = task_licfolder + sample_tag_name + "_steady_lic.png";
					// draw steady field
					saveAsPNG(steadyLicWithVaringPathlines, licFilename0);
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
					archive_o(cereal::make_nvp("n_rc_Si", n_rc_si));
					archive_o(cereal::make_nvp("txy", txy));
					archive_o(cereal::make_nvp("deform_theta_sx_sy", deform_TheteaSxSy));
					archive_o(cereal::make_nvp("observer abc", abc));
					archive_o(cereal::make_nvp("observer abc_dot", abc_dot));
					/*archive_o(cereal::make_nvp("ClusterPathlines", ClusterPathlines));*/
				}
				// do not manually close file before creal deconstructor, as cereal will preprend a ]/} to finish json class/array
				jsonOut.close();

				// output binary of vector field, segmentation, pathlines
				string velocityFilename = task_folder + sample_tag_name + ".bin";
				string pathlineFilename = task_folder + sample_tag_name + "_pathline.bin";
				string segmentationFilename = task_folder + sample_tag_name + "_segmentation.bin";
				cerealBinaryOut(rawUnsteadyFieldData, velocityFilename);
				cerealBinaryOut(flatten3DAs1Dfloat(ClusterPathlines), pathlineFilename);
				if (Si != VastisVortexType::saddle) {
					auto seg = generateSegmentationBinaryMask(n_rc_si, txy, deformMat, Xdim, Ydim, gridInterval, domainMinBoundary);
					cerealBinaryOut(flatten2DAs1Dfloat(seg), segmentationFilename);
				}
			} // for (size_t observerIndex = 0..)

		} // for n,rc sample
		});
}