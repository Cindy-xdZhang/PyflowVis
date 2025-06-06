#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <string>
#include <random>
#include <vector>
#define  _CEREAL_TOOLS_
#include "commonUtils.h"
#include "VastistasVelocityGenerator.h"






// void ConvertNoiseTextureImage2Text(const std::string& infilename, const std::string& outFile, int width, int height);

// number of result traing data = Nparamters * samplePerParameters * observerPerSetting
void generateUnsteadyFieldV0(int Nparamters, int samplePerParameters, int observerPerSetting, const std::string in_root_fodler, const std::string dataSetSplitTag);

namespace REPRODUCE {
	void GenerateSteadyVortexBoundary(int Nparamters, int samplePerParameters, const std::string in_root_fodler, const std::string dataSetSplitTag);
	void generateUnsteadyFieldMixture(int Nfield, int observerPerField, const std::string in_root_fodler, const std::string dataSetSplitTag);

}

namespace DBG_TEST {
	void testCriterion();
}


class DataSetGenBase {
public:
	void GenDataset(int Nparamters, int samplePerParameters, int observerPerSetting, const std::string& in_root_fodler, double rateVal_Train = 0.1, double rateVal_Test = 0.1);

	//son class overwrite this
	virtual void GenOneSplit(int Nparamters, int samplePerParameters, int observerPerSetting, const std::string& dataSetSplitTag) = 0;

	double minV;
	double maxV;
	Eigen::Vector2d domainRange;
	Eigen::Vector2d gridInterval;
	std::string split_root_folder;
};

class UnsteadyPathlneDataSetGenerator :public DataSetGenBase {
public:
	virtual  void GenOneSplit(int Nparamters, int samplePerParameters, int observerPerSetting, const std::string& dataSetSplitTag);

	//using VastisParamter = std::tuple<Eigen::Vector2d, Eigen::Vector2d, Eigen::Vector3d, int>;
	//direclty map one VastisParamter = std::tuple< Eigen::Vector2d rc_n, Eigen::Vector2d tx_ty,Eigen::Vector3d sxsytheta, int si>->to a data entry(for reproduce.`)
	void DeSerialize(const std::string& dest_folder, const VastisParamter& v_param, const Eigen::Vector3d& ObserverAbc, const Eigen::Vector3d& ObserverAbcDot, const std::string& SampleName);
	void classicalParametersDeserialization(const std::string& dst_folder);
	void analyticalTestCasesGeneration(const std::string& dst_folder);

	// generateUnsteadyFieldPathlineMix will have mixture steady field that has multiple(2-4) vortex & rotating zero field& static field deformed by zero (identity observer)
	void generateMixUnsteadyFieldPathline(const std::string& dest_folder, int Samples, int ObserversPerSample);

	static std::uniform_real_distribution<double> genTheta; // rotation angle's distribution
	static std::normal_distribution<double> genSx;
	static std::normal_distribution<double> genSy;
	static std::normal_distribution<double> genTx;
	static std::normal_distribution<double> genTy;
private:
	std::vector<VastisParamter> generateRandomMixtureVastisParam(const int mix);

};


