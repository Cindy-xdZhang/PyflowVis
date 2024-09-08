#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <corecrt_math_defines.h>
#include <iostream>
#include <string>
#include <random>
#include <vector>

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
	void GenDataset(int Nparamters, int samplePerParameters, int observerPerSetting, const std::string in_root_fodler, double rateVal_Train = 0.1, double rateVal_Test = 0.1);

	//son class overwrite this
	virtual void GenOneSplit(int Nparamters, int samplePerParameters, int observerPerSetting, const std::string dataSetSplitTag) {}

	double minV;
	double maxV;
	Eigen::Vector2d domainRange;
	Eigen::Vector2d gridInterval;
	std::string split_root_folder;


};

class UnsteadyPathlneDataSetGenerator :public DataSetGenBase {
	virtual  void GenOneSplit(int Nparamters, int samplePerParameters, int observerPerSetting, const std::string dataSetSplitTag);
};