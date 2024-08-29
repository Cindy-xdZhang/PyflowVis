#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <corecrt_math_defines.h>
#include <iostream>

#include <random>
#include <vector>

void ConvertNoiseTextureImage2Text(const std::string& infilename, const std::string& outFile, int width, int height);
void testKillingTransformationForRFC();
// number of result traing data = Nparamters * samplePerParameters * observerPerSetting

void generateUnsteadyField(int Nparamters, int samplePerParameters, int observerPerSetting, const std::string in_root_fodler, const std::string dataSetSplitTag);
void GenerateSteadyVortexBoundary(int Nparamters, int samplePerParameters, const std::string in_root_fodler, const std::string dataSetSplitTag);
void generateUnsteadyFieldPathline(int Nparamters, int samplePerParameters, int observerPerSetting, const std::string in_root_fodler, const std::string dataSetSplitTag);
void testCriterion();