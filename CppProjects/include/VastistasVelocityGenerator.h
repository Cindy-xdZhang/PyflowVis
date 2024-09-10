
#pragma once
#ifndef __VASTISTASVELOCITYGENERATOR_H__
#define __VASTISTASVELOCITYGENERATOR_H__
#include "VectorFieldCompute.h"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <corecrt_math_defines.h>
#include <string>
#include <vector>
#include <unordered_map>

// one VastisParamter is Eigen::Vector2d rc_n, Eigen::Vector2d tx_ty,Eigen::Vector3d sxsytheta, int si.
using VastisParamter = std::tuple<Eigen::Vector2d, Eigen::Vector2d, Eigen::Vector3d, int>;

enum class VASTIS_PARAM : unsigned char {
	VastisParamRC_N = 0,
	VastisParamTXY = 1,
	VastisParamSXSYTHETA = 2,
	VastisParamSI = 3
};


enum class VastisVortexType : unsigned char {
	saddle = 0,
	center_cw = 1,
	center_ccw = 2,
	zero_field = 3
};

enum class VastisParamRC_N : unsigned char {
	VastisParamRC = 0,
	VastisParamN = 1,
};
inline auto NormalizedVastistasV0_Fn(const double r, const double in_rc, const double in_n)
{
	const auto v0_r = r / (2 * M_PI * (in_rc * in_rc) * std::pow(std::pow(r / in_rc, 2 * in_n) + 1, 1 / in_n));
	return v0_r / (r);
}
std::vector<std::pair<double, double>> generateVastisRC_NParamters(int n, std::string mode);

class VastistasVelocityGenerator {
public:
	VastistasVelocityGenerator(int Xdim, int Ydim, Eigen::Vector2d minBondary, Eigen::Vector2d maxBondary, double rc, double n);

	// generate a deformed steady velocity slice  following the paper: Vortex Boundary Identification using Convolutional Neural Network
	SteadyVectorField2D generateSteadyField_VortexBoundaryVIS2020(double tx, double ty, double sx, double sy, double theta, VastisVortexType Si) const noexcept;
	SteadyVectorField2D generateSteadyField_VortexBoundaryVIS2020(const Eigen::Vector2d& txy, const Eigen::Vector3d sxsy_theta, VastisVortexType Si) const noexcept;


	// support mixture of multiple Vastistas profile
	SteadyVectorField2D generateSteadyFieldMixtureRobustPaper(int mixture) const noexcept;

	// only supprt ccw/cw vortex mix together, and we asking the center of two vortex is bigger than rc1 +rc2
	SteadyVectorField2D generateSteadyMixtureOFVortexBoundaryPaper(const std::vector<VastisParamter>& vectorFieldMeta) const noexcept;

	inline auto NormalizedVastistasV0(const double r) const noexcept
	{
		assert(rc > 0);
		const auto v0_r = r / (2 * M_PI * (rc * rc) * std::pow(std::pow(r / rc, 2 * n) + 1, 1 / n));
		return v0_r / (r);
	}

	inline auto getPosition(int xid, int yid) const noexcept
	{
		double x = domainBoundaryMin.x() + xid * spatialGridInterval.x();
		double y = domainBoundaryMin.y() + yid * spatialGridInterval.y();
		return Eigen::Vector2d{ x, y };
	}

private:
	int mgridDim_x;
	int mgridDim_y;
	double rc, n;

	Eigen::Vector2d spatialGridInterval;
	Eigen::Vector2d domainBoundaryMin;
	Eigen::Vector2d domainBoundaryMax;
	Eigen::Matrix2d SiMatices_[3];
};

// AnalyticalFlowCreator  is the helper class for creating UnSteadyVectorField2D
class AnalyticalFlowCreator {
public:
	AnalyticalFlowCreator(Eigen::Vector2i grid_size, int time_steps,
		Eigen::Vector2d domainBoundaryMin,
		Eigen::Vector2d domainBoundaryMax, double tmin = 0.0f, double tmax = 2 * M_PI);

	// Method to create the flow field using a lambda function
	UnSteadyVectorField2D sampleAnalyticalFunctionAsFlowField(AnalyticalFlowFunc2D lambda_func);
	UnSteadyVectorField2D createRFC(double alt = 1.0, double maxV = 1.0, double scale = 8.0);
	UnSteadyVectorField2D createBeadsFlow();
	UnSteadyVectorField2D createBeadsFlowNoContraction();
	UnSteadyVectorField2D createUnsteadyGyre();
	AnalyticalFlowFunc2D getAnalyticalFlowFieldFunction(const std::string& name, double tMin, double tMax, int numberOfTimeSteps);

	std::unordered_map<std::string, UnSteadyVectorField2D >generateAnalyticalTestSuite();
private:
	Eigen::Vector2i grid_size;
	int time_steps;
	Eigen::Vector2d domainBoundaryMin;
	Eigen::Vector2d domainBoundaryMax;
	double tmin, tmax, t_interval;
	Eigen::VectorXd t_values;
	Eigen::VectorXd x_values;
	Eigen::VectorXd y_values;
};

#endif