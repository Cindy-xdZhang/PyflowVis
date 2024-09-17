#pragma once
#ifndef __COMMONUTILS_H__
#define __COMMONUTILS_H__

#include <Eigen/Core>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <chrono>

#include "stb/stb_image.h"
#include "VectorFieldCompute.h"
void saveAsPNG(const std::vector<std::vector<Eigen::Vector3d>>& data, const std::string& filename);
void NoramlizeSpinTensor(Eigen::Matrix3d& input);
void ConvertImage2Text(const std::string& infilename, const std::string& outFile, const std::string& textureName, bool singleChannel = false);
Color mapValueToColor(double value);
double matrix2angle(const Eigen::Matrix2d& rotationMat);

#ifdef _CEREAL_TOOLS_
#include <cereal/types/tuple.hpp>
#include <cereal/cereal.hpp>
namespace cereal {

	// Specialization for saving as array
	template <class Archive>
	void save(Archive& ar, const Eigen::Vector2d& vec)
	{
		ar(vec.x(), vec.y());
	}

	template <class Archive>
	void save(Archive& ar, const Eigen::Vector3d& vec)
	{
		ar(vec.x(), vec.y(), vec.z());
	}
	template <class Archive>
	void save(Archive& ar, const Eigen::Matrix2d& vec)
	{
		ar(vec(0, 0), vec(0, 1), vec(1, 0), vec(1, 1));
	}
}

template <typename T>
auto cerealBinaryOut(T data, const std::string& dest)
{
	std::ofstream outBin(dest, std::ios::binary);
	if (!outBin.good()) [[unlikely]] {
		printf("couldn't open file: %s", dest.c_str());
		return;
		}
	cereal::BinaryOutputArchive archive_Binary(outBin);
	archive_Binary(data);
	outBin.close();
}

#endif


// Function to flatten a 2D vector to a 1D vector
inline std::vector<float> flatten2DVectorsAs1Dfloat(const std::vector<std::vector<Eigen::Vector2d>>& x2D)
{
	const size_t ydim = x2D.size();
	assert(ydim > 0);
	const size_t xdim = x2D[0].size();
	std::vector<float> result;
	result.resize(xdim * ydim * 2);
	for (size_t i = 0; i < ydim; i++)
		for (size_t j = 0; j < xdim; j++) {


			float value0 = static_cast<float>(x2D[i][j](0));
			float value1 = static_cast<float>(x2D[i][j](1));
			if (!std::isfinite(value0) || !std::isfinite(value1) || std::isnan(value0) || std::isnan(value1)) {
				printf("got infinite or nan numbers.");
			}
			else {
				result[2 * (i * xdim + j)] = value0;
				result[2 * (i * xdim + j) + 1] = value1;
			}
		}

	return result;
}

inline  std::vector<float> flatten3DVectorsAs1Dfloat(const std::vector<std::vector<std::vector<Eigen::Vector2d>>>& x3D)
{

	const size_t tdim = x3D.size();
	assert(tdim > 0);
	const size_t ydim = x3D[0].size();
	assert(ydim > 0);
	const size_t xdim = x3D[0][0].size();
	std::vector<float> result;
	result.resize(xdim * ydim * tdim * 2);
	for (size_t t = 0; t < tdim; t++)
		for (size_t i = 0; i < ydim; i++)
			for (size_t j = 0; j < xdim; j++) {
				float value0 = static_cast<float>(x3D[t][i][j](0));
				float value1 = static_cast<float>(x3D[t][i][j](1));
				if (!std::isfinite(value0) || !std::isfinite(value1) || std::isnan(value0) || std::isnan(value1)) [[unlikely]] {
					printf("got infinite or nan numbers.");
					}
				else {
					result[2 * ((t * ydim + i) * xdim + j)] = value0;
					result[2 * ((t * ydim + i) * xdim + j) + 1] = value1;
				}

			}

	return result;
}
template <typename T>
inline  std::vector<float> flatten2DvecAs1Dfloat(const std::vector<std::vector<T>>& x2D)
{

	const size_t tdim = x2D.size();
	assert(tdim > 0);
	const size_t ydim = x2D[0].size();
	assert(ydim > 0);
	std::vector<float> result(ydim * tdim);
	for (size_t t = 0; t < tdim; t++)
		for (size_t i = 0; i < ydim; i++) {
			auto value = static_cast<float>(x2D[t][i]);
			if (!std::isfinite(value) || !std::isfinite(value)) [[unlikely]] {
				printf("got infinite or nan numbers.");
				}
			else
				result[t * ydim + i] = value;
		}

	return result;
}

template <typename T>
inline  std::vector<float> flatten3DvecAs1Dfloat(const std::vector<std::vector<std::vector<T>>>& x3D)
{

	const size_t tdim = x3D.size();
	assert(tdim > 0);
	const size_t ydim = x3D[0].size();
	assert(ydim > 0);
	const size_t xdim = x3D[0][0].size();
	std::vector<float> result(xdim * ydim * tdim);
	for (size_t t = 0; t < tdim; t++)
		for (size_t i = 0; i < ydim; i++)
			for (size_t j = 0; j < xdim; j++) {
				auto value = static_cast<float>(x3D[t][i][j]);
				if (!std::isfinite(value) || !std::isfinite(value)) [[unlikely]] {
					printf("got infinite or nan numbers.");
					}
				else
					result[(t * ydim + i) * xdim + j] = value;
			}

	return result;
}

template <typename T>
inline  auto checkVectorValues(const std::vector<T>& x1D)
{

	const size_t tdim = x1D.size();
	for (size_t t = 0; t < tdim; t++) {
		auto value = static_cast<float>(x1D[t]);
		if (!std::isfinite(value) || !std::isfinite(value)) [[unlikely]] {
			printf("got infinite or nan numbers.");
			}

	}

	return;
}

inline std::string trimNumString(const std::string& numString)
{
	std::string str = numString;
	str.erase(str.find_last_not_of('0') + 1, std::string::npos);
	str.erase(str.find_last_not_of('.') + 1, std::string::npos);
	return str;
}


std::string GetTimeStamp();


#endif
