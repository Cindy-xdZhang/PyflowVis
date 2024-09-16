#pragma once
#ifndef OBSERVER_GENERATOR_H
#define OBSERVER_GENERATOR_H
#include <commonUtils.h>
#include <array>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"


namespace {
	//include bring in constexpr std::array<std::array<double, 3>, 709>colormap_tomPNG
#include "colormap_tomPNG.hpp"
}



Color mapValueToColor(double value) {
	// Ensure the value is within the [0, 1] range
	value = std::clamp(value, 0.0, 1.0);

	// Map value to corresponding index in colormap
	size_t index = static_cast<size_t>(value * (colormap_tomPNG.size() - 1));

	// Fetch the RGB values from the colormap
	auto rgb = colormap_tomPNG[index];

	return { rgb.at(0),rgb.at(1), rgb.at(2) };
}






// Function to convert a 2x2 rotation matrix to an angle
double matrix2angle(const Eigen::Matrix2d& rotationMat)
{
	// Ensure the matrix is orthogonal and its determinant is 1
	assert(rotationMat.determinant() > 0.999 && rotationMat.determinant() < 1.001);
	// Calculate the angle theta
	double theta = std::atan2(rotationMat(1, 0), rotationMat(0, 0));

	return theta;
}

std::string GetTimeStamp()
{
	const char* fmt = { "%A %c %EY" };
	const auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	std::stringstream ss;
	std::string date;
	ss << std::put_time(std::localtime(&time), fmt);
	date = ss.str();
	return date;
}

std::vector<std::vector<double>> loadSinglechanelPngFile(const std::string& filename, int& width, int& height)
{
	int n;
	unsigned char* data = stbi_load(filename.c_str(), &width, &height, &n, 1); // Load image in grayscale
	if (!data) {
		std::cerr << "Failed to load image: " << filename << std::endl;
		return {};
	}

	std::vector<std::vector<double>> texture(height, std::vector<double>(width));

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			texture[y][x] = data[y * width + x] / 255.0; // Normalize to [0, 1]
		}
	}

	stbi_image_free(data); // Free the image memory

	return texture;
}

std::vector<std::vector<std::array<double, 3>>> loadPngFile(const std::string& filename, int& width, int& height)
{
	int n;
	unsigned char* data = stbi_load(filename.c_str(), &width, &height, &n, 3); // Load image with 3 channels (RGB)
	if (!data) {
		std::cerr << "Failed to load image: " << filename << std::endl;
		return {};
	}

	std::vector<std::vector<std::array<double, 3>>> texture(height, std::vector<std::array<double, 3>>(width));

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			texture[y][x][0] = data[(y * width + x) * 3 + 0] / 255.0; // Red channel
			texture[y][x][1] = data[(y * width + x) * 3 + 1] / 255.0; // Green channel
			texture[y][x][2] = data[(y * width + x) * 3 + 2] / 255.0; // Blue channel
		}
	}

	stbi_image_free(data); // Free the image memory

	return texture;
}
void NoramlizeSpinTensor(Eigen::Matrix3d& input)
{
	Eigen::Vector3d unitAngular;
	unitAngular << input(2, 1), input(0, 2), input(1, 0);
	unitAngular.normalize();
	input << 0, -unitAngular(2), unitAngular(1),
		unitAngular(2), 0, -unitAngular(0),
		-unitAngular(1), unitAngular(0), 0;
	return;
};

void ConvertImage2Text(const std::string& infilename, const std::string& outFile, const std::string& textureName, bool SingleChannel)
{
	int width, height;
	if (SingleChannel)
	{
		auto texture = loadSinglechanelPngFile(infilename, width, height);
		std::ofstream out(outFile);
		if (!out.is_open()) {
			std::cerr << "Failed to open file for writing: " << outFile << std::endl;
			return;
		}
		constexpr int precision = 6;

		//out << "constexpr std::array<std::array<double, 64>, 64> noiseTexture = {\n";
		out << "constexpr std::array<std::array<double, " << width << ">, " << height << "> " << textureName << " = {\n";
		for (const auto& row : texture) {
			std::string beginer_string = "    std::array<double," + std::to_string(width) + ">{";
			out << beginer_string;
			for (size_t x = 0; x < row.size(); ++x) {
				out << std::fixed << std::setprecision(precision) << row[x];
				if (x < row.size() - 1)
					out << ", ";
			}
			out << " },\n";
		}
		out << "};\n";
		out.close();
	}
	else
	{
		auto texture = loadPngFile(infilename, width, height); // Load RGB image
		std::ofstream out(outFile);
		if (!out.is_open()) {
			std::cerr << "Failed to open file for writing: " << outFile << std::endl;
			return;
		}
		constexpr int precision = 6;

		out << "constexpr std::array<std::array<std::array<double, 3>, " << width << ">, " << height << "> " << textureName << " = {\n";
		for (const auto& row : texture) {
			std::string beginer_string = "    std::array<std::array<double, 3>, " + std::to_string(width) + ">{";
			out << beginer_string;
			for (size_t x = 0; x < row.size(); ++x) {
				out << "std::array<double, 3>{";
				out << std::fixed << std::setprecision(precision) << row[x][0] << ", "; // Red channel
				out << std::fixed << std::setprecision(precision) << row[x][1] << ", "; // Green channel
				out << std::fixed << std::setprecision(precision) << row[x][2];         // Blue channel
				out << "}";
				if (x < row.size() - 1)
					out << ", ";
			}
			out << " },\n";
		}
		out << "};\n";
		out.close();
	}


}

// Function to save the 2D vector as a PNG image
void saveAsPNG(const std::vector<std::vector<Eigen::Vector3d>>& data, const std::string& filename)
{
	int width = data[0].size();
	int height = data.size();

	// Create an array to hold the image data
	std::vector<unsigned char> image_data(width * height * 3); // 3 channels (RGB)

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			auto value = data[y][x];

			image_data[3 * (y * width + x) + 0] = static_cast<unsigned char>(value(0) * 255.0f); // Convert to 0-255
			image_data[3 * (y * width + x) + 1] = static_cast<unsigned char>(value(1) * 255.0f); // Convert to 0-255
			image_data[3 * (y * width + x) + 2] = static_cast<unsigned char>(value(2) * 255.0f); // Convert to 0-255pixel_value; // B
		}
	}

	// Save the image
	stbi_write_png(filename.c_str(), width, height, 3, image_data.data(), width * 3);
}

#endif
