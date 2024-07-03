#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <corecrt_math_defines.h>
#include <iostream>
#include <magic_enum/magic_enum.hpp>
#include <random>
#include <vector>
// Enum for observer function types
enum ObserverType {
    ConstTranslation = 0,
    ConstRotation,
    ConstAccRotation,
    ConstAccTranslation,
    ConstTranslationRotation,
    SinCurve,
    ArbitraryTranslation,
    StepRotation,
    StepTranslation,
    SpiralMotion
};
struct KillingComponentFunctionFactory {

    static std::function<Eigen::Vector3d(double)> getInverseObserver(const std::function<Eigen::Vector3d(double)>& funcA)
    {
        return [=](double t) -> Eigen::Vector3d {
            return -funcA(t);
        };
    }

    static std::function<Eigen::Vector3d(double)> randomObserver(int itype)
    {
        // Random device and generator
        std::random_device rd;
        std::mt19937 gen(rd());

        // Gaussian distributions for generating parameters
        std::normal_distribution<double> dist_speed(-0.2, 0.2);
        std::normal_distribution<double> dist_acc(-0.05, 0.05);
        std::normal_distribution<double> dist_rot(-0.1, 0.1);

        // Randomly generate parameters
        int direction = std::uniform_int_distribution<int>(0, 2)(gen);
        double scale = 0.5 * dist_speed(gen);
        double acc = 0.5 * dist_acc(gen);
        double rot = 0.5 * dist_rot(gen);
        double scaleA = 0.5 * dist_speed(gen);
        double scaleB = 0.5 * dist_speed(gen);
        ObserverType type = static_cast<ObserverType>(itype);
        // Return the corresponding function
        switch (type) {
        case ConstTranslation:
            return constantTranslation(direction, scale);
        case ConstAccTranslation:
            return constantAccTranslation(direction, acc);
        case ConstTranslationRotation:
            return combinedconstantTranslationRotation(direction, scale, rot);
        case ConstRotation:
            return constantRotation(scale);
        case ConstAccRotation:
            return constantAccRotation(acc);
        case SinCurve:
            return SinCurveObserver(direction, scale, rot);
        case StepRotation:
            return stepRotation(scale, 0.05 * M_PI);
        case StepTranslation:
            return stepTranslation(direction, scale, 0.05 * M_PI);
        case ArbitraryTranslation:
            return ArbitraryDirectionTranslation(scaleA, scaleB);
        case SpiralMotion:
            return spiralMotion(scale, rot);
        default:
            return constantTranslation(0, scale); // default case
        }
    }
    // constant translation velocity(killing  a or b), acc =0.
    static std::function<Eigen::Vector3d(double)> ArbitraryDirectionTranslation(double scaleA, double scaleB)
    {

        return [=](double t) {
            double tbiggerthanZero = t > 0 ? t : 0;
            return Eigen::Vector3d(scaleA * tbiggerthanZero, scaleB * tbiggerthanZero, 0);
        };
    }
    // constant translation velocity(killing  a or b), acc =0.
    static std::function<Eigen::Vector3d(double)> constantTranslation(int direction, double scale)
    {
        return [=](double t) {
            double tbiggerthanZero = t > 0 ? t : 0;
            double value = scale * tbiggerthanZero;
            if (direction == 0) {
                return Eigen::Vector3d(value, 0, 0);
            } else if (direction == 1) {
                return Eigen::Vector3d(0, value, 0);
            } else {
                return Eigen::Vector3d(value, value, 0);
            }
        };
    }
    // constant translation velocity(killing  a or b), acc =0.
    static std::function<Eigen::Vector3d(double)> SinCurveObserver(int direction, double speed, double start)
    {
        return [=](double t) {
            double tbiggerthanZero = t > 0 ? t : 0;
            double theta = speed * t + start;
            double value = 0.25 * sin(theta) * tbiggerthanZero;
            if (direction == 0) {
                return Eigen::Vector3d(value, 0, 0);
            } else if (direction == 1) {
                return Eigen::Vector3d(0, value, 0);
            } else {
                return Eigen::Vector3d(value, value, 0);
            }
        };
    }
    // constant translation velocity(killing  a or b), acc =0.
    static std::function<Eigen::Vector3d(double)> stepRotation(double validValue, double StopInterval)
    {
        return [=](double t) {
            // Calculate the period (one cycle of validValue and zero)
            double period = 2 * StopInterval;

            // Determine the position within the period
            double positionInPeriod = fmod(t, period);
            double tbiggerthanZero = t > 0 ? t : 0;
            const auto validValue2 = validValue * tbiggerthanZero;
            // If within the first half of the period, return validValue, otherwise return zero
            if (positionInPeriod < StopInterval) {
                return Eigen::Vector3d(0, 0, validValue2);
            } else {
                return Eigen::Vector3d(0, 0, 0);
            }
        };
    }

    // constant translation velocity(killing  a or b), acc =0.
    static std::function<Eigen::Vector3d(double)> stepTranslation(int direction, double validValue, double StopInterval)
    {
        return [=](double t) {
            // Calculate the period (one cycle of validValue and zero)
            double period = 2 * StopInterval;
            // Determine the position within the period
            double positionInPeriod = fmod(t, period);
            double tbiggerthanZero = t > 0 ? t : 0;
            auto value = (positionInPeriod < StopInterval && tbiggerthanZero) ? validValue : 0;
            if (direction == 0) {
                return Eigen::Vector3d(value, 0, 0);
            } else if (direction == 1) {
                return Eigen::Vector3d(0, value, 0);
            } else {
                return Eigen::Vector3d(value, value, 0);
            }
        };
    }

    // acc is the acceration of translation velocity
    static std::function<Eigen::Vector3d(double)> constantAccTranslation(int direction, double acc)
    {
        return [=](double t) {
            double tbiggerthanZero = t > 0 ? t : 0;
            auto velocity = acc * t * tbiggerthanZero;
            if (direction == 0) {
                return Eigen::Vector3d(velocity, 0, 0);
            } else if (direction == 1) {
                return Eigen::Vector3d(0, velocity, 0);
            } else {
                return Eigen::Vector3d(velocity, velocity, 0);
            }
        };
    }

    static std::function<Eigen::Vector3d(double)> combinedconstantTranslationRotation(int direction, double scale, double rot)
    {
        return [=](double t) {
            double tbiggerthanZero = t > 0 ? t : 0;
            auto velocity = scale * tbiggerthanZero;
            if (direction == 0) {
                return Eigen::Vector3d(velocity, 0, 0);
            } else if (direction == 1) {
                return Eigen::Vector3d(0, velocity, 0);
            } else {
                return Eigen::Vector3d(velocity, velocity, 0);
            }
        };
    }

    static std::function<Eigen::Vector3d(double)> constantRotation(double speed)
    {
        return [=](double t) {
            double tbiggerthanZero = t > 0 ? t : 0;
            return Eigen::Vector3d(0, 0, speed * tbiggerthanZero);
        };
    }

    static std::function<Eigen::Vector3d(double)> constantAccRotation(double acc)
    {
        return [=](double t) {
            double tbiggerthanZero = t > 0 ? t : 0;
            double velocity = acc * t * tbiggerthanZero;
            return Eigen::Vector3d(0, 0, velocity);
        };
    }
    static std::function<Eigen::Vector3d(double)> spiralMotion(double radialSpeed, double angularSpeed)
    {
        return [=](double t) {
            double tbiggerthanZero = t > 0 ? t : 0;
            // r increases linearly with t
            double r = radialSpeed * tbiggerthanZero;
            // theta increases linearly with t
            double theta = angularSpeed * tbiggerthanZero;
            // Convert polar coordinates to Cartesian coordinates
            double x = r * std::cos(theta);
            double y = r * std::sin(theta);
            // Assuming spiral motion in the XY-plane with a constant Z-component
            return Eigen::Vector3d(x, y, 0);
        };
    }
};

void ConvertNoiseTextureImage2Text(const std::string& infilename, const std::string& outFile, int width, int height);
void testKillingTransformationForRFC();
// number of result traing data = Nparamters * samplePerParameters * observerPerSetting
void generateUnsteadyField(int Nparamters, int samplePerParameters, int observerPerSetting, std::string dataSetSplitTag);
void testCriterion();