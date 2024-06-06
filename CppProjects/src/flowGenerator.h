#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <random>
#include <vector>
// Enum for observer function types
enum ObserverType {
    ConstantTranslation = 0,
    ConstantRotation = 1,
    ConstantAccRotation = 2,
    ConstantAccTranslation = 3,
    CombinedConstantTranslationRotation,
    ConstantAccTranslationRotation,
    SinCurve,
    NumTypes // This should always be the last entry
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
        std::normal_distribution<double> dist_speed(0.0, 0.25);
        std::normal_distribution<double> dist_acc(0.0, 0.1);
        std::normal_distribution<double> dist_rot(0.0, 0.1);

        // Randomly generate parameters
        int direction = std::uniform_int_distribution<int>(0, 1)(gen);
        double scale = dist_speed(gen);
        double acc = dist_acc(gen);
        double rot = dist_rot(gen);
        ObserverType type = static_cast<ObserverType>(itype);
        // Return the corresponding function
        switch (type) {
        case ConstantTranslation:
            return constantTranslation(direction, scale);
        case ConstantAccTranslation:
            return constantAccTranslation(direction, acc);
        case CombinedConstantTranslationRotation:
            return combinedconstantTranslationRotation(direction, scale, rot);
        case ConstantRotation:
            return constantRotation(scale);
        case ConstantAccRotation:
            return constantAccRotation(acc);
        case ConstantAccTranslationRotation:
            return constantAccTranslationRotation(direction, acc, rot);
        case SinCurve:
            return SinCurveObserver(direction);
        default:
            return constantTranslation(0, scale); // default case, should never hit
        }
    }

    // constant translation velocity(killing  a or b), acc =0.
    static std::function<Eigen::Vector3d(double)> SinCurveObserver(int direction)
    {
        return [=](double t) {
            double scale = 0.2 * sin(t);
            if (direction == 0) {
                return Eigen::Vector3d(scale, 0, 0);
            } else
                return Eigen::Vector3d(0, scale, 0);
        };
    }

    // constant translation velocity(killing  a or b), acc =0.
    static std::function<Eigen::Vector3d(double)> constantTranslation(int direction, double scale)
    {
        return [=](double t) {
            if (direction == 0) {
                return Eigen::Vector3d(scale, 0, 0);
            } else
                return Eigen::Vector3d(0, scale, 0);
        };
    }

    // acc is the acceration of translation velocity
    static std::function<Eigen::Vector3d(double)> constantAccTranslation(int direction, double acc)
    {
        return [=](double t) {
            auto velocity = acc * t;
            if (direction == 0) {
                return Eigen::Vector3d(velocity, 0, 0);
            } else
                return Eigen::Vector3d(0, velocity, 0);
        };
    }

    static std::function<Eigen::Vector3d(double)> combinedconstantTranslationRotation(int direction, double scale, double rot)
    {
        return [=](double t) {
            if (direction == 0) {
                return Eigen::Vector3d(scale, 0, rot);
            } else
                return Eigen::Vector3d(0, scale, rot);
        };
    }

    static std::function<Eigen::Vector3d(double)> constantRotation(double speed)
    {
        return [=](double t) {
            return Eigen::Vector3d(0, 0, speed);
        };
    }

    static std::function<Eigen::Vector3d(double)> constantAccRotation(double acc)
    {
        return [=](double t) {
            return Eigen::Vector3d(0, 0, acc * t);
        };
    }
    static std::function<Eigen::Vector3d(double)> constantAccTranslationRotation(int direction, double transACc, double rotAcc)
    {
        return [=](double t) {
            auto velocity = transACc * t;
            auto rot = rotAcc * t;
            if (direction == 0) {
                return Eigen::Vector3d(velocity, 0, rot);
            } else
                return Eigen::Vector3d(0, velocity, rot);
        };
    }
};

void testKillingTransformASteadyField();
void testKillingTransformationForRFC();
// number of result traing data = Nparamters * samplePerParameters * observerPerSetting
void generateUnsteadyField(int Nparamters, int samplePerParameters, int observerPerSetting);