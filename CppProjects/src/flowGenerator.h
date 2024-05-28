#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <random>
#include <vector>
struct KillingComponentFunctionFactory {

    static std::function<Eigen::Vector3d(double)> getInverseObserver(const std::function<Eigen::Vector3d(double)>& funcA)
    {
        return [=](double t) -> Eigen::Vector3d {
            return -funcA(t);
        };
    }

    static std::function<Eigen::Vector3d(double)> randomObserver()
    {
        // Random device and generator
        std::random_device rd;
        std::mt19937 gen(rd());

        // Distribution for selecting type
        std::uniform_int_distribution<int> dist_type(0, 5);

        // Gaussian distributions for generating parameters
        std::normal_distribution<double> dist_speed(0.0, 0.5);
        std::normal_distribution<double> dist_acc(0.0, 0.2);
        std::normal_distribution<double> dist_rot(0.0, 0.2);

        // Randomly select a type
        int type = dist_type(gen);

        // Randomly generate parameters
        int direction = std::uniform_int_distribution<int>(0, 1)(gen);
        double scale = dist_speed(gen);
        double acc = dist_acc(gen);
        double rot = dist_rot(gen);

        // Return the corresponding function
        switch (type) {
        case 0:
            return constantTranslation(direction, scale);
        case 1:
            return constantAccTranslation(direction, acc);
        case 2:
            return combinedconstantTranslationRotation(direction, scale, rot);
        case 3:
            return constantRotation(scale);
        case 4:
            return constantAccRotation(acc);
        case 5:
            return constantAccTranslationRotation(direction, acc, rot);
        default:
            return constantTranslation(0, scale); // default case, should never hit
        }
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
            auto velocityAcc = transACc * t;
            auto rotAcc = transACc * t;
            if (direction == 0) {
                return Eigen::Vector3d(velocityAcc, 0, rotAcc);
            } else
                return Eigen::Vector3d(0, velocityAcc, rotAcc);
        };
    }
};

void testKillingTransformASteadyField();
void testKillingTransformationForRFC();
// number of result traing data = Nparamters * samplePerParameters * observerPerSetting
void generateUnsteadyField(int Nparamters, int samplePerParameters, int observerPerSetting);