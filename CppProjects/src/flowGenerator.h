#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <random>
#include <vector>
struct KillingComponentFunctionFactory {

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