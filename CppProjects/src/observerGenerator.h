#pragma once
#ifndef OBSERVER_GENERATOR_H
#define OBSERVER_GENERATOR_H

#include <Eigen/Core>

struct KillingComponentFunctionFactory {
    KillingComponentFunctionFactory() {};
    // using ObserverFunction = std::function<std::pair<Eigen::Vector3d, Eigen::Vector3d>(double)>;
    //// Random device and generator
    // static std::random_device rd;
    // static std::mt19937 gen(rd());
    // static std::uniform_real_distribution<double> dist_speed(0, 0.5);
    // static std::uniform_real_distribution<double> dist_acc(0, 1.0);

    static std::function<Eigen::Vector3d(double)> arbitrayObserver(const Eigen::Vector3d& abc, const Eigen::Vector3d& abc_dot)
    {

        return [=](double t) -> Eigen::Vector3d {
            const double tbiggerthanZero = t > 0 ? t : 0;
            // Integrate velocity to get position
            auto abc_t = (abc + abc_dot * t) * tbiggerthanZero;
            return Eigen::Vector3d(abc_t);
        };
    }

    static std::function<Eigen::Vector3d(double)> getInverseObserver(const std::function<Eigen::Vector3d(double)>& funcA)
    {
        return [=](double t) -> Eigen::Vector3d {
            return -funcA(t);
        };
    }
};

#endif
