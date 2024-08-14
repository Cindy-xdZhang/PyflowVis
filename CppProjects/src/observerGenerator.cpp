#include "observerGenerator.h"

#if 0

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
// KillingComponentFunctionFactory  has problem that when there is both translation and rotation, the inverse observer function can't work.
// instead of using KillingComponentFunctionFactory, we should use something like theta_dot,theta_dotdot, cdot,cdotdot which are rotation velocity, rotation acceleration, translation velocity, translation acceleration.
// there are equivalent to killing a, killing b, killing c,  adot, bdot,cdot.
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
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<double> dist_speed(-0.5, 0.5);
        static std::uniform_real_distribution<double> dist_acc(-0.1, 0.1);
        static std::uniform_real_distribution<double> dist_rot(-0.1, 0.1);

        // Randomly generate parameters
        int direction = std::uniform_int_distribution<int>(0, 2)(gen);
        double scale = dist_speed(gen);
        double acc = dist_acc(gen);
        double rot = dist_rot(gen);
        double scaleA = dist_speed(gen);
        double scaleB = dist_speed(gen);
        ObserverType type = static_cast<ObserverType>(itype);
        // Return the corresponding function
        switch (type) {
        case ConstTranslation:
            return constantTranslation(direction, scale);
        case ConstAccTranslation:
            return constantAccTranslation(direction, acc);
        case ConstRotation:
            return constantRotation(scale);
        case ConstAccRotation:
            return constantAccRotation(acc);
        case ConstTranslationRotation:
            return combinedconstantTranslationRotation(direction, scale, rot);
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
    static std::function<Eigen::Vector3d(double)> constantTranslation(int direction, double scale)
    {
        return [=](double t) {
            double value = t > 0 ? scale : 0;
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
            double tbiggerthanZero = t > 0 ? 1 : 0;
            double theta = speed * t + start;
            double value = 0.5 * sin(theta) * tbiggerthanZero;
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
            double tbiggerthanZero = t > 0 ? 1 : 0;
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
            auto value = (positionInPeriod < StopInterval && t > 0) ? validValue : 0;
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
            auto velocity = acc * tbiggerthanZero;
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
            double tbiggerthanZero = t > 0 ? 1 : 0;
            auto velocity = 0.1 * tbiggerthanZero;
            auto rot_ = 0.1 * tbiggerthanZero;
            if (direction == 0) {
                return Eigen::Vector3d(velocity, 0, rot_);
            } else if (direction == 1) {
                return Eigen::Vector3d(0, velocity, rot_);
            } else {
                return Eigen::Vector3d(velocity, velocity, rot_);
            }
        };
    }

    static std::function<Eigen::Vector3d(double)> constantRotation(double speed)
    {
        return [=](double t) {
            double tbiggerthanZero = t > 0 ? 1 : 0;
            return Eigen::Vector3d(0, 0, speed * tbiggerthanZero);
        };
    }
    // constant translation velocity(killing  a or b), acc =0.
    static std::function<Eigen::Vector3d(double)> ArbitraryDirectionTranslation(double scaleA, double scaleB)
    {

        return [=](double t) {
            double tbiggerthanZero = t > 0 ? 1 : 0;
            return Eigen::Vector3d(scaleA * tbiggerthanZero, scaleB * tbiggerthanZero, 0);
        };
    }
    static std::function<Eigen::Vector3d(double)> constantAccRotation(double acc)
    {
        return [=](double t) {
            double tbiggerthanZero = t > 0 ? 1 : 0;
            double velocity = acc * t * tbiggerthanZero;
            return Eigen::Vector3d(0, 0, velocity);
        };
    }
    static std::function<Eigen::Vector3d(double)> spiralMotion(double radialSpeed, double angularSpeed)
    {
        return [=](double t) {
            double tbiggerthanZero = t > 0 ? 1 : 0;
            // r increases linearly with t
            double r = radialSpeed * t * tbiggerthanZero;
            // theta increases linearly with t
            double theta = angularSpeed * t * tbiggerthanZero;
            // Convert polar coordinates to Cartesian coordinates
            double x = r * std::cos(theta);
            double y = r * std::sin(theta);
            // Assuming spiral motion in the XY-plane with a constant Z-component
            return Eigen::Vector3d(x, y, 0);
        };
    }
};

#endif