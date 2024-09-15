
#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

#include <random>
#include <vector>
#include "VectorFieldCompute.h"


UnSteadyVectorField2D Tobias_ObserverTransformation(const SteadyVectorField2D& inputField, const Eigen::Vector3d& abc, const Eigen::Vector3d& abc_dot, const double tmin, const double tmax, const int timestep);
UnSteadyVectorField2D Tobias_reconstructUnsteadyField(const UnSteadyVectorField2D& inputField, const Eigen::Vector3d& abc, const Eigen::Vector3d& abc_dot);

// function killingABCtransformation transform a SteadyVectorField2D& inputField to an unsteady field by observing it with respect to KillingAbcField& observerfield
// @note: the result field has same spatial information(grid size, domain size) as the steady inputField and has same time (domain size, timesteps)  as the killing observerfield.
// inputField could be steady or unsteady field
// if  inputField  is unsteady field, the observerfield should have the same time domain as the inputField.
template <typename T = double, class InputFieldTYPE>
UnSteadyVectorField2D killingABCtransformation(const KillingAbcField& observerfield, const Eigen::Vector2d StartPosition, InputFieldTYPE& inputField);
//this is only correct when killing observer has rotation or translation only.
UnSteadyVectorField2D reconstructKillingDeformedUnsteadyField(std::function<Eigen::Vector3d(double)> predictKillingABCfunc, const UnSteadyVectorField2D& inputField);

