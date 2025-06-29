#include "ScalarField3D.h"
#include "logSystem/log.h"
#include "mydefs.h"
#include <execution>
#include "Discrete3DFlowField.h"
#include "FlowField3DUtils.h"


//#define NO_PARALLEL
namespace {
#if defined(_DEBUG ) && defined(NO_PARALLEL) 
auto policy = std::execution::seq;
#else
auto policy = std::execution::par_unseq;
#endif
}

namespace DTG {
template <typename T>
void  DTG::DiscreteScalarField3D<T>::resampleAnalytical2Gird()
{
	//	using  Scalar3DFunc=std::function<float(float,float,float, float)>;
		const int gridX =  this->mFieldInfo.XGridsize;
		const int gridY = this->mFieldInfo.YGridsize;
		const int gridZ = this->mFieldInfo.ZGridsize;
		// Total number of points in the 4D grid (excluding boundaries)
		const int totalPoints = (gridX - 2) * (gridY - 2) * ( gridZ  - 2) *  this->mFieldInfo.numberOfTimeSteps;
		const int planeStride = (gridY - 2) * (gridX - 2);
		const int rowStride = gridX - 2;

		// Create a linear index mapping each point in the 4D grid
		std::vector<int> indices(totalPoints);
		std::iota(indices.begin(), indices.end(), 0);


		// Parallel transform
		std::for_each(policy, indices.begin(), indices.end(),
			[&](int linear_idx) {
				// Calculate 4D indices directly using precomputed strides
				const int idt = linear_idx / ((gridZ - 2) * planeStride);
				linear_idx -= idt * (gridZ - 2) * planeStride;

				const int idz = linear_idx / planeStride + 1; // Offset by 1 to skip boundaries
				linear_idx -= (idz - 1) * planeStride;

				const int idy = linear_idx / rowStride + 1; // Offset by 1 to skip boundaries
				const int idx = linear_idx % rowStride + 1; // Offset by 1 to skip boundaries

				
				auto xyz_pos=this->convertGridIndex2PhysicalCoordinates(idx,idy,idz);
				auto time=this->convertTimeStep2PhysicalTime(idt);

				// Apply pointwise function and set the result
				auto value = mAnyticalFunc(xyz_pos.x(), xyz_pos.y(), xyz_pos.z(), time);
				this->SetValue(idx, idy, idz, idt, value);
			});
	return;
}

template <typename T>
std::vector<T> DTG::DiscreteScalarField3D<T>::getSliceAtPhysicalTime(double t) const
{
	double dt = mFieldInfo.Getdt();
	double continousIndex = (t - mFieldInfo.minTime) / (dt);
	int floorIndex = std::floor(continousIndex);
	int ceilIndex = std::ceil(continousIndex);

	floorIndex = std::clamp(floorIndex, 0, mFieldInfo.numberOfTimeSteps - 1);
	ceilIndex = std::clamp(ceilIndex, 0, mFieldInfo.numberOfTimeSteps - 1);
	auto resRawData = this->getSliceDataConst(floorIndex);

	if (ceilIndex != floorIndex) {
		auto resRawDataCeil = this->getSliceDataConst(ceilIndex);
		double fraction = continousIndex - (double)floorIndex;

		for (size_t i = 0; i < resRawDataCeil.size(); ++i) {
			resRawData[i] = resRawData[i] * (1.0 - fraction) + resRawDataCeil[i] * fraction;
		}
	}
	return resRawData;
}



template <typename T>
std::tuple<T, T> DTG::DiscreteScalarField3D<T>::computeMinMaxValue() const
{
	T globalMin = std::numeric_limits<T>::max();
	T globalMax = std::numeric_limits<T>::min();

	// Perform a parallel reduction over all slices.
	for (const auto& slice : dataSlices_) {
		T localMin0 = std::numeric_limits<T>::max();
		T localMax0 = std::numeric_limits<T>::min();
		// Use std::reduce with a parallel execution policy for each slice.
		T localMin = std::reduce(
			std::execution::par,
			slice.begin(), slice.end(),
			localMin0,
			[](T a, T val) -> T {
				return std::min(a, val);
			});
		T localMax = std::reduce(
			std::execution::par,
			slice.begin(), slice.end(),
			localMax0,
			[](T a, T val) -> T {
				return std::max(a, val);
			});

		// Update global min and max values.
		globalMin = std::min(globalMin, localMin);
		globalMax = std::max(globalMax, localMax);
	}

	return std::make_tuple(globalMin, globalMax);
}

template <typename T>
bool DTG::DiscreteScalarField3D<T>::HasDiscreteData() const
{
	if (dataSlices_.size() > 0)
	{
		return dataSlices_[0].size() > 0;
	}
	return false;
}

template <typename T>
void DTG::DiscreteScalarField3D<T>::SetValue(uint64_t x, uint64_t y, uint64_t z, uint64_t t, T value)
{
	// CheckBounds(x, y, z,t);
	dataSlices_[t][GetSpatialIndex(x, y, z)] = value;
}




template <typename T>
double DTG::DiscreteScalarField3D<T>::ScalarFieldReduction(const std::string& reductionType) const
{
	// Validate the reduction type
	if (reductionType != "sum" && reductionType != "average") {
		throw std::invalid_argument("Invalid reduction type. Use 'sum' or 'average'.");
	}

	// Variables for Kahan summation
	double totalSum = 0.0;
	double compensation = 0.0; // Compensates for lost low-order bits
	int totalElements = 0;
	// Iterate through each slice
	for (const auto& slice : dataSlices_) {
		totalElements += slice.size();
		for (const auto& value : slice) {
			double y = static_cast<double>(value) - compensation; // Compensate for lost bits
			double t = totalSum + y; // Tentative sum
			compensation = (t - totalSum) - y; // Update the compensation
			totalSum = t; // Update the sum
		}
	}

	if (reductionType == "average") {
		// Avoid division by zero
		if (totalElements == 0) {
			throw std::runtime_error("Cannot compute average: no elements in the field.");
		}
		auto avg = totalSum / totalElements;
		return avg;
	}
	return totalSum; // Return the sum
}

template <typename T>
void DTG:: DiscreteScalarField3D<T>::resampleASliceToFloatBuffer(std::vector<float>& buffer, Eigen::Vector2i& sliceGridSize, int SamplingAxis, double PositionConstantRatio)
{
	// Get the number of time steps.
	int numTimeSteps = mFieldInfo.numberOfTimeSteps;
	// Determine the 2D resolution of the slice depending on SamplingAxis.
	int resDim1 = 0;
	int resDim2 = 0;
	double minPos1 = 0.0, maxPos1 = 0.0;
	double minPos2 = 0.0, maxPos2 = 0.0;
	// For each sampling axis, choose the other two axes for the slice and the corresponding physical bounds.
	// SamplingAxis == 0  => Plane: x = PositionConstant, grid over y and z.
	// SamplingAxis == 1  => Plane: y = PositionConstant, grid over x and z.
	// SamplingAxis == 2  => Plane: z = PositionConstant, grid over x and y.
	const double Xrange = mFieldInfo.maxXCoordinate - mFieldInfo.minXCoordinate;
	const double Yrange = mFieldInfo.maxYCoordinate - mFieldInfo.minYCoordinate;
	const double Zrange = mFieldInfo.maxZCoordinate - mFieldInfo.minZCoordinate;
	double PlaneConstantPos;
	if (SamplingAxis == 0) {
		PlaneConstantPos = PositionConstantRatio * Xrange + mFieldInfo.minXCoordinate;

		resDim1 = mFieldInfo.YGridsize; // along Y
		resDim2 = mFieldInfo.ZGridsize; // along Z
		minPos1 = mFieldInfo.minYCoordinate;
		maxPos1 = mFieldInfo.maxYCoordinate;
		minPos2 = mFieldInfo.minZCoordinate;
		maxPos2 = mFieldInfo.maxZCoordinate;
	}
	else if (SamplingAxis == 1) {
		PlaneConstantPos = PositionConstantRatio * Yrange + mFieldInfo.minYCoordinate;
		resDim1 = mFieldInfo.XGridsize; // along X
		resDim2 = mFieldInfo.ZGridsize; // along Z
		minPos1 = mFieldInfo.minXCoordinate;
		maxPos1 = mFieldInfo.maxXCoordinate;
		minPos2 = mFieldInfo.minZCoordinate;
		maxPos2 = mFieldInfo.maxZCoordinate;
	}
	else if (SamplingAxis == 2) {
		PlaneConstantPos = PositionConstantRatio * Zrange + mFieldInfo.minZCoordinate;
		resDim1 = mFieldInfo.XGridsize; // along X
		resDim2 = mFieldInfo.YGridsize; // along Y
		minPos1 = mFieldInfo.minXCoordinate;
		maxPos1 = mFieldInfo.maxXCoordinate;
		minPos2 = mFieldInfo.minYCoordinate;
		maxPos2 = mFieldInfo.maxYCoordinate;
	}
	else {
		// Invalid SamplingAxis. Do nothing.
		buffer.clear();
		return;
	}

	// The slice will be sampled for each time step.
	// The total number of sample points in the buffer is numTimeSteps * (resDim1 * resDim2)
	int sliceSize = resDim1 * resDim2;
	sliceGridSize = { resDim1, resDim2 };
	int totalSamples = numTimeSteps * sliceSize;
	buffer.resize(totalSamples);
	// Compute the time step in physical time.
	double dt = (numTimeSteps > 1) ? (mFieldInfo.maxTime - mFieldInfo.minTime) / (numTimeSteps - 1) : 0.0;
	// Loop over all time steps and sample the corresponding slice.
	// We parallelize over time steps, assuming that interpolation is thread-safe.
#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
	for (int t = 0; t < numTimeSteps; t++) {
		double physicalTime = mFieldInfo.minTime + t * dt;
		for (int j = 0; j < resDim2; j++) {
			// Compute normalized coordinate in the second direction [0,1]
			double v = resDim2 > 1 ? static_cast<double>(j) / (resDim2 - 1) : 0.5;
			// Map normalized coordinate to physical coordinate for axis2.
			double pos2 = minPos2 + v * (maxPos2 - minPos2);

			for (int i = 0; i < resDim1; i++) {
				// Compute normalized coordinate in the first direction [0,1]
				double u = resDim1 > 1 ? static_cast<double>(i) / (resDim1 - 1) : 0.5;
				// Map normalized coordinate to physical coordinate for axis1.
				double pos1 = minPos1 + u * (maxPos1 - minPos1);

				// Set the physical coordinates for this sample.
				double xpos = 0.0, ypos = 0.0, zpos = 0.0;
				switch (SamplingAxis) {
				case 0:
					// x is fixed by PositionConstant; y and z vary.
					xpos = PlaneConstantPos;
					ypos = pos1;
					zpos = pos2;
					break;
				case 1:
					// y is fixed
					xpos = pos1;
					ypos = PlaneConstantPos;
					zpos = pos2;
					break;
				case 2:
					// z is fixed
					xpos = pos1;
					ypos = pos2;
					zpos = PlaneConstantPos;
					break;
				}
				// Get the interpolated scalar value at (xpos, ypos, zpos, physicalTime)
				T value = InterpolateValuebyPos(xpos, ypos, zpos, physicalTime);
				// Compute the index into the flattened buffer.
				int globalIndex = t * sliceSize + (j * resDim1 + i);
				buffer[globalIndex] = static_cast<float>(value);
			}
		}
	}
}



Eigen::Vector2d DTG::AttributesManager::requestScalarFieldMinMax(const std::string& requestFieldName)
{
	auto iter = mScalarFieldsMinMax.find(requestFieldName);
	if (iter == mScalarFieldsMinMax.end())
	{
		if (const auto resultScalarFieldCached = getScalarField(requestFieldName); resultScalarFieldCached&&resultScalarFieldCached->HasDiscreteData())
		{
			auto [minV, maxV] = resultScalarFieldCached->computeMinMaxValue();
			Eigen::Vector2d  res = { minV,maxV };
			mScalarFieldsMinMax.insert({ requestFieldName, res });
			return res;
		}
		else if (resultScalarFieldCached&&resultScalarFieldCached->HasAnalyticalExpression())
		{
				resultScalarFieldCached->resampleAnalytical2Gird();
				auto [minV, maxV] = resultScalarFieldCached->computeMinMaxValue();
				Eigen::Vector2d  res = { minV,maxV };
				mScalarFieldsMinMax.insert({ requestFieldName, res });
				return res;
		}else
			LOG_E("requestScalarFieldMinMax %s failed,please request the scalar field first...",requestFieldName.c_str());
		return {};
	}
	else
	{
		return iter->second;
	}

}



DTG::DiscreteScalarField3D<float>* DTG::AttributesManager::requestAnalyticalScalarField(const std::string& inputFieldName,const  FLOWFIELD_OPERATION operation)
{
	const auto requestFieldName = AttributesNameGenerate(inputFieldName, operation);
	if (const auto resultScalarFieldCached = getScalarField(requestFieldName); resultScalarFieldCached&&resultScalarFieldCached->HasAnalyticalExpression())
	{
		return resultScalarFieldCached;
	}
	else
	{
		//compute new scalar field
		if (auto* vfIterface = dynamic_cast<Discrete3DFlowField<float>*>( VectorField3DManager::GetInstance().GetVectorFieldByName(inputFieldName)); vfIterface)
		{
			auto resultField = std::make_unique<DiscreteScalarField3D<float>>(vfIterface->GetFieldInfo());
			auto dxdydz = vfIterface->GetSpatialDxDyDz();
			dxdydz *=0.1;
			Scalar3DFunc analyticalFunc= nullptr;
			if  (operation == FLOWFIELD_OPERATION::CURL_NORM)
			{
				analyticalFunc = [vfIterface, dxdydz](const float x, const float y, const float z, const float t) -> float {
					
					// Get neighboring vectors
					auto v_xPlus = vfIterface->GetVector(x + dxdydz.x(), y, z, t);
					auto v_xMinus = vfIterface->GetVector(x - dxdydz.x(), y, z, t);
					auto v_yPlus = vfIterface->GetVector(x, y + dxdydz.y(), z, t);
					auto v_yMinus = vfIterface->GetVector(x, y - dxdydz.y(), z, t);
					auto v_zPlus = vfIterface->GetVector(x, y, z + dxdydz.z(), t);
					auto v_zMinus = vfIterface->GetVector(x, y, z - dxdydz.z(), t);
					// Compute derivatives
					auto dVz_dy = (v_zPlus.y() - v_zMinus.y()) / (2.0 * dxdydz.y());
					auto dVy_dz = (v_yPlus.z() - v_yMinus.z()) / (2.0 * dxdydz.z());
					auto dVx_dz = (v_xPlus.z() - v_xMinus.z()) / (2.0 * dxdydz.z());
					auto dVz_dx = (v_zPlus.x() - v_zMinus.x()) / (2.0 * dxdydz.x());
					auto dVy_dx = (v_yPlus.x() - v_yMinus.x()) / (2.0 * dxdydz.x());
					auto dVx_dy = (v_xPlus.y() - v_xMinus.y()) / (2.0 * dxdydz.y());

					Eigen::Matrix<float, 3, 1> curl;
					curl.x() = dVz_dy - dVy_dz;
					curl.y() = dVx_dz - dVz_dx;
					curl.z() = dVy_dx - dVx_dy;
					return static_cast<float>(curl.norm());
					};
			}
			else if  (operation == FLOWFIELD_OPERATION::MAGNITUDE) {
				analyticalFunc = [vfIterface, dxdydz](const float x, const float y, const float z, const float t) -> float {
					auto vector_3d = vfIterface->GetVector(x, y, z, t);
					return static_cast<float>(vector_3d.norm()); };
			}
			else if  (operation == FLOWFIELD_OPERATION::Killing)
			{
				/*if (auto* vf_v = dynamic_cast<Discrete3DFlowField<float>*>(vfIterface); vf_v) {

					analyticalFunc = [vf_v](const float x, const float y, const float z, const float t -> float {
					
						Eigen::Vector3f Pos3d=Eigen::Vector3f(x,y,z);
						auto nabla_v = vf_v->getVelocityGradientTensor(Pos3d, t);
						Eigen::Matrix3f killingMat = nabla_v + nabla_v.transpose();
						return killingMat.squaredNorm();
						};
				}*/

			}
			else if  (operation == FLOWFIELD_OPERATION::NORM_dVdt)
			{
				analyticalFunc = [vfIterface](const float x, const float y, const float z, const float t) -> float {
		
					Eigen::Vector3f Pos3d=Eigen::Vector3f(x,y,z);
					auto dv_dt = vfIterface->getPartialDerivativeT(Pos3d, t);
					return static_cast<float>(dv_dt.norm());
					};
			}
			else if  (operation == FLOWFIELD_OPERATION::Q_criterion)
			{
				analyticalFunc =
					[vfIterface, dxdydz](const float x, const float y, const float z, const float t) -> float {
					// Retrieve neighboring vectors for central differences
					auto v_xPlus = vfIterface->GetVector(x + dxdydz.x(), y, z, t);
					auto v_xMinus = vfIterface->GetVector(x - dxdydz.x(), y, z, t);
					auto v_yPlus = vfIterface->GetVector(x, y + dxdydz.y(), z, t);
					auto v_yMinus = vfIterface->GetVector(x, y - dxdydz.y(), z, t);
					auto v_zPlus = vfIterface->GetVector(x, y, z + dxdydz.z(), t);
					auto v_zMinus = vfIterface->GetVector(x, y, z - dxdydz.z(), t);

					// Compute velocity gradients (using central differences)
					float du_dx = (v_xPlus.x() - v_xMinus.x()) / (2.0f * dxdydz.x());
					float du_dy = (v_yPlus.x() - v_yMinus.x()) / (2.0f * dxdydz.y());
					float du_dz = (v_zPlus.x() - v_zMinus.x()) / (2.0f * dxdydz.z());
					float dv_dx = (v_xPlus.y() - v_xMinus.y()) / (2.0f * dxdydz.x());
					float dv_dy = (v_yPlus.y() - v_yMinus.y()) / (2.0f * dxdydz.y());
					float dv_dz = (v_zPlus.y() - v_zMinus.y()) / (2.0f * dxdydz.z());
					float dw_dx = (v_xPlus.z() - v_xMinus.z()) / (2.0f * dxdydz.x());
					float dw_dy = (v_yPlus.z() - v_yMinus.z()) / (2.0f * dxdydz.y());
					float dw_dz = (v_zPlus.z() - v_zMinus.z()) / (2.0f * dxdydz.z());

					// Compute the symmetric part S of the velocity gradient
					float Sxx = du_dx;
					float Syy = dv_dy;
					float Szz = dw_dz;
					float Sxy = 0.5f * (du_dy + dv_dx);
					float Sxz = 0.5f * (du_dz + dw_dx);
					float Syz = 0.5f * (dv_dz + dw_dy);

					// Compute the antisymmetric part Omega of the velocity gradient
					float Omega_xy = 0.5f * (du_dy - dv_dx);
					float Omega_xz = 0.5f * (du_dz - dw_dx);
					float Omega_yz = 0.5f * (dv_dz - dw_dy);

					// Compute squared Frobenius norms of S and Omega
					float S_norm2 = Sxx * Sxx + Syy * Syy + Szz * Szz +
						2.0f * (Sxy * Sxy + Sxz * Sxz + Syz * Syz);
					float Omega_norm2 = 2.0f * (Omega_xy * Omega_xy + Omega_xz * Omega_xz + Omega_yz * Omega_yz);

					// Q-criterion: Q = 0.5*(||Omega||^2 - ||S||^2)
					return 0.5f * (Omega_norm2 - S_norm2);
					};

			}
			else if  (operation == FLOWFIELD_OPERATION::lamba2_criterion)
			{
				analyticalFunc =
					[vfIterface, dxdydz](const float x, const float y, const float z, const float t) -> float {
					// Retrieve neighboring vectors for central differences
					auto v_xPlus = vfIterface->GetVector(x + dxdydz.x(), y, z, t);
					auto v_xMinus = vfIterface->GetVector(x - dxdydz.x(), y, z, t);
					auto v_yPlus = vfIterface->GetVector(x, y + dxdydz.y(), z, t);
					auto v_yMinus = vfIterface->GetVector(x, y - dxdydz.y(), z, t);
					auto v_zPlus = vfIterface->GetVector(x, y, z + dxdydz.z(), t);
					auto v_zMinus = vfIterface->GetVector(x, y, z - dxdydz.z(), t);

					// Compute velocity gradients (using central differences)
					float du_dx = (v_xPlus.x() - v_xMinus.x()) / (2.0f * dxdydz.x());
					float du_dy = (v_yPlus.x() - v_yMinus.x()) / (2.0f * dxdydz.y());
					float du_dz = (v_zPlus.x() - v_zMinus.x()) / (2.0f * dxdydz.z());
					float dv_dx = (v_xPlus.y() - v_xMinus.y()) / (2.0f * dxdydz.x());
					float dv_dy = (v_yPlus.y() - v_yMinus.y()) / (2.0f * dxdydz.y());
					float dv_dz = (v_zPlus.y() - v_zMinus.y()) / (2.0f * dxdydz.z());
					float dw_dx = (v_xPlus.z() - v_xMinus.z()) / (2.0f * dxdydz.x());
					float dw_dy = (v_yPlus.z() - v_yMinus.z()) / (2.0f * dxdydz.y());
					float dw_dz = (v_zPlus.z() - v_zMinus.z()) / (2.0f * dxdydz.z());
					// Compute the symmetric part S of the velocity gradient
					float Sxx = du_dx;
					float Syy = dv_dy;
					float Szz = dw_dz;
					float Sxy = 0.5f * (du_dy + dv_dx);
					float Sxz = 0.5f * (du_dz + dw_dx);
					float Syz = 0.5f * (dv_dz + dw_dy);

					// Compute the antisymmetric part Omega of the velocity gradient
					float Omega_xy = 0.5f * (du_dy - dv_dx);
					float Omega_xz = 0.5f * (du_dz - dw_dx);
					float Omega_yz = 0.5f * (dv_dz - dw_dy);
					float Axx = Sxx * Sxx + Sxy * Sxy + Sxz * Sxz + Omega_xy * Omega_xy + Omega_xz * Omega_xz;
					float Ayy = Syy * Syy + Sxy * Sxy + Syz * Syz + Omega_xy * Omega_xy + Omega_yz * Omega_yz;
					float Azz = Szz * Szz + Sxz * Sxz + Syz * Syz + Omega_xz * Omega_xz + Omega_yz * Omega_yz;
					float Axy = Sxx * Sxy + Sxy * Syy + Sxz * Syz + Omega_xy * Omega_xz;
					float Axz = Sxx * Sxz + Sxy * Syz + Sxz * Szz + Omega_xz * Omega_yz;
					float Ayz = Syy * Syz + Sxy * Sxz + Syz * Szz + Omega_xy * Omega_yz;

					Eigen::Matrix3f A;
					A << Axx, Axy, Axz,
						Axy, Ayy, Ayz,
						Axz, Ayz, Azz;

					Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(A);
					Eigen::Vector3f eigenvalues = solver.eigenvalues();

					return eigenvalues(1); 
					};

			}
			else if  (operation == FLOWFIELD_OPERATION::IVD)
			{
				std::string curl_field_name = AttributesNameGenerate(inputFieldName, FLOWFIELD_OPERATION::CURL);
				auto curl_field = VectorField3DManager::GetInstance().GetVectorFieldByName(curl_field_name);
				Discrete3DFlowField<float>* vectorFieldPtr = dynamic_cast<Discrete3DFlowField<float>*>(vfIterface);
				auto active_curl_field = dynamic_cast<Discrete3DFlowField<float>*>(curl_field);
				if (vectorFieldPtr && active_curl_field) {
					// Define the pointwise operation
						// compute average of abs of curl
					const auto avg_abs_curl_slices = active_curl_field->AverageVectorFieldBySlice();
					analyticalFunc = [vectorFieldPtr, avg_abs_curl_slices](const float x, const float y, const float z, const float t) -> float {
						Eigen::Vector3f Pos3d=Eigen::Vector3f(x,y,z);
						Eigen::Matrix<float, 3, 1> curl = vectorFieldPtr->getVorticity(Pos3d, t);						
						auto ivd = (curl - avg_abs_curl_slices[t]).norm();
						return static_cast<float>(ivd);
						};

				}
			}
			else if  (operation == FLOWFIELD_OPERATION::Angular_Velocity_NORM) {
				Discrete3DFlowField<float>* vectorFieldPtr = dynamic_cast<Discrete3DFlowField<float>*>(vfIterface);

				analyticalFunc = [vectorFieldPtr](const float x, const float y, const float z, const float t) {
					Eigen::Vector3f Pos3d=Eigen::Vector3f(x,y,z);
		
					auto nabla_v = vectorFieldPtr->getVelocityGradientTensor(Pos3d, t);
					DTG::EigenVector<3, float> angularVelocity = velocityGradientTensor2AngularVelocity(nabla_v);
					return angularVelocity.norm();
					};

			}

			if (analyticalFunc)
			{
				resultField->SetAnalyticalExpression(analyticalFunc);
				mScalarFields_.insert({ requestFieldName, std::move(resultField) });
				return getScalarField(requestFieldName);
			}
			else {
				LOG_E("unknown operation,No lambda function find.");
			}
		}
		return nullptr;
	}

}



DTG::DiscreteScalarField3D<float>* DTG::AttributesManager::requestDiscreteScalarField(const std::string& inputFieldName,const  FLOWFIELD_OPERATION operation)
{
	const auto requestFieldName = AttributesNameGenerate(inputFieldName, operation);
	if (const auto resultScalarFieldCached = getScalarField(requestFieldName);resultScalarFieldCached &&resultScalarFieldCached->HasDiscreteData())
	{
		return resultScalarFieldCached;
	}
	else if(resultScalarFieldCached &&resultScalarFieldCached->HasAnalyticalExpression()){//if no discrete data but only analytical expression
		resultScalarFieldCached->resampleAnalytical2Gird();
		return resultScalarFieldCached;
	}
	else
	{
		//compute new scalar field
		if (auto* vfIterface = VectorField3DManager::GetInstance().GetVectorFieldByName(inputFieldName); vfIterface)
		{
			auto dxdydz = vfIterface->GetSpatialDxDyDz();
			auto resultField = std::make_unique<DiscreteScalarField3D<float>>(vfIterface->GetFieldInfo());

			scalarFunctionByGrid mPointwiseFunc = nullptr;
			if (operation == FLOWFIELD_OPERATION::CURL_NORM)
			{
				mPointwiseFunc = [vfIterface, dxdydz](const int x, const int y, const int z, const int t) -> float {
					// Get neighboring vectors
					auto v_xPlus = vfIterface->GetVectorAtGridDynamicCast(x + 1, y, z, t);
					auto v_xMinus = vfIterface->GetVectorAtGridDynamicCast(x - 1, y, z, t);
					auto v_yPlus = vfIterface->GetVectorAtGridDynamicCast(x, y + 1, z, t);
					auto v_yMinus = vfIterface->GetVectorAtGridDynamicCast(x, y - 1, z, t);
					auto v_zPlus = vfIterface->GetVectorAtGridDynamicCast(x, y, z + 1, t);
					auto v_zMinus = vfIterface->GetVectorAtGridDynamicCast(x, y, z - 1, t);
					// Compute derivatives
					auto dVz_dy = (v_zPlus.y() - v_zMinus.y()) / (2.0 * dxdydz.y());
					auto dVy_dz = (v_yPlus.z() - v_yMinus.z()) / (2.0 * dxdydz.z());
					auto dVx_dz = (v_xPlus.z() - v_xMinus.z()) / (2.0 * dxdydz.z());
					auto dVz_dx = (v_zPlus.x() - v_zMinus.x()) / (2.0 * dxdydz.x());
					auto dVy_dx = (v_yPlus.x() - v_yMinus.x()) / (2.0 * dxdydz.x());
					auto dVx_dy = (v_xPlus.y() - v_xMinus.y()) / (2.0 * dxdydz.y());

					Eigen::Matrix<float, 3, 1> curl;
					curl.x() = dVz_dy - dVy_dz;
					curl.y() = dVx_dz - dVz_dx;
					curl.z() = dVy_dx - dVx_dy;
					return static_cast<float>(curl.norm());
					};
			}
			else if  (operation == FLOWFIELD_OPERATION::MAGNITUDE) {
				mPointwiseFunc = [vfIterface](const int x, const int y, const int z, const int t) -> float {
					auto vector_3d = vfIterface->GetVectorAtGridDynamicCast(x, y, z, t);
					return static_cast<float>(vector_3d.norm()); };
			}
			else if  (operation == FLOWFIELD_OPERATION::Killing)
			{
				if (auto* vf_v = dynamic_cast<Discrete3DFlowField<float>*>(vfIterface); vf_v) {
					mPointwiseFunc = [vf_v](const int x, const int y, const int z, const int t) -> float {
						if (vf_v->AtSpatialBoundary(x, y, z)) [[unlikely]]
							return 0.0;
						else {
							auto nabla_v = vf_v->getVelocityGradientTensor(x, y, z, t);
							Eigen::Matrix3f killingMat = nabla_v + nabla_v.transpose();
							return killingMat.squaredNorm();
						}
						};
				}

			}
			else if  (operation == FLOWFIELD_OPERATION::NORM_dVdt)
			{
				mPointwiseFunc = [vfIterface](const int x, const int y, const int z, const int t) -> float {
					// Get neighboring vectors
					auto dv_dt = vfIterface->getPartialDerivativeT<float>(x, y, z, t);
					return static_cast<float>(dv_dt.norm());
					};
			}
			else if  (operation == FLOWFIELD_OPERATION::Q_criterion)
			{
				mPointwiseFunc =
					[vfIterface, dxdydz](const int x, const int y, const int z, const int t) -> float {
					// Retrieve neighboring vectors for central differences
					auto v_xPlus = vfIterface->GetVectorAtGridDynamicCast(x + 1, y, z, t);
					auto v_xMinus = vfIterface->GetVectorAtGridDynamicCast(x - 1, y, z, t);
					auto v_yPlus = vfIterface->GetVectorAtGridDynamicCast(x, y + 1, z, t);
					auto v_yMinus = vfIterface->GetVectorAtGridDynamicCast(x, y - 1, z, t);
					auto v_zPlus = vfIterface->GetVectorAtGridDynamicCast(x, y, z + 1, t);
					auto v_zMinus = vfIterface->GetVectorAtGridDynamicCast(x, y, z - 1, t);

					// Compute velocity gradients (using central differences)
					float du_dx = (v_xPlus.x() - v_xMinus.x()) / (2.0f * dxdydz.x());
					float du_dy = (v_yPlus.x() - v_yMinus.x()) / (2.0f * dxdydz.y());
					float du_dz = (v_zPlus.x() - v_zMinus.x()) / (2.0f * dxdydz.z());
					float dv_dx = (v_xPlus.y() - v_xMinus.y()) / (2.0f * dxdydz.x());
					float dv_dy = (v_yPlus.y() - v_yMinus.y()) / (2.0f * dxdydz.y());
					float dv_dz = (v_zPlus.y() - v_zMinus.y()) / (2.0f * dxdydz.z());
					float dw_dx = (v_xPlus.z() - v_xMinus.z()) / (2.0f * dxdydz.x());
					float dw_dy = (v_yPlus.z() - v_yMinus.z()) / (2.0f * dxdydz.y());
					float dw_dz = (v_zPlus.z() - v_zMinus.z()) / (2.0f * dxdydz.z());

					// Compute the symmetric part S of the velocity gradient
					float Sxx = du_dx;
					float Syy = dv_dy;
					float Szz = dw_dz;
					float Sxy = 0.5f * (du_dy + dv_dx);
					float Sxz = 0.5f * (du_dz + dw_dx);
					float Syz = 0.5f * (dv_dz + dw_dy);

					// Compute the antisymmetric part Omega of the velocity gradient
					float Omega_xy = 0.5f * (du_dy - dv_dx);
					float Omega_xz = 0.5f * (du_dz - dw_dx);
					float Omega_yz = 0.5f * (dv_dz - dw_dy);

					// Compute squared Frobenius norms of S and Omega
					float S_norm2 = Sxx * Sxx + Syy * Syy + Szz * Szz +
						2.0f * (Sxy * Sxy + Sxz * Sxz + Syz * Syz);
					float Omega_norm2 = 2.0f * (Omega_xy * Omega_xy + Omega_xz * Omega_xz + Omega_yz * Omega_yz);

					// Q-criterion: Q = 0.5*(||Omega||^2 - ||S||^2)
					return 0.5f * (Omega_norm2 - S_norm2);
					};

			}
			else if  (operation == FLOWFIELD_OPERATION::lamba2_criterion)
			{
				mPointwiseFunc =
					[vfIterface, dxdydz](const int x, const int y, const int z, const int t) -> float {
					// Retrieve neighboring vectors for central differences
					auto v_xPlus = vfIterface->GetVectorAtGridDynamicCast(x + 1, y, z, t);
					auto v_xMinus = vfIterface->GetVectorAtGridDynamicCast(x - 1, y, z, t);
					auto v_yPlus = vfIterface->GetVectorAtGridDynamicCast(x, y + 1, z, t);
					auto v_yMinus = vfIterface->GetVectorAtGridDynamicCast(x, y - 1, z, t);
					auto v_zPlus = vfIterface->GetVectorAtGridDynamicCast(x, y, z + 1, t);
					auto v_zMinus = vfIterface->GetVectorAtGridDynamicCast(x, y, z - 1, t);

					// Compute velocity gradients (using central differences)
					float du_dx = (v_xPlus.x() - v_xMinus.x()) / (2.0f * dxdydz.x());
					float du_dy = (v_yPlus.x() - v_yMinus.x()) / (2.0f * dxdydz.y());
					float du_dz = (v_zPlus.x() - v_zMinus.x()) / (2.0f * dxdydz.z());
					float dv_dx = (v_xPlus.y() - v_xMinus.y()) / (2.0f * dxdydz.x());
					float dv_dy = (v_yPlus.y() - v_yMinus.y()) / (2.0f * dxdydz.y());
					float dv_dz = (v_zPlus.y() - v_zMinus.y()) / (2.0f * dxdydz.z());
					float dw_dx = (v_xPlus.z() - v_xMinus.z()) / (2.0f * dxdydz.x());
					float dw_dy = (v_yPlus.z() - v_yMinus.z()) / (2.0f * dxdydz.y());
					float dw_dz = (v_zPlus.z() - v_zMinus.z()) / (2.0f * dxdydz.z());

					// Compute the symmetric part S of the velocity gradient
					float Sxx = du_dx;
					float Syy = dv_dy;
					float Szz = dw_dz;
					float Sxy = 0.5f * (du_dy + dv_dx);
					float Sxz = 0.5f * (du_dz + dw_dx);
					float Syz = 0.5f * (dv_dz + dw_dy);

					// Compute the antisymmetric part Omega of the velocity gradient
					float Omega_xy = 0.5f * (du_dy - dv_dx);
					float Omega_xz = 0.5f * (du_dz - dw_dx);
					float Omega_yz = 0.5f * (dv_dz - dw_dy);

					// Compute squared Frobenius norms of S and Omega
					float S_norm2 = Sxx * Sxx + Syy * Syy + Szz * Szz +
						2.0f * (Sxy * Sxy + Sxz * Sxz + Syz * Syz);
					float Omega_norm2 = 2.0f * (Omega_xy * Omega_xy + Omega_xz * Omega_xz + Omega_yz * Omega_yz);

					// Q-criterion: Q = 0.5*(||Omega||^2 - ||S||^2)
					return 0.5f * (Omega_norm2 - S_norm2);
					};

			}
			else if  (operation == FLOWFIELD_OPERATION::IVD)
			{

				std::string curl_field_name = AttributesNameGenerate(inputFieldName, FLOWFIELD_OPERATION::CURL);
				auto curl_field = VectorField3DManager::GetInstance().GetVectorFieldByName(curl_field_name);
				assert(curl_field );

				Discrete3DFlowField<float>* vectorFieldPtr = dynamic_cast<Discrete3DFlowField<float>*>(vfIterface);
				auto active_curl_field = dynamic_cast<Discrete3DFlowField<float>*>(curl_field);
				if (vectorFieldPtr && active_curl_field) {
					// Define the pointwise operation
						// compute average of abs of curl
					const auto avg_abs_curl_slices = active_curl_field->AverageVectorFieldBySlice();
					mPointwiseFunc = [vectorFieldPtr, avg_abs_curl_slices](const int x, const int y, const int z, const int t) -> float {
						// Get neighboring vectors
						Eigen::Matrix<float, 3, 1> curl = vectorFieldPtr->getVorticity(x, y, z, t);
						auto ivd = (curl - avg_abs_curl_slices[t]).norm();
						return static_cast<float>(ivd);
						};

				}
			}
			else if  (operation == FLOWFIELD_OPERATION::Angular_Velocity_NORM) {
				Discrete3DFlowField<float>* vectorFieldPtr = dynamic_cast<Discrete3DFlowField<float>*>(vfIterface);
				mPointwiseFunc = [vectorFieldPtr](const int x, const int y, const int z, const int t) {
					// Get neighboring vectors
					auto jacobianMat = vectorFieldPtr->getVelocityGradientTensor(x , y, z, t);
					DTG::EigenVector<3, float> angularVelocity = velocityGradientTensor2AngularVelocity(jacobianMat);
					return angularVelocity.norm();
					};

			}
	
			if (mPointwiseFunc)
			{
				// Define the pointwise operation
				VectorField3DPointwiseOperator pointwiseOperator;
				pointwiseOperator(mPointwiseFunc, *resultField);
				mScalarFields_.insert({ requestFieldName, std::move(resultField) });
				return getScalarField(requestFieldName);
			}
			else {
				LOG_E("unknown operation,No lambda function find.");
			}
		}
		return nullptr;
	}
}



DTG::DiscreteScalarField3D<float>*  DTG::AttributesManager::requestDiscreteScalarFieldFromScalarField(const std::string& inputScalarFieldName, const FLOWFIELD_OPERATION operation,const  int Pram1)
{
		const auto requestFieldName = AttributesNameGenerate(inputScalarFieldName, operation,Pram1);
		const auto resultScalarFieldCached = getScalarField(requestFieldName); 
		if (resultScalarFieldCached && resultScalarFieldCached->HasDiscreteData())
		{
			return resultScalarFieldCached;
		}
		else if (resultScalarFieldCached &&resultScalarFieldCached->HasAnalyticalExpression())
		{
				resultScalarFieldCached->resampleAnalytical2Gird();
				return resultScalarFieldCached;
		}
		else
		{
			auto inputScalarField = getScalarField(inputScalarFieldName); 
			assert(inputScalarField);
			if (!inputScalarField ->HasDiscreteData())
			{
				inputScalarField ->resampleAnalytical2Gird();
			}
			
			auto dxdydz = inputScalarField->GetSpatialDxDyDz();
			const Eigen::Vector3i res = inputScalarField->GetSpatialGridSize();
			auto resultField = std::make_unique<DiscreteScalarField3D<float>>(inputScalarField->GetFieldInfo());
			scalarFunctionByGrid mPointwiseFunc = nullptr;
			if  (operation == FLOWFIELD_OPERATION::LOCAL_AVERAGE)
			{
				mPointwiseFunc = [ inputScalarField,dxdydz,res,Pram1](const int idx, const int idy, const int idz, const int idt) -> float {
			
					const auto& NeighborhoodU = Pram1;
					int x1 = std::min(std::max(0, idx - 2*NeighborhoodU), res.x() - 1);
					int y1 = std::min(std::max(0, idy - NeighborhoodU), res.y() - 1);
					int z1 = std::min(std::max(0, idz - NeighborhoodU), res.z() - 1);
					int x2 = std::min(std::max(0, idx + 2*NeighborhoodU), res.x() - 1);
					int y2 = std::min(std::max(0, idy + NeighborhoodU), res.y() - 1);
					int z2 = std::min(std::max(0, idz + NeighborhoodU), res.z() - 1);

					float sum_magnitude= 0.0;
					float nCount = 0.0;
					for (int wz = z1; wz <= z2; ++wz)
						for (int wy = y1; wy <= y2; ++wy)
							for (int wx = x1; wx <= x2; ++wx) {
								auto vec = inputScalarField->GetValueAtGrid( wx,wy,wz,idt);
								sum_magnitude += vec;
								nCount += 1.0;
							}
					sum_magnitude /= nCount;
					return sum_magnitude;
					};

			}

			if (mPointwiseFunc)
			{
				// Define the pointwise operation
				VectorField3DPointwiseOperator pointwiseOperator;
				pointwiseOperator(mPointwiseFunc, *resultField);
				mScalarFields_.insert({ requestFieldName, std::move(resultField) });

				return getScalarField(requestFieldName);
			}
			else {
				LOG_E("unknown operation,No lambda function find.");
			}
			return nullptr;
		}
}





template class DTG:: DiscreteScalarField3D<float>; 




DTG::DiscreteScalarField3D<float>* AttributesManager::LagriangianScalarField3DTimeSeries(const std::string& inputVectorFieldname, const std::string& inputFieldName,  int nx, int ny,int nz ,int numberOfTimeSteps, float IntegrateTimeInterval, float stepSize)
{
	auto inputVectorField = dynamic_cast<DTG::Discrete3DFlowField<float>*>(VectorField3DManager::GetInstance().GetVectorFieldByName(inputVectorFieldname));
	if (inputVectorField)
	{
		const auto tmin = inputVectorField->GetMinTime();
		const auto tmax = inputVectorField->GetMaxTime();
		auto outputFiledInfo = inputVectorField->GetFieldInfo();

		const auto dt = (tmax - tmin) / (numberOfTimeSteps - 1);
		outputFiledInfo.numberOfTimeSteps = numberOfTimeSteps;
		outputFiledInfo.XGridsize = nx;
		outputFiledInfo.YGridsize = ny;
		outputFiledInfo.ZGridsize = nz;

		auto outputScalarField = std::make_unique<DiscreteScalarField3D<float>>(outputFiledInfo);

		for (int idt = 0; idt < numberOfTimeSteps; idt++)
		{
			float time = idt * dt + tmin;
			std::unique_ptr<DiscreteScalarField3D<float>>sliceScalarField = LagriangianScalarField3D(inputVectorFieldname, inputFieldName, nx,ny,nz,time, IntegrateTimeInterval, stepSize);
			if (sliceScalarField&&sliceScalarField->HasDiscreteData())
			{
				outputScalarField->setSliceData(idt, sliceScalarField->getSliceData(0));
			}
		}
		std::string requestFieldName = AttributesNameGenerate(inputFieldName, FLOWFIELD_OPERATION::LAGRANGIAN_, numberOfTimeSteps, inputVectorFieldname);
		mScalarFields_.insert({ requestFieldName, std::move(outputScalarField) });
		return getScalarField(requestFieldName);
	}
	return nullptr;
}

std::unique_ptr<DiscreteScalarField3D<float>> AttributesManager::LagriangianScalarField3D(const std::string& inputVectorFieldname, const std::string& inputFieldName, int nx, int ny, int nz, float time, float IntegrateTimeInterval, float stepSize)
{
	auto scalarFieldIn = getScalarField(inputFieldName);
	if (scalarFieldIn)
	{
		auto inputVectorField = dynamic_cast<DTG::Discrete3DFlowField<float>*>(VectorField3DManager::GetInstance().GetVectorFieldByName(inputVectorFieldname));
		if (inputVectorField)
		{
			auto seedingPoints = generateSeedPointsOnGrid4oneTime(inputVectorField->GetFieldInfo(), nx, ny, nz, time);


			const float IntegrateTimeIntervalOneDirection = IntegrateTimeInterval / (2.0);
			auto resultPathlines = inputVectorField->computeParallelPathlinesDoubleDirection(seedingPoints, stepSize, IntegrateTimeIntervalOneDirection, PathlineNumericalIntegrationMethod::RK4, false);

			std::vector<std::vector<float>> interpolatedValueOnPathlines;
			interpolateScalarFieldOnGridAlongPathlines(scalarFieldIn, resultPathlines, interpolatedValueOnPathlines);

			auto outputFiledInfo = scalarFieldIn->GetFieldInfo();
			outputFiledInfo.XGridsize = nx;
			outputFiledInfo.YGridsize = ny;
			outputFiledInfo.ZGridsize = nz;
			outputFiledInfo.numberOfTimeSteps = 1;

			auto outputScalarField = std::make_unique<DiscreteScalarField3D<float>>(outputFiledInfo);
			reduceLagriangianToScalarFieldOnGrid(resultPathlines, stepSize, interpolatedValueOnPathlines, outputScalarField.get(), nx, ny, nz);
			return  std::move(outputScalarField);
		}
		else {
			LOG_E("LagriangianScalarField3D with inputVectorFieldname=%s doesn't exist.", inputVectorFieldname.c_str());
			return nullptr;
		}
	}
	else {
		LOG_E("LagriangianScalarField3D with inputFieldName =%s doesn't exist.", inputFieldName.c_str());
		return nullptr;
	}
}

void AttributesManager::interpolateScalarFieldOnGridAlongPathlines(const DiscreteScalarField3D<float>* scalarFieldInput, const std::vector<pathline3Df>& pathlines, std::vector<std::vector<float>>& interpolatedValueOnPathlines)
{
	// interpolate scalar values along pathline points
	// input: pathlines, pathlineStartOffset, scalar field
	// output: interpolatedValueOnPathlines
	const		int numberOfPathlineSeeds = pathlines.size();
	interpolatedValueOnPathlines.resize(numberOfPathlineSeeds, std::vector<float>());
	std::vector<int>threadRange;
	threadRange.resize(numberOfPathlineSeeds);
	std::generate(threadRange.begin(), threadRange.end(), [n = 0]() mutable { return n++; });
	auto policy = std::execution::par_unseq;

	for_each(policy, threadRange.begin(), threadRange.end(), [this, &interpolatedValueOnPathlines, &scalarFieldInput, &pathlines](int pathlineIndex) {

		auto& interpolatedValueOnPathline = interpolatedValueOnPathlines[pathlineIndex];
		const int stepCounts = pathlines[pathlineIndex].size();
		interpolatedValueOnPathline.resize(stepCounts);
		for (int integrationResultIndex = 0; integrationResultIndex < stepCounts; integrationResultIndex++) {
			const PointAndTime3Df& p = pathlines[pathlineIndex][integrationResultIndex];
			interpolatedValueOnPathline[integrationResultIndex] = scalarFieldInput->GetValue(p.position, p.time);
		}
		});
}

void AttributesManager::reduceLagriangianToScalarFieldOnGrid(const std::vector<pathline3Df>& pathlines, const float dt_intergration, const std::vector<std::vector<float>>& interpolatedValueOnPathlines, DiscreteScalarField3D<float>* lagriangianedValueOnPathlines, int nx, int ny, int nz)
{
	//std::cout << "reduceToScalarFieldOnGrid" << std::endl;
	// reduction of interpolated values (along pathline) 
	// input: interpolatedValueOnPathlines
	// output: scalarFieldOutput

	const		int numberOfPathlineSeeds = pathlines.size();
	std::vector<int>threadRange;
	threadRange.resize(numberOfPathlineSeeds);
	std::generate(threadRange.begin(), threadRange.end(), [n = 0]() mutable { return n++; });

	auto policy = std::execution::par_unseq;

	for_each(policy, threadRange.begin(), threadRange.end(), [&pathlines, &interpolatedValueOnPathlines, &dt_intergration, nx, ny, nz, &lagriangianedValueOnPathlines](int pathlineIndex) {

		double lagrangian = 0;
		for (int integrationResultIndex = 0; integrationResultIndex < pathlines[pathlineIndex].size(); integrationResultIndex++) {
			double val = interpolatedValueOnPathlines[pathlineIndex][integrationResultIndex];
			lagrangian += val * dt_intergration * (1.0 / double(pathlines[pathlineIndex].size()));
		}

		const auto linearIndex = pathlineIndex;
		const int nxny = nx * ny;
		const int iz = linearIndex / nxny;
		const int remainder = linearIndex % nxny;
		const int iy = remainder / nx;
		const int ix = remainder % nx;
		lagriangianedValueOnPathlines->SetValue(ix, iy, iz, 0, lagrangian);
		});
}

void AttributesManager::renamingScalarField(const std::string& originalName, const std::string& newName)
{
	// Early exit if names are identical
	if (originalName == newName) {
		return;
	}

	// Check if original scalar field exists
	if (!mScalarFields_.contains(originalName)) {
		LOG_E("Original scalar field '%s' not found.", originalName.c_str());
		return;
	}

	// Check if new name already exists in either map
	if (mScalarFields_.contains(newName) || mScalarFieldsMinMax.contains(newName)) {
		LOG_E("New name '%s' already exists in scalar fields or min/max map.", newName.c_str());
		return;
	}
	// Rename the scalar field entry
	auto scalar_node = mScalarFields_.extract(originalName);
	scalar_node.key() = newName;
	mScalarFields_.insert(std::move(scalar_node));

	// Rename the min/max entry if it exists
	if (auto minmax_node = mScalarFieldsMinMax.extract(originalName)) {
		minmax_node.key() = newName;
		mScalarFieldsMinMax.insert(std::move(minmax_node));
	}
	LOG_I("Renamed scalar field '%s' to '%s'.", originalName.c_str(), newName.c_str());
}

}//namespace DTG