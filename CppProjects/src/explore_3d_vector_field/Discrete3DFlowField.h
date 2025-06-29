
#ifndef _DISCRETE3D_FLOWFIELD_H
#define _DISCRETE3D_FLOWFIELD_H
#pragma once
#include "PortToUsingModules.h"
#include "myTypeDefs.h"
#include "IField.h"
#include "ScalarField3D.h"
#include "vtkSmartPointer.h"
#include "vtkImageData.h"

namespace DTG {
	
	struct  KillingField3DCoefficients
	{
		Eigen::Vector3f rotationABC;
		Eigen::Vector3f tranlationDEF;
		std::string to_string()const {
			std::ostringstream oss;
			oss << "Translation_" << tranlationDEF.x() << "_" << tranlationDEF.y() << "_" << tranlationDEF.z() << "";
			oss << "_Rotation_" << rotationABC.x() << "_" << rotationABC.y() << "_" << rotationABC.z() << "";
			return oss.str();
		}
	};

	template <typename K>
	class Discrete3DFlowField;

	class IDiscreteField3d :public IField3D {
	public:

		IDiscreteField3d(const FieldSpaceTimeInfo3D& FiledInformation);
		virtual ~IDiscreteField3d() = default;


		std::vector<pathline3Df> computeParallelPathlinesDoubleDirection( const std::vector<DTG::PointAndTime3Df>&startPositions,float  stepSize,int maxIter, PathlineNumericalIntegrationMethod method, bool boundaryRElaxtion=false);
		std::vector<DTG::pathline3Df> computeParallelPathlinesDoubleDirection(const std::vector<DTG::PointAndTime3Df>& startPositions, float stepSize, float restrictIntegrationTimeIntervalOneDirection, PathlineNumericalIntegrationMethod method, bool boundaryRElaxtion/*=false*/);

		

		template <typename ScalarType = float>
		void Integrate3DPathlineDoubleDirection(const DTG::Point<3, ScalarType> & startPointInChart,const  ScalarType startTime, const  ScalarType stepSize,const  int maxIterationCount, std::vector<DTG::PointAndTime<3, ScalarType>>& results, PathlineNumericalIntegrationMethod method, bool bounaryRelaxtion = false)
		{

			std::vector<DTG::PointAndTime<3, ScalarType>> resultPointsInChartsForward = std::vector<DTG::PointAndTime<3, ScalarType>>();
			std::vector<DTG::PointAndTime<3, ScalarType>> resultPointsInChartsBackward = std::vector<DTG::PointAndTime<3, ScalarType>>();

			const DTG::Point<3, ScalarType> startPoint(startPointInChart);
			resultPointsInChartsForward.push_back(DTG::PointAndTime<3, ScalarType>(startPoint, startTime));
			if ((startTime < this->mFieldInfo.maxTime - stepSize)) {
				this->Integrate3DPathlineOneDirection(startPoint, startTime, this->mFieldInfo.maxTime, stepSize, maxIterationCount, resultPointsInChartsForward, method, bounaryRelaxtion);
			}

			if (startTime > this->mFieldInfo.minTime + stepSize) {
				this->Integrate3DPathlineOneDirection(startPoint, startTime, this->mFieldInfo.minTime, stepSize, maxIterationCount, resultPointsInChartsBackward, method, bounaryRelaxtion);
			}

			std::reverse(resultPointsInChartsBackward.begin(), resultPointsInChartsBackward.end());
			resultPointsInChartsBackward.insert(resultPointsInChartsBackward.end(), resultPointsInChartsForward.begin(), resultPointsInChartsForward.end());
			results = std::move(resultPointsInChartsBackward);
		}

		template <typename ScalarType = float>
		void Integrate3DPathlineDoubleDirection(const DTG::EigenVector<3, ScalarType>& startPointInChart, const ScalarType startTime, const  ScalarType stepSize, const  int maxIterationCount, std::vector<DTG::PointAndTime<3, ScalarType>>& results, PathlineNumericalIntegrationMethod method, bool bounaryRelaxtion = false)
		{
			 DTG::Point<3, ScalarType> StartPos={startPointInChart[0],startPointInChart[1],startPointInChart[2]};
			Integrate3DPathlineDoubleDirection( StartPos,startTime ,stepSize , maxIterationCount,results,method,bounaryRelaxtion);
		}


		virtual void Integrate3DPathlineOneDirection(const DTG::Point<3, double>& startPointInChart, double startTime, double targetIntegrationTime, double stepSize, int maxIterationCount, std::vector<DTG::PointAndTime<3, double>>& results, const PathlineNumericalIntegrationMethod method,bool bondaryRelaxtion=false) = 0;

		virtual void Integrate3DPathlineOneDirection(const DTG::Point<3, float>& startPointInChart, float startTime, float targetIntegrationTime, float stepSize, int maxIterationCount, std::vector<DTG::PointAndTime<3, float>>& results, const PathlineNumericalIntegrationMethod method,bool bondaryRelaxtion=false) = 0;

		virtual void Integrate3DStreamlineOneDirection(const DTG::Point<3, double>& startPointInChart,
			const double startTime, const double stepSize, const int maxIterationCount, std::vector<DTG::PointAndTime<3, double>>& results, const PathlineNumericalIntegrationMethod method, bool forward=true)
			= 0;

		virtual void Integrate3DStreamlineOneDirection(const DTG::Point<3, float>& startPointInChart,
			const float startTime, const float stepSize, const int maxIterationCount, std::vector<DTG::PointAndTime<3, float>>& results, const PathlineNumericalIntegrationMethod method, bool forward=true)
			= 0;



		template <typename T = float>
		DTG::EigenVector<3, T> GetVectorAtGridDynamicCast(const int x, const int y, const int z, const int timeStep) const		
		{
			if (!HasAnalyticalExpression())
			{
			const auto* derivedFloat = dynamic_cast<const Discrete3DFlowField<float>*>(this);
			const auto* derivedDouble = dynamic_cast<const Discrete3DFlowField<double>*>(this);
			if constexpr (std::is_same_v<T, float>) {
				if (derivedFloat) {
					return derivedFloat->GetVectorAtGrid(x, y, z, timeStep);
				}
				else if (derivedDouble) {
					return derivedDouble->GetVectorAtGrid(x, y, z, timeStep).template cast<float>();
				}
			}
			else if constexpr (std::is_same_v<T, double>) {
				if (derivedDouble) {
					return derivedDouble->GetVectorAtGrid(x, y, z, timeStep);
				}
				else if (derivedFloat) {
					return derivedFloat->GetVectorAtGrid(x, y, z, timeStep).template cast<double>();
				}
			}

			throw std::runtime_error("Unsupported type or dynamic cast failed");
			}
			else
			{
			  const Eigen::Vector3d pos_t=	convertGridIndex2PhysicalCoordinates(x,y,z);
			  double time=convertTimeStep2PhysicalTime(timeStep);
			  auto res=mAnyticalFunc(pos_t.x(),pos_t.y(),pos_t.z(),time);

			  return res.cast<T>();
			}
		}
		template <typename T = float>
		DTG::EigenVector<3, T> GetVector(const DTG::EigenVector<3, double>& xyz, const double time) const
		{
			return this->GetVector<T>(xyz.x(), xyz.y(), xyz.z(), time);
		}
		template <typename T= float>
		DTG::EigenVector<3, T> GetVector(const DTG::EigenVector<3, float>& xyz, const float time) const
		{
			return this->GetVector<T>(xyz.x(), xyz.y(), xyz.z(), time);
		}
		template <typename T= float>
		DTG::EigenVector<3, T> GetVector(const float x, const float y, const float z, const float time) const
		{
			if (HasAnalyticalExpression())
			{
				return mAnyticalFunc((float)x,(float)y,(float)z,(float)time).template cast<T>();
			} 
			else
			{
			const auto* derivedFloat = dynamic_cast<const Discrete3DFlowField<float>*>(this);
			const auto* derivedDouble = dynamic_cast<const Discrete3DFlowField<double>*>(this);
			if constexpr (std::is_same_v<T, float>) {
				if (derivedFloat) {
					return derivedFloat->GetSpaceAndTimeInterpolatedVector(x, y, z, time);
				}
				else if (derivedDouble) [[unlikely]]{
					return derivedDouble->GetSpaceAndTimeInterpolatedVector(x, y, z, time).template cast<float>();
				}
			}
			else if constexpr (std::is_same_v<T, double>) {
				if (derivedDouble)[[unlikely]] {
					return derivedDouble->GetSpaceAndTimeInterpolatedVector(x, y, z, time);
				}
				else if (derivedFloat) {
					return derivedFloat->GetSpaceAndTimeInterpolatedVector(x, y, z, time).template cast<double>();
				}
			}
			return {};
			}
		}

		template <typename T = float, typename K>
		Eigen::Matrix<T, 3, 3> getVelocityGradientTensor(const Eigen::Matrix<K, 3, 1>& point, K time)
		{
			const auto* derivedFloat = dynamic_cast<const Discrete3DFlowField<float>*>(this);
			const auto* derivedDouble = dynamic_cast<const Discrete3DFlowField<double>*>(this);
			if constexpr (std::is_same_v<T, float>) {
				if (derivedFloat) {
					DTG::Point<3, float> cast_point = { (float)point.x(), (float)point.y(), (float)point.z() };
					return derivedFloat->getVelocityGradientTensor(cast_point, time);
				}
				else if (derivedDouble) {
					DTG::Point<3, double> cast_point = { (double)point.x(), (double)point.y(), (double)point.z() };
					return derivedDouble->getVelocityGradientTensor(cast_point, time).template cast<float>();
				}
			}
			else if constexpr (std::is_same_v<T, double>) {
				if (derivedDouble) {
					DTG::Point<3, double> cast_point = { (double)point.x(), (double)point.y(), (double)point.z() };
					return derivedDouble->getVelocityGradientTensor(cast_point, time);
				}
				else if (derivedFloat) {
					DTG::Point<3, float> cast_point = { (float)point.x(), (float)point.y(), (float)point.z() };
					return derivedFloat->getVelocityGradientTensor(cast_point, time).template cast<double>();
				}
			}

			throw std::runtime_error("Unsupported type or dynamic cast failed");
		}

		template <typename ReturnScalarType>
		inline Eigen::Matrix<ReturnScalarType, 3, 3> getVelocityGradientTensor(const int idx, const int idy, const int idz, const int idt)
		{
			const auto* derivedFloat = dynamic_cast<const Discrete3DFlowField<float>*>(this);
			const auto* derivedDouble = dynamic_cast<const Discrete3DFlowField<double>*>(this);
			if (derivedFloat) {
				return derivedFloat->getVelocityGradientTensor(idx, idy, idz, idt).template cast<ReturnScalarType>();
			}
			else if (derivedDouble) {
				return derivedDouble->getVelocityGradientTensor(idx, idy, idz, idt).template cast<ReturnScalarType>();
			}return {};
		}

		template <typename ReturnScalarType>
		inline DTG::EigenVector<3, ReturnScalarType> getPartialDerivativeT(const int idx, const int idy, const int idz, const int idt)
		{
			const auto* derivedFloat = dynamic_cast<const Discrete3DFlowField<float>*>(this);
			const auto* derivedDouble = dynamic_cast<const Discrete3DFlowField<double>*>(this);
			if (derivedFloat) {
				return derivedFloat->getPartialDerivativeT(idx, idy, idz, idt).template cast<ReturnScalarType>();
			}
			else  if (derivedDouble) {
				return derivedDouble->getPartialDerivativeT(idx, idy, idz, idt).template cast<ReturnScalarType>();
			}return {};
		}

		template <typename ReturnScalarType, typename K>
		inline DTG::EigenVector<3, ReturnScalarType> getPartialDerivativeT(const Eigen::Matrix<K, 3, 1>& point, K time)
		{
			const auto* derivedFloat = dynamic_cast<const Discrete3DFlowField<float>*>(this);
			const auto* derivedDouble = dynamic_cast<const Discrete3DFlowField<double>*>(this);
			if (derivedFloat) {
				Eigen::Matrix<float, 3, 1> cast_point = { (float)point.x(), (float)point.y(), (float)point.z() };
				return derivedFloat->getPartialDerivativeT(cast_point, (float)time).template cast<ReturnScalarType>();
			}
			else  if (derivedDouble) {
				Eigen::Matrix<double, 3, 1> cast_point = { (double)point.x(), (double)point.y(), (double)point.z() };
				return derivedDouble->getPartialDerivativeT(cast_point, (double)time).template cast<ReturnScalarType>();
			}return {};
		}

		virtual std::unique_ptr<IDiscreteField3d> ReSampleToNewResolution(int xdim, int ydim, int zdim, int tdim) = 0;


		inline bool HasAnalyticalExpression() const
		{
			return mAnyticalFunc != nullptr;
		}
	
		void SetAnalyticalExpression(const Flow3DFunc& inAnyticalFunc){
				mAnyticalFunc= inAnyticalFunc;
		}


		 Flow3DFunc& GetAnalyticalExpression() {
			return mAnyticalFunc;

		}

		Flow3DFunc mAnyticalFunc = nullptr;
		OutOfBoundPolicy mOutOfSpaceBoundaryPolicy;
	};


	template <typename K>
	class Discrete3DFlowField : public IDiscreteField3d {
	public:
		using ScalarType = K;
		Discrete3DFlowField(const FieldSpaceTimeInfo3D& FiledInformation, bool initializeMem = true);
		~Discrete3DFlowField();

		inline void SetFlowVector(int index, int timeStep, const DTG::EigenVector<3, K>& vec)
		{
			mData.row(index + timeStep * NumberOfDataPoints) << vec(0), vec(1), vec(2);
		}

		inline void SetFlowVector(int idx, int idy, int idz, int idt, const DTG::EigenVector<3, K>& vec)
		{
			long spatial_index = idx + idy * mFieldInfo.XGridsize + idz * mFieldInfo.XGridsize * mFieldInfo.YGridsize;
			long total_points = mFieldInfo.XGridsize * mFieldInfo.YGridsize * mFieldInfo.ZGridsize;
			long total_index = spatial_index + idt * total_points;

			mData.row(total_index) << vec(0), vec(1), vec(2);
		}
		inline bool HasDiscreteData() const {
			return mData.rows()==this->GetNumberOfDataPoints()*GetNumberOfTimeSteps();
		}
		Eigen::Matrix<K, Eigen::Dynamic, 3>& GetData();
		const Eigen::Matrix<K, Eigen::Dynamic, 3>& GetDataView() const;
		DTG::EigenVector<3, K> GetSpaceAndTimeInterpolatedVector(const DTG::EigenPoint<3, K>& Position, K time) const;
		DTG::EigenVector<3, K> GetSpaceAndTimeInterpolatedVector(const DTG::EigenVector<3, K>& Position, K time) const;
		DTG::EigenVector<3, K> GetSpaceAndTimeInterpolatedVector(const K X, const K Y, const K Z, const K inTime) const;

		DTG::EigenVector<3, K> GetSpaceInterpolatedVector(const K inX, const K inY, const K inZ, const int timestep) const;
		DTG::EigenVector<3, K> GetTimeInterpolatedVector(const int index, const K time) const;
		DTG::EigenVector<3, K> GetTimeInterpolatedVector(const int x, const int y,const int z,const K time) const;
		DTG::EigenVector<3, K> GetVectorbyLinearIndex(const int index, const int timeStep) const;
		DTG::EigenVector<3, K> GetVectorAtGrid(const int x, const int y, const int z, const int timeStep) const;

		DTG::EigenVector<3, K> getPartialDerivativeT(const int idx, const int idy, const int idz, int idt) const;
		DTG::EigenVector<3, K> getPartialDerivativeT(const Eigen::Matrix<K, 3, 1>& point, K time) const;

		DTG::EigenVector<3, K> getPartialDerivativeT(const Eigen::Matrix<K, 1, 3>& point, K time) const;


		DTG::EigenVector<3, K> getPartialDerivativeX(const Eigen::Matrix<K, 3, 1>& point, K time) const;
		DTG::EigenVector<3, K> getPartialDerivativeY(const Eigen::Matrix<K, 3, 1>& point, K time) const;
		DTG::EigenVector<3, K> getPartialDerivativeZ(const Eigen::Matrix<K, 3, 1>& point, K time) const;
		DTG::EigenVector<3, K> getPartialDerivativeX(const Eigen::Matrix<K, 1, 3>& point, K time) const;
		DTG::EigenVector<3, K> getPartialDerivativeY(const Eigen::Matrix<K, 1, 3>& point, K time) const;
		DTG::EigenVector<3, K> getPartialDerivativeZ(const Eigen::Matrix<K, 1, 3>& point, K time) const;

		
		Eigen::Matrix<K, 3, 3> getVelocityGradientTensorTimeInterpolated(const int idx, const int idy, const int idz, const float time) const;

		Eigen::Matrix<K, 3, 3> getVelocityGradientTensor(const Eigen::Matrix<K, 3, 1>& point, K time) const;

		Eigen::Matrix<K, 3, 3> getVelocityGradientTensor(
			const int idx, const int idy, const int idz, const int idt) const;

		Eigen::Matrix<K, 3, 1> getVorticity(const int x, const int y, const int z, const int t) const;
		Eigen::Matrix<float, 3, 1> getVorticity(const DTG::EigenVector<3, float>& point, float time)
		{
			Eigen::Matrix<K, 3, 1> cast_point = { (K)point.x(), (K)point.y(), (K)point.z() };
			Eigen::Matrix3f jac = this->getVelocityGradientTensor(cast_point, time).cast<float>();

			auto curl = computeCurlfromGradientTensor(jac);
			return curl;
		}

		void Integrate3DPathlineOneDirection(const DTG::Point<3, double>& startPointInChart, double startTime, double targetIntegrationTime, double stepSize, int maxIterationCount, std::vector<DTG::PointAndTime<3, double>>& results, const PathlineNumericalIntegrationMethod method,bool bondaryRelaxtion=false) override;

		void Integrate3DPathlineOneDirection(const DTG::Point<3, float>& startPointInChart, float startTime, float targetIntegrationTime, float stepSize, int maxIterationCount, std::vector<DTG::PointAndTime<3, float>>& results, const PathlineNumericalIntegrationMethod method,bool bondaryRelaxtion=false) override;

		void Integrate3DStreamlineOneDirection(const DTG::Point<3, double>& startPointInChart, const double startTime, const double stepSize, const int maxIterationCount, std::vector<DTG::PointAndTime<3, double>>& results, const PathlineNumericalIntegrationMethod method, bool forward=true) override;
		void Integrate3DStreamlineOneDirection(const DTG::Point<3, float>& startPointInChart, const float startTime, const float stepSize, const int maxIterationCount, std::vector<DTG::PointAndTime<3, float>>& results, const PathlineNumericalIntegrationMethod method, bool forward=true) override;

		std::unique_ptr<IDiscreteField3d> ReSampleToNewResolution(int xdim, int ydim, int zdim, int tdim);

		Eigen::Matrix<K, Eigen::Dynamic, 3> GetSliceRawData(const int idt);
		template <typename K>
		Discrete3DFlowField<K> GetSlice(const int idt)
		{

			auto information = mFieldInfo;
			double dt = mFieldInfo.Getdt();
			double time = idt * dt + mFieldInfo.minTime;
			information.minTime = time;
			information.maxTime = time;
			information.numberOfTimeSteps = 1;
			Discrete3DFlowField<K> res(information);
			res.GetData() = this->GetSliceRawData(idt);
			return res;
		}
		DTG::Discrete3DFlowField<K> GetInterpoltedSlice(double time)
		{
			double mMaxTime = mFieldInfo.maxTime;
			double mMinTime = mFieldInfo.minTime;
			if (time > mMaxTime) {
				time = mMaxTime;
			}
			if (time < mMinTime) {
				time = mMinTime;
			}
			double timeSpan = mMaxTime - mMinTime;
			double oneTimeInterval = 0;
			double continousIndex = 0;
			const auto numOfTimsteps=GetNumberOfTimeSteps() ;
			if (numOfTimsteps> 1) {
				oneTimeInterval = timeSpan / (numOfTimsteps- 1);
				continousIndex = (time - mMinTime) / oneTimeInterval;
			}

			int floorIndex = floor(continousIndex);
			int ceilIndex = ceil(continousIndex);
			auto resRawData = this->GetSliceRawData(floorIndex);

			if (ceilIndex != floorIndex) {
				auto resRawDataCeil = this->GetSliceRawData(ceilIndex);
				double fraction = continousIndex - (double)floorIndex;
				resRawData = resRawData * (1.0 - fraction) + resRawDataCeil * fraction;
			}

			auto information = mFieldInfo;
			information.minTime = time;
			information.maxTime = time;
			information.numberOfTimeSteps = 1;

			Discrete3DFlowField res(information);
			res.GetData() = resRawData;
			return res;
		}

		std::vector<DTG::EigenVector<3, K>> AverageVectorFieldBySlice() const;


		vtkSmartPointer<vtkImageData>  ConvertDTGVectorField3DToVTKImage(double time_val);
		void SampleAnlyticalToDiscreteData();
		
	protected:
		Eigen::Matrix<K, Eigen::Dynamic, 3> mData;
	};

	template <typename T>
	std::unique_ptr<DTG::IDiscreteField3d> DuplicateVectorField(T&& inputField) // T&& is a forwarding reference
	{
		// Step 1: Get the decayed type of the input.
		// This removes references, cv-qualifiers, and converts arrays/functions to pointers.
		// If T is unique_ptr<X>&, DecayedInputType is unique_ptr<X>.
		// If T is X*, DecayedInputType is X*.
		// If T is X&, DecayedInputType is X.
		using DecayedInputType = std::decay_t<T>;

		// Step 2: Determine the actual underlying field object type.
		// This is the type that has ScalarType and mFieldInfo.
		using BareFieldType = std::remove_cv_t < // Ensure it's non-const, non-volatile
			std::conditional_t <
			// Check if DecayedInputType is a raw pointer (e.g., FieldType*)
			std::is_pointer_v<DecayedInputType>,
			std::remove_pointer_t<DecayedInputType>, // If yes, get the type it points to (FieldType)
			// Else, check if it's a smart pointer like unique_ptr (has ::element_type)
			std::conditional_t <
			requires { typename DecayedInputType::element_type; }, // Check for ::element_type
		typename DecayedInputType::element_type,               // If yes, get its element_type (FieldType)
			DecayedInputType // Otherwise, assume DecayedInputType is the field type itself (FieldType)
			>
			>
		> ;

		// Step 3: Get a const reference to the actual field object.
		const BareFieldType& field_object_ref = [&]() -> const BareFieldType& {
			if constexpr (std::is_pointer_v<DecayedInputType>) { // e.g., input is SomeType*
				if (!inputField) { // Use the original inputField parameter for the check
					throw std::runtime_error("Input raw pointer is null.");
				}
				return *inputField; // Dereference raw pointer
			}
			else if constexpr (requires { typename DecayedInputType::element_type; }&&
				requires { inputField.operator*(); }) { // e.g., input is std::unique_ptr<SomeType>
				if (!inputField) { // Use the original inputField parameter for the check
					throw std::runtime_error("Input smart pointer is null or disengaged.");
				}
				return *inputField; // Dereference smart pointer
			}
			else { // e.g., input is SomeType or SomeType&
				return inputField; // Use directly (it's already an object or reference to one)
			}
			}();

		// Step 4: Use the field_object_ref to get ScalarType and mFieldInfo
		using Scalar = typename BareFieldType::ScalarType; // Access ScalarType from the actual field type
		auto field_info = field_object_ref.mFieldInfo;    // Access mFieldInfo from the actual field object

		// Create the new field using the deduced Scalar type and retrieved field_info
		auto newField = std::make_unique<DTG::Discrete3DFlowField<Scalar>>(field_info, true);
		return newField;
	}
	template <typename T>
	std::unique_ptr<IDiscreteField3d> DuplicateVectorField(T&& inputField, bool initMem)
	{
		using DecayedInputType = std::decay_t<T>;
		using BareFieldType = std::remove_cv_t < // Ensure it's non-const, non-volatile
			std::conditional_t <
			// Check if DecayedInputType is a raw pointer (e.g., FieldType*)
			std::is_pointer_v<DecayedInputType>,
			std::remove_pointer_t<DecayedInputType>, // If yes, get the type it points to (FieldType)
			// Else, check if it's a smart pointer like unique_ptr (has ::element_type)
			std::conditional_t <
			requires { typename DecayedInputType::element_type; }, // Check for ::element_type
		typename DecayedInputType::element_type,               // If yes, get its element_type (FieldType)
			DecayedInputType // Otherwise, assume DecayedInputType is the field type itself (FieldType)
			>
			>
		> ;
		const BareFieldType& field_object_ref = [&]() -> const BareFieldType& {
			if constexpr (std::is_pointer_v<DecayedInputType>) { // e.g., input is SomeType*
				if (!inputField) { // Use the original inputField parameter for the check
					throw std::runtime_error("Input raw pointer is null.");
				}
				return *inputField; // Dereference raw pointer
			}
			else if constexpr (requires { typename DecayedInputType::element_type; }&&
				requires { inputField.operator*(); }) { // e.g., input is std::unique_ptr<SomeType>
				if (!inputField) { // Use the original inputField parameter for the check
					throw std::runtime_error("Input smart pointer is null or disengaged.");
				}
				return *inputField; // Dereference smart pointer
			}
			else { // e.g., input is SomeType or SomeType&
				return inputField; // Use directly (it's already an object or reference to one)
			}
			}();
		// Step 4: Use the field_object_ref to get ScalarType and mFieldInfo
		using Scalar = typename BareFieldType::ScalarType; // Access ScalarType from the actual field type
		auto field_info = field_object_ref.mFieldInfo;    // Access mFieldInfo from the actual field object

		auto newField = std::make_unique<Discrete3DFlowField<Scalar>>(field_info, initMem);
		return newField;
	}

	struct VectorField3DPointwiseOperator {

		template <typename scalarType>
		void operator()(const std::function<DTG::EigenVector<3, scalarType>(const int, const int, const int, const int)>& mPointwiseFunc, DTG::Discrete3DFlowField<scalarType>& result) const;
		void operator()(const std::function<float(const int, const int, const int, const int)>& mPointwiseFunc, DTG::DiscreteScalarField3D<float>& result) const;

		void operator()(const Eigen::Vector3i& gridSize, const int timeSteps, const std::function<float(const int, const int, const int, const int)>& mPointwiseFunc, DTG::DiscreteScalarField3D<float>& result) const;

		template <typename scalarType>
		void operator()(const Eigen::Vector3i& gridSize, const int timeSteps, const std::function<DTG::EigenVector<3, scalarType>(const int, const int, const int, const int)>& mPointwiseFunc, DTG::Discrete3DFlowField<scalarType>& result) const;
	};


	static inline std::size_t nextPowerOfTwo(std::size_t n)
	{

		if (n <= 1)
			return 1;
		--n;
		n |= n >> 1;
		n |= n >> 2;
		n |= n >> 4;
		n |= n >> 8;
		n |= n >> 16;
#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__)
		n |= n >> 32;
#endif
		++n;
		return n;
	}
	template <typename K>
	void HelmholtzHodgeDecomposition(
		const Discrete3DFlowField<K>& flowField,
		Discrete3DFlowField<K>& curlFreeField, // d
		Discrete3DFlowField<K>& divFreeField, // r
		Discrete3DFlowField<K>& harmonicField, // h
		const int maxPossitionIteration

	);


	std::unique_ptr<IDiscreteField3d>  CreateSteadyKillingVectorField(const KillingField3DCoefficients& coeffs, const FieldSpaceTimeInfo3D& domainInfo);


	// this is a singleton class manage vector field 3D
	class VectorField3DManager {
	public:
		static VectorField3DManager& GetInstance()
		{
			static VectorField3DManager instance;
			return instance;
		}
		void InsertVectorField(std::unique_ptr<IDiscreteField3d> vPtr, const std::string& name);

		std::vector<std::string> GetAllVectorFieldNames() const;
		IDiscreteField3d* GetVectorFieldByName(const std::string& name);
		int GetNumberOfTimestepsOfVectorField(const std::string& name);
		bool HasDiscreteVectorField(const std::string& name);

		DTG::EigenVector<3, double> GetVectorFieldMinimalRange(const std::string& name);
		DTG::EigenVector<3, double> GetVectorFieldMaximalRange(const std::string& name);

		double GetMinTimeOfVectorField(const std::string& name);
		double GetMaxTimeOfVectorField(const std::string& name);
		DTG::EigenVector<2, double> GetVectorFieldTimeRange(const std::string& name) const;
		bool IsInValidSpaceAreaOfVectorField(const std::string& name, const EigenPoint<3, double>& Position) const;
		bool IsInValidSpaceAreaOfVectorField(const std::string& name, const EigenPoint<3, float>& Position) const;
		// bool IsInValidSpaceAreaOfVectorFiled(const std::string& name, glm::vec3 Position) const;

		

	private:
		std::map<std::string, std::unique_ptr<IDiscreteField3d>> mVectorFields3DOnGrid;


		// singleton pattern
		VectorField3DManager() = default;
		~VectorField3DManager() = default;
		VectorField3DManager(const VectorField3DManager&);
		VectorField3DManager& operator=(const VectorField3DManager&);
	};

} // namespace DTG

#endif // !_DISCRETE3D_FLOWFIELD_H
