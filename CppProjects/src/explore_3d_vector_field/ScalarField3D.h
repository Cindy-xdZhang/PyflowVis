
#ifndef _DISCRETE3D_SCALARFIELD_H
#define _DISCRETE3D_SCALARFIELD_H
#pragma once
#include "PortToUsingModules.h"
#include "myTypeDefs.h"
#include "IField.h"
#include "magic_enum.hpp"
#include "logSystem/log.h"
#include <execution>
#include "FlowLineComputeUtils.h"

namespace DTG {

	enum class FLOWFIELD_OPERATION : uint32_t {
		//operation _generate scalar field
		CURL_NORM = 0,
		IVD,
		MAGNITUDE,
		NORM_dVdt,
		Killing,
		Q_criterion,
		lamba2_criterion,
		Angular_Velocity_NORM,
		LOCAL_AVERAGE,
		LAGRANGIAN_,
		LAVD,

		//operation _generate vector field
		CURL,
		Angular_Velocity,
		Material_Derivative
	};


	template <typename T>
	class DiscreteScalarField3D : public IField3D {
	public:
		DiscreteScalarField3D(const FieldSpaceTimeInfo3D& info)
			:IField3D(info)
		{
			if (info.numberOfTimeSteps>0&&mFieldInfo.XGridsize>1 &&mFieldInfo.YGridsize>1 &&mFieldInfo.ZGridsize>=1)
			{
				dataSlices_.resize(mFieldInfo.numberOfTimeSteps, std::vector<T>(mFieldInfo.XGridsize * mFieldInfo.YGridsize * mFieldInfo.ZGridsize));
			}
		}

		// Set scalar value at grid point (x, y, z)
		void SetValue(uint64_t  x, uint64_t  y, uint64_t  z, uint64_t  t, T value);

		// Get scalar value at grid point (x, y, z)
		T GetValueAtGrid(uint64_t x, uint64_t  y, uint64_t  z, uint64_t  t) const
		{
			// CheckBounds(x, y, z, t);
			return dataSlices_[t][GetSpatialIndex(x, y, z)];
		}

		T GetValue(const Eigen::Vector4d& pos_t) const
		{
			return InterpolateValuebyPos(pos_t.x(), pos_t.y(), pos_t.z(), pos_t.w());
		}
		T GetValue(const DTG::EigenVector<3, float>& pos_t, float time) const
		{
			// CheckBounds(x, y, z, t);
			return InterpolateValuebyPos(pos_t.x(), pos_t.y(), pos_t.z(), time);
		}
		T GetValue(double xpos, double ypos, double zpos, double physical_time) const
		{
			// CheckBounds(x, y, z, t);
			return InterpolateValuebyPos(xpos, ypos, zpos, physical_time);
		}

		inline T InterpolateValuebyPos(double xpos, double ypos, double zpos, double physical_time) const
		{
			auto [xIndex, yIndex, zIndex, timeIndex] = convertPhysicalPosition2Index(xpos, ypos, zpos, physical_time);
			return InterpolateBygridIndex(xIndex, yIndex, zIndex, timeIndex);
		}



		const std::vector<T>& getSliceDataConst(int t) const
		{
			return dataSlices_[t];
		}
		std::vector<T>& getSliceData(int t)
		{
			return dataSlices_[t];
		}

		std::vector<T> getSliceAtPhysicalTime(double t) const;


		void setSliceData(int t,  std::vector<T>& dataRaw) {
			dataSlices_[t]=std::move(dataRaw);
		}

		std::tuple<T, T> computeMinMaxValue() const;
		double ScalarFieldReduction(const std::string& reductionType) const;
		void resampleASliceToFloatBuffer(std::vector<float>& buffer, Eigen::Vector2i& sliceSize, int SamplingAxis, double PositionConstant);
		void SetAnalyticalExpression(const Scalar3DFunc& inAnyticalFunc) {
			mAnyticalFunc = inAnyticalFunc;
		}
		inline bool HasAnalyticalExpression()const{
			return mAnyticalFunc!=nullptr;
		}



		bool HasDiscreteData() const;
		void resampleAnalytical2Gird();

		T GetValueFromAnalyticalExpression(T xpos, T ypos, T zpos, T physical_time) const
		{
			return this->mAnyticalFunc(xpos, ypos, zpos, physical_time);
		}
			
		T GetValueAtGridFromAnalyticalExpression(uint64_t idx, uint64_t  idy, uint64_t  idz, uint64_t  idt) const
		{
			auto xyz_pos = this->convertGridIndex2PhysicalCoordinates(idx, idy, idz);
			auto physical_time = this->convertTimeStep2PhysicalTime(idt);
			return this-> mAnyticalFunc( xyz_pos.x(), xyz_pos.y(), xyz_pos.z(), physical_time);
		}

	private:
		// Perform trilinear interpolation for scalar value
		inline T InterpolateBygridIndex(double xgridindex, double ygridindex, double zgridindex, int t) const
		{
			// Grid indices and weights
			int x0 = static_cast<int>(std::floor(xgridindex));
			int y0 = static_cast<int>(std::floor(ygridindex));
			int z0 = static_cast<int>(std::floor(zgridindex));
			int x1 = x0 + 1;
			int y1 = y0 + 1;
			int z1 = z0 + 1;

			double wx = xgridindex - x0;
			double wy = ygridindex - y0;
			double wz = zgridindex - z0;

			// Clamp indices to valid range
			x0 = std::clamp(x0, 0, mFieldInfo.XGridsize - 1);
			y0 = std::clamp(y0, 0, mFieldInfo.YGridsize - 1);
			z0 = std::clamp(z0, 0, mFieldInfo.ZGridsize - 1);
			x1 = std::clamp(x1, 0, mFieldInfo.XGridsize - 1);
			y1 = std::clamp(y1, 0, mFieldInfo.YGridsize - 1);
			z1 = std::clamp(z1, 0, mFieldInfo.ZGridsize - 1);

			// Trilinear interpolation
			T c000 = GetValueAtGrid(x0, y0, z0, t);
			T c100 = GetValueAtGrid(x1, y0, z0, t);
			T c010 = GetValueAtGrid(x0, y1, z0, t);
			T c110 = GetValueAtGrid(x1, y1, z0, t);
			T c001 = GetValueAtGrid(x0, y0, z1, t);
			T c101 = GetValueAtGrid(x1, y0, z1, t);
			T c011 = GetValueAtGrid(x0, y1, z1, t);
			T c111 = GetValueAtGrid(x1, y1, z1, t);

			T c00 = c000 * (1 - wx) + c100 * wx;
			T c10 = c010 * (1 - wx) + c110 * wx;
			T c01 = c001 * (1 - wx) + c101 * wx;
			T c11 = c011 * (1 - wx) + c111 * wx;

			T c0 = c00 * (1 - wy) + c10 * wy;
			T c1 = c01 * (1 - wy) + c11 * wy;

			return c0 * (1 - wz) + c1 * wz;
		}

		inline T InterpolateBygridIndex(double xgridindex, double ygridindex, double zgridindex, double tindex) const
		{
			int tindex_floor = static_cast<int>(std::floor(tindex));
			int tindex_ceil = std::min(tindex_floor + 1, mFieldInfo.numberOfTimeSteps - 1);

			double alpha = tindex - tindex_floor;

			auto resultFloor = InterpolateBygridIndex(xgridindex, ygridindex, zgridindex, tindex_floor);

			if (tindex_floor != tindex_ceil) {
				auto resultCeil = InterpolateBygridIndex(xgridindex, ygridindex, zgridindex, tindex_ceil);
				resultFloor = resultFloor * (1 - alpha) + alpha * resultCeil;
			}
			return resultFloor;
		}
		std::vector<std::vector<T>> dataSlices_;
		Scalar3DFunc mAnyticalFunc = nullptr;
	};


	struct AttributesManager {
		//AttributeManager manage scalar fields, when user pick active field in ui it check whether scalar field exists, compute if not, then sample it flowlines/iso-surface etc.
		// !There are two status of scalar field:  scalar field of analytical expression/ discrete scalar field on grid. 
		// when request a scalar field on flowlines/iso surface-> requestAnalytical Scalar Field
		//when apply a scalar field on slice plane/user's demand to compute/request minmax of it-> requestDiscrete Scalar Field by apply field operator with scalarFunctionByGrid
	public:
		using  scalarFunctionByGrid = std::function<float(const int, const int, const int, const int)>;

		static AttributesManager& GetInstance()
		{
			static AttributesManager instance;
			return instance;
		}
		static  std::string  AttributesNameGenerate(const std::string& inputFieldName, FLOWFIELD_OPERATION operation, int Pram1=1, std::string Parm2="") {
			if (operation == FLOWFIELD_OPERATION::LOCAL_AVERAGE)[[unlikely]]
			{
				auto opName = " Avg_local" + std::to_string(Pram1);
				std::string opNameString(opName.begin(), opName.end());
				return opNameString + "(" + inputFieldName + ")";
			}
			else if (operation == FLOWFIELD_OPERATION::LAGRANGIAN_) [[unlikely]]
			{
				auto opName = " lagrangian_" + std::to_string(Pram1)+ "_"+Parm2;
				std::string opNameString(opName.begin(), opName.end());
				return opNameString + "(" + inputFieldName + ")";
			}
			else
			{
			auto opName = magic_enum::enum_name<FLOWFIELD_OPERATION >(operation);
			std::string opNameString(opName.begin(), opName.end());
			return opNameString + "(" + inputFieldName + ")";

			}
		}

		inline DTG::DiscreteScalarField3D<float>* getScalarField(const std::string& FieldName) {
			if (auto iter = mScalarFields_.find(FieldName); iter == mScalarFields_.end())
			{
				return nullptr;
			}
			else {
				return iter->second.get();
			}
		}

		DiscreteScalarField3D<float>* requestDiscreteScalarField(const std::string& inputFieldName,const FLOWFIELD_OPERATION operation);
		DiscreteScalarField3D<float>* requestAnalyticalScalarField(const std::string& inputFieldName,const FLOWFIELD_OPERATION operation);
		DTG::DiscreteScalarField3D<float>*  requestDiscreteScalarFieldFromScalarField(const std::string& inputFieldName, const FLOWFIELD_OPERATION operation,const int Pram1);


		Eigen::Vector2d requestScalarFieldMinMax(const std::string& inputFieldName);
		Eigen::Vector2d requestScalarFieldMinMax(DTG::DiscreteScalarField3D<float>* ptr){
					auto name=getFieldName(ptr);
					return requestScalarFieldMinMax(name);
			
		};


		DiscreteScalarField3D<float>* LagriangianScalarField3DTimeSeries(const std::string& inputVectorFieldname, const std::string& inputFieldName, int nx, int ny,int nz ,int numberOfTimeSteps, float IntegrateTimeInterval, float stepSize);

		//this function integrate a bunch of pathlines( seeding at time t) in some resolution-> integrate scalar value 
		std::unique_ptr<DiscreteScalarField3D<float>> LagriangianScalarField3D(const std::string&  inputVectorFieldname, const std::string& inputFieldName,  float time ,float IntegrateTimeInterval, float stepSize ){
			auto scalarFieldIn=getScalarField(inputFieldName);
			if (scalarFieldIn)
			{
				return LagriangianScalarField3D(inputVectorFieldname, inputFieldName,scalarFieldIn->GetFieldInfo().XGridsize,scalarFieldIn->GetFieldInfo().YGridsize,scalarFieldIn->GetFieldInfo().ZGridsize ,time,IntegrateTimeInterval,stepSize);
			}else{
				LOG_E("LagriangianScalarField3D with inputFieldName =%s doesn't exist.",inputFieldName.c_str());
				return nullptr;
			}
		
		}
		//this function integrate a bunch of pathlines( seeding at time t) in some resolution-> integrate scalar value 
		std::unique_ptr<DiscreteScalarField3D<float>>  LagriangianScalarField3D(const std::string&  inputVectorFieldname,const std::string& inputFieldName, int nx,int ny, int nz, float time ,float IntegrateTimeInterval,float stepSize); 

		void interpolateScalarFieldOnGridAlongPathlines(const DiscreteScalarField3D<float>* scalarFieldInput,const std::vector<pathline3Df>& pathlines,  std::vector<std::vector<float>>& interpolatedValueOnPathlines);
		void reduceLagriangianToScalarFieldOnGrid(const std::vector<pathline3Df>& pathlines, const float dt_intergration,  const std::vector<std::vector<float>>& interpolatedValueOnPathlines, DiscreteScalarField3D<float>* lagriangianedValueOnPathlines, int nx, int ny, int nz);
		
		void renamingScalarField(const std::string& originalName, const std::string& newName);

		std::string getFieldName(DTG::DiscreteScalarField3D<float>* ptr) const {
			for (const auto& [name, field] : mScalarFields_) {
				if (field.get() == ptr) {
					return name;
				}
			}
			return "";
		}
		std::unordered_map<std::string, std::unique_ptr<DTG::DiscreteScalarField3D<float>>>& getAllField() {
			return mScalarFields_;
		}
	private:
		std::unordered_map<std::string, std::unique_ptr<DTG::DiscreteScalarField3D<float>>> mScalarFields_;
		std::unordered_map<std::string, Eigen::Vector2d> mScalarFieldsMinMax;

	private:
		// singleton pattern
		AttributesManager() = default;
		~AttributesManager() = default;
		AttributesManager(const AttributesManager&) = default;
		AttributesManager& operator=(const AttributesManager&) = default;

	};


} // namespace DTG

#endif // !_DISCRETE3D_SCALARFIELD_H
