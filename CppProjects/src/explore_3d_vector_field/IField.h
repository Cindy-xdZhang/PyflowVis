
#ifndef _IFIELD_COMMON_3D_H
#define _IFIELD_COMMON_3D_H
#pragma once
#include "PortToUsingModules.h"
#include "myTypeDefs.h"
#include "Point.h"
#include <map>
#include <numeric>
#include <string>

namespace DTG {
	Eigen::Vector3f computeCurlfromGradientTensor(const Eigen::Matrix<float, 3, 3>& velocityGradientTensor);

	
	 using  Flow3DFunc = std::function<Eigen::Matrix<float, 3, 1>(float, float, float, float)>;
	 using  Scalar3DFunc = std::function<float(float, float, float, float)>;
	// pod data of 3d vector field/scalar field range:minTime,maxTIME,Timestep,Xgrid,Ygrid,Zgrid,minx,miny,minz,maxx,maxy,maxz
	struct FieldSpaceTimeInfo3D {
		double minTime = 0.0;
		double maxTime = 0.0;
		int numberOfTimeSteps = 0;
		int XGridsize = 0;
		int YGridsize = 0;
		int ZGridsize = 0;
		double minXCoordinate = 0.0;
		double minYCoordinate = 0.0;
		double minZCoordinate = 0.0;
		double maxXCoordinate = 0.0;
		double maxYCoordinate = 0.0;
		double maxZCoordinate = 0.0;
		FieldSpaceTimeInfo3D() = default;
		FieldSpaceTimeInfo3D(double minTime, double maxTime, int numberOfTimeSteps, int XGridsize, int YGridsize,
			int ZGridsize, double minXCoordinate, double minYCoordinate, double minZCoordinate, double maxXCoordinate, double maxYCoordinate, double maxZCoordinate)
			:minTime(minTime),maxTime(maxTime),numberOfTimeSteps(numberOfTimeSteps),
			XGridsize(XGridsize),
			YGridsize(YGridsize),
			ZGridsize(ZGridsize),
			minXCoordinate(minXCoordinate),
			minYCoordinate(minYCoordinate),
			minZCoordinate(minZCoordinate),
			maxXCoordinate(maxXCoordinate),
			maxYCoordinate(maxYCoordinate),
			maxZCoordinate(maxZCoordinate)
		{
		}
		FieldSpaceTimeInfo3D(Eigen::Vector4i Resolution, Eigen::Vector4d domainMin,Eigen::Vector4d domainMax)
			:minTime(domainMin.w()),
			maxTime(domainMax.w()),
			numberOfTimeSteps( Resolution.w()),
			//xyz grid
			XGridsize(Resolution.x()),
			YGridsize(Resolution.y()),
			ZGridsize(Resolution.z()),
			minXCoordinate(domainMin.x()),
			minYCoordinate(domainMin.y()),
			minZCoordinate(domainMin.z()),
			maxXCoordinate(domainMax.x()),
			maxYCoordinate(domainMax.y()),
			maxZCoordinate(domainMax.z()){		}


		double Getdt() const
		{
			return numberOfTimeSteps > 1 ? (maxTime - minTime) / (numberOfTimeSteps - 1) : std::numeric_limits<float>::infinity();
		}
		bool operator==(const FieldSpaceTimeInfo3D& other) const
		{
			return minTime == other.minTime && maxTime == other.maxTime && numberOfTimeSteps == other.numberOfTimeSteps && XGridsize == other.XGridsize && YGridsize == other.YGridsize && ZGridsize == other.ZGridsize && minXCoordinate == other.minXCoordinate && minYCoordinate == other.minYCoordinate && minZCoordinate == other.minZCoordinate && maxXCoordinate == other.maxXCoordinate && maxYCoordinate == other.maxYCoordinate && maxZCoordinate == other.maxZCoordinate;
		}

	};  
	

	//IField3D could be a physical domain with/without grid resolution
	struct IField3D {
		enum OutOfBoundPolicy {
			RepeatPolicy
		};
		IField3D(FieldSpaceTimeInfo3D info) :mFieldInfo(info)
		{
			// check domain is valid:
			if (this->mFieldInfo.minTime > this->mFieldInfo.maxTime) {
				std::swap(this->mFieldInfo.minTime, this->mFieldInfo.maxTime);
			}
			if (this->mFieldInfo.minXCoordinate > this->mFieldInfo.maxXCoordinate) {
				std::swap(this->mFieldInfo.minXCoordinate, this->mFieldInfo.maxXCoordinate);
			}
			if (this->mFieldInfo.minYCoordinate > this->mFieldInfo.maxYCoordinate) {
				std::swap(this->mFieldInfo.minYCoordinate, this->mFieldInfo.maxYCoordinate);
			}
			if (this->mFieldInfo.minZCoordinate > this->mFieldInfo.maxZCoordinate) {
				std::swap(this->mFieldInfo.minZCoordinate, this->mFieldInfo.maxZCoordinate);
			}


			const double Xrange = mFieldInfo.maxXCoordinate - mFieldInfo.minXCoordinate;
			const double Yrange = mFieldInfo.maxYCoordinate - mFieldInfo.minYCoordinate;
			const double Zrange = mFieldInfo.maxZCoordinate - mFieldInfo.minZCoordinate;
	/*		mOneXGridInterval = Xrange / (mFieldInfo.XGridsize - 1);
			mOneYGridInterval = Yrange / (mFieldInfo.YGridsize - 1);
			mOneZGridInterval = Zrange / (mFieldInfo.ZGridsize - 1);*/

			mInverse_OneXGridInterval = (mFieldInfo.XGridsize - 1) / Xrange ;
			mInverse_OneYGridInterval = (mFieldInfo.YGridsize - 1) / Yrange ;
			mInverse_OneZGridInterval =mFieldInfo.ZGridsize - 1>0? (mFieldInfo.ZGridsize - 1) / Zrange :0.0 ;
			mInverse_dt= mFieldInfo.numberOfTimeSteps >1? (mFieldInfo.numberOfTimeSteps - 1) / (mFieldInfo.maxTime-mFieldInfo.minTime):0.0;
			mXYGrid = mFieldInfo.XGridsize * mFieldInfo.YGridsize ;
			NumberOfDataPoints = mFieldInfo.XGridsize * mFieldInfo.YGridsize * mFieldInfo.ZGridsize;
		}
		DTG::EigenVector<3, double> GetDomainSize() const;
		DTG::EigenVector<3, double> GetSpatialMin() const;
		DTG::EigenVector<3, double> GetSpatialMax() const;
		inline double getXMin()const{return mFieldInfo.minXCoordinate;};
		inline double getYMin()const{return mFieldInfo.minYCoordinate;};
		inline double getZMin()const{return mFieldInfo.minZCoordinate;};
		inline double getXMax()const { return mFieldInfo.maxXCoordinate; };
		inline double getYMax()const { return mFieldInfo.maxYCoordinate; };
		inline double getZMax()const { return mFieldInfo.maxZCoordinate; };
		std::string getPrintOutGridInfo() const;
		std::string getPrintOutInfo() const;
		//vtk style functions
		DTG::EigenVector<3, double> GetOrigin()const{return GetSpatialMin();}
		DTG::EigenVector<3, double> GetSpacing()const{return GetSpatialDxDyDz();}


		double GetMinTime() const;
		double GetMaxTime()const;
		template <typename ScalarType = float>
		bool IsInValidSpaceAreaOfVectorField(const DTG::EigenPoint<3, ScalarType>& Position) const
		{
			if (Position(0) > this->mFieldInfo.maxXCoordinate || Position(0) < this->mFieldInfo.minXCoordinate) {
				return false;
			}
			if (Position(1) > this->mFieldInfo.maxYCoordinate || Position(1) < this->mFieldInfo.minYCoordinate) {
				return false;
			}
			if (Position(2) > this->mFieldInfo.maxZCoordinate || Position(2) < this->mFieldInfo.minZCoordinate) {
				return false;
			}
			return true;
		}
		template <typename ScalarType = float>
		bool IsInValidSpaceAreaOfVectorField(const DTG::EigenVector<3, ScalarType>& Position) const
		{
			if (Position(0) > this->mFieldInfo.maxXCoordinate || Position(0) < this->mFieldInfo.minXCoordinate) {
				return false;
			}
			if (Position(1) > this->mFieldInfo.maxYCoordinate || Position(1) < this->mFieldInfo.minYCoordinate) {
				return false;
			}
			if (Position(2) > this->mFieldInfo.maxZCoordinate || Position(2) < this->mFieldInfo.minZCoordinate) {
				return false;
			}
			return true;
		}



		//----------------------------------------
		//functions need grid resolution
		//----------------------------------------
		Eigen::Vector3i GetSpatialGridSize() const;
		int GetXGridSize() const { return mFieldInfo.XGridsize; }
		int GetYGridSize() const { return mFieldInfo.YGridsize; }
		int GetZGridSize() const { return mFieldInfo.ZGridsize; }
		Eigen::Vector3d GetSpatialDxDyDz() const;
		uint64_t GetNumberOfDataPoints() const;

		// Get grid dimensions as tuple
		std::tuple<int, int, int, int> GetGridSize() const
		{
			return std::make_tuple(mFieldInfo.XGridsize, mFieldInfo.YGridsize, mFieldInfo.ZGridsize, mFieldInfo.numberOfTimeSteps);
		}
		// Check if grid indices are valid 
		void CheckBounds(int x, int y, int z, int t) const
		{
			bool outOfBounday = (x < 0 || x >= mFieldInfo.XGridsize || y < 0 || y >= mFieldInfo.YGridsize || z < 0 || z >= mFieldInfo.ZGridsize || t >= mFieldInfo.numberOfTimeSteps || t < 0);
			assert(!outOfBounday);
		}
		inline bool AtSpatialBoundary(uint64_t x, uint64_t y, uint64_t z) const
		{
			return x == 0U || x == (mFieldInfo.XGridsize - 1) || y == 0U || y == (mFieldInfo.YGridsize - 1) || z == 0U || z == (mFieldInfo.ZGridsize - 1);
		}

		double convertTimeStep2PhysicalTime(int timeStep) const;
		int convertPhysicalTimeRoundedTimeStep(double time) const;
		DTG::EigenVector<3, double> convertGridIndex2PhysicalCoordinates(int x, int y, int z) const;
		DTG::EigenVector<4, double> convertGridIndex2PhysicalCoordinates(int x, int y, int z, int t) const;

		//convertPhysicalPosition2Index only works when there is Gridsize and timesteps defined.
		inline std::tuple<double, double, double, double> convertPhysicalPosition2Index(double xpos, double ypos, double zpos, double physical_time) const
		{
			// Normalize physical coordinates
			double xNorm = (xpos - mFieldInfo.minXCoordinate) / (mFieldInfo.maxXCoordinate - mFieldInfo.minXCoordinate);
			double yNorm = (ypos - mFieldInfo.minYCoordinate) / (mFieldInfo.maxYCoordinate - mFieldInfo.minYCoordinate);
			double zNorm = mFieldInfo.maxZCoordinate > mFieldInfo.minZCoordinate  ?(zpos - mFieldInfo.minZCoordinate) / (mFieldInfo.maxZCoordinate - mFieldInfo.minZCoordinate):0.0;
			double tNorm = (physical_time - mFieldInfo.minTime) / (mFieldInfo.maxTime - mFieldInfo.minTime);
			// a steady field
			if (std::abs(mFieldInfo.maxTime - mFieldInfo.minTime) < 1e-12) {
				tNorm = 0.0;
			}

			// Scale normalized values to grid indices
			double yIndex = yNorm * static_cast<double>((mFieldInfo.YGridsize - 1));
			double xIndex = xNorm * static_cast<double>((mFieldInfo.XGridsize - 1));
			double zIndex = zNorm * static_cast<double>((mFieldInfo.ZGridsize - 1));
			double timeIndex = tNorm * static_cast<double>((mFieldInfo.numberOfTimeSteps - 1));

			// Clamp indices to valid grid bounds
			xIndex = std::clamp(xIndex, 0.0, (double)mFieldInfo.XGridsize - 1.0);
			yIndex = std::clamp(yIndex, 0.0, (double)mFieldInfo.YGridsize - 1.);
			zIndex = std::clamp(zIndex, 0.0, (double)mFieldInfo.ZGridsize - 1.);
			timeIndex = std::clamp(timeIndex, 0.0, (double)mFieldInfo.numberOfTimeSteps - 1.);

			// Return as a tuple
			return std::make_tuple(xIndex, yIndex, zIndex, timeIndex);
		}

		inline Eigen::Vector3d  convertPhysicalPosition2Index(double xpos, double ypos, double zpos) const
		{
			// Normalize physical coordinates
			double xNorm = (xpos - mFieldInfo.minXCoordinate) / (mFieldInfo.maxXCoordinate - mFieldInfo.minXCoordinate);
			double yNorm = (ypos - mFieldInfo.minYCoordinate) / (mFieldInfo.maxYCoordinate - mFieldInfo.minYCoordinate);
			double zNorm = mFieldInfo.maxZCoordinate > mFieldInfo.minZCoordinate  ?(zpos - mFieldInfo.minZCoordinate) / (mFieldInfo.maxZCoordinate - mFieldInfo.minZCoordinate):0.0;
	
			// Scale normalized values to grid indices
			double yIndex = yNorm * static_cast<double>((mFieldInfo.YGridsize - 1));
			double xIndex = xNorm * static_cast<double>((mFieldInfo.XGridsize - 1));
			double zIndex = zNorm * static_cast<double>((mFieldInfo.ZGridsize - 1));

			// Clamp indices to valid grid bounds
			xIndex = std::clamp(xIndex, 0.0, (double)mFieldInfo.XGridsize - 1.0);
			yIndex = std::clamp(yIndex, 0.0, (double)mFieldInfo.YGridsize - 1.);
			zIndex = std::clamp(zIndex, 0.0, (double)mFieldInfo.ZGridsize - 1.);

			// Return as a tuple
			return Eigen::Vector3d(xIndex, yIndex, zIndex);
		}




		inline auto GetFieldInfo() const{ return mFieldInfo; }
		inline int GetNumberOfTimeSteps() const { return mFieldInfo.numberOfTimeSteps; }
		inline double Getdt() const
		{
			return mFieldInfo.Getdt();
		}
		inline double GetInversedt() const
		{
			return mInverse_dt;
		}
		// Compute flat index for (x, y, z)
		uint64_t GetSpatialIndex(uint64_t  x, uint64_t  y, uint64_t  z) const
		{
			return x + y * mFieldInfo.XGridsize + z * mXYGrid;
		}
		inline uint64_t  GetDataIndex(uint64_t  x, uint64_t  y, uint64_t  z) const
		{
			return x + y * mFieldInfo.XGridsize + z * mXYGrid;
		}
		inline uint64_t  GetDataIndex(uint64_t  x, uint64_t  y, uint64_t  z, uint64_t  t) const
		{
			return x + y * mFieldInfo.XGridsize + z *mXYGrid+ t * NumberOfDataPoints;
		}
		FieldSpaceTimeInfo3D mFieldInfo;
	protected:
		
		//for faster data access
		double		mInverse_OneXGridInterval;
		double		mInverse_OneYGridInterval;
		double		mInverse_OneZGridInterval;
		double		mInverse_dt;
		uint64_t mXYGrid; 
		uint64_t NumberOfDataPoints;

	};

	

	
	

} // namespace DTG

#endif // !_IFIELD_COMMON_3D_H
