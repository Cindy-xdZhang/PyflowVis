
#include "IField.h"
#include "logSystem/log.h"

namespace DTG {

	double DTG::IField3D::GetMinTime() const
	{
		return this->mFieldInfo.minTime;
	}

	double DTG::IField3D::GetMaxTime()const
	{
		return this->mFieldInfo.maxTime;
	}


	DTG::EigenVector<3, double> DTG::IField3D::GetSpatialMin() const
	{

		return DTG::EigenVector<3, double>({ mFieldInfo.minXCoordinate, mFieldInfo.minYCoordinate, mFieldInfo.minZCoordinate });
	}

	DTG::EigenVector<3, double> DTG::IField3D::GetSpatialMax() const
	{
		return DTG::EigenVector<3, double>({ mFieldInfo.maxXCoordinate, mFieldInfo.maxYCoordinate, mFieldInfo.maxZCoordinate });
	}

	DTG::EigenVector<3, double> DTG::IField3D::GetDomainSize() const
	{
		DTG::EigenVector<3, double> domainSize = GetSpatialMax() - GetSpatialMin();
		return domainSize;
	}
	DTG::EigenVector<3, int> DTG::IField3D::GetSpatialGridSize() const
	{
		return { mFieldInfo.XGridsize,mFieldInfo.YGridsize, mFieldInfo.ZGridsize };
	}




	uint64_t  DTG::IField3D::GetNumberOfDataPoints() const
	{
		return mFieldInfo.XGridsize * mFieldInfo.YGridsize * mFieldInfo.ZGridsize;;
	}



	DTG::EigenVector<3, double> DTG::IField3D::convertGridIndex2PhysicalCoordinates(int x, int y, int z) const
	{
		double xmin = mFieldInfo.minXCoordinate;
		double ymin = mFieldInfo.minYCoordinate;
		double zmin = mFieldInfo.minZCoordinate;
		double xmax = mFieldInfo.maxXCoordinate;
		double ymax = mFieldInfo.maxYCoordinate;
		double zmax = mFieldInfo.maxZCoordinate;

		auto thisX = double(x) / double(mFieldInfo.XGridsize - 1) * (xmax - xmin) + xmin;
		auto thisY = double(y) / double(mFieldInfo.YGridsize - 1) * (ymax - ymin) + ymin;
		auto thisZ = double(z) / double(mFieldInfo.ZGridsize - 1) * (zmax - zmin) + zmin;

		return DTG::EigenVector<3, double>({ thisX, thisY, thisZ });
	}


	DTG::EigenVector<4, double> DTG::IField3D::convertGridIndex2PhysicalCoordinates(int x, int y, int z, int t) const
	{
		double xmin = mFieldInfo.minXCoordinate;
		double ymin = mFieldInfo.minYCoordinate;
		double zmin = mFieldInfo.minZCoordinate;
		double xmax = mFieldInfo.maxXCoordinate;
		double ymax = mFieldInfo.maxYCoordinate;
		double zmax = mFieldInfo.maxZCoordinate;

		auto thisX = double(x) / double(mFieldInfo.XGridsize - 1) * (xmax - xmin) + xmin;
		auto thisY = double(y) / double(mFieldInfo.YGridsize - 1) * (ymax - ymin) + ymin;
		auto thisZ =mFieldInfo.ZGridsize - 1>0? double(z) / double(mFieldInfo.ZGridsize - 1) * (zmax - zmin) + zmin :zmin;


		auto thisT=mFieldInfo.minTime + t* mFieldInfo.Getdt();

		return DTG::EigenVector<4, double>({ thisX, thisY, thisZ ,thisT});
	}


	

	std::string IField3D::getPrintOutGridInfo() const
	{
		std::ostringstream oss;
		oss << "Grid Size:";
		oss << "  X: " << mFieldInfo.XGridsize << ",";
		oss << "  Y: " << mFieldInfo.YGridsize << ",";
		oss << "  Z: " << mFieldInfo.ZGridsize << ",";
		oss << "  Time Steps: " << mFieldInfo.numberOfTimeSteps << ",";
		oss << "  Total Data Points: " << NumberOfDataPoints << "\n";
		return oss.str();
	}

	std::string IField3D::getPrintOutInfo() const
	{
		std::ostringstream oss;
		oss << "Spatial Domain:";
		oss << "  X: [" << mFieldInfo.minXCoordinate << ", " << mFieldInfo.maxXCoordinate << "],";
		oss << "  Y: [" << mFieldInfo.minYCoordinate << ", " << mFieldInfo.maxYCoordinate << "],";
		oss << "  Z: [" << mFieldInfo.minZCoordinate << ", " << mFieldInfo.maxZCoordinate << "].";
		oss << "Grid Size:";
		oss << "  X: " << mFieldInfo.XGridsize << ",";
		oss << "  Y: " << mFieldInfo.YGridsize << ",";
		oss << "  Z: " << mFieldInfo.ZGridsize << ".";
		oss << "Temporal Domain:";
		oss << "  Time: [" << mFieldInfo.minTime << ", " << mFieldInfo.maxTime << "]";
		oss << "  Time Steps: " << mFieldInfo.numberOfTimeSteps << "";
		oss << "Derived Info:";
		oss << "  Inverse Grid Spacing (X, Y, Z): ("
			<< mInverse_OneXGridInterval << ", "
			<< mInverse_OneYGridInterval << ", "
			<< mInverse_OneZGridInterval << ")";
		oss << "  Inverse dt: " << mInverse_dt << "";
		oss << "  XY Grid Size: " << mXYGrid << "";
		oss << "  Total Data Points: " << NumberOfDataPoints << "";
		return oss.str();
	}

	Eigen::Vector3d IField3D::GetSpatialDxDyDz() const
	{
		auto DomainSize = GetDomainSize();
		Eigen::Vector3d spacing ;
		//if (mFieldInfo.XGridsize<=1||mFieldInfo.YGridsize<=1||mFieldInfo.ZGridsize<=1)
		//{
		//	LOG_W("GetSpatialDxDyDz called with some grid resolution=1.");
		//}
		spacing.x() = mFieldInfo.XGridsize > 1 ? DomainSize.x() / (double)(mFieldInfo.XGridsize - 1) :DomainSize.x() ;
		spacing.y() = mFieldInfo.YGridsize > 1 ? DomainSize.y() / (double)(mFieldInfo.YGridsize - 1) :DomainSize.y() ;
		spacing.z() = mFieldInfo.ZGridsize > 1 ? DomainSize.z() / (double)(mFieldInfo.ZGridsize - 1) :DomainSize.z() ;
		return spacing;
	}

	double IField3D::convertTimeStep2PhysicalTime(int timeStep) const
	{
		timeStep=std::clamp(timeStep,0,GetNumberOfTimeSteps()-1);
		
		return  GetNumberOfTimeSteps()>1? mFieldInfo.minTime + timeStep * mFieldInfo.Getdt():mFieldInfo.minTime ;
	}

	int IField3D::convertPhysicalTimeRoundedTimeStep(double time) const
	{

		auto floatTimeIndex=(time-mFieldInfo.minTime)*mInverse_dt;
		int timeStep=std::round(floatTimeIndex);
		timeStep=std::clamp(timeStep,0,GetNumberOfTimeSteps()-1);
		return  timeStep;
	}
} // namespace DTG

	

