
#ifndef _ISO_SURFACE_3D_H_
#define _ISO_SURFACE_3D_H_
#pragma once
#include "myVertexStructDefs.h"
#include "PortToUsingModules.h"
#include "ScalarField3D.h"
#include "vtkImageData.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include <vtkKdTreePointLocator.h>
#include "VertexArrayObject.h"
#include "Discrete3DFlowField.h"
#include "MeshIO.h"
#include <memory>

namespace DTG {

	static void convert3DScalarFieldIntSliceToVTKImageData(const DiscreteScalarField3D<float>& scalarField, int timeIndex, vtkImageData* image);
	static void convert3DScalarFieldInterpolatedSliceToVTKImageData(const DiscreteScalarField3D<float>& scalarField, double physicalTime, vtkImageData* image);
	class IsoSurfaceUtils {
	public:
		//gen iso surface
		static  vtkSmartPointer<vtkPolyData>  extractIsoSurfaceAtSeedingPosition(DTG::DiscreteScalarField3D<float>* scalarfield3DPtr, double isoValue, const Eigen::Vector3d& seedPosition, const double timeToqueryScalarField, bool closed);
		static  vtkSmartPointer<vtkPolyData>  extractIsoSurfaceAtSeedingPositionNoInterpolation(DTG::DiscreteScalarField3D<float>* scalarfield3DPtr, double isoValue, const Eigen::Vector3d& seedPosition, const int timeIndexToqueryScalarField, bool forceClosed); 
		static   vtkSmartPointer<vtkPolyData> extractIsoSurfaceFullField(	DTG::DiscreteScalarField3D<float>* scalarfield3DPtr,	double isoValue,	const double timeToqueryScalarField,	bool forceClosed);

		static  std::vector<bool> arePointsInside(const vtkSmartPointer<vtkPolyData>& closedSurface, const std::vector<Eigen::Vector3f>& points);
		static  void generateInsideOutsideField(const vtkSmartPointer<vtkPolyData>& closedSurface, DTG::DiscreteScalarField3D<float>& scalarfield3DInAndOut);
		static vtkSmartPointer<vtkPolyData> ensureClosedIsoSurface(vtkPolyData* isoSurface);
		static bool isPointInside(const vtkSmartPointer<vtkPolyData>& closedSurface, const Eigen::Vector3f& point);

		//segmentation of iso surface
		static void computeSurfaceFeatures(vtkSmartPointer<vtkPolyData>& surface);
		static std::vector<int> regionGrowingSegmentation(vtkSmartPointer<vtkPolyData>& surface, const std::vector<vtkIdType>& seedPoints, float curvatureWeight, float normalWeight, float distanceWeight,const float);

	};


	struct IrregularArea3D{
			IrregularArea3D()=default;
			IrregularArea3D(const std::function<bool(float,float,float)>& InAreaInsideFunc)
			:mAreaInsideFunc(InAreaInsideFunc){

			}

			inline bool InsideByFunc(const DTG::EigenVector<3,float>& pos){
				return  InsideByFunc(pos.x(),pos.y(),pos.z());
			}
			inline bool InsideByFunc(float x,float y,float z) {
				return  mAreaInsideFunc(x,y,z);
			}
			
			//could represent by funciont or binary mask
			std::function<bool(float,float,float)> mAreaInsideFunc=nullptr;
			double mTime;
	};


	struct ISoSurface {
	public:
		ISoSurface(vtkSmartPointer<vtkPolyData> IsosurfacePtr, float time,std::string parent_scalar_field_name) :mVTkIsoSurfaceMesh_(IsosurfacePtr),timeToqueryScalarField(time),parent_scalar_field_name(parent_scalar_field_name)
		{
			//aoi_cube_tree=std::make_unique<CubeOctTree>();
		}

		std::vector<vtkIdType> findNearestVertices(const std::vector<Eigen::Vector3f>& userPoints);
		std::vector<int> segmentationIsoSurfaceWithUserInput(const std::vector<Eigen::Vector3f>& userPoints, float curvatureWeight /*= 1.0f*/, float normalWeight /*= 1.0f*/, float distanceWeight /*= 1.0f*/, float threshold);

		void AssignVertexAttributeFromScalarField(DiscreteScalarField3D<float>* scalarField_appending, float timeToquery);


		void AssignVertexAttributeFromSegmentation(const std::vector<Eigen::Vector3f>& userPoints, float curvatureWeight = 1.0f, float normalWeight = 1.0f, float distanceWeight = 1.0f, float threshold=0.5f);


		void updateVtkPoly2VaoRenderingData();




		vtkSmartPointer<vtkPolyData> computeHighlightClosedSubIsoSurface(const std::vector<int>& segmentation, int segmentId);
		//variables 
		vtkSmartPointer<vtkPolyData>  mVTkIsoSurfaceMesh_ = nullptr;
		vtkSmartPointer<vtkKdTreePointLocator> locator = nullptr;
		float timeToqueryScalarField;
		std::vector<int>MapVAOIndex2VTK_;//multiple vao vertices might map to the sample vtk vertex
		std::vector<DTG::PureMeshGeometryVertexStruct> VaoData_;//PureMeshGeometryVertexStruct .uv[0] store time/attible from scalar field;PureMeshGeometryVertexStruct .uv[0] store clusterintg label or zero

		//temporal
		//std:: unique_ptr<CubeOctTree>aoi_cube_tree=nullptr;


		//variables for color coding
		std::string color_coding_scalar_field_name;
		std::string parent_scalar_field_name;

	};


	class IMeshAnimationTrack {
	public:
		IMeshAnimationTrack(double tmin, double tmax, int frameCount) : tmin(tmin), tmax(tmax), frameCount(frameCount) {}
		virtual ~IMeshAnimationTrack() = default;
		virtual double getFramedt() const { return frameCount > 1 ? (tmax - tmin) / (frameCount - 1) : 0.0; }
		virtual size_t getFrameCount() const { return frameCount; }
		virtual double getTMin() const { return tmin; }
		virtual double getTMax() const { return tmax; }
		virtual void clearFrames() = 0;

	protected:
		double tmin;
		double tmax;
		int frameCount;
	};

	class IsoSurfaceAnimationTrack : public IMeshAnimationTrack {
	public:
		IsoSurfaceAnimationTrack(double tmin, double tmax, int frameCount, std::vector<std::unique_ptr<ISoSurface>> iso_surface_t)
			: IMeshAnimationTrack(tmin, tmax, frameCount), iso_surface_t(std::move(iso_surface_t))
		{
		}

		std::vector<std::unique_ptr<ISoSurface>> iso_surface_t;
		const std::vector<DTG::PureMeshGeometryVertexStruct>& GetFrameMesh(double time);
		void clearFrames() override { iso_surface_t.clear(); }
	};

	






} // namespace DTG

#endif //_ISO_SURFACE_3D_H

