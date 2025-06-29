
#include "IsoSurface.h"
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkConnectivityFilter.h>
#include <vtkContourFilter.h>
#include "myVertexStructDefs.h"

#include <queue>
#include <unordered_set>
#include <random>

#include <vtkFillHolesFilter.h>
#include <vtkSelectEnclosedPoints.h>
#include <vtkFloatArray.h>
#include <vtkCleanPolyData.h>
#include "vtkPolyDataNormals.h"
#include "vtkCurvatures.h"
#include "logSystem/log.h"
#include "vtkTriangleFilter.h"
#include "vtkPointsProjectedHull.h"
#include <vtkPolygon.h>

namespace DTG{





void DTG::convert3DScalarFieldIntSliceToVTKImageData(const DiscreteScalarField3D<float>& scalarField, int timeIndex, vtkImageData* image) {

	int nx = scalarField.mFieldInfo.XGridsize;
	int ny = scalarField.mFieldInfo.YGridsize;
	int nz = scalarField.mFieldInfo.ZGridsize;

	double range_x = scalarField.mFieldInfo.maxXCoordinate - scalarField.mFieldInfo.minXCoordinate;
	double range_y = scalarField.mFieldInfo.maxYCoordinate - scalarField.mFieldInfo.minYCoordinate;
	double range_z = scalarField.mFieldInfo.maxZCoordinate - scalarField.mFieldInfo.minZCoordinate;

	const std::vector<float>& vecdata = scalarField.getSliceDataConst(timeIndex);

	image->SetDimensions(nx, ny, nz);
	image->AllocateScalars(VTK_FLOAT, 1);
	image->SetSpacing(range_x / (nx - 1), range_y / (ny - 1), range_z / (nz - 1));
	image->SetOrigin(scalarField.mFieldInfo.minXCoordinate, scalarField.mFieldInfo.minYCoordinate, scalarField.mFieldInfo.minZCoordinate);

	float* dataPtr = static_cast<float*>(image->GetScalarPointer());
	std::copy(vecdata.begin(), vecdata.end(), dataPtr);
}

void DTG::convert3DScalarFieldInterpolatedSliceToVTKImageData(const DiscreteScalarField3D<float>& scalarField, double physicalTime, vtkImageData* image)
{
	int nx = scalarField.mFieldInfo.XGridsize;
	int ny = scalarField.mFieldInfo.YGridsize;
	int nz = scalarField.mFieldInfo.ZGridsize;

	double range_x = scalarField.mFieldInfo.maxXCoordinate - scalarField.mFieldInfo.minXCoordinate;
	double range_y = scalarField.mFieldInfo.maxYCoordinate - scalarField.mFieldInfo.minYCoordinate;
	double range_z = scalarField.mFieldInfo.maxZCoordinate - scalarField.mFieldInfo.minZCoordinate;

	const std::vector<float> vecdata = scalarField.getSliceAtPhysicalTime(physicalTime);

	image->SetDimensions(nx, ny, nz);
	image->AllocateScalars(VTK_FLOAT, 1);
	image->SetSpacing(range_x / (nx - 1), range_y / (ny - 1), range_z / (nz - 1));
	image->SetOrigin(scalarField.mFieldInfo.minXCoordinate, scalarField.mFieldInfo.minYCoordinate, scalarField.mFieldInfo.minZCoordinate);

	float* dataPtr = static_cast<float*>(image->GetScalarPointer());
	std::copy(vecdata.begin(), vecdata.end(), dataPtr);
}



	

const std::vector<DTG::PureMeshGeometryVertexStruct>& DTG::IsoSurfaceAnimationTrack::GetFrameMesh(const double time)
{
	double t = std::clamp(time, tmin, tmax);

	double deltaT = (tmax - tmin) / (frameCount - 1);

	double frame_f = (t- tmin) / deltaT;
	int frame1 = static_cast<int>(floor(frame_f));
	int frame2 = std::min(frame1 + 1, frameCount - 1);
	double alpha = frame_f - frame1;
	frame1 = std::min(frame1, frameCount - 1);
	int resultFrameId = alpha > 0.5 ? frame2 : frame1;

	//interpolation is wrong! vertex of iso-surface has no correspondence between frames
	////#pragma omp parallel for
	//		for (int i = 0; i < static_cast<int>(result.size()); ++i) {
	//			const auto& v1 = mesh_surface_t[frame1][i];
	//			const auto& v2 = mesh_surface_t[frame2][i];
	//
	//			result[i]=interpolateVertex( v1,v2, (1.0 - alpha));
	//
	//		}
	resultFrameId=std::clamp(resultFrameId,0,frameCount-1);
	return 		iso_surface_t[resultFrameId]->VaoData_;
}

vtkSmartPointer<vtkPolyData> DTG::IsoSurfaceUtils::extractIsoSurfaceFullField(	DTG::DiscreteScalarField3D<float>* scalarfield3DPtr,	double isoValue,	const double timeToqueryScalarField,	bool forceClosed)
{
	STARTTIMER(MARCHING_CUBE_ISO_SURFACE_FULL)

		vtkNew<vtkImageData> image;
	convert3DScalarFieldInterpolatedSliceToVTKImageData(*scalarfield3DPtr, timeToqueryScalarField, image);

	vtkNew<vtkContourFilter> contourFilter;
	contourFilter->SetInputData(image);
	contourFilter->SetValue(0, isoValue);
	contourFilter->ComputeScalarsOff();
	contourFilter->ComputeGradientsOff();
	contourFilter->Update();

	vtkPolyData*  isoSurface = contourFilter->GetOutput();

	STOPTIMER(MARCHING_CUBE_ISO_SURFACE_FULL)

		if (forceClosed)
		{
			auto closeIsoSurface = ensureClosedIsoSurface(isoSurface);
			return closeIsoSurface;
		}
		else
		{
			return isoSurface;
		}
}
 vtkSmartPointer<vtkPolyData>  DTG::IsoSurfaceUtils::extractIsoSurfaceAtSeedingPosition(DTG::DiscreteScalarField3D<float>* scalarfield3DPtr, double isoValue, const Eigen::Vector3d& seedPosition, const double timeToqueryScalarField,bool forceClosed) {

	 STARTTIMER(MARCHING CUBE ISO-Surface)
	// apply contour filter
	vtkNew<vtkImageData> image;
	convert3DScalarFieldInterpolatedSliceToVTKImageData(*scalarfield3DPtr, timeToqueryScalarField, image);

	vtkContourFilter* contourFilter = vtkContourFilter::New();
	contourFilter->SetInputData(image);
	contourFilter->SetValue(0, isoValue);
	contourFilter->ComputeScalarsOff();
	contourFilter->ComputeGradientsOff();
	contourFilter->Update();

	// apply connectivity filter
	vtkNew<vtkConnectivityFilter> connectivity;
	connectivity->SetInputData(contourFilter->GetOutput());
	connectivity->SetExtractionModeToClosestPointRegion();

	double x = seedPosition.x();
	double y = seedPosition.y();
	double z = seedPosition.z();

	connectivity->SetClosestPoint(x, y, z);
	connectivity->Update();

	int numberOfExtractedRegions = connectivity->GetNumberOfExtractedRegions();
	vtkPolyData* isoSurface = connectivity->GetPolyDataOutput();

	STOPTIMER(MARCHING CUBE ISO-Surface)

    if (forceClosed)
    {
    auto closeIsoSurface=ensureClosedIsoSurface(isoSurface);
    return closeIsoSurface;
    }
    else
    {
        return isoSurface;
    }

}
 vtkSmartPointer<vtkPolyData>  DTG::IsoSurfaceUtils::extractIsoSurfaceAtSeedingPositionNoInterpolation(DTG::DiscreteScalarField3D<float>* scalarfield3DPtr, double isoValue, const Eigen::Vector3d& seedPosition, const int timeIndexToqueryScalarField, bool forceClosed) {

	 // apply contour filter
	 vtkNew<vtkImageData> image;
	 convert3DScalarFieldIntSliceToVTKImageData(*scalarfield3DPtr, timeIndexToqueryScalarField, image);

	 vtkContourFilter* contourFilter = vtkContourFilter::New();
	 contourFilter->SetInputData(image);
	 contourFilter->SetValue(0, isoValue);
	 contourFilter->ComputeScalarsOff();
	 contourFilter->ComputeGradientsOff();
	 contourFilter->Update();

	 // apply connectivity filter
	 vtkNew<vtkConnectivityFilter> connectivity;
	 connectivity->SetInputData(contourFilter->GetOutput());
	 connectivity->SetExtractionModeToClosestPointRegion();

	 double x = seedPosition.x();
	 double y = seedPosition.y();
	 double z = seedPosition.z();

	 connectivity->SetClosestPoint(x, y, z);
	 connectivity->Update();

	 int numberOfExtractedRegions = connectivity->GetNumberOfExtractedRegions();
	 vtkPolyData* isoSurface = connectivity->GetPolyDataOutput();

	 if (forceClosed)
	 {
		 auto closeIsoSurface = ensureClosedIsoSurface(isoSurface);
		 return closeIsoSurface;
	 }
	 else
	 {
		 return isoSurface;
	 }

 }

// Modify your existing iso-surface generation to ensure it's closed
 vtkSmartPointer<vtkPolyData> DTG::IsoSurfaceUtils::ensureClosedIsoSurface(vtkPolyData* isoSurface) {
	// Create a filter to fill holes
	vtkNew<vtkFillHolesFilter> fillHolesFilter;
	fillHolesFilter->SetInputData(isoSurface);
	// Set the maximum size of holes to fill
	// Adjust this value based on your needs
	fillHolesFilter->SetHoleSize(100.0);
	fillHolesFilter->Update();

	return fillHolesFilter->GetOutput();
}




// Test whether a point is inside the closed iso-surface but this is inefficnet
bool DTG::IsoSurfaceUtils::isPointInside(const vtkSmartPointer<vtkPolyData>& closedSurface,const Eigen::Vector3f& point) {
	// Create a point to test
	vtkNew<vtkPoints> testPoint;
	testPoint->InsertNextPoint(point.x(), point.y(), point.z());

	vtkNew<vtkPolyData> pointPolyData;
	pointPolyData->SetPoints(testPoint);

	// Create the point containment filter
	vtkNew<vtkSelectEnclosedPoints> pointsFilter;
	pointsFilter->SetSurfaceData(closedSurface);
	pointsFilter->SetInputData(pointPolyData);
	// Enable faster but less accurate testing
	pointsFilter->SetTolerance(1e-6);
	pointsFilter->Update();

	return pointsFilter->IsInside(0);

}

// Test whether multiple points are inside the closed iso-surface
std::vector<bool> DTG::IsoSurfaceUtils::arePointsInside(const vtkSmartPointer<vtkPolyData>& closedSurface,const std::vector<Eigen::Vector3f>& points)
{
	if (points.empty()) {
		return {};
	}

	// Create points to test in batch
	vtkNew<vtkPoints> testPoints;
	testPoints->SetNumberOfPoints(points.size());


	for (int i = 0; i < static_cast<int>(points.size()); i++) {
		testPoints->SetPoint(i, points[i].x(), points[i].y(), points[i].z());
	}

	vtkNew<vtkPolyData> pointsPolyData;
	pointsPolyData->SetPoints(testPoints);

	// Create the point containment filter
	vtkNew<vtkSelectEnclosedPoints> pointsFilter;
	pointsFilter->SetSurfaceData(closedSurface);
	pointsFilter->SetInputData(pointsPolyData);
	pointsFilter->SetTolerance(1e-6);

	// Enable parallel processing if available
	pointsFilter->Update();

	// Get results for all points
	std::vector<bool> results;
	results.reserve(points.size());

	vtkDataArray* insideArray = pointsFilter->GetOutput()->GetPointData()->GetArray("SelectedPoints");
	if (insideArray) {
		for (vtkIdType i = 0; i < insideArray->GetNumberOfTuples(); i++) {
			results.push_back(insideArray->GetTuple1(i) == 1);
		}
	}

	return results;
}

void IsoSurfaceUtils::generateInsideOutsideField(const vtkSmartPointer<vtkPolyData>& closedSurface, DTG::DiscreteScalarField3D<float>& scalarFieldInputOut)
{
     std::vector<Eigen::Vector3f> test_points;
     test_points.reserve(scalarFieldInputOut.GetNumberOfDataPoints());

	const double dx = (scalarFieldInputOut.mFieldInfo.maxXCoordinate - scalarFieldInputOut.mFieldInfo.minXCoordinate) / (scalarFieldInputOut.mFieldInfo.XGridsize - 1);
	const double dy = (scalarFieldInputOut.mFieldInfo.maxYCoordinate - scalarFieldInputOut.mFieldInfo.minYCoordinate) / (scalarFieldInputOut.mFieldInfo.YGridsize - 1);
	const double dz = (scalarFieldInputOut.mFieldInfo.maxZCoordinate - scalarFieldInputOut.mFieldInfo.minZCoordinate) / (scalarFieldInputOut.mFieldInfo.ZGridsize - 1);

	for (int idz = 0; idz < scalarFieldInputOut.mFieldInfo.ZGridsize; idz++) {
		float z = scalarFieldInputOut.mFieldInfo.minZCoordinate + dz * idz;
		for (int idy = 0; idy < scalarFieldInputOut.mFieldInfo.YGridsize; idy++) {
			float y = scalarFieldInputOut.mFieldInfo.minYCoordinate + dy * idy;
			for (int idx = 0; idx < scalarFieldInputOut.mFieldInfo.XGridsize; idx++) {
				float x = scalarFieldInputOut.mFieldInfo.minXCoordinate + dx * idx;
				Eigen::Vector3f pos3d({ x, y, z });
                test_points.emplace_back(pos3d);
			}
		}
	}
    auto test_result=arePointsInside(closedSurface,test_points);
    uint64_t linear_id=0U;
	for (int idz = 0; idz < scalarFieldInputOut.mFieldInfo.ZGridsize; idz++) {
		for (int idy = 0; idy < scalarFieldInputOut.mFieldInfo.YGridsize; idy++) {
			for (int idx = 0; idx < scalarFieldInputOut.mFieldInfo.XGridsize; idx++) {
                bool test_inside_thisPoint=test_result[linear_id];
                float valueToset=test_inside_thisPoint? 1.0f:0.0f;
                scalarFieldInputOut.SetValue(idx,idy,idz,0,valueToset);
                linear_id++;

			}

		}
	}


}





std::vector<int> ISoSurface::segmentationIsoSurfaceWithUserInput(const std::vector<Eigen::Vector3f>& userPoints, float curvatureWeight /*= 1.0f*/, float normalWeight /*= 1.0f*/, float distanceWeight /*= 1.0f*/,
	float threshold)
{
	if (! mVTkIsoSurfaceMesh_|| userPoints.empty()) {
		return {};
	}
	IsoSurfaceUtils::computeSurfaceFeatures(mVTkIsoSurfaceMesh_);

	std::vector<vtkIdType> seedPoints = findNearestVertices(userPoints);

	return 	IsoSurfaceUtils::regionGrowingSegmentation(mVTkIsoSurfaceMesh_, seedPoints, curvatureWeight, normalWeight, distanceWeight,threshold);
}





void IsoSurfaceUtils::computeSurfaceFeatures(vtkSmartPointer<vtkPolyData>& surface)
{
	vtkNew<vtkCurvatures> curvatures;
	curvatures->SetInputData(surface);
	curvatures->SetCurvatureTypeToGaussian();
	curvatures->Update();

	vtkNew<vtkPolyDataNormals> normals;
	normals->SetInputData(surface);
	normals->ComputePointNormalsOn();
	normals->ConsistencyOn();
	normals->Update();

	surface->GetPointData()->AddArray(curvatures->GetOutput()->GetPointData()->GetArray("Gauss_Curvature"));
	surface->GetPointData()->SetNormals(normals->GetOutput()->GetPointData()->GetNormals());
}

std::vector<vtkIdType> ISoSurface::findNearestVertices(const std::vector<Eigen::Vector3f>& userPoints)
{

	if (locator==nullptr)
	{
	locator=vtkNew<vtkKdTreePointLocator>();
	locator->SetDataSet(this->mVTkIsoSurfaceMesh_);
	locator->BuildLocator();
	}

	std::vector<vtkIdType> nearestPoints;
	nearestPoints.reserve(userPoints.size());

	for (const auto& point : userPoints) {
		double p[3] = { point.x(), point.y(), point.z() };
		vtkIdType id = locator->FindClosestPoint(p);
		nearestPoints.push_back(id);
	}

	return nearestPoints;
}

static void getConnectedVertices(
	vtkSmartPointer<vtkPolyData>& surface,
	vtkIdType pointId,
	vtkIdList* connectedVertices)
{
	vtkNew<vtkIdList> cellIds;
	surface->GetPointCells(pointId, cellIds);

	std::unordered_set<vtkIdType> vertexSet;
	for (vtkIdType i = 0; i < cellIds->GetNumberOfIds(); ++i) {
		vtkNew<vtkIdList> pointIds;
		surface->GetCellPoints(cellIds->GetId(i), pointIds);

		for (vtkIdType j = 0; j < pointIds->GetNumberOfIds(); ++j) {
			vtkIdType vertexId = pointIds->GetId(j);
			if (vertexId != pointId) {
				vertexSet.insert(vertexId);
			}
		}
	}

	connectedVertices->Reset();
	for (vtkIdType id : vertexSet) {
		connectedVertices->InsertNextId(id);
	}
}
static double computeSimilarity(
	vtkSmartPointer<vtkPolyData>& surface,
	vtkIdType user_pick_Id,//the id of vertex which is clicked by user. depth filter are apply centered at this vertex.
	vtkIdType id1,
	vtkIdType id2,
	vtkDataArray* curvatures,
	vtkDataArray* normals,
	float curvatureWeight,
	float normalWeight,
	float distanceFilter)
{
	double p1[3], p2[3];
	surface->GetPoint(user_pick_Id, p1);
	surface->GetPoint(id2, p2);
	double distance = std::sqrt(std::pow(p1[0] - p2[0], 2) +
		std::pow(p1[1] - p2[1], 2) +
		std::pow(p1[2] - p2[2], 2));
	if (distance < distanceFilter) {
		//double distanceSimilarity;

		double curv1 = curvatures->GetTuple1(id1);
		double curv2 = curvatures->GetTuple1(id2);
		double curvSimilarity = 1.0 / (1.0 + std::abs(curv1 - curv2));

		double normal1[3], normal2[3];
		normals->GetTuple(id1, normal1);
		normals->GetTuple(id2, normal2);
		double dotProduct = normal1[0] * normal2[0] + normal1[1] * normal2[1] + normal1[2] * normal2[2];
		double normalSimilarity = (dotProduct + 1.0) / 2.0;


		/*distanceSimilarity = 1.0 - distance / distanceFilter;
		constexpr double distanceWeight = 1.0f;*/
		double totalWeight = curvatureWeight + normalWeight /*+ distanceWeight*/;

		return (curvSimilarity * curvatureWeight + normalSimilarity * normalWeight /*+distanceSimilarity * distanceWeight*/) / totalWeight;
	}
	else {
		return -10000.0;
	}
}



std::vector<int> IsoSurfaceUtils::regionGrowingSegmentation(vtkSmartPointer<vtkPolyData>& surface, const std::vector<vtkIdType>& seedPoints, float curvatureWeight, float normalWeight, float distanceFilter,const float threshold)
{
	vtkIdType numPoints = surface->GetNumberOfPoints();
	std::vector<int> segmentation(numPoints, 0);
	std::vector<bool> visited(numPoints, false);

	vtkDataArray* curvatures = surface->GetPointData()->GetArray("Gauss_Curvature");
	vtkDataArray* normals = surface->GetPointData()->GetNormals();

	for (size_t i = 0; i < seedPoints.size(); ++i) {
		std::queue<vtkIdType> queue;
		queue.push(seedPoints[i]);
		segmentation[seedPoints[i]] = (i+1);
		visited[seedPoints[i]] = true;
		auto pickPointId=seedPoints[i];

		while (!queue.empty()) {
			vtkIdType currentId = queue.front();
			queue.pop();

			double currentCurv = curvatures->GetTuple1(currentId);
			double currentNormal[3];
			normals->GetTuple(currentId, currentNormal);
			double currentPoint[3];
			surface->GetPoint(currentId, currentPoint);

			vtkNew<vtkIdList> neighborIds;

			getConnectedVertices(surface, currentId, neighborIds);

			for (vtkIdType j = 0; j < neighborIds->GetNumberOfIds(); ++j) {
				vtkIdType neighborId = neighborIds->GetId(j);
				if (visited[neighborId]) continue;

				double similarity = computeSimilarity(
					surface, pickPointId, currentId, neighborId,
					curvatures, normals,
					curvatureWeight, normalWeight, distanceFilter
				);

				if (similarity > threshold) {
					queue.push(neighborId);
					segmentation[neighborId] = (i+1);
					visited[neighborId] = true;
				}
			}
		}
	}

	return segmentation;
}




void ISoSurface::AssignVertexAttributeFromScalarField(DiscreteScalarField3D<float>* scalarField_appending, float timeToquery)
{
	if (scalarField_appending)
	{
		for (auto& point : VaoData_) {
			point.uv[0] = scalarField_appending->InterpolateValuebyPos(point.position[0], point.position[1], point.position[2], timeToquery);
		}

	}
}

void ISoSurface::AssignVertexAttributeFromSegmentation(const std::vector<Eigen::Vector3f>& userPoints, float curvatureWeight /*= 1.0f*/, float normalWeight /*= 1.0f*/, float distanceWeight /*= 1.0f*/, float threshold/*=0.5f*/)
{
	auto segmentationLabel = segmentationIsoSurfaceWithUserInput(userPoints, curvatureWeight, normalWeight, distanceWeight, threshold);
	const auto size_vertex = VaoData_.size();
	std::vector<Eigen::Vector3f>aoi_points;
	aoi_points.reserve(size_vertex / 2);
	for (int v = 0; v < size_vertex; v++) {
		int pointIdInVTK = MapVAOIndex2VTK_[v];
		VaoData_[v].uv[1] = (float)segmentationLabel[pointIdInVTK];
		aoi_points.emplace_back(VaoData_[v].position[0], VaoData_[v].position[1], VaoData_[v].position[2]);
	}


	//aoi_cube_tree->buildCubeTreeFromPointsOfInteresting(aoi_points,0.02,3);
}




void ISoSurface::updateVtkPoly2VaoRenderingData()
{
	if (mVTkIsoSurfaceMesh_)
	{


		vtkCellArray* triangles = mVTkIsoSurfaceMesh_->GetPolys();
		vtkPoints* points = mVTkIsoSurfaceMesh_->GetPoints();

		// generate normals
		vtkNew<vtkPolyDataNormals> normalsGenerator;
		normalsGenerator->SetInputData(mVTkIsoSurfaceMesh_);
		normalsGenerator->ConsistencyOn(); // Enforce consistent normals (optional)
		normalsGenerator->SplittingOff(); // Prevent sharp edges from being split (optional)
		normalsGenerator->Update();
		vtkFloatArray* normals = vtkFloatArray::SafeDownCast(mVTkIsoSurfaceMesh_->GetPointData()->GetNormals());

		vtkNew<vtkIdList> idList;
		triangles->InitTraversal();
		LOG_D("surface component number of triangles: %i", (int)triangles->GetNumberOfCells());
		VaoData_.clear();
		VaoData_.reserve(triangles->GetNumberOfCells() * 3);

		//std::vector<int> indicesMapVtk2VAo;//
		MapVAOIndex2VTK_.clear();
		MapVAOIndex2VTK_.reserve(triangles->GetNumberOfCells() * 4);
		while (triangles->GetNextCell(idList)) {
			for (vtkIdType i = 0; i < idList->GetNumberOfIds(); i++) {
				vtkIdType pointIdInVTK = idList->GetId(i);
				double point[3];
				points->GetPoint(pointIdInVTK, point);

				PureMeshGeometryVertexStruct p;
				p.position[0] = point[0];
				p.position[1] = point[1];
				p.position[2] = point[2];

				p.normal[0] = 0;
				p.normal[1] = 0;
				p.normal[2] = 1;

				p.uv[0] = timeToqueryScalarField;

				if (normals) {
					double normal[3];
					normals->GetTuple(pointIdInVTK, normal);
					p.normal[0] = normal[0];
					p.normal[1] = normal[1];
					p.normal[2] = normal[2];
				}
				VaoData_.push_back(p);
				MapVAOIndex2VTK_.emplace_back(pointIdInVTK);
			}
		}
	}
}


vtkSmartPointer<vtkPolyData> extractSegmentationSubIsoSurface( vtkPolyData* isoSurfaceMesh,const std::vector<int>& segmentation,int segmentId)
{
	vtkNew<vtkPolyData> segmentationSurface;
	vtkNew<vtkPoints> newPoints;
	vtkNew<vtkCellArray> newCells;

	std::map<vtkIdType, vtkIdType> oldToNewPointId;

	vtkIdType numCells = isoSurfaceMesh->GetNumberOfCells();
	for (vtkIdType cellId = 0; cellId < numCells; ++cellId)
	{
		vtkCell* cell = isoSurfaceMesh->GetCell(cellId);
		bool cellBelongs = true;
		vtkIdList* cellPointIds = cell->GetPointIds();
		for (vtkIdType i = 0; i < cellPointIds->GetNumberOfIds(); ++i)
		{
			vtkIdType ptId = cellPointIds->GetId(i);
			if (segmentation[ptId] != segmentId)
			{
				cellBelongs = false;
				break;
			}
		}
		if (!cellBelongs)
			continue;

		vtkNew<vtkIdList> newCellPointIds;
		for (vtkIdType i = 0; i < cellPointIds->GetNumberOfIds(); ++i)
		{
			vtkIdType oldPtId = cellPointIds->GetId(i);
			vtkIdType newPtId;
			if (oldToNewPointId.find(oldPtId) == oldToNewPointId.end())
			{
				double p[3];
				isoSurfaceMesh->GetPoint(oldPtId, p);
				newPtId = newPoints->InsertNextPoint(p);
				oldToNewPointId[oldPtId] = newPtId;
			}
			else
			{
				newPtId = oldToNewPointId[oldPtId];
			}
			newCellPointIds->InsertNextId(newPtId);
		}
		newCells->InsertNextCell(newCellPointIds);
	}

	segmentationSurface->SetPoints(newPoints);
	segmentationSurface->SetPolys(newCells);
	return segmentationSurface;
}



vtkSmartPointer<vtkPolyData>  ISoSurface::computeHighlightClosedSubIsoSurface(const std::vector<int>& segmentation, int segmentId)
{
	vtkSmartPointer<vtkPolyData> sub_iso_surface=extractSegmentationSubIsoSurface(this->mVTkIsoSurfaceMesh_,segmentation,segmentId);
	vtkNew<vtkFillHolesFilter> fillHoles;
	fillHoles->SetInputData(sub_iso_surface);
	fillHoles->SetHoleSize(0.50);
	fillHoles->Update();
	vtkSmartPointer<vtkPolyData> closedSegmentationSurface = fillHoles->GetOutput();
	//render to verify
	this->mVTkIsoSurfaceMesh_=closedSegmentationSurface;
	this->updateVtkPoly2VaoRenderingData();

	//vtkSmartPointer<vtkSelectEnclosedPoints> selectEnclosedPoints=vtkSelectEnclosedPoints::New();
	//selectEnclosedPoints->SetSurfaceData(closedSegmentationSurface); // closed surface polydata
	//selectEnclosedPoints->SetTolerance(0.0);  // adjust tolerance as needed
	//selectEnclosedPoints->Update();
	// Compute bounding box for quick initial check
	/*double bounds[6];
	closedSegmentationSurface->GetBounds(bounds);*/
	// Lambda function for point inclusion test
	//return [selectEnclosedPoints, bounds, this](float x, float y, float z) {
	//	// Quick bounding box check first
	//	if (x < bounds[0] || x > bounds[1] ||
	//		y < bounds[2] || y > bounds[3] ||
	//		z < bounds[4] || z > bounds[5]) {
	//		return false;
	//	}

	//	double p[3] = { x, y, z };
	//	bool isInside = selectEnclosedPoints->IsInsideSurface(p); // returns true if inside
	//	};
	return closedSegmentationSurface;
}


#if 0

void CubeOctTree::buildCubeTreeFromPointsOfInteresting(std::vector<Eigen::Vector3f>& userPoints, const double cubeSize, int CubeGridSize)
{
	if (userPoints.empty()) return;

	// Step 1: Determine the bounding box of the points
	Eigen::Vector3f minPoint = userPoints[0];
	Eigen::Vector3f maxPoint = userPoints[0];
	for (const auto& point : userPoints) {
		minPoint = minPoint.cwiseMin(point);
		maxPoint = maxPoint.cwiseMax(point);
	}

	// Step 2: Create the root cube covering the entire bounding box
	float rootSize = std::max({ maxPoint.x() - minPoint.x(), maxPoint.y() - minPoint.y(), maxPoint.z() - minPoint.z() });
	root = std::make_unique<Node>(minPoint.x(), minPoint.y(), minPoint.z(), rootSize, CubeGridSize);

	// Step 3: Recursively subdivide the space and assign points to cubes
	buildTree(root, userPoints, cubeSize, CubeGridSize);
}

void CubeOctTree::buildRenderingGeometry(DTG::VertexArrayObjectPrototying* ObjToHoldOutputGeometry)
{
	if (!root || !ObjToHoldOutputGeometry) return; // If the tree is empty, return
	// Start traversing the tree from the root
	traverseAndAppendCubes(ObjToHoldOutputGeometry, root.get());
	ObjToHoldOutputGeometry->commit();
}

void CubeOctTree::traverseAndAppendCubes(DTG::VertexArrayObjectPrototying* ObjToHoldOutputGeometry, Node* node)
{
	if (!node) return;

	// Append the cube to the rendering geometry if it is leaf node
	if (IsleafNode(node))
	{
		// Calculate the center and size of the current cube
		float centerX = node->cube->getXMin() + node->cube->getPhysicalSize() / 2.0f;
		float centerY = node->cube->getYMin() + node->cube->getPhysicalSize() / 2.0f;
		float centerZ = node->cube->getZMin() + node->cube->getPhysicalSize() / 2.0f;
		glm::vec3 center(centerX, centerY, centerZ);
		float length = node->cube->getPhysicalSize();
		ObjToHoldOutputGeometry->appendCubeWithoutCommit(center, length);
	}

	// Recursively process child cubes
	for (const auto& child : node->children) {
		traverseAndAppendCubes(ObjToHoldOutputGeometry, child.get());
	}
}

void CubeOctTree::buildTree(std::unique_ptr<Node>& node, std::vector<Eigen::Vector3f>& points, const double cubeSize, int CubeGridSize)
{
	// If the current cube is smaller than or equal to the desired cube size, stop subdivision
	if (node->cube->getPhysicalSize() <= cubeSize) {
		node->points = std::move(points); // Assign points to this leaf node
		return;
	}

	// Step 4: Subdivide the cube into 8 children (octree)
	float halfSize = node->cube->getPhysicalSize() / 2.0f;
	auto cubeMin = node->cube->GetSpatialMin();
	float xmin = cubeMin.x();
	float ymin = cubeMin.y();
	float zmin = cubeMin.z();

	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			for (int k = 0; k < 2; ++k) {
				float childXmin = xmin + i * halfSize;
				float childYmin = ymin + j * halfSize;
				float childZmin = zmin + k * halfSize;
				node->children.push_back(std::make_unique<Node>(childXmin, childYmin, childZmin, halfSize, CubeGridSize));
			}
		}
	}

	// Step 5: Assign points to the appropriate child cubes
	for (const auto& point : points) {
		for (const auto& child : node->children) {
			if (isPointInCube(point, *child->cube)) {
				child->points.push_back(point);
			}
		}
	}

	// Step 6: Recursively build the tree for each child
	for (auto& child : node->children) {
		buildTree(child, child->points, cubeSize, CubeGridSize);
	}
}

#endif

}//namespace dtg
