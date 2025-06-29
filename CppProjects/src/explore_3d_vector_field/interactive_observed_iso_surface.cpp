/**
 * @file interactive_observed_iso_surface.cpp
 * @brief Functions for computing observer-relative isosurface transformations
 * 
 * This file contains utilities for:
 * - Transforming scalar fields between lab frame and observed frame
 * - Computing observer-relative isosurfaces animation
 */

#include "ReferenceFrame3d.h"
#include "Discrete3DFlowField.h"
#include "ScalarField.h"

//########################################################
//######## Observer-Relative Isosurface Animation#########
//########################################################
namespace DTG { // base class IFlowField3d

/**
 * @brief Transform scalar field from lab frame to observed frame
 * @param scalarFieldInput Input scalar field in lab frame
 * @param rft Reference frame transformation
 * @param scalarFieldOutput Output scalar field in observed frame
 * @param shiftRegion Whether to shift the domain bounds based on transformation
 * 
 * Transforms a 3D scalar field between coordinate frames. When shiftRegion is true,
 * computes new domain bounds to accommodate the transformed field.
 */
void RftScalarField3D(const DiscreteScalarField3D<float>& scalarFieldInput,  ReferenceFrameTransformation3D rft,DiscreteScalarField3D<float>& scalarFieldOutput,bool shiftRegion){
	if (shiftRegion)
	{

		STARTTIMER(RFT_OfScalarField3D_shiftRegion);

		Eigen::Vector3i  gridsize = scalarFieldInput.GetSpatialGridSize();
		int numberOfTimeSamples = scalarFieldInput.GetNumberOfTimeSteps();
		double tmin = scalarFieldInput.GetMinTime();
		double tmax = scalarFieldInput.GetMaxTime();

		const Eigen::Vector3f minDomainBounds = scalarFieldInput.GetSpatialMin().cast<float>();
		const Eigen::Vector3f maxDomainBounds = scalarFieldInput.GetSpatialMax().cast<float>();

		const double dx = (maxDomainBounds(0) - minDomainBounds(0)) / (gridsize(0) - 1);
		const double dy = (maxDomainBounds(1) - minDomainBounds(1)) / (gridsize(1) - 1);
		const double dz = (maxDomainBounds(2) - minDomainBounds(2)) / (gridsize(2) - 1);


		//compute transformed domain range bouding box
		std::vector<Eigen::AlignedBox3f> threadBoundingBoxes(numberOfTimeSamples);
		std::vector<int> threadRange(numberOfTimeSamples);
		std::generate(threadRange.begin(), threadRange.end(), [n = 0]() mutable { return n++; });
		for_each(policy, threadRange.begin(), threadRange.end(), [&gridsize, &threadBoundingBoxes, &minDomainBounds, dx, dy, dz, &scalarFieldInput, &rft](int timeStep) {
				Eigen::AlignedBox3f localBox;
			float time = scalarFieldInput.convertTimeStep2PhysicalTime(timeStep);
			auto transformationAtTime = rft.getTransformationLabToObserved(time);
			for (int idz = 0; idz < gridsize(2); idz++) {
				float z = minDomainBounds(2) + dz * idz;
				for (int idy = 0; idy < gridsize(1); idy++) {
					float y = minDomainBounds(1) + dy * idy;
					for (int idx = 0; idx < gridsize(0); idx++) {
						float x = minDomainBounds(0) + dx * idx;
						Eigen::Vector3f Pos3d = { x,y,z };
						Eigen::Vector3f transformed_pos = transformationAtTime(Pos3d);
						localBox.extend(transformed_pos);
					}
				}
			}
					threadBoundingBoxes[timeStep] = localBox;
			}
		);
		// Merge all thread-local bounding boxes
		Eigen::AlignedBox3f finalBox;
		for (const auto& box : threadBoundingBoxes) {
			finalBox.extend(box);
		}
		//set value from bounding box
		scalarFieldOutput.mFieldInfo.minXCoordinate = finalBox.min().x();
		scalarFieldOutput.mFieldInfo.minYCoordinate = finalBox.min().y();
		scalarFieldOutput.mFieldInfo.minZCoordinate = finalBox.min().z();

		scalarFieldOutput.mFieldInfo.maxXCoordinate = finalBox.max().x();
		scalarFieldOutput.mFieldInfo.maxYCoordinate = finalBox.max().y();
		scalarFieldOutput.mFieldInfo.maxZCoordinate = finalBox.max().z();


		for_each(policy, threadRange.begin(), threadRange.end(), [&gridsize, &minDomainBounds, dx, dy, dz, &scalarFieldInput, &scalarFieldOutput, &rft](int timeStep) {
			float time = scalarFieldInput.convertTimeStep2PhysicalTime(timeStep);
			auto transformationAtTime = rft.getTransformationObservedToLab(time);
			for (int idz = 0; idz < gridsize(2); idz++) {
				float z = scalarFieldOutput.mFieldInfo.minZCoordinate + dz * idz;
				for (int idy = 0; idy < gridsize(1); idy++) {
					float y = scalarFieldOutput.mFieldInfo.minYCoordinate  + dy * idy;
					for (int idx = 0; idx < gridsize(0); idx++) {
						float x = scalarFieldOutput.mFieldInfo.minZCoordinate+ dx * idx;
						Eigen::Vector3f Pos3d = { x,y,z };
						Eigen::Vector3f ObservedToLab_transformed_pos = transformationAtTime(Pos3d);
						auto scalar = scalarFieldInput.GetValue(ObservedToLab_transformed_pos, time);
						scalarFieldOutput.SetValue(idx, idy, idz, timeStep, scalar);
					}
				}
			}
			}
		);

		STOPTIMER(RFT_OfScalarField3D_shiftRegion);

	}
	else
	{
		STARTTIMER(RFT_OfScalarField3D);

		Eigen::Vector3i  gridsize = scalarFieldInput.GetSpatialGridSize();
		int numberOfTimeSamples = scalarFieldInput.GetNumberOfTimeSteps();
		double tmin = scalarFieldInput.GetMinTime();
		double tmax = scalarFieldInput.GetMaxTime();

		const Eigen::Vector3f minDomainBounds = scalarFieldInput.GetSpatialMin().cast<float>();
		const Eigen::Vector3f maxDomainBounds = scalarFieldInput.GetSpatialMax().cast<float>();

		const double dx = (maxDomainBounds(0) - minDomainBounds(0)) / (gridsize(0) - 1);
		const double dy = (maxDomainBounds(1) - minDomainBounds(1)) / (gridsize(1) - 1);
		const double dz = (maxDomainBounds(2) - minDomainBounds(2)) / (gridsize(2) - 1);



		std::vector<int> threadRange(numberOfTimeSamples);
		std::generate(threadRange.begin(), threadRange.end(), [n = 0]() mutable { return n++; });
		for_each(policy, threadRange.begin(), threadRange.end(), [&gridsize, &minDomainBounds, dx, dy, dz, &scalarFieldInput, &scalarFieldOutput, &rft](int timeStep) {

			float time = scalarFieldInput.convertTimeStep2PhysicalTime(timeStep);
			auto transformationAtTime = rft.getTransformationObservedToLab(time);
			for (int idz = 0; idz < gridsize(2); idz++) {
				float z = minDomainBounds(2) + dz * idz;
				for (int idy = 0; idy < gridsize(1); idy++) {
					float y = minDomainBounds(1) + dy * idy;
					for (int idx = 0; idx < gridsize(0); idx++) {
						float x = minDomainBounds(0) + dx * idx;
						Eigen::Vector3f Pos3d = { x,y,z };
						Eigen::Vector3f ObservedToLab_transformed_pos = transformationAtTime(Pos3d);
						auto scalar = scalarFieldInput.GetValue(ObservedToLab_transformed_pos, time);
						scalarFieldOutput.SetValue(idx, idy, idz, timeStep, scalar);
					}
				}
			}
			}
		);

		STOPTIMER(RFT_OfScalarField3D);
	}



}


}//namespace DTG 

/**
 * @brief  observer-relative isosurface animation  is generated by 
 * 1. transform scalar field from lab frame to observed frame
 * 2. compute isosurface within the transformed scalar field
 * 3. animate isosurface
 * 
 */
void DTG::ObjectAdapterFlowlineExploration::observerRelativeIsoSurface(){
	mFLowLineVaoGroup0->createBoolVariable("rft scalar shift domain", false, true, true, true);
	mFLowLineVaoGroup0->createAction("rft-scalar field using indicator", [this]() {
	if (auto act_scalarfield = getActiveScalarField(); act_scalarfield) {
		auto input_name = getActiveScalarFieldName();
		auto curlField = std::make_unique<DiscreteScalarField3D<float>>(act_scalarfield->GetFieldInfo());

		std::string observerFieldName = this->getActiveObserverFieldName();
		double stepSize = this->getStepsize();
		auto pos3dAndTime = GetIndicatorPos3D();
		ReferenceFrameTransformation3D transformation;
		WorldlineUtility3D::computeReferenceFrameTransformation(observerFieldName, pos3dAndTime, stepSize, transformation);
		transformation.mObservationTime = pos3dAndTime.w();
		auto boolOption=mFLowLineVaoGroup0->getBoolValue("rft scalar shift domain");
		RftScalarField3D(*act_scalarfield, transformation, *curlField,boolOption);

		string name = "rft-" + input_name;
		Insert3DScalarField(std::move(curlField), name);
	}
});

mFLowLineVaoGroup0->createAction("rft-scalar field using cached rft", [this]() {
	if (auto act_scalarfield = getActiveScalarField(); act_scalarfield) {
		auto input_name = getActiveScalarFieldName();
		auto curlField = std::make_unique<DiscreteScalarField3D<float>>(act_scalarfield->GetFieldInfo());


		ReferenceFrameTransformation3D& transformation = mActiveObserverWorldline_;
			auto boolOption=mFLowLineVaoGroup0->getBoolValue("rft scalar shift domain");
		RftScalarField3D(*act_scalarfield, transformation, *curlField,boolOption);

		string name = "rft-" + input_name;
		Insert3DScalarField(std::move(curlField), name);
	}

	objectPtr->createAction("create animation of iso-surface", [this, objectPtr]() {
		string active_scalar_field_name = getScalarFieldName(WIDGET_COMPUTE_AND_ISOSURFACE_NAME, VAR_ISOSURFACE_TARGET_SCALAR3_3D_NAME);
		auto act_scalarfield = AttributesManager::GetInstance().getScalarField(active_scalar_field_name);

		const Eigen::Vector4d pos3dAndTime = GetIndicatorPos3D();
		const Eigen::Vector3d position = { pos3dAndTime.x(), pos3dAndTime.y(), pos3dAndTime.z() };

		double isoValue = objectPtr->getValue<double>("IsoValue 3D");
		// clamp animation  time
		double animation_start_time = objectPtr->getValue<double>("animation start time");
		double animation_end_time = objectPtr->getValue<double>("animation end time");
		int animation_frame_count = objectPtr->getValue<int>("animation frames");

		animation_start_time = std::clamp(animation_start_time, act_scalarfield->GetMinTime(), act_scalarfield->GetMaxTime());
		animation_end_time = std::clamp(animation_end_time, act_scalarfield->GetMinTime(), act_scalarfield->GetMaxTime());
		if (animation_end_time <= animation_start_time)
		{
			animation_end_time = act_scalarfield->GetMaxTime();
		}

		double animation_dt = (animation_end_time - animation_start_time) / ((double)animation_frame_count - 1.0);

		auto trackName = active_scalar_field_name + std::to_string(isoValue) + "t" + std::to_string(animation_start_time) + "-" + std::to_string(animation_end_time);


		const bool disable_interpolation = objectPtr->getBoolValue("disable scalarField interpolation");
		const bool multiSeeding= objectPtr->getBoolValue("create Animation with multi-seeding");
		std::vector<Eigen::Vector3d> closePointsToComputeIsosurface;


		std::vector<std::unique_ptr<ISoSurface>> iso_surface_t(animation_frame_count);
		std::vector<int> frames(animation_frame_count);
		std::iota(frames.begin(), frames.end(), 0);
		STARTTIMER(ISO-Surface Animation)
		// Parallel loop for iso-surface computation
		std::for_each(policy, frames.begin(), frames.end(),
			[&iso_surface_t, &act_scalarfield, disable_interpolation, animation_start_time, animation_dt, active_scalar_field_name, isoValue, position,multiSeeding,&closePointsToComputeIsosurface, this](int frameId) {
				double c_time = animation_start_time + animation_dt * frameId;
				const bool updateGUiAttributeBoundary = frameId == 0;

				if (multiSeeding)
				{

					iso_surface_t[frameId] = std::move(computeIsoSurfaceMC(active_scalar_field_name,  isoValue, c_time, updateGUiAttributeBoundary, !disable_interpolation));
				}
				else//one seeding
				{
					if (disable_interpolation)
					{
						int timeIndex = act_scalarfield->convertPhysicalTimeRoundedTimeStep(c_time);
						iso_surface_t[frameId] = std::move(computeOneIsoSurfaceComponentAtSeedpointNoInterpolation(active_scalar_field_name, isoValue, position, timeIndex, updateGUiAttributeBoundary));
					}
					else
					{
						iso_surface_t[frameId] = std::move(computeOneIsoSurfaceComponentAtSeedpoint(active_scalar_field_name, isoValue, position, c_time, updateGUiAttributeBoundary));
					}
				}

			});
		STOPTIMER(ISO-Surface Animation)
		auto cached_iso_surface_animation_track = std::make_unique<IsoSurfaceAnimationTrack>(
			animation_start_time, animation_end_time, animation_frame_count,
			std::move(iso_surface_t));

		if (auto iter = mAnimationTracks_.find(trackName); iter != mAnimationTracks_.end()) {
			mAnimationTracks_.erase(trackName);
		}
		mAnimationTracks_.insert(std::make_pair(trackName, std::move(cached_iso_surface_animation_track)));
		this->notifySubscribers("update iso-surface animation track");
		});





});

}//######## Observer-Relative Isosurface Animation########
