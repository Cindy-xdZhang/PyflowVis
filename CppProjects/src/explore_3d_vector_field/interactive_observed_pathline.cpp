/**
 * @file interactive_observed_pathline.cpp
 * @brief Functions for computing and filtering observed pathlines in 3D vector fields
 * 
 * This file contains utilities for:
 * - Computing observed pathlines from lab frame pathlines using reference frame transformations
 * - Filtering pathlines based on similarity criteria
 * - Computing similarity between observed streamline and  observed   pathlines
 */

#include "ReferenceFrame3d.h"
#include "Discrete3DFlowField.h"
#include "ScalarField.h"

using PointAndTime3Df = DTG::PointAndTime<3, float>;
using pathline3Df = std::vector<DTG::PointAndTime<3, float> >;

/**
 * @brief Stores discrete values for a group of lines with time interpolation
 * 
 * Contains 2D vectors where each element represents (time, value) pairs.
 * Provides interpolation for continuous time values.
 */
struct discretefunctions1D {//values stored at a group of lines
		std::vector<std::vector<Eigen::Vector2f>> discreteValues;
		
		/**
		 * @brief Get number of time steps for a specific line
		 * @param lineId Index of the line
		 * @return Number of time steps
		 */
		int GetNumberOfTimeSteps(int lineId)const {
			return discreteValues[lineId].size();
		}
		
		/**
		 * @brief Get interpolated value at a specific time for a line
		 * @param lineID Index of the line
		 * @param inTime Time to interpolate at
		 * @return Interpolated value
		 */
		float getValue(int lineID, float inTime) const {
			const auto numOfTimsteps = GetNumberOfTimeSteps(lineID);
			if (numOfTimsteps > 1) {
				double tmin = discreteValues[lineID].front().x();
				double tmax = discreteValues[lineID].back().x();
				const double dt = (tmax - tmin) / ((double)numOfTimsteps - 1);
				auto continousIndex = 0.0;
				continousIndex = (inTime - tmin) / dt;
				int floorIndex = floor(continousIndex);
				int ceilIndex = ceil(continousIndex);
				floorIndex = std::clamp(floorIndex, 0, numOfTimsteps - 1);
				ceilIndex = std::clamp(ceilIndex, 0, numOfTimsteps - 1);
				float resultInComponents = discreteValues[lineID][floorIndex].y();
				if (ceilIndex != floorIndex) {
					// interpolate with data from ceil index
					auto ceilValue = discreteValues[lineID][ceilIndex].y();
					double fraction = continousIndex - floorIndex;
					resultInComponents = resultInComponents * (1.0 - fraction) + ceilValue * fraction;
				}
				return resultInComponents;
			}
			else
			{
				return numOfTimsteps == 1 ? discreteValues[lineID][0].y() : 0.0;
			}
		}
	};







/**
 * @brief Transform pathlines from lab frame to observed frame
 * @param Pathilnes Input pathlines in lab frame
 * @param rft Reference frame transformation
 * @return Transformed pathlines in observed frame
 */
std::vector<std::vector<DTG::PointAndTime<3, float>>>  getObservedPathlineOnCPU(const std::vector<std::unique_ptr<std::vector<DTG::PointAndTime<3, float>>>>& Pathilnes, const ReferenceFrameTransformation3D& rft) {

	std::vector<std::vector<DTG::PointAndTime<3, float>>> result;
	const int lineCount = Pathilnes.size();
	result.resize(lineCount);

	if (rft.IsValid())
	{
	#pragma omp parallel for
		for (int lineId = 0; lineId < lineCount; lineId++)
		{
			const int lineLength = Pathilnes[lineId]->size();
			result[lineId].resize(lineLength);
			//Pathilnes->result[lineId]
	#pragma omp parallel for
			for (int pointId = 0; pointId < lineLength; pointId++)
			{
				const auto& point = Pathilnes[lineId]->at(pointId);
				const auto rft_function = rft.getTransformationLabToObserved(point.time);
				const auto newPosition = rft_function(point.position);

				result[lineId][pointId] = DTG::PointAndTime<3, float>(newPosition, point.time);
			}

		}
	}
	else
	{
		LOG_E("call getObservedPathlineOnCPU with invalid observer rft.");
	}
	return result;
}

/**
 * @brief Transform a single streamline from lab frame to observed frame
 * @param streamline Input streamline in lab frame
 * @param rft Reference frame transformation
 * @param unique_time Time for the transformation
 * @return Transformed streamline in observed frame
 */
std::unique_ptr<std::vector<DTG::PointAndTime<3, float>>> getObservedStreamlineOnCPU(const std::vector<DTG::PointAndTime<3, float>>& streamline, const ReferenceFrameTransformation3D& rft, double unique_time) {
	auto result = std::make_unique<std::vector<DTG::PointAndTime<3, float>>>();
	const int lineLength = streamline.size();
	result->resize(lineLength);
	const auto rft_function = rft.getTransformationLabToObserved(unique_time);
	if (rft.IsValid())
	{
	#pragma omp parallel for
		for (int pointId = 0; pointId < lineLength; pointId++)
		{
			const auto& point = streamline.at(pointId);
			const auto newPosition = rft_function(point.position);
			result->at(pointId) = DTG::PointAndTime<3, float>(newPosition, point.time);
		}
	}
	else
	{
		LOG_E("call getObservedStreamlineOnCPU with invalid observer rft.");
	}
	return std::move(result);
}

/**
 * @brief Filter pathlines into segments based on threshold values
 * @param VectorFieldPtr Pointer to the vector field
 * @param inputPathlinesNotSegmented Input pathlines to segment
 * @param valuesPerPoint Values per point for thresholding
 * @param threshold Threshold value for filtering
 * @param minimalLengthPerSegment Minimum length for a valid segment
 * @return Segmented pathlines
 */
std::vector<std::vector<DTG::PointAndTime<3, float>>> FilterPathlineAsSegments(const Discrete3DFlowField<float>* VectorFieldPtr,const std::vector<std::unique_ptr< std::vector<DTG::PointAndTime<3, float>>>>& inputPathlinesNotSegmented, const discretefunctions1D& valuesPerPoint, float threshold, const int minimalLengthPerSegment)
{
	std::vector<std::vector<DTG::PointAndTime<3, float>>> segmentedPathlines;

	for (size_t pathlineID = 0; pathlineID < inputPathlinesNotSegmented.size(); ++pathlineID) {
		const auto& pathline = *inputPathlinesNotSegmented[pathlineID];
		std::vector<DTG::PointAndTime<3, float>> currentSegment;

		for (size_t pointID = 0; pointID < pathline.size(); ++pointID) {
			const auto point_time = pathline[pointID].time;
			auto thisPoint_value = valuesPerPoint.getValue(pathlineID, point_time);

			if (thisPoint_value <= threshold) {
				currentSegment.push_back(pathline[pointID]);
			}
			else {
				if (!currentSegment.empty() && currentSegment.size() >= static_cast<size_t>(minimalLengthPerSegment)) {
					segmentedPathlines.push_back(std::move(currentSegment));
				}
				currentSegment.clear();
			}
		}

		// Store the last segment if it meets the length requirement
		if (!currentSegment.empty() && currentSegment.size() >= static_cast<size_t>(minimalLengthPerSegment)) {
			segmentedPathlines.push_back(std::move(currentSegment));
		}
	}

	return segmentedPathlines;
}     
	
/**
 * @brief Compute bidirectional distance between two flowlines
 * @param flowline0 First flowline
 * @param axId0 Anchor point index in first flowline
 * @param flowline1 Second flowline
 * @param axId1 Anchor point index in second flowline
 * @return Average distance between corresponding points
 */
double DoubleDirectionFlowLineDistance(const std::vector<DTG::PointAndTime<3, float>>& flowline0, int axId0, const std::vector<DTG::PointAndTime<3, float>>& flowline1, int axId1) {
	int leftMinimalLength = std::min(axId0, axId1);

	int rightMinimalLength = std::min(flowline0.size() - axId0, flowline1.size() - axId1);
	double totalDistance = 0.0;
	int count = 0;
	// Compute left side distances
	if (leftMinimalLength>=1&&flowline0.size()>=leftMinimalLength&&flowline1.size()>=leftMinimalLength)
	{

	for (int step = leftMinimalLength-1; step < leftMinimalLength; step++) {
		auto pos0 = flowline0[axId0 - step].position;
		auto pos1 = flowline1[axId1 - step].position;
		Eigen::Vector3f distance = pos0 - pos1;
		totalDistance += static_cast<double>( distance.norm());
		count++;
	}
	}
	if (rightMinimalLength>=1&&flowline0.size()>=rightMinimalLength&&flowline1.size()>=leftMinimalLength)
	{
	// Compute right side distances
	for (int step = rightMinimalLength-1; step < rightMinimalLength; step++) {
		auto pos0=flowline0[axId0 + step].position;
		auto pos1=flowline1[axId1 + step].position;
		Eigen::Vector3f distance=pos0-pos1;
		totalDistance += static_cast<double>( distance.norm());
		count++;
	}

	}

	return count > 0 ? totalDistance / count : 0.0; // Return average distance

}



//########################################################
//######## Observer-Relative Pathline Filtering #########
//########################################################
/**
* @brief Compute similarity between observed pathlines and analytical streamlines
* 
* Integrates streamlines from v-u field and compares with observed pathlines
* to compute similarity metrics for filtering.
*/
void DTG::ObjectAdapterFlowlineExploration::computeStreamlineSimilarity()
{


	mFLowLineVaoGroup0->createAction("observed pathline streamline similarity", "streamline are integrated from interactively changed v-u(analytical flow field)", [this]() {
		if (const int activePathlineCount = mFlowLineCached0_.size(); activePathlineCount)
		{
			if (!mActiveObserverWorldline_.IsValid())

			{
				LOG_E("mCachedObserverWorldline_ Is Not Valid.");
				return;
			}


			auto activeFieldName = GetActive3DFieldName();
			auto activeObserverFieldname = getActiveObserverFieldName();
			auto* vfIterface_u = VectorField3DManager::GetInstance().GetVectorFieldByName(activeObserverFieldname);
			Discrete3DFlowField<float>* activeInputVectorField = GET_FLOAT_VECTOR_FLOW3D__BY_NAME(activeFieldName);
			Discrete3DFlowField<float>* vf_u = dynamic_cast<Discrete3DFlowField<float>*>(vfIterface_u);
			if (activeInputVectorField && vf_u)
			{
				Discrete3DFlowField<float> vmiusFieldAnalytical = Discrete3DFlowField<float>(activeInputVectorField->GetFieldInfo(), false);
				Flow3DFunc fieldFunction = [vf_u, activeInputVectorField](float x, float y, float z, float t)->Eigen::Matrix<float, 3, 1> {
					return activeInputVectorField->GetVector(x, y, z, t) - vf_u->GetVector(x, y, z, t);
					};
				vmiusFieldAnalytical.SetAnalyticalExpression(fieldFunction);

				const  auto tmax = activeInputVectorField->GetMaxTime();
				const  auto tmin = activeInputVectorField->GetMinTime();
				const auto gridSize = activeInputVectorField->GetSpatialGridSize();
				auto integrationLength = mFLowLineVaoGroup0->getValue<float>("integrationSegLength");
				float stepSize = this->getStepsize();
				const int maxIter = integrationLength / stepSize;
				int GridSkip = std::max(mglyph3dVAO->getValue<int>("grid skip"), 1);
				int total_streamlines = 0;
				vector<std::tuple<int, int, int, DTG::PointAndTime<3, float>>> startPoistions; // (lineID, pointID, PointAndTime)
				cached_FlowlinesSimiarity.discreteValues.clear();
				cached_FlowlinesSimiarity.discreteValues.resize(mFlowLineCached0_.size());
				startPoistions.reserve(activePathlineCount * 1024);

				STARTTIMER(observed pathline streamline similarity)

					STARTTIMER(getObservedPathlineOnCPU)
					auto observedPathline = getObservedPathlineOnCPU(mFlowLineCached0_, mActiveObserverWorldline_);
				STOPTIMER(getObservedPathlineOnCPU)
					for (int lineID = 0; lineID < mFlowLineCached0_.size(); lineID++)
					{
						const auto& raw_pathline = *mFlowLineCached0_.at(lineID);
						int computedStep = 0;
						for (int point_id = 0; point_id < raw_pathline.size(); point_id += GridSkip)
						{
							startPoistions.emplace_back(lineID, point_id, computedStep++, raw_pathline.at(point_id));
						}
						cached_FlowlinesSimiarity.discreteValues[lineID].clear();
						cached_FlowlinesSimiarity.discreteValues[lineID].resize(computedStep);

					}

				total_streamlines = startPoistions.size();
				mFlowLineCached1_.clear();
				mFlowLineCached1_.resize(total_streamlines);




				// Create a linear index mapping each point in the 4D grid
				std::vector<int> indices(total_streamlines);
				std::iota(indices.begin(), indices.end(), 0);

				//std::vector<std::vector<float>> all_streamline_vertex_distances;
				//all_streamline_vertex_distances.resize(mFlowLineCached1_.size()); // Pre-allocate for each streamline

				STARTTIMER(observed streamline integration)
					// Transform with parallel execution policy
					std::for_each(policy, indices.begin(), indices.end(),
						[&](const int inidx) {
							const auto& [PathlineID, pointIDInPathline, pointIDInValueCurve, startPointAndtimeInLabFrame] = startPoistions[inidx];
							const  DTG::Point<3, float>startPositionLabFrame = startPointAndtimeInLabFrame.position;
							const float time = startPointAndtimeInLabFrame.time;

							auto resultStreamlinesForward = std::make_unique<std::vector<DTG::PointAndTime<3, float>>>();
							auto resultStreamlineBackward = std::make_unique<std::vector<DTG::PointAndTime<3, float>>>();
							//integrate streamline of v-u in lab frame 
							vmiusFieldAnalytical.Integrate3DStreamlineOneDirection(startPositionLabFrame, time, stepSize, maxIter, *resultStreamlinesForward, PathlineNumericalIntegrationMethod::Euler);
							int aixId_streamline=0;
							if (pointIDInPathline>2)
							{
								vmiusFieldAnalytical.Integrate3DStreamlineOneDirection(startPositionLabFrame, time, stepSize, maxIter, *resultStreamlineBackward, PathlineNumericalIntegrationMethod::Euler, false);
								aixId_streamline = resultStreamlineBackward->size();
								std::reverse(resultStreamlineBackward->begin(), resultStreamlineBackward->end());
							}
							resultStreamlineBackward->insert(resultStreamlineBackward->end(), resultStreamlinesForward->begin(), resultStreamlinesForward->end());


							//compute value for this point
							const auto& observed_pathline = observedPathline.at(PathlineID);
							std::unique_ptr<std::vector<DTG::PointAndTime<3, float>>> observedStreamline = getObservedStreamlineOnCPU(*resultStreamlineBackward, mActiveObserverWorldline_, time);

							//!!!!! dis_similarity_values_on_this_streamline:
							//special index color coding for figure of paper only: when streamline is aligned with pathline, there is no depth-fighting.
							//so now for all the observedStreamline vertex, compute  the closet distance to neighboring observed_pathline points
							// A vector to store per-vertex distances for each streamline.
						// This will be a vector of vectors, where each inner vector corresponds to one streamline.
						// `dissimilarities` as passed to `MappingFlowlineAsRenderingVAO_API0` is likely expecting per-vertex data.
							//std::vector<float> dis_similarity_values_on_this_streamline;
							//dis_similarity_values_on_this_streamline.reserve(observedStreamline->size()); // Reserve space for efficiency
							//auto dis_similarity = DoubleDirectionFlowLineDistanceV0(observed_pathline, pointIDInPathline, *observedStreamline, aixId_streamline,dis_similarity_values_on_this_streamline);
							//all_streamline_vertex_distances.at(inidx)=std::move(dis_similarity_values_on_this_streamline);

							auto dis_similarity = DoubleDirectionFlowLineDistance(observed_pathline, pointIDInPathline, *observedStreamline, aixId_streamline);

							auto dv_dt = activeInputVectorField->getPartialDerivativeT(startPositionLabFrame, time).squaredNorm();
							dis_similarity += 0.1 * std::exp(-dv_dt);
							cached_FlowlinesSimiarity.discreteValues[PathlineID][pointIDInValueCurve] = { time, dis_similarity };

							mFlowLineCached1_.at(inidx) = std::move(observedStreamline);
						});
					STOPTIMER(observed streamline integration)

					STOPTIMER(observed pathline streamline similarity)


				//We don't need call the following, since observed pathline is handled by updatePathline and shader automatically,
				// the following code rendering cpu based  observed pathline just for debugging
				 //MappingFlowlineAsRenderingVAO(observedPathline, mFLowLineVaoGroup0, nullptr, tmin, tmax);

				MappingFlowlineAsRenderingVAO(mFlowLineCached1_, mFLowLineVaoGroup1.get(), nullptr, tmin, tmax);
				//MappingFlowlineAsRenderingVAO_API0(mFlowLineCached1_, mFLowLineVaoGroup1.get(), all_streamline_vertex_distances, tmin, tmax);

				//ploting streamline pathline difference:
				std::string exploreation_obj_name{ GetName<DTG::ObjectAdapterFlowlineExploration>() };
				if (auto* flowline3dwiget = dynamic_cast<ObjectAdapterFlowlineExploration*>(mScene_->getObject(exploreation_obj_name)); flowline3dwiget) {

					if (auto plotExploration = dynamic_cast<GraphWidget*>(flowline3dwiget->mChildViewer1.get()); plotExploration) {
						plotExploration->plotMappingFlowlinesCurvesWithpreComputedValue(mFlowLineCached0_, cached_FlowlinesSimiarity.discreteValues);
					}
				}
			}
		}
		});


	
	
	auto FilterOutPathlines=[this](){
		if (cached_FlowlinesSimiarity.discreteValues.size() != mFlowLineCached0_.size())
		{
			LOG_D("missing pathline and streamline distance, recomputing...");
			this->runActionRecursive("observed pathline streamline similarity");
		}
		auto activeFieldName = GetActive3DFieldName();
		Discrete3DFlowField<float>* activeInputVectorField = GET_FLOAT_VECTOR_FLOW3D__BY_NAME(activeFieldName);
		const  auto tmax = activeInputVectorField->GetMaxTime();
		const  auto tmin = activeInputVectorField->GetMinTime();
		const float dissimilarity_thresholding = mFLowLineVaoGroup0->getValue<float>("thresholding value");

		auto rest = FilterPathlineAsSegments(activeInputVectorField, mFlowLineCached0_, cached_FlowlinesSimiarity, dissimilarity_thresholding, 5);

		MappingFlowlineAsRenderingVAO(rest, mFLowLineVaoGroup0, nullptr, tmin, tmax);

	};
	mFLowLineVaoGroup0->createAction("thresholding pathline", "thresholding pathline by criterion will break them into segments", [this,FilterOutPathlines]() {
			FilterOutPathlines();

		});
});
}//######## Observer-Relative Pathline Filtering #########





