	
	
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

std::vector<std::vector<DTG::PointAndTime<3, float>>> FilterPathlineAsSegments(const Discrete3DFlowField<float>* VectorFieldPtr,
	const std::vector<std::unique_ptr< std::vector<DTG::PointAndTime<3, float>>>>& inputPathlinesNotSegmented, const discretefunctions1D& valuesPerPoint, float threshold, const int minimalLengthPerSegment)
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
								vmiusFieldAnalytical.Integrate3DStreamlineOneDirection(startPositionLabFrame, time, stepSize, maxIter, *resultStreamlineBackward, PathlineNumericalIntegrationMethod::Euler, false);
								int aixId_streamline = resultStreamlineBackward->size();
								std::reverse(resultStreamlineBackward->begin(), resultStreamlineBackward->end());
								resultStreamlineBackward->insert(resultStreamlineBackward->end(), resultStreamlinesForward->begin(), resultStreamlinesForward->end());

								//compute value for this point
								const auto& observed_pathline = observedPathline.at(PathlineID);
								std::unique_ptr<std::vector<DTG::PointAndTime<3, float>>> observedStreamline = getObservedStreamlineOnCPU(*resultStreamlineBackward, mActiveObserverWorldline_, time);

								auto value = DoubleDirectionFlowLineDistance(observed_pathline, pointIDInPathline, *observedStreamline, aixId_streamline);

								auto dv_dt = activeInputVectorField->getPartialDerivativeT(startPositionLabFrame, time).squaredNorm();
								value += 0.1 * std::exp(-dv_dt);
								cached_FlowlinesSimiarity.discreteValues[PathlineID][pointIDInValueCurve] = { time, value };

								mFlowLineCached1_.at(inidx) = std::move(observedStreamline);
							});
					STOPTIMER(observed streamline integration)


						STOPTIMER(observed pathline streamline similarity)

						MappingFlowlineAsRenderingVAO(mFlowLineCached1_, mFLowLineVaoGroup1.get(), nullptr, tmin, tmax);
					//rendering observed pathline on cpu verify we are compute correct difference
					// MappingFlowlineAsRenderingVAO(observedPathline, mFLowLineVaoGroup0, nullptr, tmin, tmax);

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

		mFLowLineVaoGroup0->createBoolVariable("interactive thresholding pathline", false, true, true, true);

		mFLowLineVaoGroup0->createVariable("thresholding value", 0.1f, ObjectVariable::SemanticType::Default, true, true, true, [this](const ObjectVariable& var) {
			std::string name{ GetName<DTG::ObjectAdapterFlowlineExploration>() };
			if (auto* flowline3dwiget = dynamic_cast<ObjectAdapterFlowlineExploration*>(mScene_->getObject(name)); flowline3dwiget) {
				auto plotExploration = dynamic_cast<GraphWidget*>(flowline3dwiget->mChildViewer1.get());
				//thresholding pathline 
				if (cached_FlowlinesSimiarity.discreteValues.size() != mFlowLineCached0_.size())
				{
					this->runActionRecursive("observed pathline streamline similarity");
				}
				auto activeFieldName = GetActive3DFieldName();
				Discrete3DFlowField<float>* activeInputVectorField = GET_FLOAT_VECTOR_FLOW3D__BY_NAME(activeFieldName);
				const  auto tmax = activeInputVectorField->GetMaxTime();
				const  auto tmin = activeInputVectorField->GetMinTime();
				const float dissimilarity_thresholding = mFLowLineVaoGroup0->getValue<float>("thresholding value");

				auto rest = FilterPathlineAsSegments(activeInputVectorField, mFlowLineCached0_, cached_FlowlinesSimiarity, dissimilarity_thresholding, 5);

				MappingFlowlineAsRenderingVAO(rest, mFLowLineVaoGroup0, nullptr, tmin, tmax);

			}//if (auto* flowline3dwiget 
			});

			mFLowLineVaoGroup0->createAction("thresholding pathline", "thresholding pathline by criterion will break them into segments", [this]() {
	std::string name{ GetName<DTG::ObjectAdapterFlowlineExploration>() };
	if (auto* flowline3dwiget = dynamic_cast<ObjectAdapterFlowlineExploration*>(mScene_->getObject(name)); flowline3dwiget) {
		auto plotExploration = dynamic_cast<GraphWidget*>(flowline3dwiget->mChildViewer1.get());
		//thresholding pathline 
		const float dissimilarity_thresholding = plotExploration->GetThresholdFitlerTopValue();
		mFLowLineVaoGroup0->updateValue<float>("thresholding value", dissimilarity_thresholding, true);

	}//if (auto* flowline3dwiget 
	});
}//######## Observer-Relative Pathline Filtering #########





//########################################################
//######## Observer-Relative Isosurface Animation#########
//########################################################
namespace DTG { // base class IFlowField3d

// compute   VelocityGradientTensor and convert to angular velocity along pathline
void DTG::WorldlineUtility3D::computeInstaneousRotationAroundPathline(const std::string& observerFieldName, const Worldline3D& pathline, std::vector<Eigen::Vector3f>& instantaneousRotation)
{
	instantaneousRotation.resize(pathline.mPathline.size());
	auto vff = VectorField3DManager::GetInstance().GetVectorFieldByName(observerFieldName);
	const auto* derivedFloatVectorField = dynamic_cast<const Discrete3DFlowField<float>*>(vff);
	auto computeInstantaneousRotation = [derivedFloatVectorField](const PointAndTime3Df& p) -> Eigen::Vector3f {
		const auto time = p.time;
		// Compute partial derivatives of the vector field
		auto gradient = derivedFloatVectorField->getVelocityGradientTensor(p.position, p.time);
		auto angularVelocity = velocityGradientTensor2AngularVelocity(gradient);
		return angularVelocity;
	};

	// Perform the computation in parallel
	std::transform(policy, pathline.mPathline.begin(), pathline.mPathline.end(), instantaneousRotation.begin(), computeInstantaneousRotation);
}

void DTG::WorldlineUtility3D::computeReferenceFrameTransformation(const std::string& observerFieldName, const Eigen::Vector4d& InStartpoint, const double& stepsize, ReferenceFrameTransformation3D& transformation)
{
	DTG::Point<3, float> pos = DTG::Point<3, float>({ (float)InStartpoint.x(), (float)InStartpoint.y(), (float)InStartpoint.z() });
	PointAndTime3Df Startpoint = PointAndTime3Df(pos, InStartpoint.w());

	Worldline3D& worldline = transformation.mWorldline;
	computeWorldLine3d(observerFieldName, Startpoint, stepsize, worldline);
	const double observation_time = InStartpoint.w();
	transformation.mObservationTime = observation_time;
	integrateReferenceFrameRotation(worldline, observation_time, transformation);
}

void DTG::WorldlineUtility3D::computeReferenceFrameTransformation(const std::string& observerFieldName, const PointAndTime3Df& startpoint, const double& stepsize, ReferenceFrameTransformation3D& transformation)
{
	Worldline3D& worldline = transformation.mWorldline;
	computeWorldLine3d(observerFieldName, startpoint, stepsize, worldline);
	const double observation_time = startpoint.time;
	transformation.mObservationTime = observation_time;
	integrateReferenceFrameRotation(worldline, observation_time, transformation);
}

template <class T>
void DTG::WorldlineUtility3D::computeReferenceFrameTransformationOfScalarField(const DiscreteScalarField3D<T>& scalarFieldInput, DiscreteScalarField3D<T>& scalarFieldOutput, const ReferenceFrameTransformation3D& transformation)
{
STARTTIMER(RFT_OfScalarField);

const int numberOfTimeSamples = scalarFieldInput.GetNumberOfTimeSteps();
const double tmin = scalarFieldInput.GetMinTime();
const double tmax = scalarFieldInput.GetMaxTime();
const double dt = scalarFieldInput.Getdt();

const double dx = (scalarFieldInput.mFieldInfo.maxXCoordinate - scalarFieldInput.mFieldInfo. minXCoordinate) / (scalarFieldInput.mFieldInfo.XGridsize - 1);
const double dy = (scalarFieldInput.mFieldInfo.maxYCoordinate - scalarFieldInput.mFieldInfo.minYCoordinate) / (scalarFieldInput.mFieldInfo.YGridsize - 1);
const double dz = (scalarFieldInput.mFieldInfo.maxZCoordinate - scalarFieldInput.mFieldInfo.minZCoordinate) / (scalarFieldInput.mFieldInfo.ZGridsize - 1);

std::vector<int> threadRange(numberOfTimeSamples);
std::generate(threadRange.begin(), threadRange.end(), [n = 0]() mutable { return n++; });

for_each(policy, threadRange.begin(), threadRange.end(), [tmin, dt, dx, dy, dz, &scalarFieldInput, &scalarFieldOutput, &transformation](int timeStep) {
	double time = tmin + dt * timeStep;

	const auto &transformationAtTime = transformation.getTransformationObservedToLab(time);
	for (int idz = 0; idz < scalarFieldInput.mFieldInfo.ZGridsize; idz++) {
		float z = scalarFieldInput.mFieldInfo.minZCoordinate + dz * idz;
		for (int idy = 0; idy < scalarFieldInput.mFieldInfo.YGridsize; idy++) {
			float y = scalarFieldInput.mFieldInfo.minYCoordinate + dy * idy;
			for (int idx = 0; idx < scalarFieldInput.mFieldInfo.XGridsize; idx++) {
				float x = scalarFieldInput.mFieldInfo.minXCoordinate + dx * idx;
				Eigen::Vector3f pos3d({ x, y, z });
				Eigen::Vector3f transformed_pos = transformationAtTime(pos3d);
				auto scalar = scalarFieldInput.InterpolateValuebyPos(transformed_pos.x(), transformed_pos.y(), transformed_pos.z(), time);
				scalarFieldOutput.SetValue(idx, idy, idz, timeStep, scalar);
			}
		}
	}
});

STOPTIMER(RFT_OfScalarField);
}

// Integrate3DPathlineDoubleDirection +computeInstaneousRotationAroundPathline
void DTG::WorldlineUtility3D::computeWorldLine3d(const std::string& observerFieldName, const PointAndTime3Df& startpoint, const double& stepsize, Worldline3D& worldline)
{
	auto& pathline = worldline.mPathline;
	const auto& FieldPtr=VectorField3DManager::GetInstance().GetVectorFieldByName(observerFieldName);
	bool relaxBoundary=FieldPtr->HasAnalyticalExpression();
	FieldPtr->Integrate3DPathlineDoubleDirection(startpoint.position, (float)startpoint.time, (float)stepsize, 20000, pathline, PathlineNumericalIntegrationMethod::RK4,relaxBoundary);

	// Get angular velocity at worldline positions
	auto& instantaneousRotation = worldline.mInstantaneousRotation;
	computeInstaneousRotationAroundPathline(observerFieldName, worldline, instantaneousRotation);
}


template void DTG::WorldlineUtility3D::computeReferenceFrameTransformationOfScalarField(const DiscreteScalarField3D<float>& input, DiscreteScalarField3D<float>& output, const ReferenceFrameTransformation3D& transformation);

template<typename T>
Eigen::Matrix<T,3,3> DTG::WorldlineUtility3D::angleVelocityExponential2Rotation(const Eigen::Matrix<T,3,1>& angularVelocity, double timeInterval_dt)
{

const auto w1 = angularVelocity(0);
const auto w2 = angularVelocity(1);
const auto w3 = angularVelocity(2);
Eigen::Matrix<T,3,3> Spintensor;
Spintensor(0, 0) = 0.0;
Spintensor(1, 0) = w3;
Spintensor(2, 0) = -w2;

Spintensor(0, 1) = -w3;
Spintensor(1, 1) = 0.0;
Spintensor(2, 1) = w1;

Spintensor(0, 2) = w2;
Spintensor(1, 2) = -w1;
Spintensor(2, 2) = 0.0;

auto NoramlizeSpinTensor = [](Eigen::Matrix<T,3,3> & input) {
	Eigen::Matrix<T,3,1> unitAngular;
	unitAngular << input(2, 1), input(0, 2), input(1, 0);
	unitAngular.normalize();
	input << 0, -unitAngular(2), unitAngular(1),
		unitAngular(2), 0, -unitAngular(0),
		-unitAngular(1), unitAngular(0), 0;
};

double fi = angularVelocity(0) * angularVelocity(0) + angularVelocity(1) * angularVelocity(1) + angularVelocity(2) * angularVelocity(2);
fi = sqrt(fi);
const double theta = fi * timeInterval_dt;

NoramlizeSpinTensor(Spintensor);
Eigen::Matrix<T,3,3>  Spi_2;
Spi_2 = Spintensor * Spintensor;
const double sinTheta = sin(theta);
const double cosTheta = cos(theta);
Eigen::Matrix<T,3,3>  I = Eigen::Matrix<T,3,3>::Identity();
Eigen::Matrix<T,3,3> expA = I + sinTheta * Spintensor + (1 - cosTheta) * Spi_2;
return expA;
}
template Eigen::Matrix<float,3,3> DTG::WorldlineUtility3D::angleVelocityExponential2Rotation(const Eigen::Matrix<float,3,1>& angularVelocity, double timeInterval_dt);
template Eigen::Matrix<double,3,3> DTG::WorldlineUtility3D::angleVelocityExponential2Rotation(const Eigen::Matrix<double,3,1>& angularVelocity, double timeInterval_dt);


void DTG::WorldlineUtility3D::integrateReferenceFrameRotation(const Worldline3D& worldline, const double& observation_time, ReferenceFrameTransformation3D& transformation){
	transformation.mObservationTime = observation_time;
	const auto& pathline = worldline.mPathline;
	Eigen::Vector2d timerange = worldline.getTimeRange();
	if (!(timerange(0) < timerange(1))) {
		LOG_E("Cannot compute transformation. Invalid time range of worldline.");
		return;
	}

	const double min_observer_time = timerange(0);
	const double max_observer_time = timerange(1);

	const int numberOfLocalTimesteps = worldline.getNumberOfTimeSteps();
	const double dt = (max_observer_time - min_observer_time) / (numberOfLocalTimesteps - 1);

	int numberOfBackwardSteps = 0;
	for (const auto& p_t : pathline) {
		if (p_t.time < transformation.mObservationTime) {
			numberOfBackwardSteps++;
		} else {
			break;
		}
	}

	// Ensure at least one forward step exists
	if (numberOfBackwardSteps == numberOfLocalTimesteps) {
		numberOfBackwardSteps--;
	}

	const auto& instantaneousRotation = worldline.mInstantaneousRotation; // Now Eigen::Vector3d
	std::vector<Eigen::Matrix3f>& integratedRotation = transformation.mIntegratedRotation;
	integratedRotation.resize(numberOfLocalTimesteps, Eigen::Matrix3f::Identity());
	integratedRotation[numberOfBackwardSteps] = Eigen::Matrix3f::Identity();

	// Integrate angular velocity using Rodrigues' formula
	// Backward
	double backward_dt = dt * -1;
	for (int pointStepIndex = numberOfBackwardSteps; pointStepIndex > 0; pointStepIndex--) {
		const Eigen::Vector3f& angularVelocity = instantaneousRotation.at(pointStepIndex - 1);
		Eigen::Matrix3f rotationMatrix = angleVelocityExponential2Rotation(angularVelocity, backward_dt);
		integratedRotation[pointStepIndex - 1] = integratedRotation[pointStepIndex] * rotationMatrix;
	}

	// Forward
	for (int pointStepIndex = numberOfBackwardSteps; pointStepIndex < numberOfLocalTimesteps - 1; pointStepIndex++) {
		const Eigen::Vector3f& angularVelocity = instantaneousRotation.at(pointStepIndex + 1);

		Eigen::Matrix3f rotationMatrix = angleVelocityExponential2Rotation(angularVelocity, dt);
		integratedRotation[pointStepIndex + 1] = integratedRotation[pointStepIndex] * rotationMatrix;
	}
}

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
});

}//######## Observer-Relative Isosurface Animation########


