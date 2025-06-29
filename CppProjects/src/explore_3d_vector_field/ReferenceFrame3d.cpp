#include "ReferenceFrame3d.h"
#include "Discrete3DFlowField.h"
#include "logSystem/log.h"
#include "mydefs.h"
#include "FlowField3DUtils.h"
namespace {
#ifndef _DEBUG
auto policy = std::execution::par_unseq;
#else
auto policy = std::execution::seq;
#endif
}

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


void DTG::WorldlineUtility3D::integrateReferenceFrameRotation(const Worldline3D& worldline, const double& observation_time, ReferenceFrameTransformation3D& transformation)
{
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


std::vector<KillingField3DCoefficients> DTG::WorldlineUtility3D::extractKillingFieldFromWorldlineForceNoRot(Discrete3DFlowField<float>*active_field, const Worldline3D& worldline) {
	const int numSteps = worldline.getNumberOfTimeSteps();
    if (numSteps == 0) return{};

    std::vector<KillingField3DCoefficients>  resutl_killingParamters;
    resutl_killingParamters.reserve(numSteps);
	for (const auto& p : worldline.mPathline) {
		// Step 1: Get gradient tensor at current point
		//auto gradient = active_field->getVelocityGradientTensor(p.position, p.time);
		//// Step 2: Compute angular velocity (3D vector) from antisymmetric part of gradient
		//Eigen::Vector3f angularVelocity = velocityGradientTensor2AngularVelocity(gradient);

		//// Store rotationABC = angularVelocity

		//// Step 3: Compute Killing field vector at this point based on angularVelocity
		//float A = angularVelocity.x();
		//float B = angularVelocity.y();
		//float C = angularVelocity.z();

		//float x = p.position.x();
		//float y = p.position.y();
		//float z = p.position.z();
		//Eigen::Vector3f killingRotABC = {
		//	A,
		//	B,
		//	C
		//};
		//// killing rotation contribution at this point
		//Eigen::Vector3f killingRotVector = {
		//	B * z - C * y,
		//	C * x - A * z,
		//	A * y - B * x
		//};
        Eigen::Vector3f killingRotVector = {0,0,0};
		//float u = D + (B * z - C * y);  // Translation X + Rotation YZ
		//float v = E + (C * x - A * z);  // Translation Y + Rotation ZX
		//float w = F + (A * y - B * x);  // Translation Z + Rotation XY
        // 
        // 
		// Step 4: Get actual field vector at this position and time
		Eigen::Vector3f actualField = active_field->GetVector(p.position, (float)p.time);

		// Step 5: The difference is translationDEF
		Eigen::Vector3f translationDEF = actualField - killingRotVector;
        resutl_killingParamters.emplace_back(killingRotVector,translationDEF);
	}
    return resutl_killingParamters;
}














Eigen::Vector2i ClampTimstep2ValidRange(const Discrete3DFlowField<float>& inputField, int timestepStart, int timestepEnd)
{
    int totalTimesteps = inputField.GetNumberOfTimeSteps();
    if (timestepEnd==-1)
    {
        timestepEnd=totalTimesteps - 1;
    }
	if (timestepStart== -1)
	{
		timestepStart= 0;
	}
    
    timestepStart = std::max(0, std::min(timestepStart, totalTimesteps - 1));
    timestepEnd = std::max(0, std::min(timestepEnd, totalTimesteps - 1));
    if (timestepStart > timestepEnd) {
        timestepEnd = timestepStart;
    }
    return Eigen::Vector2i(timestepStart, timestepEnd);
}

template <class T>
Eigen::Matrix<T, 3, 3> skewMat(const Eigen::Matrix<T, 3, 1>& input)
{
    Eigen::Matrix<T, 3, 3> res;
    res << 0, -input(2), input(1),
        input(2), 0, -input(0),
        -input(1), input(0), 0;
    return res;

}

GenericLocalOptimization3d::GenericLocalOptimization3d(int neighborhoodU, bool useSummedAreaTables)
    : NeighborhoodU(neighborhoodU)
    , UseSummedAreaTables(useSummedAreaTables)
{
}

void GenericLocalOptimization3d::compute(const Discrete3DFlowField<float>& inputField, int timestepStart, int timestepEnd, Discrete3DFlowField<float>& resultUfield, Discrete3DFlowField<float>& resultVminusUfield)
{
    DiscreteScalarField3D<float> observedTimeDerivatives(inputField.GetFieldInfo());
    observedTimeDerivatives.mFieldInfo.numberOfTimeSteps = -1;
    compute(inputField, timestepStart, timestepEnd, resultUfield, resultVminusUfield, observedTimeDerivatives);
}
void GenericLocalOptimization3d::compute(const Discrete3DFlowField<float>& inputField, int timestepStart, int timestepEnd, Discrete3DFlowField<float>& resultUfield, Discrete3DFlowField<float>& resultVminusUfield,DiscreteScalarField3D<float>& observedTimeDerivatives)
{
    if (mGridFilter)
    {
        computeWithFilter(inputField, timestepStart, timestepEnd, resultUfield, resultVminusUfield, observedTimeDerivatives);
    }else
        computeWithOutFilter(inputField, timestepStart, timestepEnd, resultUfield, resultVminusUfield, observedTimeDerivatives);
}

static int factorial(int n)
{
	int ret = 1;
	for (int i = 1; i <= n; ++i)
		ret *= i;
	return ret;
}
void ComputeDisplacementSystemMatrixCoefficients(int i, int j, int k, const Eigen::Vector3f& xx, const Eigen::Vector3f& vv, const Eigen::Matrix3f& J, Eigen::Matrix3f& f, Eigen::Matrix3f& g)
{
	f = Eigen::Matrix3f::Zero(), g = Eigen::Matrix3f::Zero();
	Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
	double x = xx.x(), y = xx.y(), z = xx.z(), u = vv.x(), v = vv.y(), w = vv.z();

	double x_i = pow(x, i), x_i1 = pow(x, i - 1);
	double y_j = pow(y, j), y_j1 = pow(y, j - 1);
	double z_k = pow(z, k), z_k1 = pow(z, k - 1);

	int fac_i = factorial(i), fac_i1 = factorial(i - 1);
	int fac_j = factorial(j), fac_j1 = factorial(j - 1);
	int fac_k = factorial(k), fac_k1 = factorial(k - 1);

	f -= x_i * y_j * z_k / (fac_i * fac_j * fac_k) * J;
	if (i != 0)
		f += u * x_i1 * y_j * z_k / (fac_i1 * fac_j * fac_k) * I;
	if (j != 0)
		f += v * x_i * y_j1 * z_k / (fac_i * fac_j1 * fac_k) * I;
	if (k != 0)
		f += w * x_i * y_j * z_k1 / (fac_i * fac_j * fac_k1) * I;

	g = x_i * y_j * z_k / (fac_i * fac_j * fac_k) * I;
}
static Eigen::Matrix3f makeEigenMatrix3f(double h00, double h01, double h02,
	double h10, double h11, double h12,
	double h20, double h21, double h22) {
	Eigen::Matrix3f mat;
	mat << h00, h01, h02,
		h10, h11, h12,
		h20, h21, h22;
	return mat;
}




void GenericLocalOptimization3d::computeWithOutFilter(const Discrete3DFlowField<float>& inputField, int timestepStart, int timestepEnd, Discrete3DFlowField<float>& resultUfield, Discrete3DFlowField<float>& resultVminusUfield, DiscreteScalarField3D<float>& observedTimeDerivatives)
{
    using namespace Eigen;
    using Float = float;
    constexpr int taylorOrder=1;//hiper paramter used for  EInvariance::Displacement
    const auto&invariance=mInvariance;
    auto info=inputField.getPrintOutGridInfo();
    LOG_D("comput observer with infor: %s" ,info.c_str());

    std::string timerString= "GGTCompute EInvariance::Objective";
	if (invariance == EInvariance::Displacement)
        timerString="GGTCompute EInvariance::Displacement";

    Vector2i timesteps = ClampTimstep2ValidRange(inputField, timestepStart, timestepEnd);
    timestepStart = timesteps(0);
    timestepEnd = timesteps(1);

    // Get the information on the domain
    double minTime = inputField.GetMinTime();
    double maxTime = inputField.GetMaxTime();
    int numberOfTimeSteps = inputField.GetNumberOfTimeSteps();
    double timeSpacing = (maxTime - minTime) * (1.0 / double(numberOfTimeSteps - 1));

    const Vector3i res = inputField.GetSpatialGridSize();
    Eigen::Vector3d minDomainBounds = inputField.GetSpatialMin();
    Eigen::Vector3d maxDomainBounds = inputField.GetSpatialMax();
    Eigen::Vector3d domainSize = (maxDomainBounds - minDomainBounds);
    Eigen::Vector3d gridSpacing(domainSize(0) * (1.0 / double(res(0) - 1)), domainSize(1) * (1.0 / double(res(1) - 1)), domainSize(2) * (1.0 / double(res(2) - 1)));

    std::array<int, 4> dims = { res(0), res(1), res(2), numberOfTimeSteps };
    const int Nx = dims[0];
    const int Ny = dims[1];
    const int Nz = dims[2];
    const int Nt = dims[3];
    std::array<double, 4> spacing = { gridSpacing(0), gridSpacing(1), gridSpacing(2), timeSpacing };
    std::array<double, 4> boundsMin = { minDomainBounds(0), minDomainBounds(1), minDomainBounds(2), minTime };
    const auto numberOfDataPoints = dims[0] * dims[1] * dims[2] * dims[3];

    // need store observed time derivatives?
    bool storeObservedTimeDerivatives = observedTimeDerivatives.GetNumberOfTimeSteps() == inputField.GetNumberOfTimeSteps();

    // NEW: copy input data and compute derivatives

    std::vector<int> timeIndices(numberOfTimeSteps);
    std::iota(timeIndices.begin(), timeIndices.end(), 0);

    // In 3D, M is a 3 ?12 matrix MTM*u=Mtv' ;  u is unknow shape 12x1; v' is dv/dt  shape is 3x1
    int systemSize = 12;
    if (invariance == EInvariance::Objective)
        systemSize = 12;
	else if (invariance == EInvariance::Displacement)
	{
		 //EApproximation::Taylor:
		systemSize = taylorOrder * taylorOrder * taylorOrder + 6 * taylorOrder * taylorOrder + 11 * taylorOrder + 6;
			
		}

    using EigenMatrixX = Eigen::MatrixXd;
    using EigenVectorX = Eigen::VectorXd;

    Eigen::Matrix<float, Eigen::Dynamic, 3>& out_data_v_minus_u = resultVminusUfield.GetData();

    STARTTIMER_STRING(timerString);
    //		// Parallel loop over time slices
    std::for_each(policy, timeIndices.begin(), timeIndices.end(),
        [&](int tid) {
            int points_computed_this_slice=0;
			std::string timerInfo = "GGT solving for timestep " + std::to_string(tid);

          	STARTTIMER_STRING(timerInfo);
            const auto time = tid * timeSpacing + minTime;

            std::vector<EigenMatrixX> _MTM(res.x() * res.y() * res.z(), EigenMatrixX(systemSize, systemSize));
            std::vector<EigenVectorX> _MTb(res.x() * res.y() * res.z(), EigenMatrixX(systemSize, 1));
            std::vector<EigenVectorX> _b(res.x() * res.y() * res.z(), EigenMatrixX(systemSize, 1));
            std::vector<bool> _HasM(res.x() * res.y() * res.z(), false);
        // chunk size, meaning each thread will process 16 iterations of the loop at a time before being assigned the next chunk.

#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
            for (int z = 0; z < res.z(); ++z)
                for (int y = 0; y < res.y(); ++y)
                    for (int x = 0; x < res.x(); ++x) {
                        EigenMatrixX MTM(systemSize, systemSize);
                        MTM.setZero();
                        EigenVectorX MTb(systemSize, 1);
                        MTb.setZero();

                        _MTM[z * res.x() * res.y() + y * res.x() + x] = MTM;
                        _MTb[z * res.x() * res.y() + y * res.x() + x] = MTb;
                        _b[z * res.x() * res.y() + y * res.x() + x] = MTb;
                    }

#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
            for (int iz = 0; iz < res.z(); ++iz) {
                double z = boundsMin[2] + iz * spacing[2];
                for (int iy = 0; iy < res.y(); ++iy) {
                    double y = boundsMin[1] + iy * spacing[1];
                    for (int ix = 0; ix < res.x(); ++ix) {
                        double x = boundsMin[0] + ix * spacing[0];
                        int linIndex = iz * res.x() * res.y() + iy * res.x() + ix;
                        {
                            Eigen::Vector3f pos = Eigen::Vector3f(x, y, z);

                            Eigen::Vector3f X = pos;
                            Eigen::Vector3f V = inputField.GetVector(pos, (float)time);
                            if (V.isZero())
                                continue;
                            points_computed_this_slice++;
                            Eigen::Matrix3f XT = skewMat(X);
                            Eigen::Matrix3f VT = skewMat(V);
                            Eigen::Matrix3f J = inputField.getVelocityGradientTensor(ix, iy, iz, tid);
                            Eigen::Matrix3f F = -J * XT + VT;
                            Eigen::Vector3f Jxv = -J * X + V;

                            // setup the matrix M (test matrix multiplication...)
                            EigenMatrixX M(3, systemSize);
                             if (invariance == EInvariance::Objective)
                            {
                                M(0, 0) = (Float)F(0, 0);
                                M(0, 1) = (Float)F(0, 1);
                                M(0, 2) = (Float)F(0, 2);
                                M(0, 3) = (Float)J(0, 0);
                                M(0, 4) = (Float)J(0, 1);
                                M(0, 5) = (Float)J(0, 2);
                                M(0, 6) = (Float)XT(0, 0);
                                M(0, 7) = (Float)XT(0, 1);
                                M(0, 8) = (Float)XT(0, 2);
                                M(0, 9) = -1;
                                M(0, 10) = 0;
                                M(0, 11) = 0;
                                M(1, 0) = (Float)F(1, 0);
                                M(1, 1) = (Float)F(1, 1);
                                M(1, 2) = (Float)F(1, 2);
                                M(1, 3) = (Float)J(1, 0);
                                M(1, 4) = (Float)J(1, 1);
                                M(1, 5) = (Float)J(1, 2);
                                M(1, 6) = (Float)XT(1, 0);
                                M(1, 7) = (Float)XT(1, 1);
                                M(1, 8) = (Float)XT(1, 2);
                                M(1, 9) = 0;
                                M(1, 10) = -1;
                                M(1, 11) = 0;
                                M(2, 0) = (Float)F(2, 0);
                                M(2, 1) = (Float)F(2, 1);
                                M(2, 2) = (Float)F(2, 2);
                                M(2, 3) = (Float)J(2, 0);
                                M(2, 4) = (Float)J(2, 1);
                                M(2, 5) = (Float)J(2, 2);
                                M(2, 6) = (Float)XT(2, 0);
                                M(2, 7) = (Float)XT(2, 1);
                                M(2, 8) = (Float)XT(2, 2);
                                M(2, 9) = 0;
                                M(2, 10) = 0;
                                M(2, 11) = -1;
                            }
							 else if (invariance == EInvariance::Displacement)
							 {
									 int index = 0;
									 for (int m = 0; m <= taylorOrder; ++m)
									 {
										 for (int i = m; i >= 0; --i)
										 {
											 for (int j = m - i; j >= 0; --j)
											 {
												 int k = m - i - j;

												 Eigen::Matrix3f f, g;
												 ComputeDisplacementSystemMatrixCoefficients(i, j, k, X, V, J, f, g);
												 M(0, index * 6 + 0) = (Float)f(0, 0);		M(1, index * 6 + 0) = (Float)f(1, 0);		M(2, index * 6 + 0) = (Float)f(2, 0);
												 M(0, index * 6 + 1) = (Float)f(0, 1);		M(1, index * 6 + 1) = (Float)f(1, 1);		M(2, index * 6 + 1) = (Float)f(2, 1);
												 M(0, index * 6 + 2) = (Float)f(0, 2);		M(1, index * 6 + 2) = (Float)f(1, 2);		M(2, index * 6 + 2) = (Float)f(2, 2);

												 M(0, index * 6 + 3) = (Float)g(0, 0);		M(1, index * 6 + 3) = (Float)g(1, 0);		M(2, index * 6 + 3) = (Float)g(2, 0);
												 M(0, index * 6 + 4) = (Float)g(0, 1);		M(1, index * 6 + 4) = (Float)g(1, 1);		M(2, index * 6 + 4) = (Float)g(2, 1);
												 M(0, index * 6 + 5) = (Float)g(0, 2);		M(1, index * 6 + 5) = (Float)g(1, 2);		M(2, index * 6 + 5) = (Float)g(2, 2);
												 index += 1;
											 }
										 }
									 }
								 
							 }


                            Eigen::MatrixXd MT = M.transpose();
                            auto dt = inputField.getPartialDerivativeT(ix, iy, iz, tid);
							if (invariance == EInvariance::Displacement)
								dt *= -1;
                            Eigen::Vector3d b(dt.x(), dt.y(), dt.z());

                            _MTM[linIndex] = MT * M;
                            _MTb[linIndex] = MT * b;
                            _HasM[linIndex] = true;

                            ///	store b for computing observed time derivatives
                            _b[linIndex] = b;
                        }
                    } // end for x
                } // end for y
            } // end for z
            std::string debugInfo = "GGT solving timestep " + std::to_string(tid) + ". points to compute this slice: " + std::to_string(points_computed_this_slice);
            LOG_D("%s",debugInfo.c_str());

            // -----------
            if (UseSummedAreaTables) {
#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
                for (int z = 0; z < res.z(); ++z)
                    for (int y = 0; y < res.y(); ++y)
                        for (int x = 1; x < res.x(); ++x) {
                            if (!_HasM[z * res.x() * res.y() + y * res.x() + (x - 1)])
                                continue;
                            _MTM[z * res.x() * res.y() + y * res.x() + x] += _MTM[z * res.x() * res.y() + y * res.x() + (x - 1)];
                            _MTb[z * res.x() * res.y() + y * res.x() + x] += _MTb[z * res.x() * res.y() + y * res.x() + (x - 1)];
                        }

#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
                for (int z = 0; z < res.z(); ++z)
                    for (int x = 0; x < res.x(); ++x)
                        for (int y = 1; y < res.y(); ++y) {
                            if (!_HasM[z * res.x() * res.y() + (y - 1) * res.x() + x])
                                continue;
                            _MTM[z * res.x() * res.y() + y * res.x() + x] += _MTM[z * res.x() * res.y() + (y - 1) * res.x() + x];
                            _MTb[z * res.x() * res.y() + y * res.x() + x] += _MTb[z * res.x() * res.y() + (y - 1) * res.x() + x];
                        }

#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
                for (int x = 0; x < res.x(); ++x)
                    for (int y = 0; y < res.y(); ++y)
                        for (int z = 1; z < res.z(); ++z) {
                            if (!_HasM[(z - 1) * res.x() * res.y() + y * res.x() + x])
                                continue;
                            _MTM[z * res.x() * res.y() + y * res.x() + x] += _MTM[(z - 1) * res.x() * res.y() + y * res.x() + x];
                            _MTb[z * res.x() * res.y() + y * res.x() + x] += _MTb[(z - 1) * res.x() * res.y() + y * res.x() + x];
                        }
            }
        // -----------
		
        //				//solving
#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
            for (int z = 0; z < res.z(); ++z) {
                for (int y = 0; y < res.y(); ++y)
                    for (int x = 0; x < res.x(); ++x) {
                        if (!_HasM[z * res.x() * res.y() + y * res.x() + x])
                            continue;

                        EigenMatrixX MTM(systemSize, systemSize);
                        MTM.setZero();
                        EigenVectorX MTb(systemSize, 1);
                        MTb.setZero();

                        int x1 = std::min(std::max(0, x - NeighborhoodU), res.x() - 1);
                        int y1 = std::min(std::max(0, y - NeighborhoodU), res.y() - 1);
                        int z1 = std::min(std::max(0, z - NeighborhoodU), res.z() - 1);
                        int x2 = std::min(std::max(0, x + NeighborhoodU), res.x() - 1);
                        int y2 = std::min(std::max(0, y + NeighborhoodU), res.y() - 1);
                        int z2 = std::min(std::max(0, z + NeighborhoodU), res.z() - 1);

                        if (UseSummedAreaTables) {
                            MTM = _MTM[z2 * res.x() * res.y() + y2 * res.x() + x2] - _MTM[z1 * res.x() * res.y() + y2 * res.x() + x2]
                                - _MTM[z2 * res.x() * res.y() + y1 * res.x() + x2] - _MTM[z2 * res.x() * res.y() + y2 * res.x() + x1]
                                + _MTM[z2 * res.x() * res.y() + y1 * res.x() + x1] + _MTM[z1 * res.x() * res.y() + y2 * res.x() + x1]
                                + _MTM[z1 * res.x() * res.y() + y1 * res.x() + x2] - _MTM[z1 * res.x() * res.y() + y1 * res.x() + x1];

                            MTb = _MTb[z2 * res.x() * res.y() + y2 * res.x() + x2] - _MTb[z1 * res.x() * res.y() + y2 * res.x() + x2]
                                - _MTb[z2 * res.x() * res.y() + y1 * res.x() + x2] - _MTb[z2 * res.x() * res.y() + y2 * res.x() + x1]
                                + _MTb[z2 * res.x() * res.y() + y1 * res.x() + x1] + _MTb[z1 * res.x() * res.y() + y2 * res.x() + x1]
                                + _MTb[z1 * res.x() * res.y() + y1 * res.x() + x2] - _MTb[z1 * res.x() * res.y() + y1 * res.x() + x1];
                        } else {
                            for (int wz = z1; wz <= z2; ++wz)
                                for (int wy = y1; wy <= y2; ++wy)
                                    for (int wx = x1; wx <= x2; ++wx) {
                                        MTM += _MTM[wz * res[0] * res[1] + wy * res[0] + wx];
                                        MTb += _MTb[wz * res[0] * res[1] + wy * res[0] + wx];
                                    }
                        }

                        Eigen::Vector3f pos = Eigen::Vector3f(boundsMin[0] + x * spacing[0], boundsMin[1] + y * spacing[1], boundsMin[2] + z * spacing[2]);
                        Eigen::Vector3f X = pos;
                        Eigen::Vector3f V = inputField.GetVector(pos, (float)time);

                        // solve
                        //const double epsilon = 1e-7; // Tiny scalar to stabilize the solve
                        //MTM.diagonal().array() += epsilon;
                        EigenVectorX uu = MTM.fullPivHouseholderQr().solve(MTb);
                        double uu0 = uu(0);
                        double uu1 = uu(1);
                        double uu2 = uu(2);
                        double uu3 = uu(3);
                        double uu4 = uu(4);
                        double uu5 = uu(5);
                        Eigen::Vector3f U123(uu0, uu1, uu2);
                        Eigen::Vector3f U456(uu3, uu4, uu5);
                        // Eigen::Matrix3f U123T = skewMat(U123);
                        Eigen::Matrix3f XT = skewMat(X);
                        Eigen::Matrix3f VT = skewMat(V);

                         Eigen::Matrix3f J;
                        // compute v_new and J_new at the center
                        Eigen::Vector3f vnew(0, 0, 0);
                        Eigen::Matrix3f Jnew = Eigen::Matrix3f::Identity();


						if (invariance == EInvariance::Objective)
						{
							vnew = V + U123.cross(X) + U456;
						}
						else if (invariance == EInvariance::Similarity)
						{
                            assert(false);
							vnew = V + U123.cross(X) - ((double)uu(12)) * X + U456;
						}
						else if (invariance == EInvariance::Affine)
						{
                            assert(false);
							double h00_1 = -uu(0);
							double h10_1 = -uu(1);
							double h20_1 = -uu(2);
							double h01_1 = -uu(3);
							double h11_1 = -uu(4);
							double h21_1 = -uu(5);
							double h02_1 = -uu(6);
							double h12_1 = -uu(7);
							double h22_1 = -uu(8);
							Eigen::Matrix3f H1 = makeEigenMatrix3f(h00_1, h01_1, h02_1,
								h10_1, h11_1, h12_1,
								h20_1, h21_1, h22_1);
							Eigen::Vector3f k1(uu(9), uu(10), uu(11));

							vnew = V + H1 * X + k1;
						}
						else if (invariance == EInvariance::Displacement)
						{
							//if (approximation == EApproximation::Taylor)
							{
								// construct velocity
								{
									Eigen::Vector3f Ft(0, 0, 0);
									for (int m = 0; m <= taylorOrder; ++m)
									{
										for (int i = 0; i <= m; ++i)
										{
											for (int j = 0; j <= m - i; ++j)
											{
												int fu_i = i, fu_j = j, fu_k = m - i - j;
												// general formular to get the linear index for f:  (i+j+k)*(i+j+k+1)*(i+j+k+2)/3 + (k+j)*(k+j+1) + 2*k
												int linear_fu = (fu_i + fu_j + fu_k) * (fu_i + fu_j + fu_k + 1) * (fu_i + fu_j + fu_k + 2) / 3 + (fu_j + fu_k) * (fu_j + fu_k + 1) + 2 * fu_k;
												double fu = uu(3 * linear_fu + 0);
												double fv = uu(3 * linear_fu + 1);
												double fw = uu(3 * linear_fu + 2);

												int fac_i = factorial(fu_i);
												int fac_j = factorial(fu_j);
												int fac_k = factorial(fu_k);
												Ft += pow(X.x(), fu_i) * pow(X.y(), fu_j) * pow(X.z(), fu_k) / (fac_i * fac_j * fac_k) * Eigen::Vector3f(fu, fv, fw);
											}
										}
									}
									vnew = V + Ft;
								}
								// construct Jacobian
								{
									Eigen::Matrix3f nablaFt = Eigen::Matrix3f::Zero();
									for (int m = 0; m <= taylorOrder; ++m)
									{
										for (int i = 0; i <= m; ++i)
										{
											for (int j = 0; j <= m - i - 1; ++j)
											{
												int fx_i = i + 1, fx_j = j, fx_k = m - i - j - 1;
												int fy_i = i, fy_j = j + 1, fy_k = m - i - j - 1;
												int fz_i = i, fz_j = j, fz_k = m - i - j;

												// general formular to get the linear index for f:  (i+j+k)*(i+j+k+1)*(i+j+k+2)/3 + (k+j)*(k+j+1) + 2*k
												int linear_fx = (fx_i + fx_j + fx_k) * (fx_i + fx_j + fx_k + 1) * (fx_i + fx_j + fx_k + 2) / 3 + (fx_j + fx_k) * (fx_j + fx_k + 1) + 2 * fx_k;
												int linear_fy = (fy_i + fy_j + fy_k) * (fy_i + fy_j + fy_k + 1) * (fy_i + fy_j + fy_k + 2) / 3 + (fy_j + fy_k) * (fy_j + fy_k + 1) + 2 * fy_k;
												int linear_fz = (fz_i + fz_j + fz_k) * (fz_i + fz_j + fz_k + 1) * (fz_i + fz_j + fz_k + 2) / 3 + (fz_j + fz_k) * (fz_j + fz_k + 1) + 2 * fz_k;
												double fxu = uu(3 * linear_fx + 0);
												double fxv = uu(3 * linear_fx + 1);
												double fxw = uu(3 * linear_fx + 2);

												double fyu = uu(3 * linear_fy + 0);
												double fyv = uu(3 * linear_fy + 1);
												double fyw = uu(3 * linear_fy + 2);

												double fzu = uu(3 * linear_fz + 0);
												double fzv = uu(3 * linear_fz + 1);
												double fzw = uu(3 * linear_fz + 2);

												int fac_i = factorial(i);
												int fac_j = factorial(j);
												int fac_k = factorial(m - i - j - 1);
												nablaFt += pow(X.x(), i) * pow(X.y(), j) * pow(X.z(), m - i - j - 1) / (fac_i * fac_j * fac_k) *
													makeEigenMatrix3f(
														fxu, fyu, fzu,
														fxv, fyv, fzv,
														fxw, fyw, fzw);
											}
										}
									}
									Jnew = J + nablaFt;
								}
							}
						}



						//output->SetVertexDataAt({ x, y, z }, vnew);
                        unsigned long idx3D = z * dims[0] * dims[1] + y * dims[0] + x;
                        unsigned long tupleIdx = (unsigned long)(tid * dims[0] * dims[1] * dims[2]) + idx3D;
                        out_data_v_minus_u(tupleIdx, 0) = vnew.x();
                        out_data_v_minus_u(tupleIdx, 1) = vnew.y();
                        out_data_v_minus_u(tupleIdx, 2) = vnew.z();

                        if (invariance == EInvariance::Objective &&storeObservedTimeDerivatives) {
                            EigenVectorX sum_b(3, 1);
                            sum_b.setZero();
                            for (int wz = z1; wz <= z2; ++wz)
                                for (int wy = y1; wy <= y2; ++wy)
                                    for (int wx = x1; wx <= x2; ++wx) {
                                        sum_b += _b[wz * res[0] * res[1] + wy * res[0] + wx];
                                    }

                            // norm of |Mx-b|^2= xTMTMx-2bTMx +btb
                            Eigen::VectorXd Mx_b_2 = uu.transpose() * MTM * uu - 2 * MTb.transpose() * uu + sum_b.transpose() * sum_b;
                            double observedTimeDerivative = Mx_b_2.norm();
                            observedTimeDerivatives.SetValue(x, y, z, tid, observedTimeDerivative);
                        }

					}//iterate x
      
            }//iterate z

            STOPTIMER_STRING(timerInfo);
        });

    std::cout << "optimization done." << std::endl;
    STOPTIMER_STRING(timerString);


    // compute (v-u)*-1.0 + v
    Eigen::Matrix<float, Eigen::Dynamic, 3>& out_actual_u = resultUfield.GetData();
    const auto& actual_v = inputField.GetDataView();
    out_actual_u = out_data_v_minus_u * -1.0 + actual_v;
    /*	for (int it = 0; it < dims[2]; it++) {
    int timeSliceOffset = it * dims[0] * dims[1] * 2;
    for (int iy = 0; iy < dims[1]; iy++) {
            int pixelRowOffset = timeSliceOffset + iy * dims[0] * 2;
            for (int ix = 0; ix < dims[0]; ix++) {
                    int tupleIdx = it * dims[0] * dims[1] + iy * dims[0] + ix;
                    int pixelOffset = pixelRowOffset + ix * 2;
                    out_actual_u(tupleIdx, 0) += vectorField[it](iy, ix)(0);
                    out_actual_u(tupleIdx, 1) += vectorField[it](iy, ix)(1);
                    out_actual_u(tupleIdx, 1) += vectorField[it](iy, ix)(1);
            }
    }
}*/

    return;
}


void GenericLocalOptimization3d::computeWithFilter(const Discrete3DFlowField<float>& inputField, int timestepStart, int timestepEnd, Discrete3DFlowField<float>& resultUfield, Discrete3DFlowField<float>& resultVminusUfield, DiscreteScalarField3D<float>& observedTimeDerivatives)
{
    using namespace Eigen;
    using Float = float;
    constexpr int taylorOrder=1;//hiper paramter used for  EInvariance::Displacement
    const auto&invariance=mInvariance;
    auto info=inputField.getPrintOutGridInfo();
    LOG_D("comput observer with infor: %s" ,info.c_str());

    std::string timerString= "GGTCompute EInvariance::Objective_withFilter";
	if (invariance == EInvariance::Displacement)
        timerString="GGTCompute EInvariance::Displacement_withFilter";


    Vector2i timesteps = ClampTimstep2ValidRange(inputField, timestepStart, timestepEnd);
    timestepStart = timesteps(0);
    timestepEnd = timesteps(1);

    // Get the information on the domain
    double minTime = inputField.GetMinTime();
    double maxTime = inputField.GetMaxTime();
    int numberOfTimeSteps = inputField.GetNumberOfTimeSteps();
    double timeSpacing = (maxTime - minTime) * (1.0 / double(numberOfTimeSteps - 1));

    const Vector3i res = inputField.GetSpatialGridSize();
    Eigen::Vector3d minDomainBounds = inputField.GetSpatialMin();
    Eigen::Vector3d maxDomainBounds = inputField.GetSpatialMax();
    Eigen::Vector3d domainSize = (maxDomainBounds - minDomainBounds);
    Eigen::Vector3d gridSpacing(domainSize(0) * (1.0 / double(res(0) - 1)), domainSize(1) * (1.0 / double(res(1) - 1)), domainSize(2) * (1.0 / double(res(2) - 1)));

    std::array<int, 4> dims = { res(0), res(1), res(2), numberOfTimeSteps };
    const int Nx = dims[0];
    const int Ny = dims[1];
    const int Nz = dims[2];
    const int Nt = dims[3];
    std::array<double, 4> spacing = { gridSpacing(0), gridSpacing(1), gridSpacing(2), timeSpacing };
    std::array<double, 4> boundsMin = { minDomainBounds(0), minDomainBounds(1), minDomainBounds(2), minTime };
    const auto numberOfDataPoints = dims[0] * dims[1] * dims[2] * dims[3];

    // need store observed time derivatives?
    bool storeObservedTimeDerivatives = observedTimeDerivatives.GetNumberOfTimeSteps() == inputField.GetNumberOfTimeSteps();

    // NEW: copy input data and compute derivatives

    std::vector<int> timeIndices(numberOfTimeSteps);
    std::iota(timeIndices.begin(), timeIndices.end(), 0);

    // In 3D, M is a 3 ?12 matrix MTM*u=Mtv' ;  u is unknow shape 12x1; v' is dv/dt  shape is 3x1
    int systemSize = 12;
    if (invariance == EInvariance::Objective)
        systemSize = 12;
	else if (invariance == EInvariance::Displacement)
	{
		 //EApproximation::Taylor:
		systemSize = taylorOrder * taylorOrder * taylorOrder + 6 * taylorOrder * taylorOrder + 11 * taylorOrder + 6;
			
		}

    using EigenMatrixX = Eigen::MatrixXd;
    using EigenVectorX = Eigen::VectorXd;

    Eigen::Matrix<float, Eigen::Dynamic, 3>& out_data_v_minus_u = resultVminusUfield.GetData();

    STARTTIMER_STRING(timerString);
    //		// Parallel loop over time slices
    std::for_each(policy, timeIndices.begin(), timeIndices.end(),
        [&](int tid) {
            const auto time = tid * timeSpacing + minTime;

            //consturct filtering mapping
			std::vector<size_t> activeIndices; // store only active grid point linear indices
			for (int iz = 0; iz < res.z(); ++iz) {
				double z = boundsMin[2] + iz * spacing[2];
				for (int iy = 0; iy < res.y(); ++iy) {
					double y = boundsMin[1] + iy * spacing[1];
					for (int ix = 0; ix < res.x(); ++ix) {
						double x = boundsMin[0] + ix * spacing[0];
						Eigen::Vector3f pos = Eigen::Vector3f(x, y, z);
						Eigen::Vector3f V = inputField.GetVector(pos, (float)time);
						if (V.isZero() || (mGridFilter && mGridFilter(ix, iy, iz, tid) == false))
							continue;
						size_t linIndex = z * res.x() * res.y() + y * res.x() + x;
						activeIndices.push_back(linIndex);
					}
				}
			}
            // Map dense linear index to sparse index (only active points)
			size_t activePoints = activeIndices.size();
            std::unordered_map<size_t, size_t> denseToSparse;
            for (size_t sparseIdx = 0; sparseIdx < activePoints; ++sparseIdx)
                denseToSparse[activeIndices[sparseIdx]] = sparseIdx;
		

            std::vector<EigenMatrixX> _MTM(activePoints, EigenMatrixX(systemSize, systemSize));
            std::vector<EigenVectorX> _MTb(activePoints, EigenMatrixX(systemSize, 1));
            std::vector<EigenVectorX> _b(activePoints, EigenMatrixX(systemSize, 1));

           std::string timerInfo = "GGT with filter solving for timestep" + std::to_string(tid)+" with points" + std::to_string(activePoints );
          	STARTTIMER_STRING(timerInfo);
        // chunk size, meaning each thread will process 16 iterations of the loop at a time before being assigned the next chunk.
#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
            for (int iz = 0; iz < res.z(); ++iz)
                for (int iy = 0; iy < res.y(); ++iy)
                    for (int ix = 0; ix < res.x(); ++ix) {
						size_t linIndex = iz * res.x() * res.y() + iy * res.x() + ix;
						// Check filter
						if (denseToSparse.find(linIndex) == denseToSparse.end())
							continue;
						size_t sparseIdx = denseToSparse[linIndex];


                        EigenMatrixX MTM(systemSize, systemSize);
                        MTM.setZero();
                        EigenVectorX MTb(systemSize, 1);
                        MTb.setZero();

                        _MTM[sparseIdx] = MTM;
                        _MTb[sparseIdx] = MTb;
                        _b[sparseIdx] = MTb;
                    }


#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
            for (int iz = 0; iz < res.z(); ++iz) {
                double z = boundsMin[2] + iz * spacing[2];
                for (int iy = 0; iy < res.y(); ++iy) {
                    double y = boundsMin[1] + iy * spacing[1];
                    for (int ix = 0; ix < res.x(); ++ix) {
                        double x = boundsMin[0] + ix * spacing[0];
                        int linIndex = iz * res.x() * res.y() + iy * res.x() + ix;
                        	// Check filter
						if (denseToSparse.find(linIndex) == denseToSparse.end())
							continue;
                        size_t sparseIdx = denseToSparse[linIndex];
                        {
                            Eigen::Vector3f pos = Eigen::Vector3f(x, y, z);

                            Eigen::Vector3f X = pos;
                            Eigen::Vector3f V = inputField.GetVector(pos, (float)time);
                     
                            Eigen::Matrix3f XT = skewMat(X);
                            Eigen::Matrix3f VT = skewMat(V);
                            Eigen::Matrix3f J = inputField.getVelocityGradientTensor(ix, iy, iz, tid);
                            Eigen::Matrix3f F = -J * XT + VT;
                            Eigen::Vector3f Jxv = -J * X + V;

                            // setup the matrix M (test matrix multiplication...)
                            EigenMatrixX M(3, systemSize);
                             if (invariance == EInvariance::Objective)
                            {
                                M(0, 0) = (Float)F(0, 0);
                                M(0, 1) = (Float)F(0, 1);
                                M(0, 2) = (Float)F(0, 2);
                                M(0, 3) = (Float)J(0, 0);
                                M(0, 4) = (Float)J(0, 1);
                                M(0, 5) = (Float)J(0, 2);
                                M(0, 6) = (Float)XT(0, 0);
                                M(0, 7) = (Float)XT(0, 1);
                                M(0, 8) = (Float)XT(0, 2);
                                M(0, 9) = -1;
                                M(0, 10) = 0;
                                M(0, 11) = 0;
                                M(1, 0) = (Float)F(1, 0);
                                M(1, 1) = (Float)F(1, 1);
                                M(1, 2) = (Float)F(1, 2);
                                M(1, 3) = (Float)J(1, 0);
                                M(1, 4) = (Float)J(1, 1);
                                M(1, 5) = (Float)J(1, 2);
                                M(1, 6) = (Float)XT(1, 0);
                                M(1, 7) = (Float)XT(1, 1);
                                M(1, 8) = (Float)XT(1, 2);
                                M(1, 9) = 0;
                                M(1, 10) = -1;
                                M(1, 11) = 0;
                                M(2, 0) = (Float)F(2, 0);
                                M(2, 1) = (Float)F(2, 1);
                                M(2, 2) = (Float)F(2, 2);
                                M(2, 3) = (Float)J(2, 0);
                                M(2, 4) = (Float)J(2, 1);
                                M(2, 5) = (Float)J(2, 2);
                                M(2, 6) = (Float)XT(2, 0);
                                M(2, 7) = (Float)XT(2, 1);
                                M(2, 8) = (Float)XT(2, 2);
                                M(2, 9) = 0;
                                M(2, 10) = 0;
                                M(2, 11) = -1;
                            }
							 else if (invariance == EInvariance::Displacement)
							 {
									 int index = 0;
									 for (int m = 0; m <= taylorOrder; ++m)
									 {
										 for (int i = m; i >= 0; --i)
										 {
											 for (int j = m - i; j >= 0; --j)
											 {
												 int k = m - i - j;

												 Eigen::Matrix3f f, g;
												 ComputeDisplacementSystemMatrixCoefficients(i, j, k, X, V, J, f, g);
												 M(0, index * 6 + 0) = (Float)f(0, 0);		M(1, index * 6 + 0) = (Float)f(1, 0);		M(2, index * 6 + 0) = (Float)f(2, 0);
												 M(0, index * 6 + 1) = (Float)f(0, 1);		M(1, index * 6 + 1) = (Float)f(1, 1);		M(2, index * 6 + 1) = (Float)f(2, 1);
												 M(0, index * 6 + 2) = (Float)f(0, 2);		M(1, index * 6 + 2) = (Float)f(1, 2);		M(2, index * 6 + 2) = (Float)f(2, 2);

												 M(0, index * 6 + 3) = (Float)g(0, 0);		M(1, index * 6 + 3) = (Float)g(1, 0);		M(2, index * 6 + 3) = (Float)g(2, 0);
												 M(0, index * 6 + 4) = (Float)g(0, 1);		M(1, index * 6 + 4) = (Float)g(1, 1);		M(2, index * 6 + 4) = (Float)g(2, 1);
												 M(0, index * 6 + 5) = (Float)g(0, 2);		M(1, index * 6 + 5) = (Float)g(1, 2);		M(2, index * 6 + 5) = (Float)g(2, 2);
												 index += 1;
											 }
										 }
									 }
								 
							 }


                            Eigen::MatrixXd MT = M.transpose();
                            auto dt = inputField.getPartialDerivativeT(ix, iy, iz, tid);
							if (invariance == EInvariance::Displacement)
								dt *= -1;
                            Eigen::Vector3d b(dt.x(), dt.y(), dt.z());

                            _MTM[sparseIdx] = MT * M;
                            _MTb[sparseIdx] = MT * b;
                            _b[sparseIdx] = b;
                        }
                    } // end for x
                } // end for y
            } // end for z
   
      //solving
#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
          for (int iz = 0; iz < res.z(); ++iz){
                for (int iy = 0; iy < res.y(); ++iy){
                    for (int ix = 0; ix < res.x(); ++ix){
                         int linIndex = iz * res.x() * res.y() + iy * res.x() + ix;
                        	// Check filter
						if (denseToSparse.find(linIndex) == denseToSparse.end())
							continue;

                        EigenMatrixX MTM(systemSize, systemSize);
                        MTM.setZero();
                        EigenVectorX MTb(systemSize, 1);
                        MTb.setZero();

                        int x1 = std::min(std::max(0, ix - NeighborhoodU), res.x() - 1);
                        int y1 = std::min(std::max(0, iy - NeighborhoodU), res.y() - 1);
                        int z1 = std::min(std::max(0, iz - NeighborhoodU), res.z() - 1);
                        int x2 = std::min(std::max(0, ix + NeighborhoodU), res.x() - 1);
                        int y2 = std::min(std::max(0, iy + NeighborhoodU), res.y() - 1);
                        int z2 = std::min(std::max(0, iz + NeighborhoodU), res.z() - 1);

                        for (int wz = z1; wz <= z2; ++wz)
                            for (int wy = y1; wy <= y2; ++wy)
                                for (int wx = x1; wx <= x2; ++wx) {
                                     int neighborlinIndex = wz * res.x() * res.y() + wy * res.x() + wx;
                                     if (denseToSparse.find(neighborlinIndex) == denseToSparse.end())
							            continue;
                                      size_t neighborSparseIdx = denseToSparse[neighborlinIndex];
                                    MTM += _MTM[neighborSparseIdx];
                                    MTb += _MTb[neighborSparseIdx];
                                }
                   

                        Eigen::Vector3f pos = Eigen::Vector3f(boundsMin[0] + ix * spacing[0], boundsMin[1] + iy * spacing[1], boundsMin[2] + iz * spacing[2]);
                        Eigen::Vector3f X = pos;
                        Eigen::Vector3f V = inputField.GetVector(pos, (float)time);

                        // solve
                        //const double epsilon = 1e-7; // Tiny scalar to stabilize the solve
                        //MTM.diagonal().array() += epsilon;
                        EigenVectorX uu = MTM.fullPivHouseholderQr().solve(MTb);
                        double uu0 = uu(0);
                        double uu1 = uu(1);
                        double uu2 = uu(2);
                        double uu3 = uu(3);
                        double uu4 = uu(4);
                        double uu5 = uu(5);
                        Eigen::Vector3f U123(uu0, uu1, uu2);
                        Eigen::Vector3f U456(uu3, uu4, uu5);
                        // Eigen::Matrix3f U123T = skewMat(U123);
                        Eigen::Matrix3f XT = skewMat(X);
                        Eigen::Matrix3f VT = skewMat(V);

                         Eigen::Matrix3f J;
                        // compute v_new and J_new at the center
                        Eigen::Vector3f vnew(0, 0, 0);
                        Eigen::Matrix3f Jnew = Eigen::Matrix3f::Identity();


						if (invariance == EInvariance::Objective)
						{
							vnew = V + U123.cross(X) + U456;
						}
						else if (invariance == EInvariance::Similarity)
						{
                            assert(false);
							vnew = V + U123.cross(X) - ((double)uu(12)) * X + U456;
						}
						else if (invariance == EInvariance::Affine)
						{
                            assert(false);
							double h00_1 = -uu(0);
							double h10_1 = -uu(1);
							double h20_1 = -uu(2);
							double h01_1 = -uu(3);
							double h11_1 = -uu(4);
							double h21_1 = -uu(5);
							double h02_1 = -uu(6);
							double h12_1 = -uu(7);
							double h22_1 = -uu(8);
							Eigen::Matrix3f H1 = makeEigenMatrix3f(h00_1, h01_1, h02_1,
								h10_1, h11_1, h12_1,
								h20_1, h21_1, h22_1);
							Eigen::Vector3f k1(uu(9), uu(10), uu(11));

							vnew = V + H1 * X + k1;
						}
						else if (invariance == EInvariance::Displacement)
						{
							//if (approximation == EApproximation::Taylor)
							{
								// construct velocity
								{
									Eigen::Vector3f Ft(0, 0, 0);
									for (int m = 0; m <= taylorOrder; ++m)
									{
										for (int i = 0; i <= m; ++i)
										{
											for (int j = 0; j <= m - i; ++j)
											{
												int fu_i = i, fu_j = j, fu_k = m - i - j;
												// general formular to get the linear index for f:  (i+j+k)*(i+j+k+1)*(i+j+k+2)/3 + (k+j)*(k+j+1) + 2*k
												int linear_fu = (fu_i + fu_j + fu_k) * (fu_i + fu_j + fu_k + 1) * (fu_i + fu_j + fu_k + 2) / 3 + (fu_j + fu_k) * (fu_j + fu_k + 1) + 2 * fu_k;
												double fu = uu(3 * linear_fu + 0);
												double fv = uu(3 * linear_fu + 1);
												double fw = uu(3 * linear_fu + 2);

												int fac_i = factorial(fu_i);
												int fac_j = factorial(fu_j);
												int fac_k = factorial(fu_k);
												Ft += pow(X.x(), fu_i) * pow(X.y(), fu_j) * pow(X.z(), fu_k) / (fac_i * fac_j * fac_k) * Eigen::Vector3f(fu, fv, fw);
											}
										}
									}
									vnew = V + Ft;
								}
								// construct Jacobian
								{
									Eigen::Matrix3f nablaFt = Eigen::Matrix3f::Zero();
									for (int m = 0; m <= taylorOrder; ++m)
									{
										for (int i = 0; i <= m; ++i)
										{
											for (int j = 0; j <= m - i - 1; ++j)
											{
												int fx_i = i + 1, fx_j = j, fx_k = m - i - j - 1;
												int fy_i = i, fy_j = j + 1, fy_k = m - i - j - 1;
												int fz_i = i, fz_j = j, fz_k = m - i - j;

												// general formular to get the linear index for f:  (i+j+k)*(i+j+k+1)*(i+j+k+2)/3 + (k+j)*(k+j+1) + 2*k
												int linear_fx = (fx_i + fx_j + fx_k) * (fx_i + fx_j + fx_k + 1) * (fx_i + fx_j + fx_k + 2) / 3 + (fx_j + fx_k) * (fx_j + fx_k + 1) + 2 * fx_k;
												int linear_fy = (fy_i + fy_j + fy_k) * (fy_i + fy_j + fy_k + 1) * (fy_i + fy_j + fy_k + 2) / 3 + (fy_j + fy_k) * (fy_j + fy_k + 1) + 2 * fy_k;
												int linear_fz = (fz_i + fz_j + fz_k) * (fz_i + fz_j + fz_k + 1) * (fz_i + fz_j + fz_k + 2) / 3 + (fz_j + fz_k) * (fz_j + fz_k + 1) + 2 * fz_k;
												double fxu = uu(3 * linear_fx + 0);
												double fxv = uu(3 * linear_fx + 1);
												double fxw = uu(3 * linear_fx + 2);

												double fyu = uu(3 * linear_fy + 0);
												double fyv = uu(3 * linear_fy + 1);
												double fyw = uu(3 * linear_fy + 2);

												double fzu = uu(3 * linear_fz + 0);
												double fzv = uu(3 * linear_fz + 1);
												double fzw = uu(3 * linear_fz + 2);

												int fac_i = factorial(i);
												int fac_j = factorial(j);
												int fac_k = factorial(m - i - j - 1);
												nablaFt += pow(X.x(), i) * pow(X.y(), j) * pow(X.z(), m - i - j - 1) / (fac_i * fac_j * fac_k) *
													makeEigenMatrix3f(
														fxu, fyu, fzu,
														fxv, fyv, fzv,
														fxw, fyw, fzw);
											}
										}
									}
									Jnew = J + nablaFt;
								}
							}
						}



						//output->SetVertexDataAt({ x, y, z }, vnew);
                        unsigned long idx3D = linIndex;
                        unsigned long tupleIdx = (unsigned long)(tid * dims[0] * dims[1] * dims[2]) + idx3D;
                        out_data_v_minus_u(tupleIdx, 0) = vnew.x();
                        out_data_v_minus_u(tupleIdx, 1) = vnew.y();
                        out_data_v_minus_u(tupleIdx, 2) = vnew.z();

                        //if (invariance == EInvariance::Objective &&storeObservedTimeDerivatives) {
                        //    EigenVectorX sum_b(3, 1);
                        //    sum_b.setZero();
                        //    for (int wz = z1; wz <= z2; ++wz)
                        //        for (int wy = y1; wy <= y2; ++wy)
                        //            for (int wx = x1; wx <= x2; ++wx) {
                        //                sum_b += _b[wz * res[0] * res[1] + wy * res[0] + wx];
                        //            }
                        //    // norm of |Mx-b|^2= xTMTMx-2bTMx +btb
                        //    Eigen::VectorXd Mx_b_2 = uu.transpose() * MTM * uu - 2 * MTb.transpose() * uu + sum_b.transpose() * sum_b;
                        //    double observedTimeDerivative = Mx_b_2.norm();
                        //    observedTimeDerivatives.SetValue(x, y, z, tid, observedTimeDerivative);
                        //}

					}//iterate x
                }//iterate y
            }//iterate z

            STOPTIMER_STRING(timerInfo);
        });

    std::cout << "optimization done." << std::endl;
    STOPTIMER_STRING(timerString);


     //compute (v-u)*-1.0 + v
    Eigen::Matrix<float, Eigen::Dynamic, 3>& out_actual_u = resultUfield.GetData();
    const auto& actual_v = inputField.GetDataView();
    out_actual_u = out_data_v_minus_u * -1.0 + actual_v;
    /*	for (int it = 0; it < dims[2]; it++) {
    int timeSliceOffset = it * dims[0] * dims[1] * 2;
    for (int iy = 0; iy < dims[1]; iy++) {
            int pixelRowOffset = timeSliceOffset + iy * dims[0] * 2;
            for (int ix = 0; ix < dims[0]; ix++) {
                    int tupleIdx = it * dims[0] * dims[1] + iy * dims[0] + ix;
                    int pixelOffset = pixelRowOffset + ix * 2;
                    out_actual_u(tupleIdx, 0) += vectorField[it](iy, ix)(0);
                    out_actual_u(tupleIdx, 1) += vectorField[it](iy, ix)(1);
                    out_actual_u(tupleIdx, 1) += vectorField[it](iy, ix)(1);
            }
    }
}*/

    return;
}

DTG::PointAndTime3Df Worldline3D::interpolatePoint(double time) const
{
    return DTG::interpolatePointOnPathline(mPathline,time);
}

Eigen::Matrix3f ReferenceFrameTransformation3D::interpolateIntegratedRotation(double time) const
{
    if (mIntegratedRotation.empty()) {
        throw std::runtime_error("cannot interpolate integrated rotation. no rotation was computed.");
    }

    double tMin = getMinTime();
    double tMax = getMaxTime();

    if (time <= tMin)
        return mIntegratedRotation.front();
    if (time >= tMax)
        return mIntegratedRotation.back();

    double griddedTime = ((time - tMin) / (tMax - tMin)) * (getNumberOfTimeSteps() - 1);
    int timeLower = std::floor(griddedTime);
    int timeUpper = timeLower + 1;

    if (timeUpper >= getNumberOfTimeSteps()) {
        timeUpper = timeLower;
    }

    double alphaTime = griddedTime - timeLower;
    return mIntegratedRotation[timeLower] * (1.0 - alphaTime) + mIntegratedRotation[timeUpper] * alphaTime;
}

std::function<Eigen::Vector3f(const Eigen::Vector3f&)> ReferenceFrameTransformation3D::getTransformationLabToObserved(double time) const
{
    auto refPoint = getReferencePoint();
    auto refPointAtTime = interpolatePoint(time);
    auto rotation = interpolateIntegratedRotation(time);

     Eigen::Matrix4f translation1 = Eigen::Matrix4f::Identity();
    translation1.block<3, 1>(0, 3) = -refPointAtTime.position;

     Eigen::Matrix4f  translation2 = Eigen::Matrix4f::Identity();
    translation2.block<3, 1>(0, 3) = refPoint.position;

     Eigen::Matrix4f  rotationMatrix = Eigen::Matrix4f::Identity();

    rotationMatrix.block<3, 3>(0, 0) = rotation.transpose();

    Eigen::Matrix4f transformationMatrix = translation2 * rotationMatrix * translation1;

    return [transformationMatrix](const Eigen::Vector3f& inputPos) -> Eigen::Vector3f {
        Eigen::Vector4f pos;
        pos << inputPos, 1.0;
        Eigen::Vector4f transformedPos = transformationMatrix * pos;
        return transformedPos.head<3>();
    };
}



std::function<Eigen::Vector3f(const Eigen::Vector3f&)> ReferenceFrameTransformation3D::getTransformationObservedToLab(double time) const
{
    const auto refPoint = getReferencePoint();
    const auto refPointAtTime = interpolatePoint(time);
    const auto rotation = interpolateIntegratedRotation(time);

    Eigen::Matrix4f translation1 = Eigen::Matrix4f::Identity();
    translation1.block<3, 1>(0, 3) = refPointAtTime.position;

    Eigen::Matrix4f translation2 = Eigen::Matrix4f::Identity();
    translation2.block<3, 1>(0, 3) = -refPoint.position;

    Eigen::Matrix4f rotationMatrix = Eigen::Matrix4f::Identity();
    rotationMatrix.block<3, 3>(0, 0) = rotation;

    Eigen::Matrix4f transformationMatrix = translation1 * rotationMatrix * translation2;

    return [transformationMatrix](const Eigen::Vector3f& inputPos) -> Eigen::Vector3f {
        Eigen::Vector4f pos;
        pos << inputPos, 1.0;
        Eigen::Vector4f transformedPos = transformationMatrix * pos;
        return transformedPos.head<3>();
    };
}
Eigen::Matrix3f  ReferenceFrameTransformation3D::getIntegratedRotationObservedToLab(double time) const
{
	const Eigen::Matrix3f rotationMatrix = interpolateIntegratedRotation(time);
    return rotationMatrix ;
}



void HadwigerKillingOptimization3d::compute(const Discrete3DFlowField<float>& inputField, int timestepStart, int timestepEnd, Discrete3DFlowField<float>& resultUfield, Discrete3DFlowField<float>& resultVminusUfield, DiscreteScalarField3D<float>& observedTimeDerivatives)
{




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





}

