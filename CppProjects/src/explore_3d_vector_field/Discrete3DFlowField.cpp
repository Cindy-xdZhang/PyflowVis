#include "Discrete3DFlowField.h"
#include "logSystem/log.h"
#include "mydefs.h"
#include <execution>
#include "vtkFloatArray.h"
#include "vtkPointData.h"

//#define NO_PARALLEL
namespace {
//#if defined(_DEBUG )|| defined(NO_PARALLEL) 
//auto policy = std::execution::seq;
//#else
auto policy = std::execution::par_unseq;
//#endif
}

namespace DTG { // base class IFlowField3d
	

IDiscreteField3d::IDiscreteField3d(const FieldSpaceTimeInfo3D& FiledInformation)
    :IField3D(FiledInformation)
{

    
    mOutOfSpaceBoundaryPolicy = IDiscreteField3d::OutOfBoundPolicy::RepeatPolicy;
    //mOutOfTimeBoundaryPolicy = IDiscreteField3d::OutOfBoundPolicy::RepeatPolicy;
}

std::vector<DTG::pathline3Df> IDiscreteField3d::computeParallelPathlinesDoubleDirection(const std::vector<DTG::PointAndTime3Df>& startPositions, float stepSize, float restrictIntegrationTimeIntervalOneDirection, PathlineNumericalIntegrationMethod method, bool boundaryRElaxtion/*=false*/){


	const auto nLines = startPositions.size();
	std::vector<pathline3Df >pathlineVecOfVec;
	pathlineVecOfVec.resize(nLines);

	const auto maxTime = this->GetMaxTime();
	const auto minTime = this->GetMinTime();
    const auto maxIter=  (this->GetMaxTime() -this->GetMinTime())/stepSize;

	auto policy = std::execution::par_unseq;

	std::transform(policy, startPositions.begin(), startPositions.end(), pathlineVecOfVec.begin(),
		[this, maxTime, minTime, method, stepSize, maxIter,restrictIntegrationTimeIntervalOneDirection](const DTG::PointAndTime3Df& startPointAndTime) {

			std::vector<DTG::PointAndTime3Df>resultPointsInChartsForward = std::vector<DTG::PointAndTime<3, float>>();
			std::vector<DTG::PointAndTime3Df>resultPointsInChartsBackward = std::vector<DTG::PointAndTime<3, float>>();
			const DTG::Point<3, float> startPoint(startPointAndTime.position);
			double startTime = startPointAndTime.time;
			resultPointsInChartsForward.push_back(startPointAndTime);
			if ((startTime < maxTime - stepSize)) {
                  const auto forward_towards_time =startTime+restrictIntegrationTimeIntervalOneDirection;

				this->Integrate3DPathlineOneDirection(startPoint, startTime, forward_towards_time, stepSize, maxIter,resultPointsInChartsForward, method);
			}

			if (startTime > minTime + stepSize) {
                const auto backwards_towards_time =startTime-restrictIntegrationTimeIntervalOneDirection;

				this->Integrate3DPathlineOneDirection(startPoint, startTime, backwards_towards_time, stepSize, maxIter, resultPointsInChartsBackward, method);
			}

			std::reverse(resultPointsInChartsBackward.begin(), resultPointsInChartsBackward.end());
			resultPointsInChartsBackward.insert(resultPointsInChartsBackward.end(), resultPointsInChartsForward.begin(), resultPointsInChartsForward.end());
			return std::move(resultPointsInChartsBackward);
		});
	return pathlineVecOfVec;



}


std::vector<DTG::pathline3Df> IDiscreteField3d::computeParallelPathlinesDoubleDirection(const std::vector<DTG::PointAndTime3Df>& startPositions, float stepSize, int maxIter, PathlineNumericalIntegrationMethod method, bool boundaryRElaxtion/*=false*/)
{
	const auto nLines = startPositions.size();
	std::vector<pathline3Df >pathlineVecOfVec;
	pathlineVecOfVec.resize(nLines);

	const auto maxTime = this->GetMaxTime();
	const auto minTime = this->GetMinTime();

	auto policy = std::execution::par_unseq;

	std::transform(policy, startPositions.begin(), startPositions.end(), pathlineVecOfVec.begin(),
		[this, maxTime, minTime, method, stepSize, maxIter](const DTG::PointAndTime3Df& startPointAndTime) {

			std::vector<DTG::PointAndTime3Df>resultPointsInChartsForward = std::vector<DTG::PointAndTime<3, float>>();
			std::vector<DTG::PointAndTime3Df>resultPointsInChartsBackward = std::vector<DTG::PointAndTime<3, float>>();
			const DTG::Point<3, float> startPoint(startPointAndTime.position);
			double startTime = startPointAndTime.time;
			resultPointsInChartsForward.push_back(startPointAndTime);
			if ((startTime < maxTime - stepSize)) {
				this->Integrate3DPathlineOneDirection(startPoint, startTime, maxTime, stepSize, maxIter, resultPointsInChartsForward, method);
			}

			if (startTime > minTime + stepSize) {
				this->Integrate3DPathlineOneDirection(startPoint, startTime, minTime, stepSize, maxIter, resultPointsInChartsBackward, method);
			}

			std::reverse(resultPointsInChartsBackward.begin(), resultPointsInChartsBackward.end());
			resultPointsInChartsBackward.insert(resultPointsInChartsBackward.end(), resultPointsInChartsForward.begin(), resultPointsInChartsForward.end());
			return std::move(resultPointsInChartsBackward);
		});
	return pathlineVecOfVec;
}

template <typename K>
DTG::Discrete3DFlowField<K>::Discrete3DFlowField(const FieldSpaceTimeInfo3D& FiledInformation, bool initMem)
    : IDiscreteField3d(FiledInformation)
{
    // Allocate Memory for mData
    if (initMem) {
        uint64_t size=(uint64_t)this->mFieldInfo.XGridsize* (uint64_t)this->mFieldInfo.YGridsize*(uint64_t)this->mFieldInfo.ZGridsize*(uint64_t)this->mFieldInfo.numberOfTimeSteps;
        this->mData.resize(size, 3); // Use resize instead of Zero for better performance
    }
}

template <typename K>
DTG::Discrete3DFlowField<K>::~Discrete3DFlowField()
{
}

template <typename K>
void DTG::Discrete3DFlowField<K>::Integrate3DPathlineOneDirection(const DTG::Point<3, double>& startPointInChart, double startTime, double targetIntegrationTime, double stepSize, int maxIterationCount, std::vector<DTG::PointAndTime<3, double>>& results, const PathlineNumericalIntegrationMethod method, bool BoundaryRelaxtion)
{

    if (results.size() < 1) {
        results.reserve(maxIterationCount);
    }

    // bool integrationOutOfChartBounds = false;
    bool outOfIntegrationTimeBounds = false;
    bool outOfspaceBounds = false;
    int iterationCount = 0;
    K integrationTimeStepSize = stepSize;

    if (targetIntegrationTime < startTime) {
        // we integrate back in time
        integrationTimeStepSize *= -1.0;
    }

   DTG::EigenVector<3,K> currentPoint = { (K)startPointInChart.x(), (K)startPointInChart.y(), (K)startPointInChart.z() };
    double currentTime = startTime;

    // do integration
    while ((!outOfIntegrationTimeBounds) && (results.size() < maxIterationCount) && (!outOfspaceBounds)) {

        // integrate until either
        // - we reached the max iteration count
        // - we reached the upper limit of the time domain
        DTG::EigenVector<3,K> newPoint;

        if (method == PathlineNumericalIntegrationMethod::Euler) {
            DTG::EigenVector<3, K> velocity = this->GetSpaceAndTimeInterpolatedVector(currentPoint, currentTime);
            newPoint = currentPoint + (velocity * integrationTimeStepSize);
        } else if (method == PathlineNumericalIntegrationMethod::RK4) {
            auto ode_fun = [this](const DTG::EigenVector<3, K>& x, double time) -> DTG::EigenVector<3, K> {
                return this->GetSpaceAndTimeInterpolatedVector(x, time);
            };

            const double h = integrationTimeStepSize;

            // coefficients
            constexpr double a21 = 0.5;
            constexpr double a31 = 0.;
            constexpr double a32 = 0.5;
            constexpr double a41 = 0.;
            constexpr double a42 = 0.;
            constexpr double a43 = 1.;

            constexpr double c2 = 0.5;
            constexpr double c3 = 0.5;
            constexpr double c4 = 1.;

            constexpr double b1 = 1. / 6.;
            constexpr double b2 = 1. / 3.;
            constexpr double b3 = b2;
            constexpr double b4 = b1;

            // 4 stages of 2 equations (i.e., 2 dimensions of the manifold and the tangent vector space)

            DTG::EigenVector<3, K> odeStepStartPoint = currentPoint;
            // stage 1
            DTG::EigenVector<3, K> k1 = ode_fun(odeStepStartPoint, currentTime);

            // stage 2
            DTG::EigenVector<3, K> stagePoint = odeStepStartPoint + (k1 * a21 * h);
            DTG::EigenVector<3, K> k2 = ode_fun(stagePoint, currentTime + c2 * h);

            // stage 3
            stagePoint = odeStepStartPoint + (a31 * k1 + a32 * k2 * h);
            DTG::EigenVector<3, K> k3 = ode_fun(stagePoint, currentTime + c3 * h);

            // stage 4
            stagePoint = odeStepStartPoint + (a41 * k1 + a42 * k2 + a43 * k3 * h);
            DTG::EigenVector<3, K> k4 = ode_fun(stagePoint, currentTime + c4 * h);

            newPoint = (odeStepStartPoint + h * (k1 * b1 + k2 * b2 + k3 * b3 + k4 * b4));

        } else if (method == PathlineNumericalIntegrationMethod::RK5) {
            auto ode_fun = [this](const DTG::EigenVector<3, K>& x, double time) -> DTG::EigenVector<3, K> {
                return this->GetSpaceAndTimeInterpolatedVector(x, time);
            };

            double h = integrationTimeStepSize;

            // coefficients
            constexpr double c2 = 1.0 / 5;
            constexpr double a21 = 1.0 / 5;
            constexpr double c3 = 3.0 / 10;
            constexpr double a31 = 3.0 / 40;
            constexpr double a32 = 9.0 / 40;
            constexpr double c4 = 4.0 / 5;
            constexpr double a41 = 44.0 / 45;
            constexpr double a42 = -56.0 / 15;
            constexpr double a43 = 32.0 / 9;
            constexpr double c5 = 8.0 / 9;
            constexpr double a51 = 19372.0 / 6561;
            constexpr double a52 = -25360.0 / 2187;
            constexpr double a53 = 64448.0 / 6561;
            constexpr double a54 = -212.0 / 729;
            constexpr double c6 = 1.0;
            constexpr double a61 = 9017.0 / 3168;
            constexpr double a62 = -355.0 / 33;
            constexpr double a63 = 46732.0 / 5247;
            constexpr double a64 = 49.0 / 176;
            constexpr double a65 = -5103.0 / 18656;
            constexpr double c7 = 1.0;
            constexpr double a71 = 35.0 / 384;
            constexpr double a72 = 0.0;
            constexpr double a73 = 500.0 / 1113;
            constexpr double a74 = 125.0 / 192;
            constexpr double a75 = -2187.0 / 6784;
            constexpr double a76 = 11.0 / 84;
            /*constexpr double b1 = 35.0 / 384;
            constexpr double b2 = 0.0;
            constexpr double b3 = 500.0 / 1113;
            constexpr double b4 = 125.0 / 192;
            constexpr double b5 = -2187.0 / 6784;
            constexpr double b6 = 11.0 / 84;*/
            constexpr double d1 = 5179.0 / 57600;
            constexpr double d2 = 0.0;
            constexpr double d3 = 7571.0 / 16695;
            constexpr double d4 = 393.0 / 640;
            constexpr double d5 = -92097.0 / 339200;
            constexpr double d6 = 187.0 / 2100;
            constexpr double d7 = 1.0 / 40;

            // 7 stages of 2 equations (i.e., 2 dimensions of the manifold and the tangent vector space)

            DTG::EigenVector<3, K> odeStepStartPoint = currentPoint;
            // stage 1
            DTG::EigenVector<3, K> k1 = ode_fun(odeStepStartPoint, currentTime);

            // stage 2
            DTG::EigenVector<3, K> stagePoint = odeStepStartPoint + (k1 * a21 * h);
            DTG::EigenVector<3, K> k2 = ode_fun(stagePoint, currentTime + c2 * h);

            // stage 3
            stagePoint = odeStepStartPoint + (a31 * k1 + a32 * k2) * h;
            DTG::EigenVector<3, K> k3 = ode_fun(stagePoint, currentTime + c3 * h);

            // stage 4
            stagePoint = odeStepStartPoint + (a41 * k1 + a42 * k2 + a43 * k3) * h;
            DTG::EigenVector<3, K> k4 = ode_fun(stagePoint, currentTime + c4 * h);

            // stage 5
            stagePoint = odeStepStartPoint + (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4) * h;
            DTG::EigenVector<3, K> k5 = ode_fun(stagePoint, currentTime + c5 * h);

            // stage 6
            stagePoint = odeStepStartPoint + (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5) * h;
            DTG::EigenVector<3, K> k6 = ode_fun(stagePoint, currentTime + c6 * h);

            // stage 7
            stagePoint = odeStepStartPoint + (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6) * h;
            DTG::EigenVector<3, K> k7 = ode_fun(stagePoint, currentTime + c7 * h);

            // compute step estimate
            DTG::EigenVector<3, K> stepEstimate = h * (k1 * d1 + k2 * d2 + k3 * d3 + k4 * d4 + k5 * d5 + k6 * d6 + k7 * d7);
            newPoint = (odeStepStartPoint + stepEstimate);
        }

        currentTime += integrationTimeStepSize;
        // check if currentTime is out of the time domain -> we are done
        if ((targetIntegrationTime > startTime) && (currentTime >= targetIntegrationTime)) {
            outOfIntegrationTimeBounds = true;
        } else if ((targetIntegrationTime < startTime) && (currentTime <= targetIntegrationTime)) {
            outOfIntegrationTimeBounds = true;
        } else if (BoundaryRelaxtion==false&& this->IsInValidSpaceAreaOfVectorField(newPoint)==false) {
            //for HasAnalyticalExpression we allow pathline points for went out of boundary
            outOfspaceBounds = true;
        } else {
            // add new point to the result list and set currentPoint to newPoint -> everything fine -> continue with the while loop
            currentPoint = newPoint;
            results.emplace_back((double)newPoint.x(), (double)newPoint.y(), (double)newPoint.z(), currentTime);
            iterationCount++;
        }
    }
}

template <typename K>
void DTG::Discrete3DFlowField<K>::Integrate3DPathlineOneDirection(const DTG::Point<3, float>& startPointInChart, float startTime, float targetIntegrationTime, float stepSize, int maxIterationCount, std::vector<DTG::PointAndTime<3, float>>& results, const PathlineNumericalIntegrationMethod method,bool bondaryRelaxtion)
{

    if (results.size() == 0) {
        results.reserve(maxIterationCount);
    }

    // bool integrationOutOfChartBounds = false;
    bool outOfIntegrationTimeBounds = false;
    bool outOfspaceBounds = false;
    int iterationCount = 0;
    K integrationTimeStepSize = stepSize;

    if (targetIntegrationTime < startTime) {
        // we integrate back in time
        integrationTimeStepSize *= -1.0;
    }

    DTG::EigenVector<3, K> currentPoint = { (K)startPointInChart.x(), (K)startPointInChart.y(), (K)startPointInChart.z() };
    double currentTime = startTime;

    // do integration
    while ((!outOfIntegrationTimeBounds) && (results.size() < maxIterationCount) && (!outOfspaceBounds)) {

        // integrate until either
        // - we reached the max iteration count
        // - we reached the upper limit of the time domain
        DTG::EigenVector<3, K> newPoint;

        if (method == PathlineNumericalIntegrationMethod::Euler) {
            DTG::EigenVector<3, K> velocity = this->GetSpaceAndTimeInterpolatedVector(currentPoint, currentTime);
            newPoint = currentPoint + (velocity * integrationTimeStepSize);
        } else if (method == PathlineNumericalIntegrationMethod::RK4) {
            auto ode_fun = [this](const DTG::EigenVector<3, K>& x, double time) -> DTG::EigenVector<3, K> {
                return this->GetSpaceAndTimeInterpolatedVector(x, time);
            };

            const double h = integrationTimeStepSize;

            // coefficients
            constexpr double a21 = 0.5;
            constexpr double a31 = 0.;
            constexpr double a32 = 0.5;
            constexpr double a41 = 0.;
            constexpr double a42 = 0.;
            constexpr double a43 = 1.;

            constexpr double c2 = 0.5;
            constexpr double c3 = 0.5;
            constexpr double c4 = 1.;

            constexpr double b1 = 1. / 6.;
            constexpr double b2 = 1. / 3.;
            constexpr double b3 = b2;
            constexpr double b4 = b1;

            // 4 stages of 2 equations (i.e., 2 dimensions of the manifold and the tangent vector space)

            DTG::EigenVector<3, K> odeStepStartPoint = currentPoint;
            // stage 1
            DTG::EigenVector<3, K> k1 = ode_fun(odeStepStartPoint, currentTime);

            // stage 2
            DTG::EigenVector<3, K> stagePoint = odeStepStartPoint + (k1 * a21 * h);
            DTG::EigenVector<3, K> k2 = ode_fun(stagePoint, currentTime + c2 * h);

            // stage 3
            stagePoint = odeStepStartPoint + (a31 * k1 + a32 * k2 * h);
            DTG::EigenVector<3, K> k3 = ode_fun(stagePoint, currentTime + c3 * h);

            // stage 4
            stagePoint = odeStepStartPoint + (a41 * k1 + a42 * k2 + a43 * k3 * h);
            DTG::EigenVector<3, K> k4 = ode_fun(stagePoint, currentTime + c4 * h);

            newPoint = (odeStepStartPoint + h * (k1 * b1 + k2 * b2 + k3 * b3 + k4 * b4));

        } else if (method == PathlineNumericalIntegrationMethod::RK5) {
            auto ode_fun = [this](const DTG::EigenVector<3, K>& x, double time) -> DTG::EigenVector<3, K> {
                return this->GetSpaceAndTimeInterpolatedVector(x, time);
            };

            double h = integrationTimeStepSize;

            // coefficients
            constexpr double c2 = 1.0 / 5;
            constexpr double a21 = 1.0 / 5;
            constexpr double c3 = 3.0 / 10;
            constexpr double a31 = 3.0 / 40;
            constexpr double a32 = 9.0 / 40;
            constexpr double c4 = 4.0 / 5;
            constexpr double a41 = 44.0 / 45;
            constexpr double a42 = -56.0 / 15;
            constexpr double a43 = 32.0 / 9;
            constexpr double c5 = 8.0 / 9;
            constexpr double a51 = 19372.0 / 6561;
            constexpr double a52 = -25360.0 / 2187;
            constexpr double a53 = 64448.0 / 6561;
            constexpr double a54 = -212.0 / 729;
            constexpr double c6 = 1.0;
            constexpr double a61 = 9017.0 / 3168;
            constexpr double a62 = -355.0 / 33;
            constexpr double a63 = 46732.0 / 5247;
            constexpr double a64 = 49.0 / 176;
            constexpr double a65 = -5103.0 / 18656;
            constexpr double c7 = 1.0;
            constexpr double a71 = 35.0 / 384;
            constexpr double a72 = 0.0;
            constexpr double a73 = 500.0 / 1113;
            constexpr double a74 = 125.0 / 192;
            constexpr double a75 = -2187.0 / 6784;
            constexpr double a76 = 11.0 / 84;
            /*constexpr double b1 = 35.0 / 384;
            constexpr double b2 = 0.0;
            constexpr double b3 = 500.0 / 1113;
            constexpr double b4 = 125.0 / 192;
            constexpr double b5 = -2187.0 / 6784;
            constexpr double b6 = 11.0 / 84;*/
            constexpr double d1 = 5179.0 / 57600;
            constexpr double d2 = 0.0;
            constexpr double d3 = 7571.0 / 16695;
            constexpr double d4 = 393.0 / 640;
            constexpr double d5 = -92097.0 / 339200;
            constexpr double d6 = 187.0 / 2100;
            constexpr double d7 = 1.0 / 40;

            // 7 stages of 2 equations (i.e., 2 dimensions of the manifold and the tangent vector space)

            DTG::EigenVector<3, K> odeStepStartPoint = currentPoint;
            // stage 1
            DTG::EigenVector<3, K> k1 = ode_fun(odeStepStartPoint, currentTime);

            // stage 2
            DTG::EigenVector<3, K> stagePoint = odeStepStartPoint + (k1 * a21 * h);
            DTG::EigenVector<3, K> k2 = ode_fun(stagePoint, currentTime + c2 * h);

            // stage 3
            stagePoint = odeStepStartPoint + (a31 * k1 + a32 * k2) * h;
            DTG::EigenVector<3, K> k3 = ode_fun(stagePoint, currentTime + c3 * h);

            // stage 4
            stagePoint = odeStepStartPoint + (a41 * k1 + a42 * k2 + a43 * k3) * h;
            DTG::EigenVector<3, K> k4 = ode_fun(stagePoint, currentTime + c4 * h);

            // stage 5
            stagePoint = odeStepStartPoint + (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4) * h;
            DTG::EigenVector<3, K> k5 = ode_fun(stagePoint, currentTime + c5 * h);

            // stage 6
            stagePoint = odeStepStartPoint + (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5) * h;
            DTG::EigenVector<3, K> k6 = ode_fun(stagePoint, currentTime + c6 * h);

            // stage 7
            stagePoint = odeStepStartPoint + (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6) * h;
            DTG::EigenVector<3, K> k7 = ode_fun(stagePoint, currentTime + c7 * h);

            // compute step estimate
            DTG::EigenVector<3, K> stepEstimate = h * (k1 * d1 + k2 * d2 + k3 * d3 + k4 * d4 + k5 * d5 + k6 * d6 + k7 * d7);
            newPoint = (odeStepStartPoint + stepEstimate);
        }

        currentTime += integrationTimeStepSize;
        // check if currentTime is out of the time domain -> we are done
        if ((targetIntegrationTime >= startTime) && (currentTime >= targetIntegrationTime)) {
            outOfIntegrationTimeBounds = true;
        } else if ((targetIntegrationTime < startTime) && (currentTime <= targetIntegrationTime)) {
            outOfIntegrationTimeBounds = true;
		}
		else if (bondaryRelaxtion == false && this->IsInValidSpaceAreaOfVectorField(newPoint) == false) {
			//for HasAnalyticalExpression we allow pathline points for went out of boundary
			outOfspaceBounds = true;
		}
		else {
            // add new point to the result list and set currentPoint to newPoint -> everything fine -> continue with the while loop
            currentPoint = newPoint;
            results.emplace_back((float)newPoint.x(), (float)newPoint.y(), (float)newPoint.z(), currentTime);
            iterationCount++;
        }
    }
}

template <typename K>
void DTG::Discrete3DFlowField<K>::Integrate3DStreamlineOneDirection(const DTG::Point<3, double>& startPointInChart, const double startTime, const double stepSize, const int maxIterationCount, std::vector<DTG::PointAndTime<3, double>>& results, const PathlineNumericalIntegrationMethod method, bool forward)
{
    if (results.size() == 0) {
        results.reserve(maxIterationCount);
    }

    bool outOfspaceBounds = false;
    int iterationCount = 0;
    K integrationStepSize =forward? stepSize:-stepSize;

   results.emplace_back(DTG::PointAndTime<3, double>(startPointInChart, startTime));
   DTG::EigenVector<3, K>currentPoint = { (K)startPointInChart.x(), (K)startPointInChart.y(), (K)startPointInChart.z() };
    // do integration
    while ((results.size() < maxIterationCount) && (!outOfspaceBounds)) {
        // integrate until either
        // - we reached the max iteration count
        // - we reached the upper limit of the time domain
          DTG::EigenVector<3, K> newPoint;
		if (method == PathlineNumericalIntegrationMethod::Euler) {
			DTG::EigenVector<3, K> velocity = this->GetSpaceAndTimeInterpolatedVector(currentPoint,startTime);
			newPoint = currentPoint + velocity * integrationStepSize;
		}
		else if (method == PathlineNumericalIntegrationMethod::RK4) {
			auto ode_fun = [this,startTime](const DTG::EigenVector<3, K>& x) -> DTG::EigenVector<3, K> {
				return this->GetSpaceAndTimeInterpolatedVector(x,startTime);
				};

			const double h = integrationStepSize;
			constexpr double a21 = 0.5, a31 = 0., a32 = 0.5, a41 = 0., a42 = 0., a43 = 1.;
			constexpr double c2 = 0.5, c3 = 0.5, c4 = 1.;
			constexpr double b1 = 1. / 6., b2 = 1. / 3., b3 = b2, b4 = b1;
			DTG::EigenVector<3, K> odeStepStartPoint = currentPoint;
			DTG::EigenVector<3, K> k1 = ode_fun(odeStepStartPoint);

			DTG::EigenVector<3, K> stagePoint = odeStepStartPoint + k1 * a21 * h;
			DTG::EigenVector<3, K> k2 = ode_fun(stagePoint);

			stagePoint = odeStepStartPoint + (a31 * k1 + a32 * k2) * h;
			DTG::EigenVector<3, K> k3 = ode_fun(stagePoint);
			stagePoint = odeStepStartPoint + (a41 * k1 + a42 * k2 + a43 * k3) * h;
			DTG::EigenVector<3, K> k4 = ode_fun(stagePoint);

			newPoint = odeStepStartPoint + h * (k1 * b1 + k2 * b2 + k3 * b3 + k4 * b4);
		}

        // check if currentTime is out of the  domain -> we are done
        if (!IsInValidSpaceAreaOfVectorField(newPoint)) {
            outOfspaceBounds = true;
        } else {
            // add new point to the result list and set currentPoint to newPoint -> everything fine -> continue with the while loop
            currentPoint = newPoint;
            DTG::Point<3, double> newPointCast = { (double)newPoint.x(), (double)newPoint.y(), (double)newPoint.z() };
            results.emplace_back(DTG::PointAndTime<3, double>(newPointCast, (double)startTime));
            iterationCount++;
        }
    }
}
template <typename K>
void DTG::Discrete3DFlowField<K>::Integrate3DStreamlineOneDirection(const DTG::Point<3, float>& startPointInChart, const float startTime, const float stepSize, const int maxIterationCount, std::vector<DTG::PointAndTime<3, float>>& results, const PathlineNumericalIntegrationMethod method, bool forward)
{
    if (results.size() < maxIterationCount) {
        results.reserve(maxIterationCount);
    }

    bool outOfspaceBounds = false;
    int iterationCount = 0;
     K integrationStepSize =forward? stepSize:-stepSize;
    results.emplace_back(DTG::PointAndTime<3, float>(startPointInChart, startTime));
   DTG::EigenVector<3, K> currentPoint = { (K)startPointInChart.x(), (K)startPointInChart.y(), (K)startPointInChart.z() };
    // do integration
    while ((results.size() < maxIterationCount) && (!outOfspaceBounds)) {
        // integrate until either
        // - we reached the max iteration count
		DTG::EigenVector<3, K>newPoint;
		if (method == PathlineNumericalIntegrationMethod::Euler) {
			DTG::EigenVector<3, K> velocity = this->GetSpaceAndTimeInterpolatedVector(currentPoint,startTime);
			newPoint = currentPoint + velocity * integrationStepSize;
		}
		else if (method == PathlineNumericalIntegrationMethod::RK4) {
			auto ode_fun = [this,startTime](const DTG::EigenVector<3, K>& x) -> DTG::EigenVector<3, K> {
				return this->GetSpaceAndTimeInterpolatedVector(x,startTime);
				};
			const double h = integrationStepSize;
			constexpr double a21 = 0.5, a31 = 0., a32 = 0.5, a41 = 0., a42 = 0., a43 = 1.;
			constexpr double c2 = 0.5, c3 = 0.5, c4 = 1.;
			constexpr double b1 = 1. / 6., b2 = 1. / 3., b3 = b2, b4 = b1;
			DTG::EigenVector<3, K> odeStepStartPoint = currentPoint;
			DTG::EigenVector<3, K> k1 = ode_fun(odeStepStartPoint);

			DTG::EigenVector<3, K> stagePoint = odeStepStartPoint + k1 * a21 * h;
			DTG::EigenVector<3, K> k2 = ode_fun(stagePoint);
			stagePoint = odeStepStartPoint + (a31 * k1 + a32 * k2) * h;
			DTG::EigenVector<3, K> k3 = ode_fun(stagePoint);
			stagePoint = odeStepStartPoint + (a41 * k1 + a42 * k2 + a43 * k3) * h;
			DTG::EigenVector<3, K> k4 = ode_fun(stagePoint);
			newPoint = odeStepStartPoint + h * (k1 * b1 + k2 * b2 + k3 * b3 + k4 * b4);
		}

        // check if currentTime is out of the  domain -> we are done
        if (!IsInValidSpaceAreaOfVectorField(newPoint)) {
            outOfspaceBounds = true;
        } else {
            // add new point to the result list and set currentPoint to newPoint -> everything fine -> continue with the while loop
            currentPoint = newPoint;
            DTG::Point<3, float> newPointCast = { (float)newPoint.x(), (float)newPoint.y(), (float)newPoint.z() };
            results.emplace_back(DTG::PointAndTime<3, float>(newPointCast, (float)startTime));
            iterationCount++;
        }
    }
}

template <typename K>
Eigen::Matrix<K, 3, 3> DTG::Discrete3DFlowField<K>::getVelocityGradientTensor(const Eigen::Matrix<K, 3, 1>& point, K time) const
{
    // Compute partial derivatives
    const auto dvdx = getPartialDerivativeX(point, time);
    const auto dvdy = getPartialDerivativeY(point, time);
    const auto dvdz = getPartialDerivativeZ(point, time);

    // Assemble the velocity gradient tensor
    Eigen::Matrix<K, 3, 3> velocityGradientTensor;
    velocityGradientTensor.col(0) = dvdx;
    velocityGradientTensor.col(1) = dvdy;
    velocityGradientTensor.col(2) = dvdz;

    return velocityGradientTensor;
}



template <typename K>
Eigen::Matrix<K, 3, 1> DTG::Discrete3DFlowField<K>::getVorticity(const int x, const int y, const int z, const int t) const
{
    auto dxdydz = GetSpatialDxDyDz();
    // Get neighboring vectors
    auto v_xPlus = GetVectorAtGrid(x + 1, y, z, t);
    auto v_xMinus = GetVectorAtGrid(x - 1, y, z, t);
    auto v_yPlus = GetVectorAtGrid(x, y + 1, z, t);
    auto v_yMinus = GetVectorAtGrid(x, y - 1, z, t);
    auto v_zPlus = GetVectorAtGrid(x, y, z + 1, t);
    auto v_zMinus = GetVectorAtGrid(x, y, z - 1, t);
    // Compute derivatives
    K dVz_dy = (v_zPlus.y() - v_zMinus.y()) / (2.0 * dxdydz.y());
    K dVy_dz = (v_yPlus.z() - v_yMinus.z()) / (2.0 * dxdydz.z());
    K dVx_dz = (v_xPlus.z() - v_xMinus.z()) / (2.0 * dxdydz.z());
    K dVz_dx = (v_zPlus.x() - v_zMinus.x()) / (2.0 * dxdydz.x());
    K dVy_dx = (v_yPlus.x() - v_yMinus.x()) / (2.0 * dxdydz.x());
    K dVx_dy = (v_xPlus.y() - v_xMinus.y()) / (2.0 * dxdydz.y());

    Eigen::Matrix<K, 3, 1> curl;
    curl.x() = dVz_dy - dVy_dz;
    curl.y() = dVx_dz - dVz_dx;
    curl.z() = dVy_dx - dVx_dy;

    return curl;
}

template <typename K>
inline Eigen::Matrix<K, Eigen::Dynamic, 3>& DTG::Discrete3DFlowField<K>::GetData()
{
    return mData;
}

template <typename K>
const Eigen::Matrix<K, Eigen::Dynamic, 3>& DTG::Discrete3DFlowField<K>::GetDataView() const
{
    return mData;
}

template <typename K>
inline DTG::EigenVector<3, K> DTG::Discrete3DFlowField<K>::GetSpaceAndTimeInterpolatedVector(const DTG::EigenPoint<3, K>& Position, K time) const
{
    return this->GetSpaceAndTimeInterpolatedVector(Position(0), Position(1), Position(2), time);
}
template <typename K>
inline DTG::EigenVector<3, K> DTG::Discrete3DFlowField<K>::GetSpaceAndTimeInterpolatedVector(const DTG::EigenVector<3, K>& Position, K time) const
{
    return this->GetSpaceAndTimeInterpolatedVector(Position(0), Position(1), Position(2), time);
}

template <typename K>
DTG::EigenVector<3, K> DTG::Discrete3DFlowField<K>::GetSpaceAndTimeInterpolatedVector(const K X, const K Y, const K Z, const K inTime) const
{
	if (HasAnalyticalExpression())
	{
		DTG::EigenVector<3, K> res = mAnyticalFunc(X, Y, Z, inTime).cast<K>();
		return res;
	}
	else
	{
    const double mMinTime = mFieldInfo.minTime;
    double Time = inTime;
    double continousIndex = 0;
    const auto numOfTimsteps=GetNumberOfTimeSteps() ;
    if (numOfTimsteps> 1) {
        continousIndex = (Time - mMinTime)* mInverse_dt;
    }

	int floorIndex = static_cast<int>(floor(continousIndex));
	int ceilIndex = static_cast<int>(ceil(continousIndex));
	floorIndex = std::clamp(floorIndex, 0,numOfTimsteps-1);
	ceilIndex = std::clamp(ceilIndex,0, numOfTimsteps-1);

    auto resultInComponents = GetSpaceInterpolatedVector(X, Y, Z, floorIndex);

    if (ceilIndex != floorIndex) {
        EigenVector<3, K> pCeil = GetSpaceInterpolatedVector(X, Y, Z, ceilIndex);

        K fraction = continousIndex - (K)floorIndex;
        resultInComponents = resultInComponents * (1.0 - fraction) + pCeil * fraction;
    }

    return resultInComponents;
    }

}
template <typename K>
DTG::EigenVector<3, K> DTG::Discrete3DFlowField<K>::GetTimeInterpolatedVector(const int x, const int y, const int z, const K time) const
{
    const int index=x+y*this->mFieldInfo.XGridsize+z*mXYGrid;
    return GetTimeInterpolatedVector(index  ,time);

}
template <typename K>
inline DTG::EigenVector<3, K> DTG::Discrete3DFlowField<K>::GetTimeInterpolatedVector(const int index, const K inTime) const
{

	const double mMinTime = mFieldInfo.minTime;
	double Time = inTime;
	double continousIndex = 0;
	const auto numOfTimsteps = GetNumberOfTimeSteps();
	if (numOfTimsteps > 1) {
		continousIndex = (Time - mMinTime) * mInverse_dt;
	}

    int floorIndex = floor(continousIndex);
    int ceilIndex = ceil(continousIndex);
	floorIndex = std::clamp(floorIndex, 0, numOfTimsteps  - 1);
	ceilIndex = std::clamp(ceilIndex, 0, numOfTimsteps - 1);
    const auto&NumberOfDataPoints =GetNumberOfDataPoints();
    auto resultInComponents = EigenVector<3, K>(mData.row(NumberOfDataPoints * floorIndex + index));

    if (ceilIndex != floorIndex) {
        // interpolate with data from ceil index
        int dataIndex0Ceil = NumberOfDataPoints * ceilIndex + index;
        EigenVector<3, K> pCeil(mData.row(dataIndex0Ceil));
        K fraction = continousIndex - (K)floorIndex;
        resultInComponents = resultInComponents * (1.0 - fraction) + pCeil * fraction;
    }

    return resultInComponents;
}

template <typename K>
DTG::EigenVector<3, K> DTG::Discrete3DFlowField<K>::GetVectorbyLinearIndex(const int index, const int timeStep) const
{
    const auto&NumberOfDataPoints =GetNumberOfDataPoints();
    if (index + timeStep * NumberOfDataPoints >= NumberOfDataPoints * GetNumberOfTimeSteps()) {
        LOG_E("getVector at Invalid memory.");
        return DTG::EigenVector<3, K>();
    }
    return mData.row(index + timeStep * NumberOfDataPoints);
}

template <typename K>
DTG::EigenVector<3, K> DTG::Discrete3DFlowField<K>::GetVectorAtGrid(const int x, const int y, const int z, const int timeStep) const
{
	uint64_t index = (uint64_t)x + (uint64_t)y * mFieldInfo.XGridsize + (uint64_t)z * mXYGrid + (uint64_t)timeStep * NumberOfDataPoints;
	return mData.row(index);
}


template <typename K>
DTG::EigenVector<3, K> DTG::Discrete3DFlowField<K>::GetSpaceInterpolatedVector(const K X, const K Y, const K Z, const int timestep) const
{
   
    K ContinuousXIndex, ContinuousYIndex, ContinuousZIndex;


    ContinuousXIndex = (X - mFieldInfo.minXCoordinate) * mInverse_OneXGridInterval;//* mInverse_OneXGridInterval =/dx
    ContinuousYIndex = (Y - mFieldInfo.minYCoordinate) * mInverse_OneYGridInterval;//* mInverse_OneYGridInterval =/dy
    ContinuousZIndex = (Z - mFieldInfo.minZCoordinate) * mInverse_OneZGridInterval;//* mInverse_OneZGridInterval =/dz

	int x0 = std::clamp(static_cast<int>(std::floor(ContinuousXIndex)), 0, mFieldInfo.XGridsize - 1);
	int x1 = std::clamp(x0 + 1, 0, mFieldInfo.XGridsize - 1);
	K alphaX = ContinuousXIndex - x0;

	int y0 = std::clamp(static_cast<int>(std::floor(ContinuousYIndex)), 0, mFieldInfo.YGridsize - 1);
	int y1 = std::clamp(y0 + 1, 0, mFieldInfo.YGridsize - 1);
	K alphaY = ContinuousYIndex - y0;

	int z0 = std::clamp(static_cast<int>(std::floor(ContinuousZIndex)), 0, mFieldInfo.ZGridsize - 1);
	int z1 = std::clamp(z0 + 1, 0, mFieldInfo.ZGridsize - 1);
	K alphaZ = ContinuousZIndex - z0;
	auto V000 = GetVectorAtGrid(x0, y0, z0, timestep);
	auto V100 = GetVectorAtGrid(x1, y0, z0, timestep);
	auto V010 = GetVectorAtGrid(x0, y1, z0, timestep);
	auto V110 = GetVectorAtGrid(x1, y1, z0, timestep);
	auto V001 = GetVectorAtGrid(x0, y0, z1, timestep);
	auto V101 = GetVectorAtGrid(x1, y0, z1, timestep);
	auto V011 = GetVectorAtGrid(x0, y1, z1, timestep);
	auto V111 = GetVectorAtGrid(x1, y1, z1, timestep);
	K wx[2] = { 1 - alphaX, alphaX };
	K wy[2] = { 1 - alphaY, alphaY };
	K wz[2] = { 1 - alphaZ, alphaZ };
	EigenVector<3, K> res =
		(wx[0] * wy[0] * wz[0]) * V000 +
		(wx[1] * wy[0] * wz[0]) * V100 +
		(wx[0] * wy[1] * wz[0]) * V010 +
		(wx[1] * wy[1] * wz[0]) * V110 +
		(wx[0] * wy[0] * wz[1]) * V001 +
		(wx[1] * wy[0] * wz[1]) * V101 +
		(wx[0] * wy[1] * wz[1]) * V011 +
		(wx[1] * wy[1] * wz[1]) * V111;

    return res;


}

template <typename K>
Eigen::Matrix<K, Eigen::Dynamic, 3> DTG::Discrete3DFlowField<K>::GetSliceRawData(const int idt)
{
    auto NofPoints = GetNumberOfDataPoints();

    // Result matrix to store the slice data
    Eigen::Matrix<K, Eigen::Dynamic, 3> resData = Eigen::Matrix<K, Eigen::Dynamic, 3>::Zero(NofPoints, 3);

    // Assuming mData is organized with timesteps in contiguous blocks:
    // size = number_of_points_per_timestep * number_of_timesteps x 3
    Eigen::Index startRow = static_cast<Eigen::Index>(idt) * NofPoints;

    // Copy the relevant slice from mData to resData
    resData = mData.block(startRow, 0, NofPoints, 3);

    return resData;
}

template <typename K>
Eigen::Matrix<K, 3, 3> DTG::Discrete3DFlowField<K>::getVelocityGradientTensorTimeInterpolated(const int idx, const int idy, const int idz, const float time) const{
	const double mMinTime = this->GetMinTime();
	const double  one_dived_dt=this->GetInversedt();
    const auto NumberOfTimeSteps=this->GetNumberOfTimeSteps();
	double continousIndex = 0;
	if (NumberOfTimeSteps > 1) {
		continousIndex = (time - mMinTime) * one_dived_dt;
	}
	int floorIndex = floor(continousIndex);
	int ceilIndex = ceil(continousIndex);
	floorIndex = std::clamp(floorIndex, 0, NumberOfTimeSteps - 1);
	ceilIndex = std::clamp(ceilIndex, 0, NumberOfTimeSteps - 1);
	Eigen::Matrix<K, 3, 3> J = this->getVelocityGradientTensor(idx, idy, idz, floorIndex);
	if (ceilIndex != floorIndex) {
		Eigen::Matrix<K, 3, 3> JCeil = this->getVelocityGradientTensor(idx, idy, idz, ceilIndex);
		double fraction = continousIndex - floorIndex;
		J = J * (1.0 - fraction) + JCeil * fraction;
	}
    return J;

}
template <typename K>
DTG::EigenVector<3, K> DTG::Discrete3DFlowField<K>::getPartialDerivativeZ(const Eigen::Matrix<K, 1, 3>& point, K time) const
{
	if (mFieldInfo.ZGridsize <= 1) [[unlikely]]
	{
		return { 0.0,0.0,0.0 };
	}
	K dZ = (mFieldInfo.maxZCoordinate - mFieldInfo.minZCoordinate) / (K)(mFieldInfo.ZGridsize - 1);

	auto point_top = point;
	point_top(2) += dZ;
	if (point_top(2) > mFieldInfo.maxZCoordinate) {
		point_top(2) = mFieldInfo.maxZCoordinate;
	}

	auto point_bottom = point;
	point_bottom(2) -= dZ;
	if (point_bottom(2) < mFieldInfo.minZCoordinate) {
		point_bottom(2) = mFieldInfo.minZCoordinate;
	}

	DTG::EigenVector<3, K> v_top = GetSpaceAndTimeInterpolatedVector(point_top, time);
	DTG::EigenVector<3, K> v_bottom = GetSpaceAndTimeInterpolatedVector(point_bottom, time);
	DTG::EigenVector<3, K> dvdz = (v_top - v_bottom) * (1.0 / (point_top(2) - point_bottom(2)));
	return dvdz;
}
template <typename K>
DTG::EigenVector<3, K> DTG::Discrete3DFlowField<K>::getPartialDerivativeZ(const Eigen::Matrix<K, 3, 1>& point, K time) const
{
	if (mFieldInfo.ZGridsize <= 1) [[unlikely]]
	{
		return { 0.0,0.0,0.0 };
	}
	K dZ = (mFieldInfo.maxZCoordinate - mFieldInfo.minZCoordinate) / (K)(mFieldInfo.ZGridsize - 1);

	auto point_top = point;
	point_top(2) += dZ;
	if (point_top(2) > mFieldInfo.maxZCoordinate) {
		point_top(2) = mFieldInfo.maxZCoordinate;
	}

	auto point_bottom = point;
	point_bottom(2) -= dZ;
	if (point_bottom(2) < mFieldInfo.minZCoordinate) {
		point_bottom(2) = mFieldInfo.minZCoordinate;
	}

	DTG::EigenVector<3, K> v_top = GetSpaceAndTimeInterpolatedVector(point_top, time);
	DTG::EigenVector<3, K> v_bottom = GetSpaceAndTimeInterpolatedVector(point_bottom, time);
	DTG::EigenVector<3, K> dvdz = (v_top - v_bottom) * (1.0 / (point_top(2) - point_bottom(2)));
	return dvdz;
}
template <typename K>
DTG::EigenVector<3, K> DTG::Discrete3DFlowField<K>::getPartialDerivativeY(const Eigen::Matrix<K, 1, 3>& point, K time) const
{
	if (mFieldInfo.YGridsize <= 1) [[unlikely]]
	{
		return { 0.0,0.0,0.0 };
	}
	K dY = (mFieldInfo.maxYCoordinate - mFieldInfo.minYCoordinate) / (K)(mFieldInfo.YGridsize - 1);

	auto point_north = point;
	point_north(1) += dY;
	if (point_north(1) > mFieldInfo.maxYCoordinate) {
		point_north(1) = mFieldInfo.maxYCoordinate;
	}

	auto point_south = point;
	point_south(1) -= dY;
	if (point_south(1) < mFieldInfo.minYCoordinate) {
		point_south(1) = mFieldInfo.minYCoordinate;
	}

	DTG::EigenVector<3, K> v_north = GetSpaceAndTimeInterpolatedVector(point_north, time);
	DTG::EigenVector<3, K> v_south = GetSpaceAndTimeInterpolatedVector(point_south, time);
	DTG::EigenVector<3, K> dvdy = (v_north - v_south) * (1.0 / (point_north(1) - point_south(1)));
	return dvdy;
}
template <typename K>
DTG::EigenVector<3, K> DTG::Discrete3DFlowField<K>::getPartialDerivativeY(const Eigen::Matrix<K, 3, 1>& point, K time) const
{
	if (mFieldInfo.YGridsize <= 1) [[unlikely]]
	{
		return { 0.0,0.0,0.0 };
	}
	K dY = (mFieldInfo.maxYCoordinate - mFieldInfo.minYCoordinate) / (K)(mFieldInfo.YGridsize - 1);

	auto point_north = point;
	point_north(1) += dY;
	if (point_north(1) > mFieldInfo.maxYCoordinate) {
		point_north(1) = mFieldInfo.maxYCoordinate;
	}

	auto point_south = point;
	point_south(1) -= dY;
	if (point_south(1) < mFieldInfo.minYCoordinate) {
		point_south(1) = mFieldInfo.minYCoordinate;
	}

	DTG::EigenVector<3, K> v_north = GetSpaceAndTimeInterpolatedVector(point_north, time);
	DTG::EigenVector<3, K> v_south = GetSpaceAndTimeInterpolatedVector(point_south, time);
	DTG::EigenVector<3, K> dvdy = (v_north - v_south) * (1.0 / (point_north(1) - point_south(1)));
	return dvdy;
}
template <typename K>
DTG::EigenVector<3, K> DTG::Discrete3DFlowField<K>::getPartialDerivativeX(const Eigen::Matrix<K, 1, 3>& point, K time) const
{
	if (mFieldInfo.XGridsize <= 1) [[unlikely]]
	{
		return { 0.0,0.0,0.0 };
	}
	K dX = (mFieldInfo.maxXCoordinate - mFieldInfo.minXCoordinate) / (K)(mFieldInfo.XGridsize - 1);
	auto point_east = point;
	point_east(0) += dX;
	if (point_east(0) > mFieldInfo.maxXCoordinate) {
		point_east(0) = mFieldInfo.maxXCoordinate;
	}

	auto point_west = point;
	point_west(0) -= dX;
	if (point_west(0) < mFieldInfo.minXCoordinate) {
		point_west(0) = mFieldInfo.minXCoordinate;
	}

	DTG::EigenVector<3, K> v_east = GetSpaceAndTimeInterpolatedVector(point_east, time);
	DTG::EigenVector<3, K> v_west = GetSpaceAndTimeInterpolatedVector(point_west, time);
	DTG::EigenVector<3, K> dvdx = (v_east - v_west) * (1.0 / (point_east(0) - point_west(0)));
	return dvdx;
}
template <typename K>
DTG::EigenVector<3, K> DTG::Discrete3DFlowField<K>::getPartialDerivativeX(const Eigen::Matrix<K, 3, 1>& point, K time) const
{
	if (mFieldInfo.XGridsize <= 1) [[unlikely]]
	{
		return { 0.0,0.0,0.0 };
	}
	K dX = (mFieldInfo.maxXCoordinate - mFieldInfo.minXCoordinate) / (K)(mFieldInfo.XGridsize - 1);
	auto point_east = point;
	point_east(0) += dX;
	if (point_east(0) > mFieldInfo.maxXCoordinate) {
		point_east(0) = mFieldInfo.maxXCoordinate;
	}

	auto point_west = point;
	point_west(0) -= dX;
	if (point_west(0) < mFieldInfo.minXCoordinate) {
		point_west(0) = mFieldInfo.minXCoordinate;
	}

	DTG::EigenVector<3, K> v_east = GetSpaceAndTimeInterpolatedVector(point_east, time);
	DTG::EigenVector<3, K> v_west = GetSpaceAndTimeInterpolatedVector(point_west, time);
	DTG::EigenVector<3, K> dvdx = (v_east - v_west) * (1.0 / (point_east(0) - point_west(0)));
	return dvdx;
}


template <typename K>
Eigen::Matrix<K, 3, 3> DTG::Discrete3DFlowField<K>::getVelocityGradientTensor(const int idx, const int idy, const int idz, const int idt) const
{
    if (this->HasAnalyticalExpression())
    {
		const DTG::EigenVector<4, double>query_position = this->convertGridIndex2PhysicalCoordinates(idx, idy, idz, idt);
		const auto query_position_x = query_position.x();
		const auto query_position_y = query_position.y();
		const auto query_position_z = query_position.z();
		const auto query_position_t = query_position.w();
        
        auto dxdydz=GetSpatialDxDyDz();
        const auto dx=dxdydz.x();
        const auto dy=dxdydz.y();
        const auto dz=dxdydz.z();


		DTG::EigenVector<3, float> v_x_next = mAnyticalFunc(query_position_x + 0.1 * dx, query_position_y, query_position_z, query_position_t);
		DTG::EigenVector<3, float> v_x_prev = mAnyticalFunc(query_position_x - 0.1 * dx, query_position_y, query_position_z, query_position_t);
		DTG::EigenVector<3, float> v_y_next = mAnyticalFunc(query_position_x, query_position_y + 0.1 * dy, query_position_z, query_position_t);
		DTG::EigenVector<3, float> v_y_prev = mAnyticalFunc(query_position_x, query_position_y - 0.1 * dy, query_position_z, query_position_t);
		DTG::EigenVector<3, float> v_z_next = mAnyticalFunc(query_position_x, query_position_y, query_position_z + 0.1 * dz, query_position_t);
		DTG::EigenVector<3, float> v_z_prev = mAnyticalFunc(query_position_x, query_position_y, query_position_z - 0.1 * dz, query_position_t);
		DTG::EigenVector<3, float> dv_dx = (v_x_next - v_x_prev) *mInverse_OneXGridInterval*5.0;  //  diff/(0.2*dx)
		DTG::EigenVector<3, float> dv_dy = (v_y_next - v_y_prev) *mInverse_OneYGridInterval*5.0;
		DTG::EigenVector<3, float> dv_dz = (v_z_next - v_z_prev) *mInverse_OneZGridInterval*5.0;

		Eigen::Matrix<K, 3, 3> grad;
		grad.col(0) = dv_dx.cast<K>();
		grad.col(1) = dv_dy.cast<K>();
		grad.col(2) = dv_dz.cast<K>();

		return grad;
    }
    else
    {

		// Compute neighbor indices with clamping at the boundaries.
		int idx_prev = (idx - 1 < 0) ? 0 : idx - 1;
		int idx_next = (idx + 1 >= mFieldInfo.XGridsize) ? mFieldInfo.XGridsize - 1 : idx + 1;

		int idy_prev = (idy - 1 < 0) ? 0 : idy - 1;
		int idy_next = (idy + 1 >= mFieldInfo.YGridsize) ? mFieldInfo.YGridsize - 1 : idy + 1;

		int idz_prev = (idz - 1 < 0) ? 0 : idz - 1;
		int idz_next = (idz + 1 >= mFieldInfo.ZGridsize) ? mFieldInfo.ZGridsize - 1 : idz + 1;

		DTG::EigenVector<3, K> v_x_next = GetVectorAtGrid(idx_next, idy, idz, idt);
		DTG::EigenVector<3, K> v_x_prev = GetVectorAtGrid(idx_prev, idy, idz, idt);

		DTG::EigenVector<3, K> v_y_next = GetVectorAtGrid(idx, idy_next, idz, idt);
		DTG::EigenVector<3, K> v_y_prev = GetVectorAtGrid(idx, idy_prev, idz, idt);

		DTG::EigenVector<3, K> v_z_next = GetVectorAtGrid(idx, idy, idz_next, idt);
		DTG::EigenVector<3, K> v_z_prev = GetVectorAtGrid(idx, idy, idz_prev, idt);
		DTG::EigenVector<3, K> dv_dx = ((v_x_next - v_x_prev) * mInverse_OneXGridInterval) / (static_cast<K>(idx_next - idx_prev) );//diff/dt or diff/(2dt)-> diff *(1/dt) /
		DTG::EigenVector<3, K> dv_dy = ((v_y_next - v_y_prev) * mInverse_OneYGridInterval) / (static_cast<K>(idy_next - idy_prev) );
		DTG::EigenVector<3, K> dv_dz = ((v_z_next - v_z_prev) * mInverse_OneZGridInterval) / (static_cast<K>(idz_next - idz_prev) );

		Eigen::Matrix<K, 3, 3> grad;
		grad.col(0) = dv_dx;
		grad.col(1) = dv_dy;
		grad.col(2) = dv_dz;

		return grad;
        
    
    }

}


template <typename K>
DTG::EigenVector<3, K> DTG::Discrete3DFlowField<K>::getPartialDerivativeT(const int idx, const int idy, const int idz, int idt) const
{
	if (mFieldInfo.numberOfTimeSteps > 1)
	{
		const K dt = mFieldInfo.Getdt();
		auto idt_next = idt + 1;
		if (idt_next >= mFieldInfo.numberOfTimeSteps - 1) {
			idt_next = mFieldInfo.numberOfTimeSteps - 1;
		}

		auto idt_before = idt - 1;
		if (idt_before <= 0) {
			idt_before = 0;
		}

		DTG::EigenVector<3, K> v_before = GetVectorAtGrid(idx, idy, idz, idt_before);
		DTG::EigenVector<3, K> v_next = GetVectorAtGrid(idx, idy, idz, idt_next);
		DTG::EigenVector<3, K> dvdt = (v_next - v_before) / (dt * (idt_next - idt_before));
		return dvdt;
	}
	else
	{
		return { 0,0,0 };
	}


}
template <typename K>
DTG::EigenVector<3, K> DTG::Discrete3DFlowField<K>::getPartialDerivativeT(const Eigen::Matrix<K, 1, 3>& point, K time) const
{
	if (mFieldInfo.numberOfTimeSteps > 1)
	{
		const K dt = mFieldInfo.Getdt();
		auto time_next = time + 0.5 * dt;
		if (time_next > mFieldInfo.maxTime) {
			time_next = mFieldInfo.maxTime;
		}

		auto time_before = time - 0.5 * dt;
		if (time_next < mFieldInfo.minTime) {
			time_before = mFieldInfo.minTime;
		}

		DTG::EigenVector<3, K> v_before = GetSpaceAndTimeInterpolatedVector(point, time_before);
		DTG::EigenVector<3, K> v_next = GetSpaceAndTimeInterpolatedVector(point, time_next);
		DTG::EigenVector<3, K> dvdt = (v_next - v_before) / (time_next - time_before);
		return dvdt;
	}
	else
	{
		return { 0,0,0 };
	}

}
template <typename K>
DTG::EigenVector<3, K> DTG::Discrete3DFlowField<K>::getPartialDerivativeT(const Eigen::Matrix<K, 3, 1>& point, K time) const
{
	if (mFieldInfo.numberOfTimeSteps > 1)
	{
		const K dt = mFieldInfo.Getdt();
		if (dt == K(0.0))
		{
			return { 0.0f,0.0f,0.0f };
		}
		auto time_next = time + 0.5 * dt;
		if (time_next > mFieldInfo.maxTime) {
			time_next = mFieldInfo.maxTime;
		}

		auto time_before = time - 0.5 * dt;
		if (time_next < mFieldInfo.minTime) {
			time_before = mFieldInfo.minTime;
		}

		DTG::EigenVector<3, K> v_before = GetSpaceAndTimeInterpolatedVector(point, time_before);
		DTG::EigenVector<3, K> v_next = GetSpaceAndTimeInterpolatedVector(point, time_next);
		DTG::EigenVector<3, K> dvdt = (v_next - v_before) / (time_next - time_before);
		return dvdt;
	}
	else
	{
		return { 0,0,0 };
	}
}


template <typename K>
std::vector<DTG::EigenVector<3, K>> DTG::Discrete3DFlowField<K>::AverageVectorFieldBySlice() const
{
	std::vector<DTG::EigenVector<3, K>> res;
	const int totalElements = GetNumberOfDataPoints();
	// Iterate through each slice
	for (int slice = 0; slice < mFieldInfo.numberOfTimeSteps; slice++) {

		// Variables for Kahan summation for each component
		double totalSumX = 0.0, totalSumY = 0.0, totalSumZ = 0.0;
		double compensationX = 0.0, compensationY = 0.0, compensationZ = 0.0;
		for (int grid_id = 0; grid_id < totalElements; grid_id++) {
			double vector_x = mData(slice * totalElements + grid_id, 0);
			double vector_y = mData(slice * totalElements + grid_id, 1);
			double vector_z = mData(slice * totalElements + grid_id, 2);
			// Handle X component
			double yX = static_cast<double>(vector_x) - compensationX;
			double tX = totalSumX + yX;
			compensationX = (tX - totalSumX) - yX;
			totalSumX = tX;

			// Handle Y component
			double yY = static_cast<double>(vector_y) - compensationY;
			double tY = totalSumY + yY;
			compensationY = (tY - totalSumY) - yY;
			totalSumY = tY;

			// Handle Z component
			double yZ = static_cast<double>(vector_z) - compensationZ;
			double tZ = totalSumZ + yZ;
			compensationZ = (tZ - totalSumZ) - yZ;
			totalSumZ = tZ;
		}

		// Compute averages for each component
		double averageX = totalSumX / totalElements;
		double averageY = totalSumY / totalElements;
		double averageZ = totalSumZ / totalElements;
		DTG::EigenVector<3, K> sliceAverage(averageX, averageY, averageZ);

		res.push_back(sliceAverage);
	}

	// Return the average as a 3D vector
	return res;
}

template <typename K>
vtkSmartPointer<vtkImageData>  DTG::Discrete3DFlowField<K>::ConvertDTGVectorField3DToVTKImage(double time_val)
{
	auto dims_arr = this->GetSpatialGridSize();

	Eigen::Vector3d spacing_eigen = this->GetSpatialDxDyDz();
	Eigen::Vector3d origin_eigen = this->GetOrigin();

	vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
	imageData->SetDimensions(dims_arr[0], dims_arr[1], dims_arr[2]);
	imageData->SetSpacing(static_cast<double>(spacing_eigen.x()),
		static_cast<double>(spacing_eigen.y()),
		static_cast<double>(spacing_eigen.z()));
	imageData->SetOrigin(static_cast<double>(origin_eigen.x()),
		static_cast<double>(origin_eigen.y()),
		static_cast<double>(origin_eigen.z()));

	vtkSmartPointer<vtkFloatArray> vectorData = vtkSmartPointer<vtkFloatArray>::New();
	vectorData->SetNumberOfComponents(3); // x, y, z
	vectorData->SetNumberOfTuples(static_cast<vtkIdType>(dims_arr[0]) * dims_arr[1] * dims_arr[2]);
	vectorData->SetName("VelocityVectors");

	float sim_time = static_cast<float>(time_val);
	vtkIdType tupleIdx = 0;
	for (int k = 0; k < dims_arr[2]; ++k) {
		for (int j = 0; j < dims_arr[1]; ++j) {
			for (int i = 0; i < dims_arr[0]; ++i) {
				Eigen::Vector3f pos = this->convertGridIndex2PhysicalCoordinates(i, j, k).cast<float>();
				Eigen::Vector3f vec = this->GetVector(pos, sim_time);
				float vtkVec[3] = { vec.x(), vec.y(), vec.z() };
				vectorData->SetTuple(tupleIdx++, vtkVec);
			}
		}
	}
	imageData->GetPointData()->SetVectors(vectorData);
	return imageData;
}









void DTG::VectorField3DPointwiseOperator::operator()(const std::function<float(const int, const int, const int, const int)>& mPointwiseFunc, DTG::DiscreteScalarField3D<float>& result) const
{
    const auto gridSize = result.GetSpatialGridSize();
    const int timeSteps = result.GetNumberOfTimeSteps();
    this->operator()(gridSize, timeSteps, mPointwiseFunc, result);
}
void DTG::VectorField3DPointwiseOperator::operator()(const Eigen::Vector3i& gridSize, const int timeSteps, const std::function<float(const int, const int, const int, const int)>& mPointwiseFunc, DTG::DiscreteScalarField3D<float>& result) const
{


	const int gridX = gridSize.x();
	const int gridY = gridSize.y();
	const int gridZ = gridSize.z();
	if (gridX <= 2 || gridY <= 2) {
		LOG_W("Fail to use VectorField3DPointwiseOperator due to gridX<= 2 ||gridY<= 2.");
		return;
	}
	if (gridZ >= 3)
	{
        const int interiorZ = std::max(gridZ - 2, 0);
		// Total number of points in the 4D grid (excluding boundaries)
		const int totalPoints = (gridSize.x() - 2) * (gridSize.y() - 2) * (interiorZ)*timeSteps;
		const int planeStride = (gridY - 2) * (gridX - 2);
		const int rowStride = gridX - 2;
		// Create a linear index mapping each point in the 4D grid
		std::vector<int> indices(totalPoints);
		std::iota(indices.begin(), indices.end(), 0);
		/// Parallel transform
		std::for_each(policy, indices.begin(), indices.end(),
			[&](int idx) {
				// Calculate 4D indices directly using precomputed strides
				const int t = idx / ((gridZ - 2) * planeStride);
				idx -= t * (gridZ - 2) * planeStride;

				const int z = idx / planeStride + 1; // Offset by 1 to skip boundaries
				idx -= (z - 1) * planeStride;

				const int y = idx / rowStride + 1; // Offset by 1 to skip boundaries
				const int x = idx % rowStride + 1; // Offset by 1 to skip boundaries

				// Apply pointwise function and set the result
				auto value = mPointwiseFunc(x, y, z, t);
				result.SetValue(x, y, z, t, value);
			});
	}
	else
	{
		constexpr int interiorZ = 1;
        constexpr int zindx=0;
		// Total number of points in the 4D grid (excluding boundaries)
		const int totalPoints = (gridSize.x() - 2) * (gridSize.y() - 2) * (interiorZ)*timeSteps;
		const int planeStride = (gridY - 2) * (gridX - 2);
		const int rowStride = gridX - 2;
		// Create a linear index mapping each point in the 4D grid
		std::vector<int> indices(totalPoints);
		std::iota(indices.begin(), indices.end(), 0);
		/// Parallel transform
		std::for_each(policy, indices.begin(), indices.end(),
			[&](int idx) {
				// Calculate 4D indices directly using precomputed strides
				const int t = idx / (planeStride);
				idx -= t * planeStride;

				const int y = idx / rowStride + 1; // Offset by 1 to skip boundaries
				const int x = idx % rowStride + 1; // Offset by 1 to skip boundaries

				// Apply pointwise function and set the result
				auto value = mPointwiseFunc(x, y, zindx, t);
				result.SetValue(x, y, zindx, t, value);
			});


	}

}

template <typename scalarType>
void DTG::VectorField3DPointwiseOperator::operator()(const std::function<DTG::EigenVector<3, scalarType>(const int, const int, const int, const int)>& mPointwiseFunc, DTG::Discrete3DFlowField<scalarType>& result) const
{
    const auto gridSize = result.GetSpatialGridSize();
    const int timeSteps = result.GetNumberOfTimeSteps();
    this->operator()(gridSize, timeSteps, mPointwiseFunc, result);
}
template <typename scalarType>
void VectorField3DPointwiseOperator::operator()(const Eigen::Vector3i& gridSize, const int timeSteps, const std::function<DTG::EigenVector<3, scalarType>(const int, const int, const int, const int)>& mPointwiseFunc, DTG::Discrete3DFlowField<scalarType>& result) const
{
    result.GetData().setZero();
    // Total number of points in the 4D grid (excluding boundaries)
    const int gridX = gridSize.x();
    const int gridY = gridSize.y();
    const int gridZ = gridSize.z();
	if (gridX <= 2 || gridY <= 2) {
		LOG_W("Fail to use VectorField3DPointwiseOperator due to gridX<= 2 ||gridY<= 2.");
		return;
	}
	if (gridZ >= 3)
	{
    // Total number of points in the 4D grid (excluding boundaries)
    const int totalPoints = (gridSize.x() - 2) * (gridSize.y() - 2) * (gridSize.z() - 2) * timeSteps;
    const int planeStride = (gridY - 2) * (gridX - 2);
    const int rowStride = gridX - 2;

    // Create a linear index mapping each point in the 4D grid
    std::vector<int> indices(totalPoints);
    std::iota(indices.begin(), indices.end(), 0);

    // Transform with parallel execution policy
    std::for_each(policy, indices.begin(), indices.end(),
        [&](const int inidx) {
            int idx = inidx;
            // Convert linear index back to 4D (t, z, y, x)
            /*int t = idx / ((gridSize.z() - 2) * (gridSize.y() - 2) * (gridSize.x() - 2));
            idx %= (gridSize.z() - 2) * (gridSize.y() - 2) * (gridSize.x() - 2);
            int z = idx / ((gridSize.y() - 2) * (gridSize.x() - 2)) + 1;
            idx %= (gridSize.y() - 2) * (gridSize.x() - 2);
            int y = idx / (gridSize.x() - 2) + 1;
            int x = idx % (gridSize.x() - 2) + 1;*/
            // Calculate 4D indices directly using precomputed strides
            const int t = idx / ((gridZ - 2) * planeStride);
            idx -= t * (gridZ - 2) * planeStride;

            const int z = idx / planeStride + 1; // Offset by 1 to skip boundaries
            idx -= (z - 1) * planeStride;

            const int y = idx / rowStride + 1; // Offset by 1 to skip boundaries
            const int x = idx % rowStride + 1; // Offset by 1 to skip boundaries

            // Apply pointwise function and set the result
            result.SetFlowVector(x, y, z, t, mPointwiseFunc(x, y, z, t));
        });
    }
	else
	{
		constexpr int interiorZ = 1;
		constexpr int zindx = 0;
		// Total number of points in the 4D grid (excluding boundaries)
		const int totalPoints = (gridSize.x() - 2) * (gridSize.y() - 2) * (interiorZ)*timeSteps;
		const int planeStride = (gridY - 2) * (gridX - 2);
		const int rowStride = gridX - 2;
		// Create a linear index mapping each point in the 4D grid
		std::vector<int> indices(totalPoints);
		std::iota(indices.begin(), indices.end(), 0);
		/// Parallel transform
		std::for_each(policy, indices.begin(), indices.end(),
			[&](int idx) {
				// Calculate 4D indices directly using precomputed strides
				const int t = idx / (planeStride);
				idx -= t * planeStride;

				const int y = idx / rowStride + 1; // Offset by 1 to skip boundaries
				const int x = idx % rowStride + 1; // Offset by 1 to skip boundaries

				// Apply pointwise function and set the result
				const auto& value = mPointwiseFunc(x, y, zindx, t);
				result.SetFlowVector(x, y, zindx, t, value);
			});

	}


}

inline int idx3D(int ix, int iy, int iz, int Nx, int Ny, int Nz)
{
    return iz * (Ny * Nx) + iy * (Nx) + ix;
}

/**
 * Helmholtz–Hodge Decomposition in a periodic 3D domain:
 *   v = d + r + h,
 * where:
 *   - h is the uniform ("zero-frequency") mode in a fully periodic domain,
 *   - d = grad(phi) (curl-free),
 *   - r is divergence-free,
 *   - the domain is Nx x Ny x Nz with periodic boundary conditions.
 */
template <typename K>
void HelmholtzHodgeDecomposition(
    const Discrete3DFlowField<K>& flowField, Discrete3DFlowField<K>& curlFreeField, Discrete3DFlowField<K>& divFreeField, Discrete3DFlowField<K>& harmonicField, const int maxPossitionIteration)
{
    // Get grid dimensions and number of time steps.
    auto res = flowField.GetSpatialGridSize();
    int Nx = res.x();
    int Ny = res.y();
    int Nz = res.z();
    int Nt = flowField.GetNumberOfTimeSteps();

    // Temporary arrays for a single time step
    std::vector<K> vX(Nx * Ny * Nz), vY(Nx * Ny * Nz), vZ(Nx * Ny * Nz);
    std::vector<K> divV(Nx * Ny * Nz);
    std::vector<K> phi(Nx * Ny * Nz, K(0)); // Poisson potential
    std::vector<K> phiNew(Nx * Ny * Nz, K(0)); // buffer for Jacobi iteration

    // Arrays for final results: d, r, h
    std::vector<K> dX(Nx * Ny * Nz), dY(Nx * Ny * Nz), dZ(Nx * Ny * Nz);
    std::vector<K> rX(Nx * Ny * Nz), rY(Nx * Ny * Nz), rZ(Nx * Ny * Nz);
    std::vector<K> hX(Nx * Ny * Nz), hY(Nx * Ny * Nz), hZ(Nx * Ny * Nz);

    // A small lambda to wrap periodic indices
    auto wrap = [&](int c, int N) {
        // In C++, (c % N) can be negative if c<0, so let's fix that:
        int w = c % N;
        return (w < 0) ? (w + N) : w;
    };

    // For each time step, do the decomposition
    for (int it = 0; it < Nt; ++it) {

        // 1) Load velocity into vX, vY, vZ
        // ---------------------------------------------------------------------
#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
        for (int ix = 0; ix < Nx; ++ix) {
            for (int iy = 0; iy < Ny; ++iy) {
                for (int iz = 0; iz < Nz; ++iz) {
                    int idx = idx3D(ix, iy, iz, Nx, Ny, Nz);
                    auto v = flowField.GetVectorAtGrid(ix, iy, iz, it);
                    vX[idx] = v[0];
                    vY[idx] = v[1];
                    vZ[idx] = v[2];
                }
            }
        }

        // --------------------------------------------------------------
        // 2) Compute global average velocity (the zero-frequency mode)
        // --------------------------------------------------------------
        double sumX = K(0), sumY = K(0), sumZ = K(0);
        int totalSize = Nx * Ny * Nz;
        for (int i = 0; i < totalSize; ++i) {
            sumX += vX[i];
            sumY += vY[i];
            sumZ += vZ[i];
        }
        sumX /= double(totalSize);
        sumY /= double(totalSize);
        sumZ /= double(totalSize);

        std::vector<K> vXp(totalSize), vYp(totalSize), vZp(totalSize);

        // This uniform velocity field is our harmonic field candidate:
        // --------------------------------------------------------------
        // 3) Subtract h from v, so v' = v - h
        // --------------------------------------------------------------
#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
        for (int i = 0; i < totalSize; ++i) {
            hX[i] = sumX;
            hY[i] = sumY;
            hZ[i] = sumZ;
            vXp[i] = vX[i] - hX[i];
            vYp[i] = vY[i] - hY[i];
            vZp[i] = vZ[i] - hZ[i];
        }

        // --------------------------------------------------------------
        // 4) Compute divergence of v': divV = d/dx(vXp) + d/dy(vYp) + d/dz(vZp)
        //    Use central differences with periodic boundary wrap.
        std::fill(divV.begin(), divV.end(), K(0));

#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
        for (int ix = 0; ix < Nx; ++ix) {
            for (int iy = 0; iy < Ny; ++iy) {
                for (int iz = 0; iz < Nz; ++iz) {
                    int ixp = wrap(ix + 1, Nx);
                    int ixm = wrap(ix - 1, Nx);
                    int iyp = wrap(iy + 1, Ny);
                    int iym = wrap(iy - 1, Ny);
                    int izp = wrap(iz + 1, Nz);
                    int izm = wrap(iz - 1, Nz);

                    int c = idx3D(ix, iy, iz, Nx, Ny, Nz);
                    int xp = idx3D(ixp, iy, iz, Nx, Ny, Nz);
                    int xm = idx3D(ixm, iy, iz, Nx, Ny, Nz);
                    int yp = idx3D(ix, iyp, iz, Nx, Ny, Nz);
                    int ym = idx3D(ix, iym, iz, Nx, Ny, Nz);
                    int zp = idx3D(ix, iy, izp, Nx, Ny, Nz);
                    int zm = idx3D(ix, iy, izm, Nx, Ny, Nz);

                    K ddx = (vXp[xp] - vXp[xm]) * K(0.5);
                    K ddy = (vYp[yp] - vYp[ym]) * K(0.5);
                    K ddz = (vZp[zp] - vZp[zm]) * K(0.5);
                    divV[c] = ddx + ddy + ddz;
                }
            }
        }

        // --------------------------------------------------------------
        // 5) Solve Poisson: Laplacian(phi) = divV, with periodic BC
        //    We'll do a naive Jacobi iteration. Also re-center phi each iteration
        //    to ensure mean(phi)=0, because in a periodic domain phi is only
        //    determined up to an additive constant.
        // --------------------------------------------------------------
        std::fill(phi.begin(), phi.end(), K(0));
        std::fill(phiNew.begin(), phiNew.end(), K(0));

        // the possion solver is extremely slow,  don't iterate for too long
        int maxIter = std::min(maxPossitionIteration, 10000);

        K tolerance = K(1e-6);

        for (int iter = 0; iter < maxIter; ++iter) {
            K maxDiff = K(0);

#ifdef NDEBUG
#pragma omp parallel
#endif
            {
                // Each thread has a local maximum difference
                K localMaxDiff = K(0);

#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
                for (int ix = 0; ix < Nx; ++ix) {
                    for (int iy = 0; iy < Ny; ++iy) {
                        for (int iz = 0; iz < Nz; ++iz) {
                            int ixp = wrap(ix + 1, Nx);
                            int ixm = wrap(ix - 1, Nx);
                            int iyp = wrap(iy + 1, Ny);
                            int iym = wrap(iy - 1, Ny);
                            int izp = wrap(iz + 1, Nz);
                            int izm = wrap(iz - 1, Nz);

                            int c = idx3D(ix, iy, iz, Nx, Ny, Nz);
                            int xp = idx3D(ixp, iy, iz, Nx, Ny, Nz);
                            int xm = idx3D(ixm, iy, iz, Nx, Ny, Nz);
                            int yp = idx3D(ix, iyp, iz, Nx, Ny, Nz);
                            int ym = idx3D(ix, iym, iz, Nx, Ny, Nz);
                            int zp = idx3D(ix, iy, izp, Nx, Ny, Nz);
                            int zm = idx3D(ix, iy, izm, Nx, Ny, Nz);

                            K sumN = phi[xp] + phi[xm]
                                + phi[yp] + phi[ym]
                                + phi[zp] + phi[zm];

                            K newVal = (sumN - divV[c]) / K(6);

                            K diffVal = std::abs(newVal - phi[c]);
                            if (diffVal > localMaxDiff) {
                                localMaxDiff = diffVal;
                            }
                            phiNew[c] = newVal;
                        }
                    }
                }

                // Write local maxDiff back in a critical section
#ifdef NDEBUG
#pragma omp critical
#endif
                {
                    if (localMaxDiff > maxDiff) {
                        maxDiff = localMaxDiff;
                    }
                }
            } // end parallel region

            // Re-center phi => mean(phi)=0
            {
                K sumPhi = K(0);

#ifdef NDEBUG
#pragma omp parallel
#endif
                {
                    K localSum = K(0);

#ifdef NDEBUG
#pragma omp for nowait
#endif
                    for (int i = 0; i < totalSize; ++i) {
                        localSum += phiNew[i];
                    }

#ifdef NDEBUG
#pragma omp critical
#endif
                    {
                        sumPhi += localSum;
                    }
                }

                K meanPhi = sumPhi / K(totalSize);

#ifdef NDEBUG
#pragma omp parallel for
#endif
                for (int i = 0; i < totalSize; ++i) {
                    phiNew[i] -= meanPhi;
                }
            }

            phi.swap(phiNew);

            if (maxDiff < tolerance) {
                // Converged
                break;
            }
        }

        // --------------------------------------------------------------
        // 6) Compute d = grad(phi) with periodic indexing, then r = v' - d
        // --------------------------------------------------------------
#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
        for (int ix = 0; ix < Nx; ++ix) {
            for (int iy = 0; iy < Ny; ++iy) {
                for (int iz = 0; iz < Nz; ++iz) {
                    int ixp = wrap(ix + 1, Nx);
                    int ixm = wrap(ix - 1, Nx);
                    int iyp = wrap(iy + 1, Ny);
                    int iym = wrap(iy - 1, Ny);
                    int izp = wrap(iz + 1, Nz);
                    int izm = wrap(iz - 1, Nz);

                    int c = idx3D(ix, iy, iz, Nx, Ny, Nz);
                    int xp = idx3D(ixp, iy, iz, Nx, Ny, Nz);
                    int xm = idx3D(ixm, iy, iz, Nx, Ny, Nz);
                    int yp = idx3D(ix, iyp, iz, Nx, Ny, Nz);
                    int ym = idx3D(ix, iym, iz, Nx, Ny, Nz);
                    int zp = idx3D(ix, iy, izp, Nx, Ny, Nz);
                    int zm = idx3D(ix, iy, izm, Nx, Ny, Nz);

                    dX[c] = (phi[xp] - phi[xm]) * K(0.5);
                    dY[c] = (phi[yp] - phi[ym]) * K(0.5);
                    dZ[c] = (phi[zp] - phi[zm]) * K(0.5);
                }
            }
        }

        for (int i = 0; i < totalSize; ++i) {
            rX[i] = vXp[i] - dX[i];
            rY[i] = vYp[i] - dY[i];
            rZ[i] = vZp[i] - dZ[i];
        }

        // --------------------------------------------------------------
        // 7) Finally, v = d + r + h
        //    Store these fields in curlFreeField, divFreeField, harmonicField
        // --------------------------------------------------------------
#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
        for (int ix = 0; ix < Nx; ++ix) {
            for (int iy = 0; iy < Ny; ++iy) {
                for (int iz = 0; iz < Nz; ++iz) {
                    int idx = idx3D(ix, iy, iz, Nx, Ny, Nz);

                    // d
                    curlFreeField.SetFlowVector(
                        ix, iy, iz, it,
                        DTG::EigenVector<3, K>(dX[idx], dY[idx], dZ[idx]));
                    // r
                    divFreeField.SetFlowVector(
                        ix, iy, iz, it,
                        DTG::EigenVector<3, K>(rX[idx], rY[idx], rZ[idx]));
                    // h
                    harmonicField.SetFlowVector(
                        ix, iy, iz, it,
                        DTG::EigenVector<3, K>(hX[idx], hY[idx], hZ[idx]));
                }
            }
        }
        LOG_D("hhd for step %d done", it);
    } // end for each time step
}

// ReSampleToNewResolution by Trilinear (Spatial) + Linear (Temporal) (resamplingMethod = 1)
template <typename K>
std::unique_ptr<IDiscreteField3d> Discrete3DFlowField<K>::ReSampleToNewResolution(int xdim, int ydim, int zdim, int tdim)
{

    auto fieldInformation = mFieldInfo;
    fieldInformation.XGridsize = xdim;
    fieldInformation.YGridsize = ydim;
    fieldInformation.ZGridsize = zdim;
    fieldInformation.numberOfTimeSteps = tdim;

    if (auto duplicate = std::make_unique<Discrete3DFlowField<K>>(fieldInformation); duplicate) {

        auto& newData = duplicate->GetData();
        newData.resize(xdim * ydim * zdim * tdim, 3);

        K xMin = static_cast<K>(mFieldInfo.minXCoordinate);
        K xMax = static_cast<K>(mFieldInfo.maxXCoordinate);
        K yMin = static_cast<K>(mFieldInfo.minYCoordinate);
        K yMax = static_cast<K>(mFieldInfo.maxYCoordinate);
        K zMin = static_cast<K>(mFieldInfo.minZCoordinate);
        K zMax = static_cast<K>(mFieldInfo.maxZCoordinate);
        K tMin = static_cast<K>(mFieldInfo.minTime);
        K tMax = static_cast<K>(mFieldInfo.maxTime);

		// Avoid division by zero for the case xdim=1, etc.
		K xCount = (xdim > 1) ? static_cast<K>(xdim - 1) : static_cast<K>(1);
		K yCount = (ydim > 1) ? static_cast<K>(ydim - 1) : static_cast<K>(1);
		K zCount = (zdim > 1) ? static_cast<K>(zdim - 1) : static_cast<K>(1);
		K tCount = (tdim > 1) ? static_cast<K>(tdim - 1) : static_cast<K>(1);

		const bool  analyticalExpress = HasAnalyticalExpression();
		const  bool  hasDiscreteData = HasDiscreteData();
		if (!analyticalExpress && !hasDiscreteData)
		{
			LOG_E("Fail to create resampled vector field due to no valid data.");
			return nullptr;
		}

        //  Loop over the new resolution in x, y, z, t, fill newData
#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
        for (int it = 0; it < tdim; ++it) {
            for (int iz = 0; iz < zdim; ++iz) {
                for (int iy = 0; iy < ydim; ++iy) {
                    for (int ix = 0; ix < xdim; ++ix) {
                        // Parametric ratio in [0,1] for time
                        K tRatio = (tdim > 1) ? (static_cast<K>(it) / tCount) : static_cast<K>(0.0);
                        // Physical time
                        K timeVal = tMin + (tMax - tMin) * tRatio;
                        K zRatio = (zdim > 1) ? (static_cast<K>(iz) / zCount) : static_cast<K>(0.0);
                        K zVal = zMin + (zMax - zMin) * zRatio;
                        K yRatio = (ydim > 1) ? (static_cast<K>(iy) / yCount) : static_cast<K>(0.0);
                        K yVal = yMin + (yMax - yMin) * yRatio;
                        K xRatio = (xdim > 1) ? (static_cast<K>(ix) / xCount) : static_cast<K>(0.0);
                        K xVal = xMin + (xMax - xMin) * xRatio;

                        DTG::EigenVector<3, K> vec;

                        // 1: tri-linear in space + linear in time
                        //    We can reuse the existing function
                        //    (makes sense if it already does tri-linear + linear time)

                        vec =  this->GetSpaceAndTimeInterpolatedVector(xVal, yVal, zVal, timeVal);

                        int linearIndex = duplicate->GetDataIndex(ix, iy, iz, it);
                        newData(linearIndex, 0) = vec(0);
                        newData(linearIndex, 1) = vec(1);
                        newData(linearIndex, 2) = vec(2);
                    }
                }
            }
        }

        return std::move(duplicate);
    }
    LOG_E("Fail to create resampled vector field.");
    return nullptr;
}


template <typename K>
void Discrete3DFlowField<K>::SampleAnlyticalToDiscreteData()
{
	const bool  analyticalExpression = HasAnalyticalExpression();
	if (!analyticalExpression)
	{
		LOG_E("Fail to create resampled vector field due to no valid analyticalExpression.");
		return;
	}

	const auto& fieldInformation = mFieldInfo;
	const auto& xdim = fieldInformation.XGridsize;
	const auto& ydim = fieldInformation.YGridsize;
	const auto& zdim = fieldInformation.ZGridsize;
	const auto& tdim = fieldInformation.numberOfTimeSteps;

	auto& newData = GetData();
	newData.resize(xdim * ydim * zdim * tdim, 3);

	K xMin = static_cast<K>(mFieldInfo.minXCoordinate);
	K xMax = static_cast<K>(mFieldInfo.maxXCoordinate);
	K yMin = static_cast<K>(mFieldInfo.minYCoordinate);
	K yMax = static_cast<K>(mFieldInfo.maxYCoordinate);
	K zMin = static_cast<K>(mFieldInfo.minZCoordinate);
	K zMax = static_cast<K>(mFieldInfo.maxZCoordinate);
	K tMin = static_cast<K>(mFieldInfo.minTime);
	K tMax = static_cast<K>(mFieldInfo.maxTime);

	// Avoid division by zero for the case xdim=1, etc.
	K xCount = (xdim > 1) ? static_cast<K>(xdim - 1) : static_cast<K>(1);
	K yCount = (ydim > 1) ? static_cast<K>(ydim - 1) : static_cast<K>(1);
	K zCount = (zdim > 1) ? static_cast<K>(zdim - 1) : static_cast<K>(1);
	K tCount = (tdim > 1) ? static_cast<K>(tdim - 1) : static_cast<K>(1);


	//  Loop over the new resolution in x, y, z, t, fill newData
#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 16)
#endif
	for (int it = 0; it < tdim; ++it) {
		for (int iz = 0; iz < zdim; ++iz) {
			for (int iy = 0; iy < ydim; ++iy) {
				for (int ix = 0; ix < xdim; ++ix) {
					// Parametric ratio in [0,1] for time
					K tRatio = (tdim > 1) ? (static_cast<K>(it) / tCount) : static_cast<K>(0.0);
					// Physical time
					K timeVal = tMin + (tMax - tMin) * tRatio;
					K zRatio = (zdim > 1) ? (static_cast<K>(iz) / zCount) : static_cast<K>(0.0);
					K zVal = zMin + (zMax - zMin) * zRatio;
					K yRatio = (ydim > 1) ? (static_cast<K>(iy) / yCount) : static_cast<K>(0.0);
					K yVal = yMin + (yMax - yMin) * yRatio;
					K xRatio = (xdim > 1) ? (static_cast<K>(ix) / xCount) : static_cast<K>(0.0);
					K xVal = xMin + (xMax - xMin) * xRatio;

					DTG::EigenVector<3, K> vec = mAnyticalFunc(xVal, yVal, zVal, timeVal).cast<K>();

					int linearIndex =GetDataIndex(ix, iy, iz, it);
					newData(linearIndex, 0) = vec(0);
					newData(linearIndex, 1) = vec(1);
					newData(linearIndex, 2) = vec(2);
				}
			}
		}
	}

}


Eigen::Vector3f computeCurlfromGradientTensor(const Eigen::Matrix<float, 3, 3>& velocityGradientTensor)
{
    // Compute the curl from the velocity gradient tensor
    Eigen::Vector3f curl;

    curl(0) = velocityGradientTensor(2, 1) - velocityGradientTensor(1, 2);
    curl(1) = velocityGradientTensor(0, 2) - velocityGradientTensor(2, 0);
    curl(2) = velocityGradientTensor(1, 0) - velocityGradientTensor(0, 1);

    return curl;
}




DTG::IDiscreteField3d* DTG::VectorField3DManager::GetVectorFieldByName(const std::string& name)
{
	auto iter = this->mVectorFields3DOnGrid.find(name);
	if (iter == this->mVectorFields3DOnGrid.end())
		return nullptr;
	return iter->second.get();
}

double DTG::VectorField3DManager::GetMinTimeOfVectorField(const std::string& name)
{
	auto iter = this->mVectorFields3DOnGrid.find(name);
	if (iter == this->mVectorFields3DOnGrid.end()) {
		LOG_E("Query Faild as no Vector field: %s", name.c_str());
		return -1;
	}
	return iter->second.get()->GetMinTime();
}

double DTG::VectorField3DManager::GetMaxTimeOfVectorField(const std::string& name)
{
	auto iter = this->mVectorFields3DOnGrid.find(name);
	if (iter == this->mVectorFields3DOnGrid.end()) {
		LOG_E("Query Faild as no Vector field: %s", name.c_str());
		return -1;
	}
	return iter->second.get()->GetMaxTime();
}

int DTG::VectorField3DManager::GetNumberOfTimestepsOfVectorField(const std::string& name)
{
	auto iter = this->mVectorFields3DOnGrid.find(name);
	if (iter == this->mVectorFields3DOnGrid.end()) {
		LOG_E("Query Faild as no Vector field: %s", name.c_str());
		return -1;
	}
	return iter->second.get()->GetNumberOfTimeSteps();
}

bool DTG::VectorField3DManager::HasDiscreteVectorField(const std::string& name)
{
	if (mVectorFields3DOnGrid.find(name) != mVectorFields3DOnGrid.end())
		return true;
	else
		return false;
}
void DTG::VectorField3DManager::InsertVectorField(std::unique_ptr<IDiscreteField3d> vPtr, const std::string& name)
{
	if (mVectorFields3DOnGrid.find(name) != mVectorFields3DOnGrid.end()) {
		LOG_W("Vector Field '%s' to insert already exists. Prevous data will be overwrite.", name.c_str());
		mVectorFields3DOnGrid.erase(name);
	}
	mVectorFields3DOnGrid.insert(std::make_pair(name, std::move(vPtr)));
}

std::vector<std::string> DTG::VectorField3DManager::GetAllVectorFieldNames() const
{
	std::vector<std::string> temp;
	for (const auto& [key, value] : mVectorFields3DOnGrid) {
		temp.push_back(key);
	}

	return temp;
}

DTG::EigenVector<3, double> DTG::VectorField3DManager::GetVectorFieldMinimalRange(const std::string& name)
{
	auto iter = this->mVectorFields3DOnGrid.find(name);
	if (iter == this->mVectorFields3DOnGrid.end()) {
		LOG_E("Query Faild as no Vector field: %s", name.c_str());
		return EigenVector<3, double>();
	}
	return iter->second.get()->GetSpatialMin();
}

DTG::EigenVector<3, double> DTG::VectorField3DManager::GetVectorFieldMaximalRange(const std::string& name)
{
	auto iter = this->mVectorFields3DOnGrid.find(name);
	if (iter == this->mVectorFields3DOnGrid.end()) {
		LOG_E("Query Faild as no Vector field: %s", name.c_str());
		return EigenVector<3, double>();
	}
	return iter->second.get()->GetSpatialMax();
}

bool DTG::VectorField3DManager::IsInValidSpaceAreaOfVectorField(const std::string& name, const EigenPoint<3, double>& Position) const
{
	auto iter = this->mVectorFields3DOnGrid.find(name);
	if (iter == this->mVectorFields3DOnGrid.end()) {
		LOG_E("Query Faild as no Vector field: %s", name.c_str());
		return false;
	}
	return iter->second.get()->IsInValidSpaceAreaOfVectorField(Position);
}
bool DTG::VectorField3DManager::IsInValidSpaceAreaOfVectorField(const std::string& name, const EigenPoint<3, float>& Position) const
{
	EigenPoint<3, double> Pos = { Position(0), Position(1), Position(2) };
	return IsInValidSpaceAreaOfVectorField(name, Pos);
}

DTG::EigenVector<2, double> DTG::VectorField3DManager::GetVectorFieldTimeRange(const std::string& name) const
{
	auto iter = this->mVectorFields3DOnGrid.find(name);
	if (iter == this->mVectorFields3DOnGrid.end()) {
		LOG_E("Query Faild as no Vector field: %s", name.c_str());
		return EigenVector<2, double>();
	}
	auto tmin = iter->second->GetMinTime();
	auto tmax = iter->second->GetMaxTime();
	return { tmin, tmax };
}

std::unique_ptr<DTG::IDiscreteField3d> CreateSteadyKillingVectorField(const KillingField3DCoefficients& coeffs, const FieldSpaceTimeInfo3D& domainInfo)
{
	auto newKillingField = std::make_unique<Discrete3DFlowField<float>>(domainInfo, false);
	newKillingField->SetAnalyticalExpression([coeffs](float x, float y, float z, float t) -> Eigen::Vector3f {
		// Extract rotation coefficients (A, B, C)
		float A = coeffs.rotationABC.x();
		float B = coeffs.rotationABC.y();
		float C = coeffs.rotationABC.z();

		// Extract translation coefficients (D, E, F)
		float D = coeffs.tranlationDEF.x();
		float E = coeffs.tranlationDEF.y();
		float F = coeffs.tranlationDEF.z();

		// Calculate Killing field components
		float u = D + (B * z - C * y);  // Translation X + Rotation YZ
		float v = E + (C * x - A * z);  // Translation Y + Rotation ZX
		float w = F + (A * y - B * x);  // Translation Z + Rotation XY

		return Eigen::Vector3f(u, v, w);
		});

	return std::move(newKillingField);
}


}

namespace DTG {
template void DTG::HelmholtzHodgeDecomposition<float>(const Discrete3DFlowField<float>& flowField, Discrete3DFlowField<float>& curlFreeField, /* d */ Discrete3DFlowField<float>& divFreeField, /* r */ Discrete3DFlowField<float>& harmonicField, const int maxPossitionIteration);


template class DTG::Discrete3DFlowField<double>;
template void VectorField3DPointwiseOperator::operator() < double > (
	const std::function<EigenVector<3, double>(const int, const int, const int, const int)>& mPointwiseFunc,
	Discrete3DFlowField<double>& result) const;
template void VectorField3DPointwiseOperator::operator() < double > (
	const Eigen::Vector3i& gridSize,
	const int timeSteps,
	const std::function<EigenVector<3, double>(const int, const int, const int, const int)>& mPointwiseFunc,
	Discrete3DFlowField<double>& result) const;


template class DTG::Discrete3DFlowField<float>;
template void VectorField3DPointwiseOperator::operator()<float>(
    const std::function<EigenVector<3, float>(const int, const int, const int, const int)>& mPointwiseFunc,
    Discrete3DFlowField<float>& result) const;
template void VectorField3DPointwiseOperator::operator()<float>(
    const Eigen::Vector3i& gridSize,
    const int timeSteps,
    const std::function<EigenVector<3, float>(const int, const int, const int, const int)>& mPointwiseFunc,
    Discrete3DFlowField<float>& result) const;

}
