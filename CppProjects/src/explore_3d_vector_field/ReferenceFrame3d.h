
#ifndef _REFERENCE_FRAME_3D_H_
#define _REFERENCE_FRAME_3D_H_
#pragma once
#include "PortToUsingModules.h"

#include "myTypeDefs.h"
#include "Point.h"
#include <map>
#include <string>
#include <numeric>
#include <execution>
#include "Discrete3DFlowField.h"
#include "IField.h"

namespace DTG {
			

	
	struct Worldline3D {
		std::vector<PointAndTime3Df > mPathline;
		std::vector<Eigen::Vector3f > mInstantaneousRotation;

		double getMinTime() const {
			if (mPathline.empty()) {
				return 0.0;
			}
			return mPathline.front().time;
		}

		double getMaxTime() const {
			if (mPathline.empty()) {
				return 0.0;
			}
			return mPathline.back().time;
		}

		Eigen::Vector2d getTimeRange() const {
			return Eigen::Vector2d(getMinTime(), getMaxTime());
		}

		int getNumberOfTimeSteps() const {
			return mPathline.size();
		}

		PointAndTime3Df interpolatePoint(double time) const;
	};


	class ReferenceFrameTransformation3D {
	public:
		bool IsValid() const{
			return mWorldline.getNumberOfTimeSteps()>2;
		}
		inline  double getMinTime() const {
			return mWorldline.getMinTime();
		}

		inline double getMaxTime() const {
			return mWorldline.getMaxTime();
		}

		inline 	Eigen::Vector2d getTimeRange() const {
			return mWorldline.getTimeRange();
		}

		int getNumberOfTimeSteps() const {
			return mWorldline.getNumberOfTimeSteps();
		}

		inline PointAndTime3Df getReferencePoint() const {
			return interpolatePoint(mObservationTime);
		}

		inline  PointAndTime3Df interpolatePoint(double time) const {
			return mWorldline.interpolatePoint(time);
		}
		Eigen::Matrix3f interpolateIntegratedRotation(double time) const;
		std::function<Eigen::Vector3f(const Eigen::Vector3f&)> getTransformationLabToObserved(double time) const;

		std::function<Eigen::Vector3f(const Eigen::Vector3f&)> getTransformationObservedToLab(double time) const;

		Eigen::Matrix3f getIntegratedRotationObservedToLab(double time) const;

		Worldline3D mWorldline;
		//mIntegratedRotation is set by function integrateReferenceFrameRotation
		std::vector<Eigen::Matrix3f> mIntegratedRotation;
		double mObservationTime;
		
	};
		



	namespace WorldlineUtility3D {
		template<typename T>
		Eigen::Matrix<T,3,3> angleVelocityExponential2Rotation(const Eigen::Matrix<T,3,1>& angularVelocity, double timeInterval_dt);

		void computeWorldLine3d(const std::string& observerFieldName, const DTG::PointAndTime3Df& startpoint, const double& stepsize, Worldline3D& worldline);
		void computeInstaneousRotationAroundPathline(const std::string& observerFieldName, const Worldline3D& pathline, std::vector<Eigen::Vector3f>& instantaneousRotation);
		void integrateReferenceFrameRotation(const Worldline3D& worldline, const double& observation_time, ReferenceFrameTransformation3D& transformation);
		void computeReferenceFrameTransformation(const std::string& observerFieldName, const DTG::PointAndTime3Df& startpoint, const double& stepsize, ReferenceFrameTransformation3D& transformation);
		void computeReferenceFrameTransformation(const std::string& observerFieldName, const Eigen::Vector4d& InStartpoint, const double& stepsize, ReferenceFrameTransformation3D& transformation);
		template<class T>
		void computeReferenceFrameTransformationOfScalarField(const DiscreteScalarField3D<T>& input, DiscreteScalarField3D<T>& output, const ReferenceFrameTransformation3D& transformation);

		std::vector<KillingField3DCoefficients> extractKillingFieldFromWorldlineForceNoRot(Discrete3DFlowField<float>* active_field, const Worldline3D& worldline);
	}//namespace WorldlineUtility3D



	Eigen::Vector2i ClampTimstep2ValidRange(const Discrete3DFlowField<float>& inputField, int timestepStart, int timestepEnd);

	template<class T>
	Eigen::Matrix<T, 3, 3> make_Matrix3d(const Eigen::Matrix<T, 3, 1>& c0, const Eigen::Matrix<T, 3, 1>& c1, const Eigen::Matrix<T, 3, 1>& c2)
	{
		Eigen::Matrix<T, 3, 3>M;
		M << c0.x(), c1.x(), c2.x(),
			c0.y(), c1.y(), c2.y(),
			c0.z(), c1.z(), c2.z();
		return M;
	}
	void RftScalarField3D(const DiscreteScalarField3D<float>& scalarFieldInput,  ReferenceFrameTransformation3D rft,DiscreteScalarField3D<float>& scalarFieldOutput,bool shiftRegion=false);

	

	enum class EInvariance {
		Objective,
		Similarity,
		Affine,
		Displacement

	};


	class GenericLocalOptimization3d {
	public:

		GenericLocalOptimization3d() = default;
		GenericLocalOptimization3d(int  neighborhoodU, bool useSummedAreaTables);
		~GenericLocalOptimization3d() {}
		void compute(const Discrete3DFlowField<float>& inputField, int timestepStart, int timestepEnd, Discrete3DFlowField<float>& resultUfield, Discrete3DFlowField<float>& resultVminusUfield);
		void compute(const Discrete3DFlowField<float>& inputField, int timestepStart, int timestepEnd, Discrete3DFlowField<float>& resultUfield, Discrete3DFlowField<float>& resultVminusUfield, DiscreteScalarField3D<float>& observedTimeDerivatives);
		
		void setFilter(const std::function<bool(int,int,int,int)>&GridFilter){
			mGridFilter=GridFilter;
		}
		void setInvariance(EInvariance  Invariance){
		 mInvariance =Invariance;
		}
	protected:
		void computeWithOutFilter(const Discrete3DFlowField<float>& inputField, int timestepStart, int timestepEnd, Discrete3DFlowField<float>& resultUfield, Discrete3DFlowField<float>& resultVminusUfield, DiscreteScalarField3D<float>& observedTimeDerivatives);
		void computeWithFilter(const Discrete3DFlowField<float>& inputField, int timestepStart, int timestepEnd, Discrete3DFlowField<float>& resultUfield, Discrete3DFlowField<float>& resultVminusUfield, DiscreteScalarField3D<float>& observedTimeDerivatives);

		std::function<bool(int,int,int,int)>mGridFilter=nullptr;
		EInvariance  mInvariance=EInvariance::Objective;
		int NeighborhoodU = 10;
		bool UseSummedAreaTables = true;

	};

    class HadwigerKillingOptimization3d {
            public:
                HadwigerKillingOptimization3d () = default;
				HadwigerKillingOptimization3d(double weight_killing, double weight_regular):weight_killing(weight_killing),weight_regular(weight_regular){
					}
                ~HadwigerKillingOptimization3d () { }
                void compute(const Discrete3DFlowField<float>& inputField, int timestepStart, int timestepEnd, Discrete3DFlowField<float>& resultUfield, Discrete3DFlowField<float>& resultVminusUfield, DiscreteScalarField3D<float>& observedTimeDerivatives);
	protected:
                double weight_killing=1.0;
				double weight_regular=0.001;

        };



} // namespace DTG


#endif // !_REFERENCE_FRAME_3D_H_
