#include"transformation.h"
#include"commonUtils.h"





// this function is similar to but 0different from killingABCtransformation, this function assume inputField is some unsteady field get deformed by an observer represented by
//  predictKillingABCfunc, reconstructKillingDeformedUnsteadyField will remove the transformation imposed by predictKillingABCfunc.
UnSteadyVectorField2D reconstructKillingDeformedUnsteadyField(std::function<Eigen::Vector3d(double)> predictKillingABCfunc, const UnSteadyVectorField2D& inputField)
{
	const auto tmax = inputField.tmax;
	const auto tmin = inputField.tmin;
	const double dt = (inputField.tmax - inputField.tmin) / ((double)inputField.timeSteps - 1.0);
	if (inputField.analyticalFlowfunc_) {

		UnSteadyVectorField2D outputField;
		outputField.spatialDomainMaxBoundary = inputField.getSpatialMaxBoundary();
		outputField.spatialDomainMinBoundary = inputField.getSpatialMinBoundary();
		outputField.spatialGridInterval = inputField.spatialGridInterval;
		outputField.XdimYdim = inputField.XdimYdim;
		outputField.tmin = tmin;
		outputField.tmax = tmax;
		outputField.timeSteps = inputField.timeSteps;
		outputField.field.resize(inputField.timeSteps);

		outputField.analyticalFlowfunc_ = [=](const Eigen::Vector2d& posX, double t) -> Eigen::Vector2d {
			auto abc_t = predictKillingABCfunc(t);

			const double floatingTimeStep = (t - tmin) / dt;
			const int timestep_floor = std::clamp((int)std::floor(floatingTimeStep), 0, inputField.timeSteps - 1);
			const int timestep_ceil = std::clamp((int)std::floor(floatingTimeStep) + 1, 0, inputField.timeSteps - 1);
			const double ratio = floatingTimeStep - timestep_floor;

			const auto Q_t = inputField.Q_t[timestep_floor] * (1 - ratio) + inputField.Q_t[timestep_ceil] * ratio;
			const auto Q_transpose = Q_t.transpose();
			const auto c_t = inputField.c_t[timestep_floor] * (1 - ratio) + inputField.c_t[timestep_ceil] * ratio;

			Eigen::Vector2d xStar = Q_t * posX + c_t;
			Eigen::Vector2d v_star_xstar = inputField.getVectorAnalytical({ xStar(0), xStar(1) }, t);

			// compute Qdot=Q(t)*Omega(t) where Omega(t) is the anti-symmetric matrix of the angular velocity vector
			Eigen::Matrix2d Spintensor;
			Spintensor(0, 0) = 0.0;
			Spintensor(1, 0) = -abc_t(2);
			Spintensor(0, 1) = abc_t(2);
			Spintensor(1, 1) = 0.0;

			Eigen::Matrix2d Q_dot = Q_t * Spintensor;
			Eigen::Vector2d translationTdot = { -abc_t.x(), -abc_t.y() };
			Eigen::Vector2d v_at_pos = Q_transpose * (v_star_xstar - Q_dot * posX - translationTdot);
			return v_at_pos;
			};
		outputField.resampleFromAnalyticalExpression();
		return outputField;
	}
	else {
		printf("reconstructUnsteadyField only support analyticalFlowfunc_");
		return {};
	}
}


template <typename T = double, class InputFieldTYPE>
UnSteadyVectorField2D killingABCtransformation(const KillingAbcField& observerfield, const Eigen::Vector2d StartPosition, InputFieldTYPE& inputField)
{
	const auto tmin = observerfield.tmin;
	const auto tmax = observerfield.tmax;
	const auto dt = observerfield.dt;
	const int timestep = observerfield.timeSteps;

	observerfield.spatialDomainMaxBoundary = inputField.getSpatialMaxBoundary();
	observerfield.spatialDomainMinBoundary = inputField.getSpatialMinBoundary();

	std::vector<Eigen::Vector2d> pathVelocitys;
	std::vector<Eigen::Vector3d> pathPositions;
	bool suc = PathhlineIntegrationRK4(StartPosition, observerfield, tmin, tmax, dt, pathVelocitys, pathPositions);
	assert(suc);

	int validPathSize = pathPositions.size();

	std::vector<Eigen::Matrix3d> observerRotationMatrices;
	observerRotationMatrices.resize(timestep);
	observerRotationMatrices[0] = Eigen::Matrix3d::Identity();

	std::vector<Eigen::Matrix4d> observertransformationMatrices;
	observertransformationMatrices.resize(timestep);
	observertransformationMatrices[0] = Eigen::Matrix4d::Identity();
	const auto observerStartPoint = pathPositions.at(0);

	for (size_t i = 1; i < validPathSize; i++) {
		const double t = observerfield.tmin + i * observerfield.dt;
		const Eigen::Vector3d abc = observerfield.killingABCfunc_(t);
		const auto c_ = abc(2);
		// this abs is important, otherwise flip sign of c_ will cause the spin tensor and rotation angle theta to be flipped simultaneously,
		// two flip sign cancel out the result stepInstanenousRotation never  change.
		//
		// theta is just scalar measure of how many degree the observer rotate with out direction. the rotation angle encoded in Spintensor
		const auto theta = dt * std::abs(c_);
		Eigen::Matrix3d Spintensor;
		Spintensor(0, 0) = 0.0;
		Spintensor(1, 0) = c_;
		Spintensor(2, 0) = 0;

		Spintensor(0, 1) = -c_;
		Spintensor(1, 1) = 0.0;
		Spintensor(2, 1) = 0.0;

		Spintensor(0, 2) = 0;
		Spintensor(1, 2) = 0.0;
		Spintensor(2, 2) = 0.0;
		NoramlizeSpinTensor(Spintensor);
		Eigen::Matrix3d Spi_2;
		Spi_2 = Spintensor * Spintensor;
		double sinTheta = sin(theta);
		double cosTheta = cos(theta);
		Eigen::Matrix3d I = Eigen::Matrix<double, 3, 3>::Identity();
		Eigen::Matrix3d stepInstanenousRotation = I + sinTheta * Spintensor + (1 - cosTheta) * Spi_2;
		// get the rotation matrix of observer, which is the Q(t)^T of frame transformation x*=Q(t)x+c(t)
		observerRotationMatrices[i] = stepInstanenousRotation * observerRotationMatrices[i - 1];
		const auto& stepRotation = observerRotationMatrices[i];
		// compute observer's relative transformation as M=T(position)*integral of [ R(matrix_exponential(spinTensor))] * T(-startPoint)
		// then observer bring transformation  M-1=T(startPoint)*integral of [ R(matrix_exponential(spinTensor))]^T * T(-position)

		auto tP1 = pathPositions.at(i);

		// eigen << fill data in row major regardless of storage order

		Eigen::Matrix4d inv_translationPostR;
		inv_translationPostR << 1, 0, 0, -tP1(0), // first rowm
			0, 1, 0, -tP1(1), // second row
			0, 0, 1, 0,
			0.0, 0, 0, 1;

		Eigen::Matrix4d inv_translationPreR;
		inv_translationPreR << 1, 0, 0, observerStartPoint(0), // first row
			0, 1, 0, observerStartPoint(1), // second row
			0, 0, 1, 0,
			0.0, 0, 0, 1;

		// this Q_t  is exactly the frame transformation x*=Q(t)x+c(t)
		Eigen::Matrix4d Q_t_transpose = Eigen::Matrix4d::Zero();
		Q_t_transpose(0, 0) = stepRotation(0, 0);
		Q_t_transpose(0, 1) = stepRotation(0, 1);
		Q_t_transpose(0, 2) = stepRotation(0, 2);
		Q_t_transpose(1, 0) = stepRotation(1, 0);
		Q_t_transpose(1, 1) = stepRotation(1, 1);
		Q_t_transpose(1, 2) = stepRotation(1, 2);
		Q_t_transpose(2, 0) = stepRotation(2, 0);
		Q_t_transpose(2, 1) = stepRotation(2, 1);
		Q_t_transpose(2, 2) = stepRotation(2, 2);
		Q_t_transpose(3, 3) = 1.0;

		// Eigen::Matrix4f  ObserverTransformation = translationPostR * (Q[i] * translationPreR);
		//  combine translation and rotation into final transformation
		Eigen::Matrix4d InvserseTransformation = inv_translationPreR * (Q_t_transpose.transpose() * inv_translationPostR);

		observertransformationMatrices[i] = InvserseTransformation;
	}
	const auto lastPushforward = observerRotationMatrices[validPathSize - 1];
	const auto lastTransformation = observertransformationMatrices[validPathSize - 1];
	for (size_t i = validPathSize; i < timestep; i++) {
		observertransformationMatrices[i] = lastTransformation;
		observerRotationMatrices[i] = lastPushforward;
		// pathVelocitys.emplace_back(0.0f, 0.0f);
		pathPositions.emplace_back(pathPositions.back());
	}

	UnSteadyVectorField2D resultField;
	resultField.spatialDomainMaxBoundary = inputField.getSpatialMaxBoundary();
	resultField.spatialDomainMinBoundary = inputField.getSpatialMinBoundary();
	resultField.spatialGridInterval = inputField.spatialGridInterval;
	resultField.XdimYdim = inputField.XdimYdim;
	resultField.tmin = tmin;
	resultField.tmax = tmax;
	resultField.timeSteps = timestep;
	resultField.field.resize(timestep);

	if (inputField.analyticalFlowfunc_) {
		const Eigen::Vector2d Os = { pathPositions[0].x(), pathPositions[0].y() };
		resultField.Q_t.resize(timestep);
		resultField.c_t.resize(timestep);
		for (size_t i = 0; i < timestep; i++) {
			//  frame transformation is F(x):x*=Q(t)x+c(t)  or x*=T(Os) *Q*T(-Pt)*x
			//  =>F(x):x* = Q(t)*(x-pt)+Os= Qx-Q*pt+Os -> c=-Q*pt+Os  // => F^(-1)(x)= Q^T (x-c)= Q^T *( x+Q*pt-Os)
			resultField.Q_t[i] = observerRotationMatrices[i].transpose().block<2, 2>(0, 0);
			auto& Q_t = resultField.Q_t[i];
			const Eigen::Vector2d position_t = { pathPositions[i].x(), pathPositions[i].y() };
			Eigen::Vector2d c_t = Os - Q_t * position_t;
			resultField.c_t[i] = c_t;
		}

		// resultField.analyticalFlowfunc_ = [inputField, observerfield, resultField, dt, observerRotationMatrices, pathPositions](const Eigen::Vector2d& pos, double t) -> Eigen::Vector2d {
		//     double tmin = observerfield.tmin;
		//     const double floatingTimeStep = (t - tmin) / dt;
		//     const int timestep_floor = std::clamp((int)std::floor(floatingTimeStep), 0, observerfield.timeSteps - 1);
		//     const int timestep_ceil = std::clamp((int)std::floor(floatingTimeStep) + 1, 0, observerfield.timeSteps - 1);
		//     const double ratio = floatingTimeStep - timestep_floor;
		//    const Eigen::Matrix2d Q_t = resultField.Q_t[timestep_floor] * (1 - ratio) + resultField.Q_t[timestep_ceil] * ratio;
		//    auto Q_transpose = Q_t.transpose();
		//    auto c_t = resultField.c_t[timestep_floor] * (1 - ratio) + resultField.c_t[timestep_ceil] * ratio;
		//    // => F^(-1)(x)= Q^T (x-c)= Q^T *( x+Q*pt-Os)
		//    Eigen ::Vector2d F_inverse_x_2d = Q_transpose * (pos - c_t);
		//    auto v = inputField.getVectorAnalytical(F_inverse_x_2d, t);
		//    auto u = observerfield.getVector(F_inverse_x_2d(0), F_inverse_x_2d(1), t);
		//    Eigen::Vector2d vminusu_lab_frame;
		//    vminusu_lab_frame << v(0) - u(0), v(1) - u(1);
		//    // then w*(x,t)=Q(t)* w( F^-1(x),t )= Q(t)*w( Q^T *( x +Q*pt-Os),t )
		//    return Q_t * vminusu_lab_frame;
		//};
		resultField.analyticalFlowfunc_ = [inputField, observerfield, resultField, dt](const Eigen::Vector2d& pos, double t) -> Eigen::Vector2d {
			double tmin = observerfield.tmin;
			const double floatingTimeStep = (t - tmin) / dt;
			const int timestep_floor = std::clamp((int)std::floor(floatingTimeStep), 0, observerfield.timeSteps - 1);
			const int timestep_ceil = std::clamp((int)std::floor(floatingTimeStep) + 1, 0, observerfield.timeSteps - 1);
			const double ratio = floatingTimeStep - timestep_floor;

			const Eigen::Matrix2d Q_t = resultField.Q_t[timestep_floor] * (1 - ratio) + resultField.Q_t[timestep_ceil] * ratio;
			auto Q_transpose = Q_t.transpose();
			auto c_t = resultField.c_t[timestep_floor] * (1 - ratio) + resultField.c_t[timestep_ceil] * ratio;
			// => F^(-1)(x)= Q^T (x-c)= Q^T *( x+Q*pt-Os)
			Eigen::Vector2d F_inverse_x_2d = Q_transpose * (pos - c_t);
			auto v = inputField.getVectorAnalytical(F_inverse_x_2d, t);
			auto u = observerfield.getVector(F_inverse_x_2d(0), F_inverse_x_2d(1), t);
			const Eigen::Vector3d abc = observerfield.killingABCfunc_(t);
			const auto c_ = abc(2);
			Eigen::Matrix2d Spintensor;
			Spintensor(0, 0) = 0.0;
			Spintensor(1, 0) = -c_;
			Spintensor(0, 1) = c_;
			Spintensor(1, 1) = 0.0;
			Eigen::Matrix2d Q_dot = Q_t * Spintensor;
			Eigen::Vector2d translationTdot = { -abc.x(), -abc.y() };

			Eigen::Vector2d res = Q_t * v + Q_dot * F_inverse_x_2d + translationTdot;
			return res;
			};
		resultField.resampleFromAnalyticalExpression();
	}
	else {
		printf("error...");
	}
	return resultField;
}

// const Eigen::Vector3d& abc, const Eigen::Vector3d& abc_dot represents the xdot,ydot,theta_dot, xdotdot,ydotdot,theta_dotdot of paper "Roboust reference frame..."
UnSteadyVectorField2D Tobias_ObserverTransformation(const SteadyVectorField2D& inputField, const Eigen::Vector3d& abc, const Eigen::Vector3d& abc_dot, const double tmin, const double tmax, const int timestep)
{
	const auto dt = (tmax - tmin) / ((double)(timestep)-1.0);
	UnSteadyVectorField2D resultField;
	resultField.spatialDomainMaxBoundary = inputField.getSpatialMaxBoundary();
	resultField.spatialDomainMinBoundary = inputField.getSpatialMinBoundary();
	resultField.spatialGridInterval = inputField.spatialGridInterval;
	resultField.XdimYdim = inputField.XdimYdim;
	resultField.tmin = tmin;
	resultField.tmax = tmax;
	resultField.timeSteps = timestep;
	resultField.field.resize(timestep);

	// Q(0)=I ->theta(0)=0; translation(0)=0;
	std::vector<Eigen::Vector2d> Velocities(timestep);
	std::vector<double> AngularVelocities(timestep);
	resultField.Q_t.resize(timestep);
	resultField.c_t.resize(timestep);
	resultField.Q_t[0] = Eigen::Matrix2d::Identity();
	resultField.c_t[0] = Eigen::Vector2d::Zero();

	// rotation
	double theta = 0;
	double angularVelocity = abc(2);
	AngularVelocities[0] = { angularVelocity };

	// translation
	Eigen::Vector2d translation_c_t = { 0, 0 };
	Eigen::Vector2d translation_cdot = { abc(0), abc(1) };
	Velocities[0] = translation_cdot;
	Eigen::Vector2d translation_cdotdot = { abc_dot(0), abc_dot(1) };

	for (size_t i = 1; i < timestep; i++) {
		theta = theta + dt * angularVelocity;
		angularVelocity = angularVelocity + dt * abc_dot(2);
		AngularVelocities[i] = angularVelocity;

		// translation
		translation_c_t = translation_c_t + dt * translation_cdot;
		translation_cdot = translation_cdot + dt * translation_cdotdot;
		Velocities[i] = translation_cdot;

		Eigen::Matrix2d rotQ;
		rotQ << cos(theta), -sin(theta),
			sin(theta), cos(theta);
		resultField.Q_t[i] = rotQ;
		resultField.c_t[i] = translation_c_t;
	}

	if (inputField.analyticalFlowfunc_) {

		resultField.analyticalFlowfunc_ = [inputField, tmin, resultField, dt, Velocities, AngularVelocities](const Eigen::Vector2d& pos, double t) -> Eigen::Vector2d {
			const double floatingTimeStep = (t - tmin) / dt;
			const int timestep_floor = std::clamp((int)std::floor(floatingTimeStep), 0, resultField.timeSteps - 1);
			const int timestep_ceil = std::clamp((int)std::floor(floatingTimeStep) + 1, 0, resultField.timeSteps - 1);
			const double ratio = floatingTimeStep - timestep_floor;

			const Eigen::Matrix2d Q_t = resultField.Q_t[timestep_floor] * (1 - ratio) + resultField.Q_t[timestep_ceil] * ratio;
			auto Q_transpose = Q_t.transpose();
			auto c_t = resultField.c_t[timestep_floor] * (1 - ratio) + resultField.c_t[timestep_ceil] * ratio;
			// => F^(-1)(x)= Q^T (x-c)= Q^T *( x+Q*pt-Os)
			Eigen::Vector2d F_inverse_x_2d = Q_transpose * (pos - c_t);

			auto v = inputField.getVectorAnalytical(F_inverse_x_2d, t);
			auto Velocity = Velocities[timestep_floor] * (1 - ratio) + Velocities[timestep_ceil] * ratio;
			auto AngularVelocity = AngularVelocities[timestep_floor] * (1 - ratio) + AngularVelocities[timestep_ceil] * ratio;

			Eigen::Matrix2d Spintensor;
			Spintensor(0, 0) = 0.0;
			Spintensor(1, 0) = -AngularVelocity;
			Spintensor(0, 1) = AngularVelocity;
			Spintensor(1, 1) = 0.0;
			Eigen::Matrix2d Q_dot = Q_t * Spintensor;
			Eigen::Vector2d translationTdot = Velocity;
			Eigen::Vector2d res = Q_t * v + Q_dot * F_inverse_x_2d + translationTdot;
			return res;
			};
		resultField.resampleFromAnalyticalExpression();
	}
	else {
		printf("error...");
	}
	return resultField;
}
// this function is similar to but 0
UnSteadyVectorField2D Tobias_reconstructUnsteadyField(const UnSteadyVectorField2D& inputField, const Eigen::Vector3d& abc, const Eigen::Vector3d& abc_dot)
{
	const auto tmax = inputField.tmax;
	const auto tmin = inputField.tmin;
	const double dt = (inputField.tmax - inputField.tmin) / ((double)inputField.timeSteps - 1.0);
	if (inputField.analyticalFlowfunc_) {

		UnSteadyVectorField2D outputField;
		outputField.spatialDomainMaxBoundary = inputField.getSpatialMaxBoundary();
		outputField.spatialDomainMinBoundary = inputField.getSpatialMinBoundary();
		outputField.spatialGridInterval = inputField.spatialGridInterval;
		outputField.XdimYdim = inputField.XdimYdim;
		outputField.tmin = tmin;
		outputField.tmax = tmax;
		outputField.timeSteps = inputField.timeSteps;
		outputField.field.resize(inputField.timeSteps);

		// Q(0)=I ->theta(0)=0; translation(0)=0;
		std::vector<Eigen::Vector2d> translation_c_t_list(inputField.timeSteps);
		std::vector<Eigen::Vector2d> Velocities(inputField.timeSteps);
		std::vector<double> AngularVelocities(inputField.timeSteps);
		std::vector<Eigen::Matrix2d> Q_t_list(inputField.timeSteps);
		translation_c_t_list[0] = { 0, 0 };
		// rotation
		double theta = 0;
		double angularVelocity = abc(2);
		AngularVelocities[0] = angularVelocity;
		Q_t_list[0] = Eigen::Matrix2d::Identity();

		// translation
		Eigen::Vector2d translation_c_t = { 0, 0 };
		Eigen::Vector2d translation_cdot = { abc(0), abc(1) };
		Velocities[0] = translation_cdot;
		Eigen::Vector2d translation_cdotdot = { abc_dot(0), abc_dot(1) };
		for (size_t i = 1; i < inputField.timeSteps; i++) {
			theta = theta + dt * angularVelocity;
			angularVelocity = angularVelocity + dt * abc_dot(2);
			AngularVelocities[i] = angularVelocity;
			Eigen::Matrix2d rotQ;
			rotQ << cos(theta), -sin(theta),
				sin(theta), cos(theta);
			Q_t_list[i] = rotQ;

			// translation
			translation_c_t = translation_c_t + dt * translation_cdot;
			translation_cdot = translation_cdot + dt * translation_cdotdot;
			Velocities[i] = translation_cdot;
			translation_c_t_list[i] = translation_c_t;
		}

		outputField.analyticalFlowfunc_ = [=](const Eigen::Vector2d& posX, double t) -> Eigen::Vector2d {
			const double floatingTimeStep = (t - tmin) / dt;
			const int timestep_floor = std::clamp((int)std::floor(floatingTimeStep), 0, inputField.timeSteps - 1);
			const int timestep_ceil = std::clamp((int)std::floor(floatingTimeStep) + 1, 0, inputField.timeSteps - 1);
			const double ratio = floatingTimeStep - timestep_floor;

			const auto Q_t = Q_t_list[timestep_floor] * (1 - ratio) + Q_t_list[timestep_ceil] * ratio;
			const auto c_t = translation_c_t_list[timestep_floor] * (1 - ratio) + translation_c_t_list[timestep_ceil] * ratio;
			const auto Velocity = Velocities[timestep_floor] * (1 - ratio) + Velocities[timestep_ceil] * ratio;
			const auto AngularVelocity = AngularVelocities[timestep_floor] * (1 - ratio) + AngularVelocities[timestep_ceil] * ratio;

			const auto Q_transpose = Q_t.transpose();

			Eigen::Vector2d xStar = Q_t * posX + c_t;
			Eigen::Vector2d v_star_xstar = inputField.getVectorAnalytical({ xStar(0), xStar(1) }, t);

			// compute Qdot=Q(t)*Omega(t) where Omega(t) is the anti-symmetric matrix of the angular velocity vector
			Eigen::Matrix2d Spintensor;
			Spintensor(0, 0) = 0.0;
			Spintensor(1, 0) = -AngularVelocity;
			Spintensor(0, 1) = AngularVelocity;
			Spintensor(1, 1) = 0.0;

			Eigen::Matrix2d Q_dot = Q_t * Spintensor;
			Eigen::Vector2d translationTdot = Velocity;
			Eigen::Vector2d v_at_pos = Q_transpose * (v_star_xstar - Q_dot * posX - translationTdot);
			return v_at_pos;
			};
		outputField.resampleFromAnalyticalExpression();
		return outputField;
	}
	else {
		printf("reconstructUnsteadyField only support analyticalFlowfunc_");
		return {};
	}
}




//template UnSteadyVectorField2D killingABCtransformation<double, UnSteadyVectorField2D>(const KillingAbcField& observerfield, const Eigen::Vector2d StartPosition, UnSteadyVectorField2D& inputField);
//template UnSteadyVectorField2D killingABCtransformation<double, SteadyVectorField2D>(const KillingAbcField& observerfield, const Eigen::Vector2d StartPosition, SteadyVectorField2D& inputField);
//
