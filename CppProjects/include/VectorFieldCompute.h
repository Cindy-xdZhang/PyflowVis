#pragma once
#ifndef __VECTOR_COMPUTE_H___
#define __VECTOR_COMPUTE_H___
#include <Eigen/Dense>
#include <functional>
#include <vector>
#include <array>
#include <random>
// for steady field  second parameter (t) is ignored.
using AnalyticalFlowFunc2D = std::function<Eigen::Vector2d(const Eigen::Vector2d&, double)>;
using AnalyticalFlowFunc3D = std::function<Eigen::Vector3d(const Eigen::Vector3d&, double)>;
using velocityFieldData = std::vector<std::vector<Eigen::Vector2d>>;
// save 3d position, vorticity,IVD,distance(to start seeding point)
using PathlinePointInfo = std::vector<double>;
enum class PATHLINE_POINT_INFO : unsigned char {
	POSITION_X = 0,
	POSITION_Y = 1,
	TIME_T = 2,
	IVD = 3,
	DISTANCE_OR_LABEL = 4,
	VELOCITY_MAGNITUDE = 5,
	NABLA_U00,
	NABLA_U01,
	NABLA_U11
};
enum class PATHLINE_SEEDING_SAMPLING : unsigned char {
	GRID_CROSS_SEEDING = 0,
	RECTANGULAR_CLUSTER4_RANDOM
};


using Color = Eigen::Vector3d;

template <int Component>
__forceinline int VecComponentAddressTrans(const int x, const int y, const int mgridDim_x)
{
	static_assert(Component == 1 || Component == 2);
	const int GridPointFlatternIdx = x + y * mgridDim_x;
	if constexpr (Component == 1) {
		return 2 * GridPointFlatternIdx;
	}
	else if constexpr (Component == 2) {
		return 2 * GridPointFlatternIdx + 1;
	}
};

template <typename T>
T bilinear_interpolate(const std::vector<std::vector<T>>& vector_field, double x, double y)
{
	x = std::clamp(x, double(0), double(vector_field[0].size() - 1));
	y = std::clamp(y, double(0), double(vector_field.size() - 1));

	int x0 = static_cast<int>(x);
	int y0 = static_cast<int>(y);

	int x1 = std::min(x0 + 1, static_cast<int>(vector_field[0].size() - 1));
	int y1 = std::min(y0 + 1, static_cast<int>(vector_field.size() - 1));

	double tx = x - x0;
	double ty = y - y0;

	T v00 = vector_field[y0][x0];
	T v01 = vector_field[y0][x1];
	T v10 = vector_field[y1][x0];
	T v11 = vector_field[y1][x1];
	T a = v00 * (1 - tx) + v01 * tx;
	T b = v10 * (1 - tx) + v11 * tx;
	return a * (1 - ty) + b * ty;
}

template <typename T, int N>
T bilinear_interpolate(const std::array<std::array<T, N>, N>& vector_field, double x, double y)
{
	x = std::clamp(x, double(0), double(N - 1));
	y = std::clamp(y, double(0), double(N - 1));

	int x0 = static_cast<int>(x);
	int y0 = static_cast<int>(y);

	int x1 = std::min(x0 + 1, static_cast<int>(N - 1));
	int y1 = std::min(y0 + 1, static_cast<int>(N - 1));

	double tx = x - x0;
	double ty = y - y0;

	T v00 = vector_field[y0][x0];
	T v01 = vector_field[y0][x1];
	T v10 = vector_field[y1][x0];
	T v11 = vector_field[y1][x1];
	T a = v00 * (1 - tx) + v01 * tx;
	T b = v10 * (1 - tx) + v11 * tx;
	return a * (1 - ty) + b * ty;
}

template <typename T>
T trilinear_interpolate(const std::vector<std::vector<std::vector<T>>>& vector_field, double x, double y, double t)
{
	int t0 = std::clamp(t, double(0), double(vector_field.size() - 1));
	int t1 = std::min(t0 + 1, static_cast<int>(vector_field.size() - 1));
	auto value_t0 = bilinear_interpolate(vector_field[t0], x, y);
	auto value_t1 = bilinear_interpolate(vector_field[t1], x, y);
	double tx = t - t0;
	auto retValue = value_t0 * (1 - tx) + value_t1 * (tx);
	return retValue;
}


struct IUnsteadField2D {
public:
	virtual Eigen::Vector2d getVector(int x, int y, int t) const = 0;
	virtual Eigen::Vector2d getVector(double x, double y, double t) const = 0;
	inline Eigen::Vector2d getVector(Eigen::Vector2d pos, double t) const
	{
		return this->getVector(pos.x(), pos.y(), t);
	}

	Eigen::Vector2d getSpatialMinBoundary() const { return spatialDomainMinBoundary; }
	Eigen::Vector2d getSpatialMaxBoundary() const { return spatialDomainMaxBoundary; }

public:
	// for anayltical field this variables are not set
	Eigen::Vector2d spatialDomainMinBoundary;
	Eigen::Vector2d spatialDomainMaxBoundary;
	Eigen::Vector2d spatialGridInterval;
	Eigen::Vector2i XdimYdim;
	// for steady field variables (tmin, tmax,timesteps) are not set
	double tmin, tmax;
	int timeSteps = -1;
	AnalyticalFlowFunc2D analyticalFlowfunc_ = nullptr;
	inline Eigen::Vector2d getVectorAnalytical(const Eigen::Vector2d& pos, double t) const
	{
		assert(this->analyticalFlowfunc_);
		return this->analyticalFlowfunc_(pos, t);
	}
};

struct SteadyVectorField2D : public IUnsteadField2D {
	SteadyVectorField2D() = default;
	SteadyVectorField2D(const std::vector<std::vector<Eigen::Vector2d>>& ifield, const Eigen::Vector2d& ispatialmin, const Eigen::Vector2d& ispatialMax, const Eigen::Vector2i& ixdimydim)
		: field(ifield)
	{
		this->spatialDomainMinBoundary = ispatialmin;
		this->spatialDomainMaxBoundary = ispatialMax;
		this->XdimYdim = ixdimydim;
		this->spatialGridInterval = (spatialDomainMaxBoundary - spatialDomainMinBoundary).cwiseQuotient(Eigen::Vector2d(XdimYdim.x() - 1, XdimYdim.y() - 1));
	}
	std::vector<std::vector<Eigen::Vector2d>> field; // first y dimension then x dimension



	Eigen::Vector2d getVector(int x, int y) const
	{
		return field[y][x];
	}
	inline Eigen::Vector2d getVector(double x, double y) const
	{
		const double inverse_grid_interval_x = 1.0f / (double)spatialGridInterval(0);
		const double inverse_grid_interval_y = 1.0f / (double)spatialGridInterval(1);
		double doubleIndicesX = (x - spatialDomainMinBoundary(0)) * inverse_grid_interval_x;
		double doubleIndicesY = (y - spatialDomainMinBoundary(1)) * inverse_grid_interval_y;
		return bilinear_interpolate(field, doubleIndicesX, doubleIndicesY);
	}
	Eigen::Vector2d getVector(int x, int y, int t) const override
	{
		return getVector(x, y);
	}
	Eigen::Vector2d getVector(double x, double y, double t) const override
	{
		return getVector(x, y);
	}
	inline Eigen::Vector2d getVector(const Eigen::Vector2d& pos, double t) const
	{
		return getVector(pos.x(), pos.y());
	}
	inline Eigen::Vector2d getVector(const Eigen::Vector2d& pos) const
	{
		return getVector(pos.x(), pos.y());
	}
	// parameter x,y are the physical position in the vector field domain.
	// SteadyVectorField2D  might have analytical expression, then when query value from out of boundary is return valid value.


	inline bool resampleFromAnalyticalExpression()
	{
		if (this->analyticalFlowfunc_) [[likely]] {
			const auto ydim = this->XdimYdim.y();
			const auto xdim = this->XdimYdim.x();
			this->field.resize(ydim);

			for (int y = 0; y < ydim; y++) {
				// y slice i
				this->field[y].resize(xdim);
				for (int x = 0; x < ydim; x++) {
					// y slice
					Eigen::Vector2d pos = { x * this->spatialGridInterval(0) + this->spatialDomainMinBoundary(0), y * this->spatialGridInterval(1) + this->spatialDomainMinBoundary(1) };
					Eigen::Vector2d analyticalVec = this->analyticalFlowfunc_(pos, 0);

					if (std::isnan(analyticalVec(0)) || std::isnan(analyticalVec(1))) {
						printf("get nan during resampleFromAnalyticalExpression.");
						return false;
					}
					this->field[y][x] = analyticalVec;
				}
			}

			return true;
			}
		assert(false);
		return false;
	}
};

class UnSteadyVectorField2D : public IUnsteadField2D {
public:
	std::vector<std::vector<std::vector<Eigen::Vector2d>>> field; // first t, then y dimension then x dimension


	inline Eigen::Vector2d getVector(int x, int y, int t) const
	{
		return field[t][y][x];
	}
	inline Eigen::Vector2d getVector(double x, double y, double t) const
	{
		const double inverse_grid_interval_t = (double)(timeSteps) / (tmax - tmin);

		const double inverse_grid_interval_x = 1.0f / (double)spatialGridInterval(0);
		const double inverse_grid_interval_y = 1.0f / (double)spatialGridInterval(1);
		double floatIndicesX = (x - spatialDomainMinBoundary(0)) * inverse_grid_interval_x;
		double floatIndicesY = (y - spatialDomainMinBoundary(1)) * inverse_grid_interval_y;
		double floatIndicesT = (t - tmin) * inverse_grid_interval_t;

		// Trilinear interpolation
		auto floorX = static_cast<int>(std::floor(floatIndicesX));
		auto floorY = static_cast<int>(std::floor(floatIndicesY));
		auto floorT = static_cast<int>(std::floor(floatIndicesT));
		// if tx is bigger than one need to clamp
		double tx = floatIndicesX - floorX;
		double ty = floatIndicesY - floorY;
		double tt = floatIndicesT - floorT;

		int x0 = std::clamp(floorX, 0, static_cast<int>(field[0][0].size() - 1));
		int x1 = std::clamp(floorX + 1, 0, static_cast<int>(field[0][0].size() - 1));
		int y0 = std::clamp(floorY, 0, static_cast<int>(field[0].size() - 1));
		int y1 = std::clamp(floorY + 1, 0, static_cast<int>(field[0].size() - 1));
		int t0 = std::clamp(floorT, 0, static_cast<int>(field.size() - 1));
		int t1 = std::clamp(floorT + 1, 0, static_cast<int>(field.size() - 1));

		const auto& c000 = field[t0][y0][x0];
		const auto& c100 = field[t0][y0][x1];
		const auto& c010 = field[t0][y1][x0];
		const auto& c110 = field[t0][y1][x1];
		const auto& c001 = field[t1][y0][x0];
		const auto& c101 = field[t1][y0][x1];
		const auto& c011 = field[t1][y1][x0];
		const auto& c111 = field[t1][y1][x1];

		const auto c00 = c000 * (1 - tx) + c100 * tx;
		const auto c10 = c010 * (1 - tx) + c110 * tx;
		const auto c01 = c001 * (1 - tx) + c101 * tx;
		const auto c11 = c011 * (1 - tx) + c111 * tx;

		Eigen::Vector2d c0 = c00 * (1 - ty) + c10 * ty;
		Eigen::Vector2d c1 = c01 * (1 - ty) + c11 * ty;

		return c0 * (1 - tt) + c1 * tt;
	}

	inline Eigen::Vector2d getVector(Eigen::Vector2d pos, double t) const
	{
		return getVector(pos(0), pos(1), t);
	}

	inline const SteadyVectorField2D getVectorfieldSliceAtTime(int t) const
	{
		SteadyVectorField2D vecfield;
		vecfield.spatialDomainMinBoundary = spatialDomainMinBoundary;
		vecfield.spatialDomainMaxBoundary = spatialDomainMaxBoundary;
		vecfield.spatialGridInterval = spatialGridInterval;
		if (t >= 0 && t < timeSteps) {
			vecfield.field = field[static_cast<int>(t)];
		}
		else {
			assert(false);
		}

		return vecfield;
	}
	inline const auto getSliceDataAtTime(int t) const
	{

		if (t >= 0 && t < timeSteps) {
			return field[static_cast<int>(t)];
		}
		assert(false);
		return std::vector<std::vector<Eigen::Vector2d>>();
	}
	inline bool resampleFromAnalyticalExpression()
	{
		if (this->analyticalFlowfunc_) {

			// this->timeSteps = timeSteps;
			// this->spatialGridInterval = (this->spatialDomainMaxBoundary - this->spatialDomainMinBoundary).cwiseQuotient(Eigen::Vector2d(GridX - 1, GridY - 1));
			this->field.resize(this->timeSteps);
			const auto xdim = this->XdimYdim.x();
			const auto ydim = this->XdimYdim.y();
			const auto dt = (tmax - tmin) / (double)(timeSteps - 1);
			for (size_t i = 0; i < this->timeSteps; i++) {
				// time slice i
				this->field[i].resize(ydim);
				const double physical_t_this_slice = tmin + i * dt;
				for (size_t j = 0; j < ydim; j++) {
					// y slice
					this->field[i][j].resize(xdim);
					for (size_t k = 0; k < xdim; k++) {
						Eigen::Vector2d pos = { k * this->spatialGridInterval(0) + this->spatialDomainMinBoundary(0), j * this->spatialGridInterval(1) + this->spatialDomainMinBoundary(1) };
						Eigen::Vector2d analyticalVec = this->analyticalFlowfunc_(pos, physical_t_this_slice);

						this->field[i][j][k] = analyticalVec;
					}
				}
			}

			return true;
		}
		return false;
	}

	// if this unsteady field is generated from another field u by reference frame transformation x*=Q(t)x+c(t), then we can save the Q(t) and c(t) for future use.
	std::vector<Eigen::Matrix2d> Q_t;
	std::vector<Eigen::Vector2d> c_t;
	//// theta_t is generated by matrix2agnle(Q_t)
	// std::vector<Eigen::Matrix3d> theta_t;
};
class KillingAbcField : public IUnsteadField2D {
public:
	KillingAbcField(std::function<Eigen::Vector3d(double)>& func, int n, double tmin, double tmax)
	{
		this->tmin = tmin;
		this->tmax = tmax;
		this->timeSteps = n;
		dt = (tmax - tmin) / (double)(n - 1);
		assert(n > 1);
		assert(dt > 0.0);
		centerPos = { 0.0, 0.0 };
		this->killingABCfunc_ = func;
	}
	double dt;
	Eigen::Vector2d centerPos;
	std::function<Eigen::Vector3d(double)> killingABCfunc_ = nullptr;
	template <class Archive>
	void serialize(Archive& ar)
	{
		// std::vector<Eigen::Vector3d> abcs_;
		// abcs_.reserve(timeSteps);
		// for (size_t i = 0; i < timeSteps; i++) {
		//     double time = this->tmin + i * this->dt;
		//     Eigen::Vector3d abc = this->killingABCfunc_(time);
		//     abcs_.push_back(abc);
		// }
		// ar(CEREAL_NVP(tmin), CEREAL_NVP(tmax), CEREAL_NVP(timeSteps), CEREAL_NVP(dt) /*, CEREAL_NVP(centerPos)*/);
		// ar(CEREAL_NVP(abcs_));
	}

	// give a curve of a(t),b(t),c(t), killing vector(x,t) = a(t)+b(t)+ [0,-c(t);c(t),0]*(x-o), where o is the center of the plane.
	inline Eigen::Vector2d getKillingVector(const Eigen::Vector2d& queryPos, double t) const
	{
		if (this->killingABCfunc_)
			[[likely]] {
			Eigen::Vector3d abc = this->killingABCfunc_(t);

			Eigen::Vector2d uv = { abc(0), abc(1) };
			auto ra = queryPos - centerPos;

			const auto c = abc(2);
			//[0,-c(t);c(t),0]*(ra)
			Eigen::Vector2d c_componnet = { ra(1) * -c, ra(0) * c };
			return uv + c_componnet;
			}
		assert(false);
		return {};
	}
	virtual Eigen::Vector2d getVector(int x, int y, int t) const
	{
		// we don't need this function
		assert(false);
		return getKillingVector({ (double)x, (double)y }, t);
	};

	virtual Eigen::Vector2d getVector(double x, double y, double t) const
	{
		return getKillingVector({ (double)x, (double)y }, t);
	};
	UnSteadyVectorField2D resample2UnsteadyField(
		const Eigen::Vector2i& gridSize,
		const Eigen::Vector2d& domainBoundaryMin,
		const Eigen::Vector2d& domainBoundaryMax)
	{
		UnSteadyVectorField2D vectorField2d;

		vectorField2d.spatialDomainMinBoundary = domainBoundaryMin;
		vectorField2d.spatialDomainMaxBoundary = domainBoundaryMax;
		vectorField2d.timeSteps = this->timeSteps;
		vectorField2d.tmin = this->tmin;
		vectorField2d.tmax = this->tmax;

		Eigen::Vector2d spatialGridInterval;
		spatialGridInterval(0) = (domainBoundaryMax(0) - domainBoundaryMin(0)) / (gridSize(0) - 1);
		spatialGridInterval(1) = (domainBoundaryMax(1) - domainBoundaryMin(1)) / (gridSize(1) - 1);
		vectorField2d.spatialGridInterval = spatialGridInterval;

		vectorField2d.field.resize(this->timeSteps, std::vector<std::vector<Eigen::Vector2d>>(gridSize.y(), std::vector<Eigen::Vector2d>(gridSize.x())));

		for (int t = 0; t < timeSteps; ++t) {
			double time = this->tmin + t * this->dt;
			for (int y = 0; y < gridSize.y(); ++y) {
				double posY = domainBoundaryMin(1) + y * spatialGridInterval(1);
				for (int x = 0; x < gridSize.x(); ++x) {
					double posX = domainBoundaryMin(0) + x * spatialGridInterval(0);
					vectorField2d.field[t][y][x] = getVector(posX, posY, time);
				}
			}
		}

		return vectorField2d;
	}
};
//////////////////////////////////////////////////////////////////////////////
/////////////// functional to compute criterion from vector field/////////////
//////////////////////////////////////////////////////////////////////////////
enum class VORTEX_CRITERION {
	NONE = 0,
	CURL,
	Q_CRITERION,
	LAMBDA2_CRITERION,
	IVD_CRITERION,
	DELTA_CRITERION,
	SUJUDI_HAIMES_CRITERION,
	SOBEL_EDGE_DETECTION,
	VELICITY_MAGINITUDE
	// LAVD wip
};

template <class T>
std::pair<double, double> computeMinMax(const std::vector<T>& values)
{
	if (values.empty()) {
		throw std::invalid_argument("The input vector is empty.");
	}

	T minVal = std::numeric_limits<T>::infinity();
	T maxVal = -std::numeric_limits<T>::infinity();

	for (T val : values) {
		if (val < minVal) {
			minVal = val;
		}
		if (val > maxVal) {
			maxVal = val;
		}
	}

	return { minVal, maxVal };
}
template <class T>
std::pair<double, double> computeMinMax(const std::vector<std::vector<T>>& values)
{
	if (values.empty() || values[0].empty()) {
		throw std::invalid_argument("The input vector is empty.");
	}

	double minVal = std::numeric_limits<double>::infinity();
	double maxVal = -std::numeric_limits<double>::infinity();
	//constexpr handle T is eigen::Vector
	if constexpr (std::is_same_v<T, Eigen::Vector2d> || std::is_same_v<T, Eigen::Vector3d> || std::is_same_v<T, Eigen::VectorXd>) {
		for (const auto& row : values) {
			for (const auto& vec : row) {
				double norm = vec.norm();
				if (norm < minVal) {
					minVal = norm;
				}
				if (norm > maxVal) {
					maxVal = norm;
				}
			}
		}
	}
	else
	{
		for (const auto& row : values) {
			for (T val : row) {
				if (val < minVal) {
					minVal = val;
				}
				if (val > maxVal) {
					maxVal = val;
				}
			}
		}

	}

	return { minVal, maxVal };
}

template <class T, int N>
std::pair<T, T> computeMinMax(const std::array<std::array<T, N>, N>& values)
{
	if (values.empty() || values[0].empty()) {
		throw std::invalid_argument("The input vector is empty.");
	}

	T minVal = std::numeric_limits<T>::infinity();
	T maxVal = -std::numeric_limits<T>::infinity();
	int rowid = 0;
	for (const auto& row : values) {
		for (T val : row) {
			if (val < minVal) {
				minVal = val;
			}
			if (val > maxVal) {
				maxVal = val;
			}
		}
		rowid += 1;
	}

	return { minVal, maxVal };
}

inline std::vector<std::vector<double>> ComputeCurl(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim, double SpatialGridIntervalX, double SpatialGridIntervalY)
{
	std::vector<std::vector<double>> curl(Ydim, std::vector<double>(Xdim, 0.0f));
	const double inverse_grid_interval_x = 1.0f / (double)SpatialGridIntervalX;
	const double inverse_grid_interval_y = 1.0f / (double)SpatialGridIntervalY;
	// Calculate curl (vorticity) of the vector field
	for (int y = 1; y < Ydim - 1; ++y) {
		for (int x = 1; x < Xdim - 1; ++x) {
			Eigen::Vector2d dv_dx = (vecfieldData[y][x + 1] - vecfieldData[y][x - 1]) * 0.5f * inverse_grid_interval_x;
			Eigen::Vector2d du_dy = (vecfieldData[y + 1][x] - vecfieldData[y - 1][x]) * 0.5f * inverse_grid_interval_y;
			double curl_ = dv_dx(1) - du_dy(0);
			curl[y][x] = curl_;
		}
	}
	return curl;
}

// Function to compute the Q criterion for a 2D steady vector field slice
inline std::vector<std::vector<double>> ComputeQCriterion(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim, double SpatialGridIntervalX, double SpatialGridIntervalY)
{
	std::vector<std::vector<double>> Q(Ydim, std::vector<double>(Xdim, 0.0));
	const double inverse_grid_interval_x = 1.0 / SpatialGridIntervalX;
	const double inverse_grid_interval_y = 1.0 / SpatialGridIntervalY;

	for (int y = 1; y < Ydim - 1; ++y) {
		for (int x = 1; x < Xdim - 1; ++x) {
			Eigen::Vector2d du_dx = (vecfieldData[y][x + 1] - vecfieldData[y][x - 1]) * 0.5 * inverse_grid_interval_x;
			Eigen::Vector2d dv_dy = (vecfieldData[y + 1][x] - vecfieldData[y - 1][x]) * 0.5 * inverse_grid_interval_y;
			Eigen::Matrix2d gradient;
			gradient << du_dx(0), du_dx(1),
				dv_dy(0), dv_dy(1);

			Eigen::Matrix2d S = 0.5 * (gradient + gradient.transpose());
			Eigen::Matrix2d Omega = 0.5 * (gradient - gradient.transpose());

			double Q_value = 0.5 * (Omega.squaredNorm() - S.squaredNorm());
			Q[y][x] = Q_value;
		}
	}
	return Q;
}
inline std::vector<std::vector<double>> ComputeDeltaCriterion(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim, double SpatialGridIntervalX, double SpatialGridIntervalY)
{
	std::vector<std::vector<double>> delta(Ydim, std::vector<double>(Xdim, 0.0));
	const double inverse_grid_interval_x = 1.0 / SpatialGridIntervalX;
	const double inverse_grid_interval_y = 1.0 / SpatialGridIntervalY;

	for (int y = 1; y < Ydim - 1; ++y) {
		for (int x = 1; x < Xdim - 1; ++x) {
			Eigen::Vector2d dv_dx = (vecfieldData[y][x + 1] - vecfieldData[y][x - 1]) * 0.5 * inverse_grid_interval_x;
			Eigen::Vector2d du_dy = (vecfieldData[y + 1][x] - vecfieldData[y - 1][x]) * 0.5 * inverse_grid_interval_y;
			Eigen::Matrix2d Jacobian;
			Jacobian << dv_dx(0), dv_dx(1),
				du_dy(0), du_dy(1);
			auto J2 = Jacobian * Jacobian;
			auto Q = -0.5 * J2.trace();
			auto R = Jacobian.determinant();
			double detlaVal = std::pow(Q / 3.0, 3.0) + std::pow(R / 2.0, 2.0);
			delta[y][x] = detlaVal;
		}
	}
	return delta;
}

// Function to compute the lambda_2 criterion for a 2D steady vector field slice
inline std::vector<std::vector<double>> ComputeLambda2Criterion(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim, double SpatialGridIntervalX, double SpatialGridIntervalY)
{
	std::vector<std::vector<double>> lambda2(Ydim, std::vector<double>(Xdim, 0.0));
	const double inverse_grid_interval_x = 1.0 / SpatialGridIntervalX;
	const double inverse_grid_interval_y = 1.0 / SpatialGridIntervalY;

	for (int y = 1; y < Ydim - 1; ++y) {
		for (int x = 1; x < Xdim - 1; ++x) {
			Eigen::Vector2d du_dx = (vecfieldData[y][x + 1] - vecfieldData[y][x - 1]) * 0.5 * inverse_grid_interval_x;
			Eigen::Vector2d dv_dy = (vecfieldData[y + 1][x] - vecfieldData[y - 1][x]) * 0.5 * inverse_grid_interval_y;
			Eigen::Matrix2d gradient;
			gradient << du_dx(0), du_dx(1),
				dv_dy(0), dv_dy(1);

			Eigen::Matrix2d S = 0.5 * (gradient + gradient.transpose());
			Eigen::Matrix2d Omega = 0.5 * (gradient - gradient.transpose());

			Eigen::Matrix2d S2_ADD_OMEGA2 = S * S + Omega * Omega;
			Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(S2_ADD_OMEGA2);
			Eigen::Vector2d eigenvalues = solver.eigenvalues();
			lambda2[y][x] = eigenvalues(1); // The second largest eigenvalue
		}
	}
	return lambda2;
}

// Function to compute Instantaneous Vorticity Deviation (IVD) for a 2D steady vector field slice
inline std::vector<std::vector<double>> ComputeIVD(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim, double SpatialGridIntervalX, double SpatialGridIntervalY)
{
	std::vector<std::vector<double>> IVD(Ydim, std::vector<double>(Xdim, 0.0));
	const double inverse_grid_interval_x = 1.0 / SpatialGridIntervalX;
	const double inverse_grid_interval_y = 1.0 / SpatialGridIntervalY;
	auto curlField = ComputeCurl(vecfieldData, Xdim, Ydim, SpatialGridIntervalX, SpatialGridIntervalY);
	double averageCurl = 0.0;
	for (const auto& row : curlField) {
		double sumRow = 0.0;
		for (auto val : row) {
			sumRow += val;
		}
		averageCurl += sumRow;
	}
	averageCurl /= (Xdim - 2) * (Ydim - 2);

	for (int y = 1; y < Ydim - 1; ++y) {
		for (int x = 1; x < Xdim - 1; ++x) {
			Eigen::Vector2d dv_dx = (vecfieldData[y][x + 1] - vecfieldData[y][x - 1]) * 0.5 * inverse_grid_interval_x;
			Eigen::Vector2d du_dy = (vecfieldData[y + 1][x] - vecfieldData[y - 1][x]) * 0.5 * inverse_grid_interval_y;
			double vorticity = dv_dx(1) - du_dy(0);

			IVD[y][x] = std::abs(vorticity - averageCurl);
		}
	}
	return IVD;
}

inline std::vector<std::vector<Eigen::Matrix2d>> ComputeNablaU(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim, double SpatialGridIntervalX, double SpatialGridIntervalY)
{

	std::vector<std::vector<Eigen::Matrix2d>> NablaU(Ydim, std::vector<Eigen::Matrix2d>(Xdim);
	const double inverse_grid_interval_x = 1.0 / SpatialGridIntervalX;
	const double inverse_grid_interval_y = 1.0 / SpatialGridIntervalY;

	for (int y = 0; y < Ydim; ++y) {
		for (int x = 0; x < Xdim ++x) {
			int xPlus1 = std::min(x + 1, )

				Eigen::Vector2d du_dx = (vecfieldData[y][x + 1] - vecfieldData[y][x - 1]) * 0.5 * inverse_grid_interval_x;
			Eigen::Vector2d dv_dy = (vecfieldData[y + 1][x] - vecfieldData[y - 1][x]) * 0.5 * inverse_grid_interval_y;
			Eigen::Matrix2d gradient;
			gradient << du_dx(0), du_dx(1),
				dv_dy(0), dv_dy(1);

			NablaU[y][x] = gradient;
		}
	}

	return NablaU;
}

inline std::vector<std::vector<double>> ComputeSujudiHaimes(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim, double SpatialGridIntervalX, double SpatialGridIntervalY)
{
	std::vector<std::vector<double>> sujudiHaimes(Ydim, std::vector<double>(Xdim, 0.0));
	const double inverse_grid_interval_x = 1.0 / SpatialGridIntervalX;
	const double inverse_grid_interval_y = 1.0 / SpatialGridIntervalY;

	for (int y = 1; y < Ydim - 1; ++y) {
		for (int x = 1; x < Xdim - 1; ++x) {
			Eigen::Vector2d dv_dx = (vecfieldData[y][x + 1] - vecfieldData[y][x - 1]) * 0.5 * inverse_grid_interval_x;
			Eigen::Vector2d du_dy = (vecfieldData[y + 1][x] - vecfieldData[y - 1][x]) * 0.5 * inverse_grid_interval_y;
			Eigen::Matrix2d gradient;
			gradient << dv_dx(0), dv_dx(1),
				du_dy(0), du_dy(1);

			auto JV = gradient * vecfieldData[y][x];
			auto V = vecfieldData[y][x];
			// check JV and v is paralell?
			bool paralllel = JV.dot(V) == JV.norm() * V.norm();
			sujudiHaimes[y][x] = paralllel ? 1.0 : 0.0;
		}
	}
	return sujudiHaimes;
}

// Function to compute Lagrangian-Averaged Vorticity Deviation (LAVD) for a 2D unsteady vector field
inline std::vector<std::vector<double>> ComputeLAVD(const std::vector<std::vector<std::vector<Eigen::Vector2d>>>& vecfieldData, int Xdim, int Ydim, int timeSteps, double SpatialGridIntervalX, double SpatialGridIntervalY, double tmin, double tmax)
{
	/*  std::vector<std::vector<double>> LAVD(Ydim, std::vector<double>(Xdim, 0.0));
	  const double dt = (tmax - tmin) / static_cast<double>(timeSteps - 1);

	  for (int y = 0; y < Ydim; ++y) {
		  for (int x = 0; x < Xdim; ++x) {
			  double lavd_sum = 0.0;

			  for (int t = 0; t < timeSteps; ++t) {
				  Eigen::Vector2d dv_dx = (vecfieldData[t][y][x + 1] - vecfieldData[t][y][x - 1]) * 0.5 * (1.0 / SpatialGridIntervalX);
				  Eigen::Vector2d du_dy = (vecfieldData[t][y + 1][x] - vecfieldData[t][y - 1][x]) * 0.5 * (1.0 / SpatialGridIntervalY);
				  double vorticity = dv_dx(1) - du_dy(0);

				  lavd_sum += std::abs(vorticity) * dt;
			  }

			  LAVD[y][x] = lavd_sum;
		  }
	  }
	  return LAVD;*/
	return {};
}

inline std::vector<std::vector<double>> ComputeFTLE(
	const std::vector<std::vector<Eigen::Vector2d>>& vecfieldDataStart,
	const std::vector<std::vector<Eigen::Vector2d>>& vecfieldDataEnd,
	int Xdim, int Ydim,
	double SpatialGridIntervalX, double SpatialGridIntervalY,
	double deltaT)
{
	std::vector<std::vector<double>> FTLE(Ydim, std::vector<double>(Xdim, 0.0));
	const double inverse_grid_interval_x = 1.0 / SpatialGridIntervalX;
	const double inverse_grid_interval_y = 1.0 / SpatialGridIntervalY;

	for (int y = 1; y < Ydim - 1; ++y) {
		for (int x = 1; x < Xdim - 1; ++x) {
			// Compute flow map gradient
			Eigen::Matrix2d flowMapGradient;

			// dx/dx0
			flowMapGradient(0, 0) = (vecfieldDataEnd[y][x + 1](0) - vecfieldDataEnd[y][x - 1](0)) * 0.5 * inverse_grid_interval_x;
			// dy/dx0
			flowMapGradient(1, 0) = (vecfieldDataEnd[y][x + 1](1) - vecfieldDataEnd[y][x - 1](1)) * 0.5 * inverse_grid_interval_x;
			// dx/dy0
			flowMapGradient(0, 1) = (vecfieldDataEnd[y + 1][x](0) - vecfieldDataEnd[y - 1][x](0)) * 0.5 * inverse_grid_interval_y;
			// dy/dy0
			flowMapGradient(1, 1) = (vecfieldDataEnd[y + 1][x](1) - vecfieldDataEnd[y - 1][x](1)) * 0.5 * inverse_grid_interval_y;

			// Compute Cauchy-Green strain tensor
			Eigen::Matrix2d cauchyGreen = flowMapGradient.transpose() * flowMapGradient;

			// Compute eigenvalues
			Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(cauchyGreen);
			double maxEigenvalue = eigensolver.eigenvalues().maxCoeff();

			// Compute FTLE
			FTLE[y][x] = std::log(std::sqrt(maxEigenvalue)) / std::abs(deltaT);
		}
	}

	return FTLE;
}

inline std::vector<std::vector<double>> ComputeVelocityMagniture(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim)
{
	std::vector<std::vector<double>> mag(Ydim, std::vector<double>(Xdim, 0.0f));
	// Calculate curl (vorticity) of the vector field
	for (int y = 0; y < Ydim; ++y) {
		for (int x = 0; x < Xdim; ++x) {
			mag[y][x] = vecfieldData[y][x].norm();
		}
	}
	return mag;
}

// Function to compute the Sobel edge detection for a 2D vector field slice
inline std::vector<std::vector<double>> ComputeSobelEdge(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim)
{
	std::vector<std::vector<double>> edgeMagnitude(Ydim, std::vector<double>(Xdim, 0.0));

	// Sobel kernels
	Eigen::Matrix3d sobelX;
	sobelX << -1, 0, 1,
		-2, 0, 2,
		-1, 0, 1;

	Eigen::Matrix3d sobelY;
	sobelY << -1, -2, -1,
		0, 0, 0,
		1, 2, 1;

	for (int y = 1; y < Ydim - 1; ++y) {
		for (int x = 1; x < Xdim - 1; ++x) {
			double gx1 = 0.0;
			double gy1 = 0.0;
			double gx2 = 0.0;
			double gy2 = 0.0;

			// Apply Sobel kernels to both components of the vector field
			for (int i = -1; i <= 1; ++i) {
				for (int j = -1; j <= 1; ++j) {
					gx1 += sobelX(i + 1, j + 1) * vecfieldData[y + i][x + j](0);
					gy1 += sobelY(i + 1, j + 1) * vecfieldData[y + i][x + j](0);

					gx2 += sobelX(i + 1, j + 1) * vecfieldData[y + i][x + j](1);
					gy2 += sobelY(i + 1, j + 1) * vecfieldData[y + i][x + j](1);
				}
			}

			// Compute the magnitude of the gradient
			double gradientMagnitude1 = std::sqrt(gx1 * gx1 + gy1 * gy1);
			double gradientMagnitude2 = std::sqrt(gx2 * gx2 + gy2 * gy2);

			// Combine the magnitudes from both components
			edgeMagnitude[y][x] = std::sqrt(gradientMagnitude1 * gradientMagnitude1 + gradientMagnitude2 * gradientMagnitude2);
		}
	}

	return edgeMagnitude;
}

inline auto computeTargetCrtierion(const std::vector<std::vector<Eigen::Vector2d>>& vecfieldData, int Xdim, int Ydim, double SpatialGridIntervalX, double SpatialGridIntervalY, VORTEX_CRITERION criterionENUM)
{
	switch (criterionENUM) {
	case VORTEX_CRITERION::Q_CRITERION:
		return ComputeQCriterion(vecfieldData, Xdim, Ydim, SpatialGridIntervalX, SpatialGridIntervalY);
	case VORTEX_CRITERION::LAMBDA2_CRITERION:
		return ComputeLambda2Criterion(vecfieldData, Xdim, Ydim, SpatialGridIntervalX, SpatialGridIntervalY);
	case VORTEX_CRITERION::IVD_CRITERION:
		return ComputeIVD(vecfieldData, Xdim, Ydim, SpatialGridIntervalX, SpatialGridIntervalY);
	case VORTEX_CRITERION::DELTA_CRITERION:
		return ComputeDeltaCriterion(vecfieldData, Xdim, Ydim, SpatialGridIntervalX, SpatialGridIntervalY);
	case VORTEX_CRITERION::SUJUDI_HAIMES_CRITERION:
		return ComputeSujudiHaimes(vecfieldData, Xdim, Ydim, SpatialGridIntervalX, SpatialGridIntervalY);
	case VORTEX_CRITERION::SOBEL_EDGE_DETECTION:
		return ComputeSobelEdge(vecfieldData, Xdim, Ydim);
	case VORTEX_CRITERION::VELICITY_MAGINITUDE:
		return ComputeVelocityMagniture(vecfieldData, Xdim, Ydim);
	case VORTEX_CRITERION::CURL:
	default:
		return ComputeCurl(vecfieldData, Xdim, Ydim, SpatialGridIntervalX, SpatialGridIntervalY);
	}
}

// lic LICAlgorithms
std::vector<std::vector<double>> randomNoiseTexture(int width, int height);
std::vector<std::vector<Eigen::Vector3d>> LICAlgorithm(const SteadyVectorField2D& vecfield, const int licImageSizeX, const int licImageSizeY, double stepSize, int MaxIntegrationSteps, VORTEX_CRITERION criterionlColorBlend = VORTEX_CRITERION::NONE);
std::vector<std::vector<std::vector<Eigen::Vector3d>>> LICAlgorithm_UnsteadyField(const UnSteadyVectorField2D& vecfield, const int licImageSizeX, const int licImageSizeY, double stepSize, int MaxIntegrationSteps, VORTEX_CRITERION curlColorBlend = VORTEX_CRITERION::NONE);

bool PathhlineIntegrationRK4(const Eigen::Vector2d& StartPosition, const IUnsteadField2D& inputField, const double tstart, const double tend, const double dt_, std::vector<Eigen::Vector2d>& pathVelocitys, std::vector<Eigen::Vector3d>& pathPositions);

bool PathhlineIntegrationRK4v2(const Eigen::Vector2d& StartPosition, const IUnsteadField2D& inputField, const double tstart, const double targetIntegrationTime, const double dt_, std::vector<Eigen::Vector3d>& pathPositions);





namespace GroupSeeding {
	inline Eigen::Vector2d JittorReSeeding(const Eigen::Vector2d& preSeeding, Eigen::Vector2d domainMin, Eigen::Vector2d domainMax) {
		static std::random_device rd;
		static  std::mt19937 rng(rd());
		const auto domainRange = domainMax - domainMin;
		//plane center vector:
		Eigen::Vector2d Center = domainMin + 0.5 * domainRange;
		Eigen::Vector2d  Direction = Center - preSeeding;
		std::uniform_real_distribution<> disX(0.00001, 0.5);
		double shift = disX(rng);

		Eigen::Vector2d   seeding = preSeeding + shift * Direction;

		return seeding;
	}

	inline std::vector<Eigen::Vector2d> generateSeedingsRec(double xmin, double xmax, double ymin, double ymax, int K) {
		static std::random_device rd;
		static  std::mt19937 rng(rd());
		assert(xmax - xmin > 0);
		assert(ymax - ymin > 0);
		std::vector<Eigen::Vector2d> seedings;
		seedings.reserve(K);

		// Define distributions for x and y coordinates within the rectangle
		std::uniform_real_distribution<> disX(xmin, xmax);
		std::uniform_real_distribution<> disY(ymin, ymax);

		for (int i = 0; i < K; ++i) {
			// Generate random (x, y) within the defined rectangle
			double x = disX(rng);
			double y = disY(rng);
			seedings.emplace_back(x, y);
		}

		return seedings;
	}
	inline  std::vector<Eigen::Vector2d>  RecTangular4ClusterSampling(int samplesPercluster, Eigen::Vector2d domainMin, Eigen::Vector2d domainMax) {
		const auto domainRange = domainMax - domainMin;
		constexpr int NClusters = 4;
		std::array<Eigen::Vector2d, 4> domainCenters{
			Eigen::Vector2d(0.25 * domainRange.x() + domainMin.x(), 0.25 * domainRange.y() + domainMin.y()), //-1,-1
			Eigen::Vector2d(0.25 * domainRange.x() + domainMin.x(), 0.75 * domainRange.y() + domainMin.y()), // 1,1
			Eigen::Vector2d(0.75 * domainRange.x() + domainMin.x(), 0.25 * domainRange.y() + domainMin.y()), // 1,-1
			Eigen::Vector2d(0.75 * domainRange.x() + domainMin.x(), 0.75 * domainRange.y() + domainMin.y()) //-1,1
		};
		std::vector<Eigen::Vector2d> res;
		res.reserve(NClusters * samplesPercluster);
		for (int i = 0; i < NClusters; i++) {
			// the domain d is 1/5*range,(1/10 in one direction).
			Eigen::Vector2d domainCenter = domainCenters[i];

			auto Domain_minx = std::max(domainCenter.x() - 0.25 * domainRange.x(), domainMin.x());
			auto Domain_maxx = std::min(domainCenter.x() + 0.25 * domainRange.x(), domainMax.x());
			auto Domain_miny = std::max(domainCenter.y() - 0.25 * domainRange.y(), domainMin.y());
			auto Domain_maxy = std::min(domainCenter.y() + 0.25 * domainRange.y(), domainMax.y());
			std::vector<Eigen::Vector2d>  seedings = generateSeedingsRec(Domain_minx, Domain_maxx, Domain_miny, Domain_maxy, samplesPercluster);
			res.insert(res.end(), seedings.begin(), seedings.end());
		}
		return res;
	}
	inline  std::vector<Eigen::Vector2d> generateSeedingsCross(double xCenter, double yCenter, double dx, double dy) {
		std::vector<Eigen::Vector2d> seedings;
		seedings.emplace_back(xCenter - dx, yCenter);
		seedings.emplace_back(xCenter, yCenter - dy);
		seedings.emplace_back(xCenter + dx, yCenter);
		seedings.emplace_back(xCenter, yCenter + dy);
		return seedings;
	}

	inline  std::vector<Eigen::Vector2d>  GridCrossSampling(int gx, int gy, Eigen::Vector2d domainMin, Eigen::Vector2d domainMax) {
		Eigen::Vector2d domainRange = domainMax - domainMin;
		//need to get rid of points that are too close to boundary
		Eigen::Vector2d SamplingDomainStart = domainMin + 0.1 * domainRange;
		Eigen::Vector2d SamplingDomainEnd = domainMin + 0.9 * domainRange;
		Eigen::Vector2d SamplingDomainRange = SamplingDomainEnd - SamplingDomainStart;

		const double sampleGrid_dx = SamplingDomainRange.x() / (gx - 1);
		const double sampleGrid_dy = SamplingDomainRange.y() / (gy - 1);

		const double sampleCross_dx = sampleGrid_dx / 3.0;
		const double sampleCross_dy = sampleGrid_dy / 3.0;
		std::vector<Eigen::Vector2d> res;
		res.reserve(gx * gy * 4);
		for (size_t i = 0; i < gy; i++)
			for (size_t j = 0; j < gx; j++)
			{
				const double centerPointX = SamplingDomainStart.x() + sampleGrid_dx * j;
				const double centerPointY = SamplingDomainStart.y() + sampleGrid_dy * i;
				std::vector<Eigen::Vector2d>  seedings = generateSeedingsCross(centerPointX, centerPointY, sampleCross_dx, sampleCross_dy);
				res.insert(res.end(), seedings.begin(), seedings.end());
			};

		return res;
	}



}





std::vector<std::vector<PathlinePointInfo>> PathlineIntegrationInfoCollect2D(const UnSteadyVectorField2D& inputField, int KLines, const int outputPathlineLength, PATHLINE_SEEDING_SAMPLING sampleMEthod);

std::vector<std::vector<PathlinePointInfo>> PathlineIntegrationInfoCollect2D(const UnSteadyVectorField2D& inputField, int KLines, const Eigen::Matrix2d& deformMat, const Eigen::Vector3d& meta_rc_n_si, const Eigen::Vector2d& txy, const int outputPathlineLength, PATHLINE_SEEDING_SAMPLING sampleMEthod);
#endif