
#include "VastistasVelocityGenerator.h"
#include "flowGenerator.h"




int main()
{

	auto generator = UnsteadyPathlneDataSetGenerator();
	generator.GenDataset(15, 20, 100, "../data/dbgPathline/");

	// generateUnsteadyFieldPathline(5, 1, 1, "../data/Pathline/", "train");
	//  generateUnsteadyField(15, 20, 100, "../data/Robust/", "train"); // 30000
	//  generateUnsteadyField(10, 10, 10, "../data/Robust/", "validation"); // 1000

	//generateUnsteadyFieldPathline(2, 10, 20, "../data/PathlineZeroField/", "train");
	//generateUnsteadyFieldPathline(1, 10, 5, "../data/PathlineZeroField/", "validation"); // 200
	// generateUnsteadyFieldPathline(3, 20, 20, "../data/Pathline2/", "test"); // 200

	// GenerateSteadyVortexBoundary(30, 60, "../data/Steady/", "train");
	// GenerateSteadyVortexBoundary(10, 20, "../data/Steady/", "validation");
	// GenerateSteadyVortexBoundary(5, 60, "../data/Steady/", "test");
	return 0;
}
