
#include "VastistasVelocityGenerator.h"
#include "flowGenerator.h"

int main()
{

    // testKillingTransformationForRFC();
    // generateUnsteadyField(15, 20, 100, "../data/DebugRobust/", "train"); // 30000
    // generateUnsteadyField(4, 20, 10, "../data/DebugRobust/", "validation"); // 800
    // generateUnsteadyField(4, 20, 10, "../data/DebugRobust/", "test");
    //     testCriterion();

    GenerateSteadyVortexBoundary(30, 60, "../data/Steady/", "train");
    GenerateSteadyVortexBoundary(10, 20, "../data/Steady/", "validation");
    GenerateSteadyVortexBoundary(10, 20, "../data/Steady/", "test");
    return 0;
}