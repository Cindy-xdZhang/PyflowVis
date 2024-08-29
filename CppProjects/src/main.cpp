
#include "VastistasVelocityGenerator.h"
#include "flowGenerator.h"

int main()
{

    // generateUnsteadyFieldPathline(5, 1, 1, "../data/Pathline/", "train");
    //  generateUnsteadyField(15, 20, 100, "../data/Robust/", "train"); // 30000
    //  generateUnsteadyField(10, 10, 10, "../data/Robust/", "validation"); // 1000

    generateUnsteadyFieldMixture(100, 30, "../data/RobustMix/", "train"); // 3000
    generateUnsteadyFieldMixture(10, 20, "../data/RobustMix/", "validation"); // 200
    generateUnsteadyFieldMixture(10, 20, "../data/RobustMix/", "test"); // 200

    // GenerateSteadyVortexBoundary(30, 60, "../data/Steady/", "train");
    // GenerateSteadyVortexBoundary(10, 20, "../data/Steady/", "validation");
    // GenerateSteadyVortexBoundary(5, 60, "../data/Steady/", "test");
    return 0;
}