
#include "VastistasVelocityGenerator.h"
#include "flowGenerator.h"

int main()
{

    // testKillingTransformationForRFC();
    generateUnsteadyField(10, 10, 30, "../data/DebugRobust/", "train"); // 3000
    generateUnsteadyField(4, 20, 10, "../data/DebugRobust/", "validation"); // 800
    generateUnsteadyField(4, 20, 10, "../data/DebugRobust/", "test");
    //   testCriterion();
    return 0;
}