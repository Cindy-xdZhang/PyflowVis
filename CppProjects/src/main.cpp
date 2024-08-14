
#include "VastistasVelocityGenerator.h"
#include "flowGenerator.h"

int main()
{

    // testKillingTransformationForRFC();
    generateUnsteadyField(2, 2, 6, "../data/Debug/", "train");
    // generateUnsteadyField(3, 2, 10, "../data/Debug/", "validation"); // 800
    // generateUnsteadyField(3, 2, 10, "../data/Debug/", "test"); // 800
    //  testCriterion();
    return 0;
}