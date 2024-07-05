
#include "VastistasVelocityGenerator.h"
#include "flowGenerator.h"

int main()
{

    // testKillingTransformationForRFC();
    generateUnsteadyField(5, 5, 5, "../data/Debug/", "train");
    generateUnsteadyField(3, 2, 5, "../data/Debug/", "validation"); // 800
    generateUnsteadyField(3, 2, 5, "../data/Debug/", "test"); // 800
    // testCriterion();
    return 0;
}