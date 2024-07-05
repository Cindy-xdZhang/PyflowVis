
#include "VastistasVelocityGenerator.h"
#include "flowGenerator.h"

int main()
{

    // testKillingTransformationForRFC();
    generateUnsteadyField(20, 10, 10, "train");
    generateUnsteadyField(4, 2, 5, "validation"); // 800
    generateUnsteadyField(4, 2, 5, "test"); // 800
    // testCriterion();
    return 0;
}