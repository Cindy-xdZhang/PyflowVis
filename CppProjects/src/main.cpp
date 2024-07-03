
#include "VastistasVelocityGenerator.h"
#include "flowGenerator.h"

int main()
{

    // testKillingTransformationForRFC();
    generateUnsteadyField(15, 10, 5, "train");
    generateUnsteadyField(4, 2, 5, "validation"); // 800
    generateUnsteadyField(4, 2, 5, "test"); // 800
    // testCriterion();
    return 0;
}