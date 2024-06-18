
#include "VastistasVelocityGenerator.h"
#include "flowGenerator.h"

int main()
{

    // testKillingTransformationForRFC();
    // generateUnsteadyField(12, 20, 40, "train");
    generateUnsteadyField(2, 20, 20, "validation"); // 800
    generateUnsteadyField(2, 20, 20, "test"); // 800
    //  testCriterion();
    return 0;
}