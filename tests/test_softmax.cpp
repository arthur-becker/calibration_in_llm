#include <assert.h>

#include "utils/softmax.h"
#include "utils/position_result/position_result.h"
#include "utils/position_result/position_full_result.h"



void test_case(const PositionFullResult &logits){
    PositionFullResult proba = softmax(logits);

    assert(logits.getCorrectToken() == proba.getCorrectToken());
    assert(logits.getTokenData().size() == proba.getTokenData().size());

    float sum = 0.0;
    for(float value : proba.getTokenData()){
        sum += value;
    }
    assert(sum >= 0.99);
    assert(sum <= 1.01);
}

int main(){
    float logits_vector1[] = {-30.0, 21.0, 111.4, 16.0};
    PositionFullResult logits1(
        std::vector<float>(logits_vector1, logits_vector1 + 4),
        1
    );
    test_case(logits1);
    printf("Test case 1 passed\n");

    float logits_vector2[] = {0.3, 0.1, 0.1, 0.05};
    PositionFullResult logits2(
        std::vector<float>(logits_vector2, logits_vector2 + 4),
        2
    );
    test_case(logits2);
    printf("Test case 2 passed\n");

    float logits_vector3[] = {13653.2, 424.775, 4342.1, 543.5, 1.111, -54.0};
    PositionFullResult logits3(
        std::vector<float>(logits_vector3, logits_vector3 + 6),
        0
    );
    test_case(logits3);
    printf("Test case 3 passed\n");

    return 0;
}