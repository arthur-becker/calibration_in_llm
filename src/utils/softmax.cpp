#include <algorithm>
#include <assert.h>
#include <cmath>

#include "utils/position_result/position_full_result.h"
#include "utils/position_result/position_result.h"
#include "utils/softmax.h"

PositionFullResult softmax(const PositionFullResult & logits) {
    std::vector<float> logits_vector = logits.getTokenData();
    std::vector<float> proba_vector(logits_vector.size());

    assert(!logits_vector.empty());

    float max_logit = *std::max_element(logits_vector.begin(), logits_vector.end());

    double sum_exp = 0.0;
    for (size_t i = 0; i < logits_vector.size(); i++) {
        // Subtract for numerical stability
        const float logit = logits_vector[i] - max_logit;

        const float exp_logit = expf(logit);
        sum_exp += exp_logit;
        proba_vector[i] = exp_logit;
    }

    for(size_t i = 0; i < proba_vector.size(); i++){
        proba_vector[i] /= sum_exp;
    }

    PositionFullResult proba(proba_vector, logits.getCorrectToken());
    return proba;
};