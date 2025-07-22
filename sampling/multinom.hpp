#pragma once
#include <random>
#include <chrono>

inline void sample_multinomial_c(unsigned int* counts,
                                 int K,
                                 unsigned int n,
                                 const double* probs) noexcept
{
   // Use time-based seed for better randomness
    static thread_local std::mt19937 gen = []() {
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        return std::mt19937(static_cast<unsigned int>(seed));
    }();

    unsigned int remainder = n;
    double prob_sum_remaining = 0.0;
    
    // Compute total probability sum
    for (int i = 0; i < K; ++i) {
        prob_sum_remaining += probs[i];
    }
    
    // Handle degenerate case
    if (prob_sum_remaining <= 0.0) {
        for (int i = 0; i < K; ++i) {
            counts[i] = 0;
        }
        return;
    }
    
    // Sequential binomial sampling with CONDITIONAL probabilities
    for (int i = 0; i < K - 1; ++i) {
        if (remainder == 0) {
            counts[i] = 0;
            continue;
        }
        
        // KEY FIX: Use conditional probability
        double conditional_prob = probs[i] / prob_sum_remaining;
        
        // Safety clamps
        if (conditional_prob < 0.0) conditional_prob = 0.0;
        if (conditional_prob > 1.0) conditional_prob = 1.0;
        
        std::binomial_distribution<unsigned int> dist(remainder, conditional_prob);
        unsigned int draw = dist(gen);
        counts[i] = draw;
        
        // Update for next iteration
        remainder -= draw;
        prob_sum_remaining -= probs[i];
    }
    counts[K - 1] = remainder;
}