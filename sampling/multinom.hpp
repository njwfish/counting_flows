// multinom.hpp
#pragma once
#include <random>

// Inline “pure C++” multinomial sampler with no Python/GIL involvement.
// `noexcept` guarantees to Cython that it never throws.
inline void sample_multinomial_c(unsigned int* counts,
                                 int K,
                                 unsigned int n,
                                 const double* probs) noexcept
{
    // thread_local so each OS thread has its own RNG state
    static thread_local std::mt19937 gen(12345);

    unsigned int remainder = n;
    for (int i = 0; i < K - 1; ++i) {
        // construct a binomial_dist for this step
        std::binomial_distribution<unsigned int> dist(remainder, probs[i]);
        unsigned int draw = dist(gen);
        counts[i] = draw;
        remainder -= draw;
    }
    counts[K - 1] = remainder;
}
