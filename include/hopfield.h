#pragma once
#include <vector>
#include <iostream>

class HopfieldNetwork {
private:
    std::vector<std::vector<int>> weights;
    size_t size;

public:
    explicit HopfieldNetwork(size_t input_size) : size(input_size) {
        weights.resize(size, std::vector<int>(size, 0));
    }

    void train(const std::vector<std::vector<int>>& patterns) {
        for (const auto& p : patterns) {
            for (size_t i = 0; i < size; ++i) {
                for (size_t j = 0; j < size; ++j) {
                    if (i != j) {
                        weights[i][j] += p[i] * p[j];
                    }
                }
            }
        }
    }

    std::vector<int> recall(const std::vector<int>& input, int iterations = 5) {
        std::vector<int> state = input;

        for (int it = 0; it < iterations; ++it) {
            for (size_t i = 0; i < size; ++i) {
                int sum = 0;
                for (size_t j = 0; j < size; ++j) {
                    sum += weights[i][j] * state[j];
                }
                state[i] = sum >= 0 ? 1 : -1;
            }
        }

        return state;
    }
};
