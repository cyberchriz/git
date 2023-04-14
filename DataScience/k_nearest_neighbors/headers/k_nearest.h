// k nearest neighbors (classification algorithm)
// author: 'cyberchriz' (Christian Suer)

#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>

template<typename T>
struct TrainingSample {
    std::vector<T> sample;
    int label;
};

template<typename T>
class KNearest {
public:
    // parametric constructor
    KNearest(int k = 3) : k_(k) {};

    // add training sample(s) with corresponding label(s)
    void add_samples(const std::vector<std::vector<T>>& training_samples, const std::vector<int>& labels);
    void add_samples(const std::vector<TrainingSample<T>>& training_samples);
    void add_samples(const TrainingSample<T>& sample);
    void add_samples(const std::vector<T>& sample, T label);

    // predict new label classification from a test sample
    int predict(const std::vector<T>& test_sample) const;

private:
    // internal copy of all training data added via the add_samples methods
    std::vector<std::vector<T>> training_samples_;
    
    std::vector<int> labels_;
    int k_;

    // method to calculate distance between two vectors
    T distance(const std::vector<T>& x1, const std::vector<T>& x2) const;

    // method to return the number of unique labels in the training set
    int num_classes() const;

    // struct to store distances and corresponding indices of training samples
    struct DistanceIndex {
        T distance;
        int index;
        bool operator<(const DistanceIndex& other) const {
            return distance < other.distance;
        }
    };
};

// reference to .cpp file required for the compiler because of template class
#include "../sources/k_nearest.cpp"