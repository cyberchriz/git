#include "../headers/k_nearest.h"

template<typename T>
void KNearest<T>::add_samples(const std::vector<std::vector<T>>& training_samples, const std::vector<int>& labels) {
    for (unsigned int i = 0; i < training_samples.size(); ++i) {
        add_samples(training_samples[i], labels[i]);
    }
}

template<typename T>
void KNearest<T>::add_samples(const std::vector<TrainingSample<T>>& training_samples) {
    for (const auto& sample : training_samples) {
        add_samples(sample.sample, sample.label);
    }
}

template<typename T>
void KNearest<T>::add_samples(const TrainingSample<T>& sample) {
    add_samples(sample.sample, sample.label);
}

template<typename T>
void KNearest<T>::add_samples(const std::vector<T>& sample, T label) {
    training_samples_.push_back(sample);
    labels_.push_back(label);
}

template<typename T>
int KNearest<T>::predict(const std::vector<T>& test_sample) const {
    // create priority queue to store distances and corresponding indices of training samples
    std::priority_queue<DistanceIndex> distance_queue;

    // calculate distance between test sample and all training samples
    for (unsigned int i = 0; i < training_samples_.size(); ++i) {
        T dist = distance(test_sample, training_samples_[i]);
        distance_queue.push({ dist, static_cast<int>(i) });
    }

    // determine k nearest neighbors
    std::vector<int> k_indices;
    for (int i = 0; i < k_; ++i) {
        k_indices.push_back(distance_queue.top().index);
        distance_queue.pop();
    }

    // determine label of k nearest neighbors
    std::vector<int> k_labels;
    for (const auto& index : k_indices) {
        k_labels.push_back(labels_[index]);
    }

    // determine most common label
    int max_count = 0;
    int most_common_label = -1;
    for (int label : k_labels) {
        int count = std::count(k_labels.begin(), k_labels.end(), label);
        if (count > max_count) {
            max_count = count;
            most_common_label = label;
        }
    }

    return most_common_label;
}

template<typename T>
T KNearest<T>::distance(const std::vector<T>& x1, const std::vector<T>& x2) const {
    T sum = 0;
    for (unsigned int i = 0; i < x1.size(); ++i) {
        sum += pow(x1[i] - x2[i], 2);
    }
    return sqrt(sum);
}

template<typename T>
int KNearest<T>::num_classes() const {
    std::vector<int> unique_labels;
    for (const auto& label : labels_) {
        if (std::find(unique_labels.begin(), unique_labels.end(), label) == unique_labels.end()) {
            unique_labels.push_back(label);
        }
    }
    return static_cast<int>(unique_labels.size());
}
