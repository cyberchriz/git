// author: Christian Suer (github.con/cyberchriz/git/DataScience)

#pragma once
#include <vector>
#include <unordered_map>
#include <cmath>

template<typename T>
class NaiveBayes{
public:
    NaiveBayes();
    ~NaiveBayes();
    
    // Add a training example
    void addExample(const std::vector<T>& features, const std::string& label);
    
    // Train the model
    void train();
    
    // Classify a new example
    std::string classify(const std::vector<T>& features);
    
private:
    // The counts for each label
    std::unordered_map<std::string, int> labelCounts_;
    
    // The counts for each feature and label
    std::unordered_map<std::string, std::unordered_map<T, int>> featureCounts_;
    
    // The total counts for each feature
    std::unordered_map<std::string, int> featureTotalCounts_;
    
    // The total number of training examples
    int numExamples_;
};

#include "../sources/naive_bayes.cpp"
