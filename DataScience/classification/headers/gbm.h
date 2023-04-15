// author: Christian Suer
// github.com/cyberchriz/git/DataScience

// Gradient Boosting Machines (GBM) algorithm

#pragma once
#include <vector>
#include <cmath>
#include <iostream>

template <typename T>
class GBM {
public:
    GBM(int numTrees, int maxDepth, T learningRate);
    void train(const std::vector<std::vector<T>>& X, const std::vector<T>& y);
    std::vector<T> predict(const std::vector<std::vector<T>>& X) const;
    
private:
    struct TreeNode {
        T splitValue;
        int featureIndex;
        TreeNode* leftChild;
        TreeNode* rightChild;
    };
    
    std::vector<TreeNode*> trees_;
    int numTrees_;
    int maxDepth_;
    T learningRate_;
    
    T calculateError(const std::vector<T>& predicted, const std::vector<T>& actual) const;
    TreeNode* buildTree(const std::vector<std::vector<T>>& X, const std::vector<T>& y, const std::vector<int>& sampleIndices, int depth);
    T calculateSplitValue(const std::vector<std::vector<T>>& X, const std::vector<T>& y, const std::vector<int>& sampleIndices, int featureIndex) const;
    void traverseTree(const TreeNode* node, const std::vector<T>& sample, T& prediction) const;
    std::vector<T> predictSubset(const std::vector<std::vector<T>>& X, const std::vector<int>& sampleIndices, int featureIndex, T splitValue, bool leftBranch) const;
};

#include "../sources/gbm.cpp"
