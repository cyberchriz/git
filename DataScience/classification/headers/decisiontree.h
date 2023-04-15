// author: Christian Suer (github.com/cyberchriz/git)

// The code implements a decision tree classifier using the ID3 algorithm for splitting nodes.

// Preprocessor directives
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

// Main class
template <typename T>
class DecisionTree {
public:
    // Constructor
    DecisionTree();

    // Destructor
    ~DecisionTree();

    // Fit the decision tree to the training data X with the corresponding labels y
    void fit(const std::vector<std::vector<T>>& X, const std::vector<T>& y);

    // Predict the class labels for the test data
    std::vector<T> predict(const std::vector<std::vector<T>>& X) const;

    std::vector<T> DecisionTree<T>::get_unique_values(const std::vector<T>& v) const;

private:
    struct Node {
        bool is_leaf;
        T label;
        int feature_index;
        T split_value;
        Node* left_child;
        Node* right_child;
    };

    Node* root;

    // Recursive function to build the decision tree
    // (using the training data X and the corresponding labels y)
    Node* build_tree(const std::vector<std::vector<T>>& X, const std::vector<T>& y);

    // Recursive function to traverse the decision tree and predict the labels for the test data X
    void traverse_tree(const Node* node, const std::vector<std::vector<T>>& X, std::vector<T>& predictions) const;

    // Calculate the entropy of the target variable y
    // entropy = - sum(p * log2(p))
    // where p is the proportion of samples in each class
    double calculate_entropy(const std::vector<T>& y) const;

    // Find the feature that maximizes the information gain and the corresponding split value
    void find_best_split(const std::vector<std::vector<T>>& X, const std::vector<T>& y, int& best_feature, T& best_split_value) const;

    // Split the dataset based on the best feature and split value
    std::pair<std::vector<std::vector<T>>, std::vector<T>> split_dataset(const std::vector<std::vector<T>>& X, const std::vector<T>& y, int feature_index, T split_value) const;
};


// include corresponding .cpp file (this is required due to the main class being a template class
// -> implementations can't be separate)
#include "../sources/decisiontree.cpp"