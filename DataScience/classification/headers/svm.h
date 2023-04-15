#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <cstdlib>

template<typename T>
class SVM {
public:
    SVM(int num_features, T c, T tol, int max_iter);
    void train(const std::vector<std::vector<T>>& X, const std::vector<T>& y);
    T predict(const std::vector<T>& x) const;

private:
    T kernel(const std::vector<T>& x1, const std::vector<T>& x2) const;

    int num_features_;
    T c_;
    T tol_;
    int max_iter_;
    std::vector<T> alpha_;
    std::vector<std::vector<T>> X_;
    T b_;
};

#include "../sources/svm.cpp"
