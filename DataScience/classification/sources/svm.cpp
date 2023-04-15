#include "../headers/svm.h"

// constructor
template<typename T>
SVM<T>::SVM(int num_features, T c, T tol, int max_iter) :
    num_features_(num_features),
    c_(c),
    tol_(tol),
    max_iter_(max_iter),
    alpha_(num_features),
    b_(0) {}

template<typename T>
void SVM<T>::train(const std::vector<std::vector<T>>& X, const std::vector<T>& y) {
    int n = X.size();
    X_ = X;

    // Create kernel matrix
    std::vector<std::vector<T>> K(n, std::vector<T>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            K[i][j] = kernel(X[i], X[j]);
        }
    }

    // Initialize alpha vector
    alpha_.assign(n, 0);

    // SMO algorithm
    int iter = 0;
    while (iter < max_iter_) {
        bool alpha_changed = false;
        for (int i = 0; i < n; i++) {
            T Ei = b_;
            for (int j = 0; j < n; j++) {
                Ei += alpha_[j] * y[j] * K[i][j];
            }
            Ei -= y[i];

            if ((y[i] * Ei < -tol_ && alpha_[i] < c_) ||
                (y[i] * Ei > tol_ && alpha_[i] > 0)) {
                int j = i;
                while (j == i) {
                    j = rand() % n;
                }

                T Ej = b_;
                for (int k = 0; k < n; k++) {
                    Ej += alpha_[k] * y[k] * K[j][k];
                }
                Ej -= y[j];

                T old_alpha_i = alpha_[i];
                T old_alpha_j = alpha_[j];
                T L, H;
                if (y[i] != y[j]) {
                    L = std::max((T)0, alpha_[j] - alpha_[i]);
                    H = std::min(c_, c_ + alpha_[j] - alpha_[i]);
                } else {
                    L = std::max((T)0, alpha_[i] + alpha_[j] - c_);
                    H = std::min(c_, alpha_[i] + alpha_[j]);
                }
                if (L == H) continue;

                T eta = 2 * K[i][j] - K[i][i] - K[j][j];
                if (eta >= 0) continue;

                alpha_[j] -= y[j]
                alpha_[j] -= y[j] * (Ei - Ej) / eta;
                alpha_[j] = std::min(std::max(alpha_[j], L), H);

                if (std::abs(alpha_[j] - old_alpha_j) < tol_) continue;

                alpha_[i] += y[i] * y[j] * (old_alpha_j - alpha_[j]);

                T b1 = b_ - Ei - y[i] * (alpha_[i] - old_alpha_i) * K[i][i] - y[j] * (alpha_[j] - old_alpha_j) * K[i][j];
                T b2 = b_ - Ej - y[i] * (alpha_[i] - old_alpha_i) * K[i][j] - y[j] * (alpha_[j] - old_alpha_j) * K[j][j];
                if (0 < alpha_[i] && alpha_[i] < c_) {
                    b_ = b1;
                } else if (0 < alpha_[j] && alpha_[j] < c_) {
                    b_ = b2;
                } else {
                    b_ = (b1 + b2) / 2;
                }

                alpha_changed = true;
            }
        }
        if (!alpha_changed) {
            break;
        }
        iter++;
    }
}

template<typename T>
T SVM<T>::predict(const std::vector<T>& x) const {
    T sum = 0;
    for (int i = 0; i < alpha_.size(); i++) {
        sum += alpha_[i] * kernel(x, X[i]);
    }
    sum -= b_;
    return sum >= 0 ? 1 : -1;
}

template<typename T>
T SVM<T>::kernel(const std::vector<T>& x1, const std::vector<T>& x2) const {
    T dot_product = 0;
    for (int i = 0; i < x1.size(); i++) {
        dot_product += x1[i] * x2[i];
    }
    return dot_product;
}
