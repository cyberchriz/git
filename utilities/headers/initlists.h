#pragma once
#include <vector>

// helper method to convert a std::initializer_list<T> to std::vector<T>
template<typename T>
std::vector<T> initlist_to_vector(const std::initializer_list<T>& list){
    std::vector<int> vector(list.size());
    auto iterator = list.begin();
    for (int n=0;iterator!=list.end();n++, iterator++){
        vector[n] = *iterator;
    }
    return vector;
}

// helper method to convert a std::vector<T> to std::initializer_list<T>
template<typename T>
std::initializer_list<T> vector_to_initlist(const std::vector<T>& vec) {
    std::initializer_list<T> init_list;
    for (auto& elem : vec) {
        init_list = {std::initializer_list<int>{elem}};
    }
}