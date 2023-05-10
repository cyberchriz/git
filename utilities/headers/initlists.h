#pragma once
#include <vector>

template<typename T>
std::vector<T> initlist_to_vector(std::initializer_list<T> initlist){
    std::vector<T> result(initlist.size());
    int n=0;
    std::for_each(initlist.begin(), initlist.end(), [&result, &n](auto item) {
        result[n] = item;
        n++;
    });
    return result;
}