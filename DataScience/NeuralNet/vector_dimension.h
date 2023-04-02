#pragma once
#include <utility>
#include <vector>

template <typename T>
std::pair<std::size_t, std::vector<std::size_t>> vector_dimension(const std::vector<T>& v) {
    std::vector<std::size_t> sizes;
    std::size_t dimensions = 1;
    for (const auto& i : v) {
        if constexpr(std::is_same<T, std::vector<typename T::value_type>>::value) {
            auto res = vector_dimension(i);
            dimensions = res.first + 1;
            sizes.insert(sizes.end(), res.second.begin(), res.second.end());
        }
    }
    sizes.insert(sizes.begin(), v.size());
    return std::make_pair(dimensions, sizes);
}
template <>
std::pair<std::size_t, std::vector<std::size_t>> vector_dimension(const std::vector<unsigned char>& v) {
    return std::make_pair(1, std::vector<std::size_t>{v.size()});
}

