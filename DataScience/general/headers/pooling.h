#pragma once
#include "array.h"

// forward declaration
template<typename T> class Array;
template<typename T> class Matrix;
template<typename T> class Vector;

// Array pooling class
template<typename T>
class Pooling {
    protected:
        // methods that are visible for class Array and Matrix
        void max(const std::vector<int> slider_shape, const std::vector<int> stride_shape, Array<T>& target);
        void max(const std::initializer_list<int> slider_shape, const std::initializer_list<int> stride_shape, Array<T>& target);
        void average(const std::vector<int> slider_shape, const std::vector<int> stride_shape, Array<T>& target);
        void average(const std::initializer_list<int> slider_shape, const std::initializer_list<int> stride_shape, Array<T>& target);        
        friend class Array<T>;
        friend class Matrix<T>;
    protected:
        // methods that are visible for class Matrix
        void max(const int slider_width, const int slider_height, const int stride_rows, const int stride_cols, Matrix<T>& target);
        void average(const int slider_width, const int slider_height, const int stride_rows, const int stride_cols, Matrix<T>& target);
        friend class Matrix<T>;
    protected:
        // methods that are visible for class Vector
        void max(const int slider_elements, const int stride, Vector<T>& target);
        void average(const int slider_elements, const int stride, Vector<T>& target);
        friend class Vector<T>;
    public:
        // constructor
        Pooling() : arr(nullptr){}
        Pooling(Array<T>* arr) : arr(arr){}
    private:           
        Array<T>* arr;
};