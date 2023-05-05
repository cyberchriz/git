#pragma once
#include "datastructures.h"

// forward declaration
template<typename T> class Array;
template<typename T> class Matrix;
template<typename T> class Vector;

// Array pooling class
template<typename T>
class Pooling {
    public:
        Array<T> max(const std::initializer_list<int> slider_shape, const std::initializer_list<int> stride_shape){
            // confirm valid slider shape
            if (slider_shape.size() != source.dimensions){
                throw std::invalid_argument("slider shape for avg pooling must have same number of dimensions as the layer it is acting upon");
            }
            // confirm valid stride shape
            if (stride_shape.size() != source.dimensions){
                throw std::invalid_argument("stride shape for avg pooling must have same number of dimensions as the layer it is acting upon");
            }    
            // get target shape
            std::vector<int> shape;
            auto iterator = source.shape.begin();
            int d=0;
            for (;d<source.dimensions;d++, iterator++){
                shape.emplace_back((*iterator - slider_shape[d])/stride_shape[d]);
            }
            // create target
            std::unique_ptr<Array<T>> target = std::make_unique<Array<T>>(shape);
            // create a sliding box for pooling
            Array<double> slider = Array<double>(slider_shape);
            // iterate over target
            for (int j=0;j<target.get_elements();j++){
                // get associated index
                std::vector<int> index_j = target.get_index(j);
                // get corresponding source index
                std::vector<int> index_i;
                for (int d=0;d<target.dimensions;d++){
                    index_i.push_back(index_j[d] * *(stride_shape.begin()+d));
                }
                // iterate over elements of the slider
                Vector<int> index_slider;
                for (int n=0;n<slider.get_elements();n++){
                    // get multidimensional index of the slider element
                    index_slider = Vector<int>::asVector(slider.get_index(n));
                    // assing slider value from the element with the index of the sum of index_i+index_slider
                    slider.set(n, layer[l-1].h.get((index_i+index_slider).flatten()));
                }
                target.set(j,slider.max());
            }
            return std::move(*target);
        }

        Array<T> average(const std::initializer_list<int> slider_shape, const std::initializer_list<int> stride_shape){
            // confirm valid slider shape
            if (slider_shape.size() != source.dimensions){
                throw std::invalid_argument("slider shape for avg pooling must have same number of dimensions as the layer it is acting upon");
            }
            // confirm valid stride shape
            if (stride_shape.size() != source.dimensions){
                throw std::invalid_argument("stride shape for avg pooling must have same number of dimensions as the layer it is acting upon");
            }    
            // get target shape
            std::vector<int> shape;
            auto iterator = source.shape.begin();
            int d=0;
            for (;d<source.dimensions;d++, iterator++){
                shape.emplace_back((*iterator - slider_shape[d])/stride_shape[d]);
            }
            // create target
            std::unique_ptr<Array<T>> target = std::make_unique<Array<T>>(shape);
            // create a sliding box for pooling
            Array<double> slider = Array<double>(slider_shape);
            // iterate over target
            for (int j=0;j<target.get_elements();j++){
                // get associated index
                std::vector<int> index_j = target.get_index(j);
                // get corresponding source index
                std::vector<int> index_i;
                for (int d=0;d<target.dimensions;d++){
                    index_i.push_back(index_j[d] * *(stride_shape.begin()+d));
                }
                // iterate over elements of the slider
                Vector<int> index_slider;
                for (int n=0;n<slider.get_elements();n++){
                    // get multidimensional index of the slider element
                    index_slider = Vector<int>::asVector(slider.get_index(n));
                    // assing slider value from the element with the index of the sum of index_i+index_slider
                    slider.set(n, layer[l-1].h.get((index_i+index_slider).flatten()));
                }
                target.set(j,slider.mean());
            }
            return std::move(*target);
        }
    public:
        // constructor
        Pooling() : source(nullptr){}
        Pooling(Array<T>* source) : source(source){}
    private:           
        Array<T>* source;
};