#pragma once
#include "array.h"

// forward declaration
template<typename T> class Vector;

// result struct for histograms
template<typename T>
struct HistogramResult{
    private:
        struct Histogram_bar{
            T _lower_boundary;
            T _upper_boundary;
            int _abs_count=0;
            double _rel_count;
        };
    public:    
        T _min, _max;
        T _bar_width;
        T _width;
        uint _bars;
        std::unique_ptr<Histogram_bar[]> _bar;
        HistogramResult() : _bars(0) {};
        HistogramResult(uint bars) : _bars(bars) {
            _bar = std::make_unique<Histogram_bar[]>(bars);
        }
        ~HistogramResult(){};
};

template<typename T>
class Histogram{
    private:
        Vector<T>* _source;
    public:
        HistogramResult<T> get(uint bars) const;
        // constructor
        Histogram() : _source(nullptr){}
        Histogram(Vector<T>* source) : _source(source){}
};

// returns a histogram of the source vector data
// with the specified number of bars and returns the 
// result as type struct Histogram<T>'
template <typename T>
HistogramResult<T> Histogram<T>::get(uint bars) const {
    std::unique_ptr<typename Array<T>::Histogram> histogram = std::make_unique<typename Array<T>::Histogram>(bars);
    // get min and max value from sample
    histogram->_min = _source->_data[0];
    histogram->_max = _source->_data[0];
    for (int i=0;i<_source->_elements;i++){
        histogram->_min=std::fmin(histogram->_min, _source->_data[i]);
        histogram->_max=std::fmax(histogram->_max, _source->_data[i]);
    }

    // get histogram x-axis scaling
    histogram->_width = histogram->_max - histogram->_min;
    histogram->_bar_width = histogram->_width / bars;
    
    // set histogram x values, initialize count to zero
    for (int i=0;i<bars;i++){
        histogram->_bar[i]._lower_boundary = histogram->_min + histogram->_bar_width * i;
        histogram->_bar[i]._upper_boundary = histogram->_min + histogram->_bar_width * (i+1);
        histogram->_bar[i]._count=0;
    }

    // count absolute occurences per histogram bar
    for (int i=0;i<_source->_elements;i++){
        histogram->_bar[int((_source->_data[i]-histogram->_min)/histogram->_bar_width)]._abs_count++;
    }

    // convert to relative values
    for (int i=0;i<bars;i++){
        histogram->bar[i]._rel_count=histogram->_bar[i]._abs_count/_source->_elements;
    }
    return std::move(*histogram);
}
