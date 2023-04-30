#pragma once
#include "datastructures.h"

// forward declaration
template<typename T> class Vector;

// result struct for histograms
template<typename T>
struct HistogramResult{
    private:
        struct Histogrambar{
            T lower_boundary;
            T upper_boundary;
            int abs_count=0;
            double rel_count;
        };
    public:    
        T min, max;
        T bar_width;
        T _width;
        uint bars;
        std::unique_ptr<Histogrambar[]> bar;
        HistogramResult() : bars(0) {};
        HistogramResult(uint bars) : bars(bars) {
            bar = std::make_unique<Histogrambar[]>(bars);
        }
        ~HistogramResult(){};
};

template<typename T>
class Histogram{
    private:
        Vector<T>* source;
    public:
        HistogramResult<T> get(uint bars) const;
        // constructor
        Histogram() : source(nullptr){}
        Histogram(Vector<T>* source) : source(source){}
};

// returns a histogram of the source vector data
// with the specified number of bars and returns the 
// result as type struct Histogram<T>'
template <typename T>
HistogramResult<T> Histogram<T>::get(uint bars) const {
    std::unique_ptr<typename Array<T>::Histogram> histogram = std::make_unique<typename Array<T>::Histogram>(bars);
    // get min and max value from sample
    histogram->min = source->data[0];
    histogram->max = source->data[0];
    for (int i=0;i<source->_elements;i++){
        histogram->min=std::fmin(histogram->min, source->data[i]);
        histogram->max=std::fmax(histogram->max, source->data[i]);
    }

    // get histogram x-axis scaling
    histogram->_width = histogram->max - histogram->min;
    histogram->bar_width = histogram->_width / bars;
    
    // set histogram x values, initialize count to zero
    for (int i=0;i<bars;i++){
        histogram->bar[i].lower_boundary = histogram->min + histogram->bar_width * i;
        histogram->bar[i].upper_boundary = histogram->min + histogram->bar_width * (i+1);
        histogram->bar[i].abs_count=0;
    }

    // count absolute occurences per histogram bar
    for (int i=0;i<source->_elements;i++){
        histogram->bar[int((source->data[i]-histogram->min)/histogram->bar_width)].abs_count++;
    }

    // convert to relative values
    for (int i=0;i<bars;i++){
        histogram->bar[i].rel_count=histogram->bar[i].abs_count/source->_elements;
    }
    return std::move(*histogram);
}
