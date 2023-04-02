//+------------------------------------------------------------------+
//|      sample distribution (convert sample array to histogram)     |
//+------------------------------------------------------------------+

// author: Christian Suer

#pragma once
#include <cmath>
#include <vector>
using namespace std;

void histogram(vector<double> sample,vector<double>& histogram_x,vector<double> histogram_y, double& bar_width, bool relative=true){
    // get min and max value from sample
    double samplemax=sample[0];
    double samplemin=sample[0];
    int elements=sample.size();
    for (int i=0;i<elements;i++){
        samplemax=fmax(samplemax,sample[i]);
        samplemin=fmin(samplemin,sample[i]);
    }

    // get histogram x-axis scaling
    double range = samplemax-samplemin;
    int steps = histogram_x.size();
    bar_width = range / steps;

    // initialize with zeros
    fill(histogram_x.begin(), histogram_x.end(),0);
    fill(histogram_y.begin(), histogram_x.end(),0);
    
    // set histogram x values
    for (int i=0;i<steps;i++){
        histogram_x[i]=samplemin+i*bar_width;
    }

    // count occurences per histogram bar
    for (int i=0;i<elements;i++){
        histogram_y[int((sample[i]-samplemin)/bar_width)]++;
    }

    // optional: convert to relative values
    if (relative){
        for (int i=0;i<steps;i++){
            histogram_y[i]/=elements;
        }
    }
}