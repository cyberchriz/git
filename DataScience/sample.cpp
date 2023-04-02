#pragma once
#include "sample.h"
#include <numeric>
#include <vector>
#include <cmath>
#include "distributions/headers/random_distributions.h"

// arithmetic mean
template<typename T>
double Sample<T>::mean(){
    double sum=0;
    for (int n=0;n<elements;n++){
        sum+=data_vect[n];
    }
    return sum/elements;
}

// median
template<typename T>
double Sample<T>::median(){
    // make a copy
    std::vector<double> sorted_copy = this->sort();
    if (elements%2>__DBL_EPSILON__){ //=odd number
        return sorted_copy[(int)floor(double(elements)/2)];
    }
    else{
        return (sorted_copy[elements/2]+sorted_copy[elements/2+1])/2;
    }
}

// weighted average of a 1d double std::vector
template<typename T>
double Sample<T>::weighted_average(bool as_series=false){
    double weight, weight_sum, sum=0;
    if (!as_series) //=indexing from zero, lower index means lower attributed weight
    {
        for (int n=0;n<elements;n++)
        {
        weight++;
        weight_sum+=weight;
        sum+=weight*data_vect[n];
        }
        return sum/(elements*weight_sum+__DBL_MIN__);
    }
    else
    {
        for (int n=elements-1;n>=0;n--)
        {
        weight++;
        weight_sum+=weight;
        sum+=weight*data_vect[n];
        }
        return sum/(elements*weight_sum+__DBL_MIN__);
    }   
}     

// ranking
template<typename T>
std::vector<int> Sample<T>::ranking(bool low_to_high) {
    // initialize ranks
    std::vector<int> rank(elements);
    for (int n=0;n<elements;n++){
        rank[n]=n;
    }
    bool ranking_completed=false;
    while (!ranking_completed){
        ranking_completed=true; //=let's assume this until a wrong order is found
        for (int i=0;i<elements-1;i++){
            // pairwise comparison:
            if (low_to_high){
                if (data_vect[rank[i]]>data_vect[rank[i+1]]){
                    ranking_completed=false;
                    int higher_ranking=rank[i+1];
                    rank[i+1]=rank[i];
                    rank[i]=higher_ranking;
                }
            }
            else{
                if (data_vect[rank[i]]<data_vect[rank[i+1]]){
                    ranking_completed=false;
                    int lower_ranking=rank[i+1];
                    rank[i+1]=rank[i];
                    rank[i]=lower_ranking;
                }
            }            
        }
    }  
    return rank;
}

// exponential smoothing (for time series)
template<typename T>
std::vector<T> Sample<T>::exponential_smoothing(bool as_series=false){
    double alpha=2/(elements);
    std::vector<T> result(elements);
    
    if (as_series){
        result[elements-1]=this->mean();
        for (int n=elements-2;n>=0;n--){
            result[n] = alpha*(data_vect[n]-result[n+1]) + result[n+1];
        }     
    }
    else{
        result[0]=this->mean();
        for (int n=1;n<elements;n++){
            result[n] = alpha*(data_vect[n]-result[n-1]) + result[n-1];
        }
    }
    return result;
}

// sample variance
template<typename T>
double Sample<T>::variance(){
    double sample_mean = this->mean();
    double mdev2sum=0;
    for (int n=0;n<elements;n++){
        mdev2sum+=pow(data_vect[n]-sample_mean,2);
    }
    return mdev2sum/elements;
}

// standard deviation from any sample
template<typename T>
double Sample<T>::stddev(std::vector<T>& sample){
    int size=sample.size();
    double sample_mean = 0;
    for (int i=0;i<size;i++){
        sample_mean+=sample[i];
    }
    sample_mean/=size;
    double mdev2sum=0;
    for (int n=0;n<size;n++){
        mdev2sum+=pow(sample[n]-sample_mean,2);
    }
    return sqrt(mdev2sum/size);            
}

// standard deviation from sample as received from parametric object constructor
template<typename T>
double Sample<T>::stddev(){
    return sqrt(this->variance());
}        

// count the number of appearances of a given value within a numbers array
template<typename T>
unsigned int Sample<T>::find(T value,int index_from,int index_to){
    int findings=0;
    int index_end=fmin(index_to,data_vect.size()-1);;
    for (int n=index_from;n<=index_end;n++){
        if (data_vect[n]==value){findings++;}
    }
    return findings;
}                

// augmented Dickey-Fuller unit root test for stationarity
template<typename T>
double Sample<T>::Dickey_Fuller(){
    // make a copy
    std::vector<double> data_copy = data_vect;
    std::vector<double> diff = Sample(data_copy).stationary(integer,1);
    data_copy.erase(data_copy.begin());
    Sample* corr = new Sample(data_copy,diff);
    corr->correlation();
    return corr->Pearson_R*sqrt((double)(elements-1)/(1-pow(corr->Pearson_R,2)));  
    }

// correlation
template<typename T>
void Sample<T>::correlation(){
    // get empirical array autocorrelation (Pearson coefficient R), assumimg linear dependence
    x_mean=Sample(x_vect).mean();
    y_mean=Sample(y_vect).mean();
    covariance=0;
    for (int n=0;n<elements;n++){covariance+=(x_vect[n]-x_mean)*(y_vect[n]-y_mean);}
    double stdev_x=stddev(x_vect);
    double stdev_y=stddev(y_vect);
    Pearson_R = covariance/(stdev_x*stdev_y);   

        // get r_squared (coefficient of determination) assuming linear dependence
    double x_mdev2_sum=0,y_mdev2_sum=0,slope_numerator=0;
    for (int n=0;n<elements;n++)
        {
        x_mdev2_sum+=pow(x_vect[n]-x_mean,2); //=slope denominator
        y_mdev2_sum+=pow(y_vect[n]-y_mean,2); //=SST
        slope_numerator+=(x_vect[n]-x_mean)*(y_vect[n]-y_mean);
        }
    slope=slope_numerator/x_mdev2_sum;
    y_intercept=y_mean-slope*x_mean;
    // get regression line values
    double y_pred[elements];
    double SSE=0,SSR=0;
    for (int n=0;n<elements;n++)
        {
        y_pred[n]=y_intercept+slope*x_vect[n];
        // get sum of squared (y-^y) //=SSE   
        SSE+=pow(y_pred[n]-y_mean,2);
        //SSR+=pow(y_vect[n]-y_pred[n],2);
        };
    r_squared = SSE/(fmax(y_mdev2_sum,__DBL_MIN__)); //=SSE/SST, equal to 1-SSR/SST

    // Spearman correlation, assuming non-linear monotonic dependence
    std::vector<int> rank_x = Sample(x_vect).ranking();
    std::vector<int> rank_y = Sample(y_vect).ranking();
    double numerator=0;
    for (int n=0;n<elements;n++){
        numerator+=6*pow(rank_x[n]-rank_y[n],2);
    }
    Spearman_Rho=1-numerator/(elements*(pow(elements,2)-1));
    // test significance against null hypothesis
    double fisher_transform=0.5*log((1+Spearman_Rho)/(1-Spearman_Rho));
    z_score=sqrt((elements-3)/1.06)*fisher_transform;
    t_score=Spearman_Rho*sqrt((elements-2)/(1-pow(Spearman_Rho,2)));      
    correlation_completed=true;       
};            

// time series stationary transformation
template<typename T>
std::vector<T> Sample<T>::stationary(DIFFERENCING method,double degree,double fract_exponent){
    // make a copy
    std::vector<double> target = data_vect;
    if (method==integer){
        for (int d=1;d<=(int)degree;d++){ //=loop allows for higher order differencing
            for (int t=target.size()-1;t>0;t--){
                if (target[t-1]!=0){target[t]-=target[t-1];}
            }
            target.erase(target.begin()); //remove first element
        }
    }
    if (method==logreturn){
        for (int d=1;d<=round(degree);d++){ //=loop allows for higher order differencing
            for (int t=target.size()-1;t>0;t--){
                if (target[t-1]!=0){
                    target[t]=log(__DBL_MIN__+fabs(target[t]/(target[t-1]+__DBL_MIN__)));
                }
            }
            target.erase(target.begin()); //remove first element
        }     
    }
    if (method==fractional){
        for (int t=target.size()-1;t>0;t--){
            if (target[t-1]!=0){
                double stat=log(__DBL_MIN__+fabs(data_vect[t]/data_vect[t-1])); //note: DBL_MIN and fabs are used to avoid log(x<=0)
                double non_stat=log(fabs(data_vect[t])+__DBL_MIN__);
                target[t]=degree*stat+pow((1-degree),fract_exponent)*non_stat;
            }
        }
        target.erase(target.begin()); //remove first element      
    }
    if (method==deltamean){
        int elements=data_vect.size();
        double sum=0;
        for (int i=0;i<elements;i++){
            sum+=data_vect[i];
        }
        double x_mean=sum/elements;
        for (int t=elements-1;t>0;t--){
            target[t]-=x_mean;
        }
    }
    return target;
}

// std::vector sort via pairwise comparison
template<typename T>
std::vector<T> Sample<T>::sort(bool ascending){
    // make a copy
    std::vector<T> data_copy = data_vect;
    bool completed=false;
    while (!completed){
        completed=true; //let's assume this until proven otherwise
        for (int i=0;i<elements-1;i++){
            if(ascending){
                if (data_copy[i]>data_copy[i+1]){
                    completed=false;
                    double temp=data_copy[i];
                    data_copy[i]=data_copy[i+1];
                    data_copy[i+1]=temp;
                }
            }
            else{
                if (data_copy[i]<data_copy[i+1]){
                    completed=false;
                    double temp=data_copy[i];
                    data_copy[i]=data_copy[i+1];
                    data_copy[i+1]=temp;
                }
            }
        }
    }
    return data_copy;
}

// random shuffle (for 1-dimensional type <double> std::vectors)
template<typename T>
std::vector<T> Sample<T>::shuffle(){
    std::vector<T> result(elements); //=make a copy
    for (int i=0;i<elements;i++){
        result[i]=data_vect[i];
    }
    for (int i=0;i<elements;i++){
        int new_position=floor(rand_uni()*elements);
        T temp=data_vect[new_position];
        result[new_position]=data_vect[i];
        data_vect[i]=temp;
    }
}

// log transformation
template<typename T>
std::vector<T> Sample<T>::log_transform(){
    std::vector<T> target(elements);
    for (int n=0;n<elements;n++){
        target[n]=log(data_vect[n]);
    }
    return target;
}

// Engle-Granger test for cointegration
template<typename T>
double Sample<T>::Engle_Granger(){
    std::vector<double> log_x(elements), log_y(elements);
    for (int i=0;i<elements;i++){
        log_x[i]=log(x_vect[i]);
        log_y[i]=log(y_vect[i]);
    }
    Sample(log_x,log_y) res;
    res.linear_regression();
    return Sample(res.residuals).Dickey_Fuller();
}        

// predict y-values given new x-values, assuming polynomial dependence
template<typename T>
double Sample<T>::polynomial_predict(T x) {
    if (!poly_reg_completed){return NAN;}
    double y_pred = 0;
    for (int i = 0; i < coefficients.size(); i++) {
        y_pred += coefficients[i] * pow(x, i);
    }
    return y_pred;
}

// calculate mean squared error, assuming polynomial dependence
template<typename T>
double Sample<T>::polynomial_MSE() {
    if (!poly_reg_completed){return NAN;}
    double RSS = 0;
    for (int i = 0; i < x_vect.size(); i++) {
        double y_pred = polynomial_predict(x_vect[i]);
        RSS += pow(y_vect[i] - y_pred, 2);
    }        
    return RSS / (x_vect.size() - coefficients.size());
}

// check goodness of fit (based on r_squared)
template<typename T>
bool Sample<T>::isGoodFit(double threshold) {
    if (!poly_reg_completed && !lin_reg_completed){return NULL;}
    return r_squared > threshold;
}

// Polynomial regression
template<typename T>
void Sample<T>::polynomial_regression(int power) {
    // Create matrix of x values raised to different powers
    std::vector<std::vector<double>> X(elements, std::vector<double>(power + 1));
    for (int i = 0; i < elements; i++) {
        for (int j = 0; j <= power; j++) {
            X[i][j] = pow(x_vect[i], j);
        }
    }
    // Perform normal equation
    std::vector<double> beta(power + 1);
    for (int i = 0; i <= power; i++) {
        for (int j = 0; j <= power; j++) {
            double sum = 0;
            for (int k = 0; k < elements; k++) {
                sum += X[k][i] * X[k][j];
            }
            X[i][j] = sum;
        }
        double sum = 0;
        for (int k = 0; k < elements; k++) {
            sum += y_vect[k] * X[k][i];
        }
        beta[i] = sum;
    }
    // Get R-squared value
    double SS_res = 0, SS_tot = 0;
    double y_mean = accumulate(y_vect.begin(), y_vect.end(), 0.0) / elements;
    for (int i = 0; i < elements; i++) {
        double y_pred = 0;
        for (int j = 0; j <= power; j++) {
            y_pred += beta[j] * pow(x_vect[i], j);
        }
        SS_res += pow(y_vect[i] - y_pred, 2);
        SS_tot += pow(y_vect[i] - y_mean, 2);
    }
    r_squared = 1 - SS_res / SS_tot;

    // store the results
    coefficients = beta;
    standard_deviation = sqrt(SS_res / (elements - power - 1));
    poly_reg_completed = true;
}

// linear regression
template<typename T>
void Sample<T>::linear_regression(){
    y_regression.resize(elements);
    residuals.resize(elements);
    // get mean for x any y values
    double x_mean, y_mean=0;
    for (int i=0;i<elements;i++){
        x_mean+=x_vect[i];
        y_mean+=y_vect[i];
    }
    x_mean/=elements;
    y_mean/=elements;
    // get sum of squared mean deviations
    double x_mdev2_sum=0,y_mdev2_sum=0,slope_numerator=0;
    for (int n=0;n<elements;n++)
        {
        x_mdev2_sum+=pow(x_vect[n]-x_mean,2); //=slope denominator
        y_mdev2_sum+=pow(y_vect[n]-y_mean,2); //=SST
        slope_numerator+=(x_vect[n]-x_mean)*(y_vect[n]-y_mean);
        }
    slope=slope_numerator/(x_mdev2_sum+__DBL_MIN__);
    y_intercept=y_mean-slope*x_mean;
    double SST=0,SSR=0;
    for (int n=0;n<elements;n++)
        {
        y_regression[n]=y_intercept+slope*x_vect[n];
        residuals[n]=y_vect[n]-y_regression[n];
        SST+=pow(y_vect[n]-y_mean,2);
        SSR+=pow(y_vect[n]-y_regression[n],2);
        }
    r_squared=1-SSR/(SST+__DBL_MIN__);
    lin_reg_completed=true;
}

template<typename T>
Histogram<T> Sample<T>::histogram(unsigned int bars){
    Histogram histogram(bars);
    // get min and max value from sample
    histogram.min=data_vect[0];
    histogram.max=data_vect[0];
    for (int i=0;i<elements;i++){
        histogram.min=fmin(histogram.min,data_vect[i]);
        histogram.max=fmax(histogram.max,data_vect[i]);
    }

    // get histogram x-axis scaling
    T range = histogram.max-histogram.min;
    double bar_width = double(range) / bar;
    
    // set histogram x values, initialize count to zero
    for (int i=0;i<bars;i++){
        histogram.bar[i].from=histogram.min+histogram.bar_width*i;
        histogram.bar[i].to=histogram.min+histogram.bar_width*(i+1);
        histogram.bar[i].count=0;
    }

    // count absolute occurences per histogram bar
    for (int i=0;i<elements;i++){
        histogram.bar[int((data_vect[i]-histogram.min)/histogram.bar_width)].abs_count++;
    }

    // convert to relative values
    for (int i=0;i<bars;i++){
        histogram.bar[i].release_count=histogram.bar[i].abs_count/elements;
    }
    return histogram;
}

// getter functions for private members
template<typename T>
double Sample<T>::get_Pearson_R(){
    if(!correlation_completed){
        correlation();
    }
    return Pearson_R;
}

template<typename T>
double Sample<T>::get_slope(){
    if(!correlation_completed){
        correlation();
    }
    return slope;
}

template<typename T>
double Sample<T>::get_y_intercept(){
    if(!correlation_completed){
        correlation();
    }
    return y_intercept;
}

template<typename T>
double Sample<T>::get_r_squared(){
    if(!correlation_completed && !lin_reg_completed && !poly_reg_completed){
        return NAN;
    }
    return r_squared;
}

template<typename T>
double Sample<T>::get_z_score(){
    if (!correlation_completed){
        correlation();
    }
    return z_score;
}

template<typename T>
double Sample<T>::get_t_score(){
    if (!correlation_completed){
        correlation();
    }
    return t_score;
}

template<typename T>
double Sample<T>::get_Spearman_Rho(){
    if (!correlation_completed){
        correlation();
    }
    return Spearman_Rho;
}

template<typename T>
double Sample<T>::get_x_mean(){
    if (!lin_reg_completed){
        linear_regression();
    }
    return x_mean;
}

template<typename T>
double Sample<T>::get_y_mean(){
    if (!lin_reg_completed){
        linear_regression();
    }
    return y_mean;
}   

template<typename T>
double Sample<T>::get_covariance(){
    if (!correlation_completed){
        correlation();
    }
    return covariance;
}




