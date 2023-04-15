#include "../headers/sample.h"

// returns the arithmetic mean of a sample
//that has been provided with the parametric constructor
template<typename T>
double Sample<T>::mean(){
    double sum=0;
    for (int n=0;n<elements;n++){
        sum+=data[n];
    }
    return sum/elements;
}

// returns the median of a sample that has been provided with the parametric constructor
template<typename T>
double Sample<T>::median(){
    // make a copy
    T* sorted_copy = this->sort();
    if (elements%2>__DBL_EPSILON__){ //=median has odd index
        return sorted_copy[(int)std::floor(double(elements)/2)];
    }
    else{ //=median has even index
        return (sorted_copy[elements/2]+sorted_copy[elements/2+1])/2;
    }
}

// returns the weighted average of a sample
// that has been provided with the parametric constructor
template<typename T>
double Sample<T>::weighted_average(bool as_series){
    double weight=0, weight_sum, sum=0;
    if (!as_series){ //=indexing from zero, lower index means lower attributed weight
        for (int n=0;n<elements;n++){
            weight++;
            weight_sum+=weight;
            sum+=weight*data[n];
        }
        return sum/(elements*weight_sum+__DBL_MIN__);
    }
    else {
        for (int n=elements-2;n>=0;n--) {
            weight++;
            weight_sum+=weight;
            sum+=weight*data[n];
        }
        return sum/(elements*weight_sum+__DBL_MIN__);
    }   
}     

// returns a vector of integers with that represent the individual rankings of values from
// a sample (numeric vector or array) that has been provided with the parametric constructor
// default: rank from low to high (ascending=true)
// pass "false" in order to rank in descending order
template<typename T>
int* Sample<T>::ranking(bool ascending) {
    // initialize ranks
    int rank[elements];
    for (int n=0;n<elements;n++){
        rank[n]=n;
    }
    bool ranking_completed=false;
    while (!ranking_completed){
        ranking_completed=true; //=let's assume this until a wrong order is found
        for (int i=0;i<elements-1;i++){
            // pairwise comparison:
            if (ascending){
                if (data[rank[i]]>data[rank[i+1]]){
                    ranking_completed=false;
                    int higher_ranking=rank[i+1];
                    rank[i+1]=rank[i];
                    rank[i]=higher_ranking;
                }
            }
            else{
                if (data[rank[i]]<data[rank[i+1]]){
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

// returns a modified copy of a numeric data sample (provided with the parametric constructor)
// with exponential smoothing (e.g. for time series)
template<typename T>
T* Sample<T>::exponential_smoothing(bool as_series){
    double alpha=2/(elements);
    T result[elements];

    if (as_series){
        result[elements-1]=this->mean();
        for (int n=elements-2;n>=0;n--){
            result[n] = alpha*(data[n]-result[n+1]) + result[n+1];
        }     
    }
    else{
        result[0]=this->mean();
        for (int n=1;n<elements;n++){
            result[n] = alpha*(data[n]-result[n-1]) + result[n-1];
        }
    }
    return result;
}

// returns the variance of a sample (numeric vector or array)
// that has been provided with the parametric constructor
template<typename T>
double Sample<T>::variance(){
    double sample_mean = this->mean();
    double mdev2sum=0;
    for (int n=0;n<elements;n++){
        mdev2sum+=pow(data[n]-sample_mean,2);
    }
    return mdev2sum/elements;
}

// returns the standard deviation of a sample (numeric vector or array)
// that has been provided with the parametric constructor
template<typename T>
double Sample<T>::stddev(){
    return sqrt(this->variance());
}        

// count the number of appearances of a given value within a sample
// (numeric vector or array) that has been provided with the parametric constructor
template<typename T>
unsigned int Sample<T>::find(T value,int index_from,int index_to){
    int findings=0;
    int index_end=fmin(index_to,data_vect.size()-1);
    int index_start=fmax(0,index_from);
    for (int n=index_start;n<=index_end;n++){
        if (data[n]==value){findings++;}
    }
    return findings;
}                

// performs an augmented Dickey-Fuller test
// (=unit root test for stationarity)
// on a sample (numeric vector or array)
// that has been provided with the parametric constructor;
// The test returns a p-value, which is used to determine whether or not
// the null hypothesis that the dataset has a unit root
// (=implying that the sample is non-stationary and has a trend) is rejected.
// If the p-value is less than a chosen significance level (usually 0.05),
// then the null hypothesis is rejected and it is concluded that the
// time series dataset does not have a unit root and is stationary.
template<typename T>
double Sample<T>::Dickey_Fuller(){
    // make a copy
    T data_copy[elements];
    for (int n=0;n<elements;n++){
        data_copy[n] = data[n];
    }
    // get a stationary transformation
    T* diff = Sample(data_copy).stationary(integer,1);
    // erase the first element of data_copy by shifting its pointer by 4 bytes
    data_copy+=4;
    // correlate the data with their stationary transformation
    double Pearson_R = Sample2d<T>(data_copy,diff).get_Pearson_R();
    return Pearson_R*std::sqrt((double)(elements-1)/(1-std::pow(Pearson_R,2)));  
}

// takes two samples (numeric vectors or arrays of same size) provided with the parametric
// constructor and performs a correlation test;
// Pearson method: assuming linear correlation
// Spearman: assuming non-linear monotonous correlation
// the results can be accessed via the following public methods:
//  - double Sample<typename T>::get_Pearson_R()
//  - double Sample<typename T>::get_Spearman_Rho()
// as intermediary steps, the algorithm will calculate additional values, accessible via:
//  - double Sample<typename T>::get_slope()
//  - double Sample<typename T>::get_y_intercept()
//  - double Sample<typename T>::get_x_mean()
//  - double Sample<typename T>::get_y_mean()
//  - double Sample<typename T>::get_covariance()
//  - double Sample<typename T>::get_r_squared()
//  - double Sample<typename T>::get_z_score()
//  - double Sample<typename T>::get_t_score()
template<typename T>
void Sample2d<T>::correlation(){
    // get empirical array autocorrelation (Pearson coefficient R), assumimg linear dependence
    x_mean=Sample<T>(x_data).mean();
    y_mean=Sample<T>(y_data).mean();
    covariance=0;
    for (int n=0;n<elements;n++){
        covariance+=(x_data[n]-x_mean)*(y_data[n]-y_mean);
    }
    double stdev_x=Sample<T>(x_data).stddev();
    double stdev_y=Sample<T>(y_data).stddev();
    Pearson_R = covariance/(stdev_x*stdev_y);   

    // get r_squared (coefficient of determination) assuming linear dependence
    double x_mdev2_sum=0,y_mdev2_sum=0,slope_numerator=0;
    for (int n=0;n<elements;n++){
        x_mdev2_sum+=std::pow(x_data[n]-x_mean,2); //=slope denominator
        y_mdev2_sum+=std::pow(y_data[n]-y_mean,2); //=SST
        slope_numerator+=(x_data[n]-x_mean)*(y_data[n]-y_mean);
    }
    slope=slope_numerator/x_mdev2_sum;
    y_intercept=y_mean-slope*x_mean;
    // get regression line values
    double y_pred[elements];
    double SSE=0,SSR=0;
    for (int n=0;n<elements;n++){
        y_pred[n]=y_intercept+slope*x_data[n];
        // get sum of squared (y-^y) //=SSE   
        SSE+=std::pow(y_pred[n]-y_mean,2);
        //SSR+=pow(y_data[n]-y_pred[n],2);
    };
    r_squared = SSE/(std::fmax(y_mdev2_sum,__DBL_MIN__)); //=SSE/SST, equal to 1-SSR/SST

    // Spearman correlation, assuming non-linear monotonic dependence
    std::vector<int> rank_x = Sample<T>x_data.ranking();
    std::vector<int> rank_y = Sample<T>y_data.ranking();
    double numerator=0;
    for (int n=0;n<elements;n++){
        numerator+=6*std::pow(rank_x[n]-rank_y[n],2);
    }
    Spearman_Rho=1-numerator/(elements*(std::pow(elements,2)-1));
    // test significance against null hypothesis
    double fisher_transform=0.5*log((1+Spearman_Rho)/(1-Spearman_Rho));
    z_score=sqrt((elements-3)/1.06)*fisher_transform;
    t_score=Spearman_Rho*sqrt((elements-2)/(1-std::pow(Spearman_Rho,2)));      
    correlation_completed=true;       
};            

// perform a stationary transformation (e.g. for time series)
// on a sample (numeric vector or array) that has been provided
// with the parametric constructor
template<typename T>
T* Sample<T>::stationary(DIFFERENCING method,double degree,double fract_exponent){
    // make a copy
    T target(elements);
    for (int n=0;n<elements;n++){
        target[n]=data[n];
    }
    if (method==integer){
        for (int d=1;d<=(int)degree;d++){ //=loop allows for higher order differencing
            for (int t=elements-1;t>0;t--){
                if (target[t-1]!=0){target[t]-=target[t-1];}
            }
            target.erase(target.begin()); //remove first element
        }
    }
    if (method==logreturn){
        for (int d=1;d<=round(degree);d++){ //=loop allows for higher order differencing
            for (int t=elements-1;t>0;t--){
                if (target[t-1]!=0){
                    target[t]=log(__DBL_MIN__+std::fabs(target[t]/(target[t-1]+__DBL_MIN__)));
                }
            }
            target.erase(target.begin()); //remove first element
        }     
    }
    if (method==fractional){
        for (int t=target.size()-1;t>0;t--){
            if (target[t-1]!=0){
                double stat=log(__DBL_MIN__+fabs(data[t]/data[t-1])); //note: DBL_MIN and fabs are used to avoid log(x<=0)
                double non_stat=log(fabs(data[t])+__DBL_MIN__);
                target[t]=degree*stat+pow((1-degree),fract_exponent)*non_stat;
            }
        }
        target.erase(target.begin()); //remove first element      
    }
    if (method==deltamean){
        double sum=0;
        for (int i=0;i<elements;i++){
            sum+=data[i];
        }
        double x_mean=sum/elements;
        for (int t=elements-1;t>0;t--){
            target[t]-=x_mean;
        }
    }
    return target;
}

// sorts a given sample (numeric vector or array) thas has
// been provided with the parametric constructor
// via pairwise comparison
// default: ascending order;
// set 'false' flag for reverse order sorting
template<typename T>
T* Sample<T>::sort(bool ascending){
    // make a copy
    T data_copy[elements];
    for (int n=0;n<elements;n++){
        data_copy[n] = data[n];
    }
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

// returns a randomly shuffled copy of a sample
// (=numeric vector or array) that has been provided
// with the parametric constructor
template<typename T>
T* Sample<T>::shuffle(){
    // make a copy
    std::vector<T> result = *data_vect;
    for (int i=0;i<elements;i++){
        int new_position=std::floor(Random<double>::uniform()*elements);
        T temp=data[new_position];
        result[new_position]=result[i];
        result[i]=temp;
    }
}

// performs a logarithmic transformation of the sample
// (numeric vector or array)
// that has been provided via the parametric constructor
template<typename T>
T* Sample<T>::log_transform(){
    T target[elements];
    for (int n=0;n<elements;n++){
        target[n]=std::log(data[n]);
    }
    return target;
}

// performs an Engle-Granger test in order to test
// the given sample (numeric vector or array, as provided
// with the parametric constructor) for cointegration,
// i.e. checking time-series data for a long-term relationship.
// The test was proposed by Clive Granger and Robert Engle in 1987.
// If the returned p-value is less than a chosen significance level (typically 0.05),
//  it suggests that the two time series are cointegrated and have a long-term relationship.
template<typename T>
double Sample2d<T>::Engle_Granger(){
    double log_x[elements], log_y[elements];
    for (int i=0;i<elements;i++){
        log_x[i]=std::log(x_data[i]);
        log_y[i]=std::log(y_data[i]);
    }
    std::unique_ptr<Sample<T>> res(new Sample<T>(log_x,log_y));
    res.linear_regression();
    return Sample<T>(res.residuals).Dickey_Fuller();
}        

// predict y-values given new x-values,
// assuming a polynomial dependence of two samples
// (as provided via the parametric constructor);
// for this to work, Sample<T>::polynomial_regression(int power)
// must be executed first! Otherwise the method will return "NaN"!
template<typename T>
T Sample<T>::polynomial_predict(T x) {
    if (!poly_reg_completed){return NAN;}
    double y_pred = 0;
    for (int i = 0; i < coefficients.size(); i++) {
        y_pred += coefficients[i] * std::pow(x, i);
    }
    return y_pred;
}

// calculate mean squared error,
// assuming a polynomial dependence of two samples
// (as provided via the parametric constructor);
// for this to work, Sample<T>::polynomial_regression(int power)
// must be executed first! Otherwise the method will return "NaN"!
template<typename T>
double Sample<T>::polynomial_MSE() {
    if (!poly_reg_completed){return NAN;}
    double RSS = 0;
    for (int i = 0; i < x_vect.size(); i++) {
        double y_pred = polynomial_predict(x_data[i]);
        RSS += std::pow(y_data[i] - y_pred, 2);
    }        
    return RSS / (x_data.size() - coefficients.size());
}

// check goodness of fit, based on the r_squared value
// of the correlation of two samples (as provided via the
// parametric constructor) relative to a specified confidence
// level (usually 0.95);
// for this to work, either Sample<T>::linear_regression()
// or Sample<T>::polynomial_regression() must be
// executed first! Otherwise the function will return NaN!
template<typename T>
bool Sample<T>::isGoodFit(double threshold) {
    if (!poly_reg_completed && !lin_reg_completed){return NAN;}
    return r_squared > threshold;
}

// performs polynomial regression (to the specified power)
// on a datapoint sample (numeric x & y vectors or arrays)
// as provided via the parametric constructor;
// this allows to predict new y values for new x values
// via the method Sample<T>::polynomial_predict(T x);
template<typename T>
void Sample<T>::polynomial_regression(int power) {
    // Create matrix of x values raised to different powers
    double[elements][power+1];
    for (int i=0; i<elements; i++) {
        for (int p = 0; p <= power; p++) {
            X[i][p] = std::pow(x_data[i], p);
        }
    }
    // Perform normal equation
    double beta[power+1];
    for (int i = 0; i <= power; i++) {
        for (int j = 0; j <= power; j++) {
            double sum = 0;
            for (int k = 0; k < elements; k++) {
                sum += X[k][i] * X[k][j];
            }
            X[i][j] = sum;
        }
        beta[i] = 0;
        for (int k = 0; k < elements; k++) {
            beta[i] += y_data[k] * X[k][i];
        }
    }
    // Get R-squared value
    double SS_res = 0, SS_tot = 0;
    double y_mean = accumulate(y_data.begin(), y_data.end(), 0.0) / elements;
    for (int i = 0; i < elements; i++) {
        double y_pred = 0;
        for (int j = 0; j <= power; j++) {
            y_pred += beta[j] * pow(x_data[i], j);
        }
        SS_res += std::pow(y_data[i] - y_pred, 2);
        SS_tot += std::pow(y_data[i] - y_mean, 2);
    }
    r_squared = 1 - SS_res / SS_tot;

    // store the results
    coefficients = beta;
    standard_deviation = std::sqrt(SS_res / (elements - power - 1));
    poly_reg_completed = true;
}

// performs linear regression on a datapoint sample
// (numeric x & y vectors or arrays)
// as provided via the parametric constructor;
// this allows to predict new y values for new x values
// via the method Sample<T>::linear_predict(T x);
template<typename T>
void Sample<T>::linear_regression(){
    // get mean for x any y values
    double x_mean, y_mean=0;
    for (int i=0;i<elements;i++){
        x_mean+=x_data[i];
        y_mean+=y_data[i];
    }
    x_mean/=elements;
    y_mean/=elements;
    // get sum of squared mean deviations
    double x_mdev2_sum=0,y_mdev2_sum=0,slope_numerator=0;
    for (int n=0;n<elements;n++){
        x_mdev2_sum+=std::pow(x_data[n]-x_mean,2); //=slope denominator
        y_mdev2_sum+=std::pow(y_data[n]-y_mean,2); //=SST
        slope_numerator+=(x_data[n]-x_mean)*(y_data[n]-y_mean);
    }
    slope=slope_numerator/(x_mdev2_sum+__DBL_MIN__);
    y_intercept=y_mean-slope*x_mean;
    double SST=0,SSR=0;
    for (int n=0;n<elements;n++)
        {
        y_regression[n]=y_intercept+slope*x_data[n];
        residuals[n]=y_data[n]-y_regression[n];
        SST+=std::pow(y_data[n]-y_mean,2);
        SSR+=std::pow(y_data[n]-y_regression[n],2);
        }
    r_squared=1-SSR/(SST+__DBL_MIN__);
    lin_reg_completed=true;
}

// predict y-values given new x-values,
// assuming a linear dependence of two samples
// (as provided via the parametric constructor);
template<typename T>
T Sample<T>::linear_predict(T x){
    if (!this->linear_reg_completed){this->linear_regression();}
    return slope * x + y_intercept;
}

// returns a histogram with the specified number of bars
// (datatype: 'struct Histogram<T>')
// from the data supplied via the parametric constructor
template<typename T>
Histogram<T> Sample<T>::histogram(uint bars){
    Histogram histogram(bars);
    // get min and max value from sample
    histogram.min=data[0];
    histogram.max=data[0];
    for (int i=0;i<elements;i++){
        histogram.min=std::fmin(histogram.min,data[i]);
        histogram.max=std::fmax(histogram.max,data[i]);
    }

    // get histogram x-axis scaling
    T range = histogram.max-histogram.min;
    double bar_width = double(range) / bars;
    
    // set histogram x values, initialize count to zero
    for (int i=0;i<bars;i++){
        histogram.bar[i].lower_boundary=histogram.min+histogram.bar_width*i;
        histogram.bar[i].upper_boundary=histogram.min+histogram.bar_width*(i+1);
        histogram.bar[i].count=0;
    }

    // count absolute occurences per histogram bar
    for (int i=0;i<elements;i++){
        histogram.bar[int((data[i]-histogram.min)/histogram.bar_width)].abs_count++;
    }

    // convert to relative values
    for (int i=0;i<bars;i++){
        histogram.bar[i].release_count=histogram.bar[i].abs_count/elements;
    }
    return histogram;
}

// returns the Pearson R value from
// assumed linear correlation of the datapoints
// given via the parametric constructor
template<typename T>
double Sample2d<T>::get_Pearson_R(){
    if(!correlation_completed){
        correlation();
    }
    return Pearson_R;
}

// returns the slope from linear regression
// of the datapoints given via the parametric constructor 
template<typename T>
double Sample<T>::get_slope(){
    if(!lin_reg_completed){
        linear_regression();
    }
    return slope;
}

// returns the slope of an assumed linear correlation
// of the x and y datapoints given via the parametric constructor 
template<typename T>
double Sample2d<T>::get_slope(){
    if(!correlation_completed){
        correlation();
    }
    return slope;
}

// returns the intercept point on the y-axis
// from linear regression of the data
// given via the parametric constructor
template<typename T>
double Sample<T>::get_y_intercept(){
    if(!lin_reg_completed){
        linear_regression();
    }
    return y_intercept;
}

// returns the intercept point on the y-axis given
// an assumed linear correlation of the x and y datapoints
// given via the parametric constructor
template<typename T>
double Sample2d<T>::get_y_intercept(){
    if(!correlation_completed){
        correlation();
    }
    return y_intercept;
}

// returns the r_squared value of linear or polynomial
// regression -> run Sample<T>::linear_regression() or
// Sample<T>::polynomial_regression(int power) first
// for this to work! will otherwise return NaN!
template<typename T>
double Sample<T>::get_r_squared(){
    if(!lin_reg_completed && !poly_reg_completed){
        return NAN;
    }
    return r_squared;
}

// returns the r_squared value of an assumed linear
// correlation
template<typename T>
double Sample2d<T>::get_r_squared(){
    if(!correlation_completed){
        correlation();
    }
    return r_squared;
}

// returns the Z-score of the correlation of the datapoints
// provided via the parametric constructor
// the Z-score (aka "standard score") is calculated
// to determine the significance of the correlation coefficient R;
// Z = r * sqrt(n - 2) / sqrt(1 - r^2)
// if the sample size is small, the Student's T-score can be
// used instead: Sample<T>::get_t_score();
template<typename T>
double Sample2d<T>::get_z_score(){
    if (!correlation_completed){
        correlation();
    }
    return z_score;
}

// returns the Student's T-score of the correlation of the datapoints
// provided via the parametric constructor;
// like the Z-score, the T-score is calculated to determine the significance
// of the correlation coefficient R;
// The T-score is preferred with the sample size is small.
// T = r * sqrt(n - 2) / sqrt(1 - r^2) * sqrt((n - 1) / (1 - r^2))
template<typename T>
double Sample2d<T>::get_t_score(){
    if (!correlation_completed){
        correlation();
    }
    return t_score;
}

// returns the Spearman Rho value of an
// assumed non-linear monotonous dependence of
// the datapoints provided via the parametric constructor
template<typename T>
double Sample2d<T>::get_Spearman_Rho(){
    if (!correlation_completed){
        correlation();
    }
    return Spearman_Rho;
}

// returns the arrithmetic mean of the x_values of the
// datapoint that have been provided via the parametric constructor
template<typename T>
double Sample2d<T>::get_x_mean(){
    return Sample<T>(x_data).mean();
}

// returns the arrithmetic mean of the y_values of the
// datapoint that have been provided via the parametric constructor
template<typename T>
double Sample2d<T>::get_y_mean(){
    return Sample<T>(y_data).mean();
}   

// returns the covariance of the correlation of the datapoints that
// have been provided via the parametric constructor;
// Covariance is a statistical measure that describes how two variables
// are related to each other by measuring the degree to which the values
// of one variable change in relation to the values of another variable.
// cov(X,Y) = E[(X - E[X])(Y - E[Y])]
// with E[X] as the expected value (i.e., the mean) of X
// and E[Y] as the expected value of Y
// If the covariance is positive, it means that when one variable increases,
// the other variable tends to increase as well (and vice versa).
//If the covariance is zero, there's no linear relationship between the two variables.
template<typename T>
double Sample2d<T>::get_covariance(){
    if (!correlation_completed){
        correlation();
    }
    return covariance;
}