#pragma once
#include <vector>
#include <cmath>

// list of time series differencing methods for stationarity transformation
enum DIFFERENCING
  {
   integer=1,
   logreturn=2,
   fractional=3,
   deltamean=4,
   original=5
  };

// data structure for sample histograms
// histo_x: right(=upper) border of given bar 
template<typename T>
struct Histogram_bar{
    T from,
    T to,
    int abs_count,
    double rel_count
};

// histogram structure
template<typename T>
struct Histogram{
    T min;
    T max;
    T bar_width;
    vector<Histogram_bar<T>> bar;
    Histogram(int bars){
        for (int n=0;n<bars;n++){
            bar.push_back(Histogram_bar<T>);
        }
    }
};

// Sample class declaration
template<typename T>
class Sample{
    static_assert(std::is_same<T, float>::value ||
        std::is_same<T, double>::value ||
        std::is_same<T, double double>::value ||
        std::is_same<T, char>::value ||
        std::is_same<T, unsigned char>::value ||
        std::is_same<T, short>::value ||
        std::is_same<T, unsigned short>::value ||
        std::is_same<T, int>::value ||
        std::is_same<T, unsigned int>::value ||
        std::is_same<T, long>::value ||
        std::is_same<T, long long>::value ||
        std::is_same<T, unsigned long>::value,
        "T must be a numeric type");       
    private:
        std::vector<T> x_vect;
        std::vector<T> y_vect;
        std::vector<T> data_vect;
        std::vector<double> coefficients; //for polynomial regression
        std::vector<double> y_regression;
        std::vector<double> residuals;        
        int elements;
        double Pearson_R;
        double slope;
        double y_intercept;
        double r_squared;
        double z_score;
        double t_score;
        double Spearman_Rho; 
        double x_mean;
        double y_mean;   
        double covariance;
        double standard_deviation;
        double RSS;        
        bool lin_reg_completed=false;
        bool poly_reg_completed=false;
        bool correlation_completed=false;
    protected:
    public:
        double mean(); // arithmetic mean
        double median(); // median     
        double weighted_average(bool as_series=false); // weighted average of a 1d double std::vector 
        std::vector<int> ranking(bool low_to_high=true); // ranking
        std::vector<T> exponential_smoothing(bool as_series=false); // exponential smoothing (for time series)
        double variance(); // variance
        double stddev(std::vector<T>& sample); // standard deviation from any sample
        double stddev(); // standard deviation from sample as received from parametric object constructor     
        unsigned int find(T value,int index_from=0,int index_to=__INT_MAX__); // count the number of appearances of a given value within a numbers array
        double Dickey_Fuller(); // augmented Dickey-Fuller unit root test for stationarity{
        void correlation(); // Pearson correlation
        std::vector<T> stationary(DIFFERENCING method=integer,double degree=1,double fract_exponent=2); // time series stationary transformation
        std::vector<T> sort(bool ascending=true); // std::vector sort via pairwise comparison
        std::vector<T> shuffle(); // random shuffle (for 1-dimensional type <double> std::vectors)
        std::vector<T> log_transform(); // log transformation
        double Engle_Granger(); // Engle-Granger test for cointegration
        double polynomial_predict(T x); // predict y-values given new x-values, assuming polynomial dependence
        double polynomial_MSE(); // calculate mean squared error, assuming polynomial dependence
        bool isGoodFit(double threshold=0.95); // check goodness of fit (based on r_squared)
        void polynomial_regression(int power=5); // Polynomial regression
        void linear_regression(); // linear regression
        double get_Pearson_R(); // getter functions for private members
        double get_slope(); 
        double get_y_intercept();
        double get_r_squared();
        double get_z_score();
        double get_t_score();
        double get_Spearman_Rho();
        double get_x_mean();
        double get_y_mean();
        double get_covariance();
        Histogram<T> histogram(unsigned int bars);
        // non-parametric constructor
        Sample(){};
        // parametric constructor for single std::vector-type sample
        Sample(std::vector<T> data_vect){
            this->data_vect = data_vect;
            elements = data_vect.size();
        }
        // parametric constructor for two std::vector-type samples
        Sample(std::vector<T> x_vect, std::vector<T> y_vect){
            this->x_vect = x_vect;
            this->y_vect = y_vect;
            elements = fmin(x_vect.size(),y_vect.size());
        }       
        // parametric constructor for single array-type sample
        Sample(T *data_vect){
            elements = sizeof(data_vect)/sizeof(T);
            for (int i=0;i<elements;i++){
                this->data_vect.push_back(data_vect[i]);
            }
        }
        // parametric constructor for two array-type samples
        Sample(T *x_vect, T *y_vect){
            elements = fmin(sizeof(x_vect)/sizeof(T),sizeof(y_vect)/sizeof(T));
            for (int i=0;i<elements;i++){
                this->x_vect.push_back(x_vect[i]);
                this->y_vect.push_back(y_vect[i]);
            }
        }            
        // destructor
        ~Sample(){};
};






