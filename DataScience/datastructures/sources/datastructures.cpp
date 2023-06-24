#include "../headers/datastructures.h"

// +=================================+   
// | getters & setters               |
// +=================================+

// assigns a value to an array element via
// a std::initializer_list<int> index
template<typename T>
void Array<T>::set(const std::initializer_list<int>& index, const T value){
    this->data[get_element(index)] = value;
}

// assigns a value to an array element, with index parameter
// as const std::vector<int>&
template<typename T>
void Array<T>::set(const std::vector<int>& index, const T value){
    this->data[get_element(index)] = value;
}

// assigns a value to an array element, with index parameter
// as a 1-dimensionsal const Array<int>&
template<typename T>
void Array<T>::set(const Array<int>& index, const T value){
    // check valid index
    if (index.get_dimensions() != 1){
        Log::log(LOG_LEVEL_WARNING,
            "index for method Array<T>::set() must be a 1-dimensionsal const Array<int>& but has ",
            index.get_dimensions(), " dimensions");
        return;
    }
    // copy index Array to std::vector
    std::vector<int> temp(index.get_size());
    for (int i=0;i<index.get_size();i++){
        temp[i] = index[i];
    }
    // set value
    this->data[get_element(temp)] = value;
}

// assigns a value to a data element via its flattened index
template<typename T>
void Array<T>::set(const int index, const T value){
    // check valid index
    if (index<0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'void Array<T>::set(const int index, const T value)': index is ",
            index, " but must be positive");
        return;
    }
    else if (index>=this->data_elements){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'void Array<T>::set(const int index, const T value)': index is ",
            index, " but the Array has only ", this->data_elements, " elements");
        return;
    }
    // set value
    this->data[index] = value;
};

// assigns a value to an element of a two-dimensional
// Array via row and column index
template<typename T>
void Array<T>::set(const int row, const int col, const T value){
    // check valid indices
    if (row<0 || col <0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'void Array<T>::set(const int row, const int col, const T value)': ",
            "row and column indices must be positive but are [", row, "][", col, "]");
        return;
    }
    if (row>=this->dim_size[0]){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'void Array<T>::set(const int row, const int col, const T value)': ",
            "row index is ", row, "but the Array has only ", this->dim_size[0], " rows (index from 0, ",
            "therefore ", this->dim_size[0]-1, " is the allowed maximum)");
        return;
    }
    if (col>=this->dim_size[1]){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'void Array<T>::set(const int row, const int col, const T value)': ",
            "column index is ", col, "but the Array has only ", this->dim_size[1], " columns (index from 0, ",
            "therefore ", this->dim_size[1]-1, " is the allowed maximum)");
        return;
    }    
    if (this->dimensions != 2){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'void Array<T>::set(const int row, const int col, const T value)': ",
            "can only be used for 2d Arrays but Array has ", this->dimensions, " dimensions");
        return;
    }
    // set value
    this->data[std::fmin(this->data_elements-1,col + row * this->dim_size[1])] = value;
}

// returns the value of an array element via its index
// (with index as type const std::initializer_list<int>&)
template<typename T>
T Array<T>::get(const std::initializer_list<int>& index) const {
    int element=this->get_element(index);
    return this->data[element];
}

// returns the value of an array element via
// its index (with index as type const std::vector<int>&)
template<typename T>
T Array<T>::get(const std::vector<int>& index) const {
    int element=get_element(index);
    return this->data[element];
}

// returns the value of an array element via its index
// (with index as a 1-dimensional const Array<int>&)
template<typename T>
T Array<T>::get(const Array<int>& index) const {
    // check valid index dimensions
    if (index.get_elements() != this->dimensions){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'T Array<T>::get(const Array<int>& index)' ",
            "with ", index.get_elements(), "d index parameter for a ", this->dimensions,
            "d array -> will return NAN");
        return T(NAN);
    }
    // copy Array to std::vector
    std::vector<int> temp(index.get_size());
    for (int i=0;i<index.get_size();i++){
        temp[i] = index[i];
    }    
    int element=get_element(temp);
    return this->data[element];
}

// returns the value of a 2d Array element via row and column indices
template<typename T>
T Array<T>::get(const int row, const int col) const {
    // check valid indices
    if (row<0 || col <0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'void Array<T>::set(const int row, const int col, const T value)': ",
            "row and column indices must be positive but are [", row, "][", col, "]",
            " --> will return NAN");
        return NAN;
    }
    if (row>=this->dim_size[0]){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'void Array<T>::set(const int row, const int col, const T value)': ",
            "row index is ", row, "but the Array has only ", this->dim_size[0], " rows (index from 0, ",
            "therefore ", this->dim_size[0]-1, " is the allowed maximum)",
            " --> will return NAN");
        return NAN;
    }
    if (col>=this->dim_size[1]){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'void Array<T>::set(const int row, const int col, const T value)': ",
            "column index is ", col, "but the Array has only ", this->dim_size[1], " columns (index from 0, ",
            "therefore ", this->dim_size[1]-1, " is the allowed maximum)",
            " --> will return NAN");
        return NAN;
    }    
    if (this->dimensions != 2){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'void Array<T>::set(const int row, const int col, const T value)': ",
            "can only be used for 2d Arrays but Array has ", this->dimensions, " dimensions",
            " --> will return NAN");
        return NAN;
    }
    // get value
    return this->data[std::fmin(this->data_elements-1,col + row * this->dim_size[1])];
}

// returns the shape of the array as std::string
template<typename T>
std::string Array<T>::get_shapestring() const {
    const static int MAX_DIM = 10;
    std::string result = "{";
    for (int i=0;i<std::min(this->dimensions, MAX_DIM);i++){
        result += std::to_string(this->dim_size[i]);
        if (i!=this->dimensions-1){
            result += ',';
        }
        if (i==MAX_DIM-1){
            result += " ...";
        }
    }
    return result + "}";
}

// flattens a multidimensional index (represented by the values of
// a std::initializer_list<int>) to a one-dimensional index (=as a scalar)
template<typename T>
int Array<T>::get_element(const std::initializer_list<int>& index) const {
    // check for invalid index dimensions
    if (int(index.size()) != this->dimensions){
        Log::log(LOG_LEVEL_WARNING,
            "method 'get_element()' has been used with invalid index dimensions; the corresponding array has ",
            this->dimensions, " dimensions (shape: ", this->get_shapestring(), "), result will be int(T(NAN))");
        return int(T(NAN));
    }
    // deal with the special case of single dimension Arrays
    if (this->dimensions == 1){
        return *(index.begin());
    }
    // initialize result to number of elements belonging to last dimension
    int result = *(index.end()-1);
    // initialize iterator to counter of second last dimension
    auto iterator = index.end()-2;
    // initialize dimension index to second last dimension
    int i = this->dimensions-2;
    // decrement iterator down to first dimension
    for (; iterator >= index.begin(); i--, iterator--){
        // initialize amount to add to count in dimension i
        int add = *iterator;
        // multiply by product of sizes of dimensions higher than i
        int s=this->dimensions-1;
        for(; s >i; s--){
            add *= this->dim_size[s];
        }
        // add product to result 
        result += add;
    }
    // limit to valid boundaries,
    // i.e. double-check that the result isn't higher than the total
    // number of elements in the corresponding data buffer (this->data[])
    result=std::min(this->data_elements-1,result);
    return result;
}

// flattens a multidimensional index (given as type const std::vector<int>&)
// to a one-dimensional index (=as a scalar)
template<typename T>
int Array<T>::get_element(const std::vector<int>& index) const {
    if (!index_isvalid(index)) return 0;
    if (this->dimensions == 1) return index[0];
    // initialize result to number of elements belonging to last dimension
    int result = index[this->dimensions-1];
    // initialize dimension index to second last dimension
    int i = this->dimensions-2;
    for (; i>=0;i--){
        // initialize amount to add to count in dimension i
        int add = index[i];
        // multiply by product of sizes of dimensions higher than i
        int s=this->dimensions-1;
        for(; s > i; s--){
            add *= this->dim_size[s];
        }
        // add product to result;
        result += add;
    }
    // limit to valid boundaries
    result=std::min(this->data_elements-1,result);
    return result;
}

// converts a one-dimensional ('flattened') index back into
// its multi-dimensional equivalent
template<typename T>
std::vector<int> Array<T>::get_index(int element) const {
    std::vector<int> result(this->dimensions);
    // check if the array is not initialized (i.e. dimensions=0)
    if (this->dimensions<=0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'std::vector<int> Array<T>::get_index(int element)': ",
            "the array isn't properly initialized (dimensions=", this->dimensions, ")");
        return result;
    }    
    // check valid element index
    if (element <0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'std::vector<int> Array<T>::get_index(int element)': ",
            "element index must be positive but is ", element);
        return result;
    }
    if (element>=this->data_elements){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'std::vector<int> Array<T>::get_index(int element)': ",
            "element index is ", element, " but the Array has only ", this->data_elements, "elements",
            " (indexing start from zero, therefore ", this->data_elements-1, " is the highest allowed value)");
    }
    // deal with the special case of single dimension arrays
    if (this->dimensions == 1){
        result[0] = element;
        return result;
    }
    // initialize iterator to counter of second last dimension
    auto iterator = result.end()-1;
    // initialize dimension index to last dimension
    int i = this->dimensions-1;
    // decrement iterator down to first dimension
    int _element = element;
    for (; iterator >= result.begin(); i--, iterator--){
        // calculate index for this dimension
        result[i] = _element % this->dim_size[i];
        // divide flattened_index by size of this dimension
        _element /= this->dim_size[i];
    }
    return result;
}

// Return the subspace size
template<typename T>
int Array<T>::get_subspace(int dimension) const {
    // check valid dimension argument
    if (dimension<0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'int Array<T>::get_subspace(int dimension)': ",
            " dimension argument is ", dimension, "but must be positive --> result will be 0");
        return 0;
    }
    if (dimension>this->dimensions){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'int Array<T>::get_subspace(int dimension)': ",
            " dimension argument is ", dimension, "but must the Array has only ", this->dimensions,
            " dimensions (indexing starts from zero, therefore ", this->dimensions-1,
            " is the highest allowed value) --> result will be 0");
        return 0;
    }
    // return subspace size
    return this->subspace_size[dimension];
}

// +=================================+   
// | Fill, Initialize                |
// +=================================+

// fill entire array with given value
template<typename T>
void Array<T>::fill_values(const T value){
    for (int i=0;i<this->data_elements;i++){
        this->data[i]=value;
    }
}

// initialize all values of the array with zeros
template<typename T>
void Array<T>::fill_zeros(){
    this->fill_values(0);
}

// fill with identity matrix
template<typename T>
void Array<T>::fill_identity(){
    // initialize with zeros
    this->fill_values(0);
    // get size of smallest dimension
    int max_index=this->get_size(0);
    for (int i=1; i<dimensions; i++){
        max_index=std::min(max_index,this->get_size(i));
    }
    std::vector<int> index(dimensions);
    // add 'ones' of identity matrix
    for (int i=0;i<max_index;i++){
        for (int d=0;d<dimensions;d++){
            index[d]=i;
        }
        this->set(index,1);
    }
}

// fill with values from a random normal distribution
template<typename T>
void Array<T>::fill_random_gaussian(const T mu, const T sigma){
    for (int i=0; i<this->data_elements; i++){
        this->data[i] = Random<T>::gaussian(mu,sigma);
    }           
}

// fill with values from a random uniform distribution
template<typename T>
void Array<T>::fill_random_uniform(const T min, const T max){
    for (int i=0; i<this->data_elements;i++){
        this->data[i] = Random<T>::uniform(min,max);
    }
}
// fills the array with a continuous
// range of numbers (with specified start parameter
// referring to the zero position and a step parameter)
// in all dimensions
template<typename T>
void Array<T>::fill_range(const T start, const T step){
    // deal with the special case of 1-dimensional Arrays
    if (this->dimensions==1){
        for (int i=0;i<this->data_elements;i++){
            this->data[i]=start+i*step;
        }
        return;
    }
    // for all Arrays with >1 dimensions:
    std::vector<int> index(this->dimensions);
    int zero_distance;
    for (int i=0;i<this->data_elements;i++){
        index = this->index(i);
        zero_distance = 0;
        for (int ii=0;ii<index.size();ii++){
            zero_distance = std::max(zero_distance,index[ii]);
        }
        this->set(index, start + zero_distance*step);
    }
}

// randomly sets a specified fraction of the values to zero
// and retains the rest
template<typename T>
void Array<T>::fill_dropout(double ratio){
    // check valid ratio
    if (ratio >1 || ratio <0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'void Array<T>::fill_dropout(double ratio)': "
            "ratio argument must be between 0-1 but is ", ratio,
            " --> argument will be clipped to fit this range");
    }
    double valid_ratio = std::fmax(std::fmin(ratio,1.0),0);
    for (int i=0;i<this->data_elements;i++){
        this->data[i] *= Random<double>::uniform() > valid_ratio;
    }
}

// randomly sets the specified fraction of the values to zero
// and the rest to 1 (default: 0.5, i.e. 50%)
template<typename T>
void Array<T>::fill_random_binary(double ratio){
    // check valid ratio
    if (ratio >1 || ratio <0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'void Array<T>::fill_random_binary(double ratio)': "
            "ratio argument must be between 0-1 but is ", ratio,
            " --> argument will be clipped to fit this range");
    }
    double valid_ratio = std::fmax(std::fmin(ratio,1.0),0);    
    for (int i=0;i<this->data_elements;i++){
        this->data[i] = Random<double>::uniform() > valid_ratio;
    }
}

// randomly sets the specified fraction of the values to -1
// and the rest to +1 (default: 0.5, i.e. 50%)
template<typename T>
void Array<T>::fill_random_sign(double ratio){
    // check valid ratio
    if (ratio >1 || ratio <0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'void Array<T>::fill_random_sign(double ratio)': "
            "ratio argument must be between 0-1 but is ", ratio,
            " --> argument will be clipped to fit this range");
    }
    double valid_ratio = std::fmax(std::fmin(ratio,1.0),0);    
    for (int i=0;i<this->data_elements;i++){
        this->data[i] = Random<double>::uniform() > valid_ratio ? 1 : -1;
    }
}

// fill with normal "Xavier" weight initialization
// (by Xavier Glorot & Bengio) for tanh activation
template<typename T>
void Array<T>::fill_Xavier_normal(int fan_in, int fan_out){
    for (int i=0;i<this->data_elements;i++){
        // get a random number from a normal distribution with zero mean and variance one
        this->data[i] = Random<double>::gaussian(0.0,1.0);
        // apply Xavier normal formula
        this->data[i] *= sqrt(6/sqrt(double(fan_in+fan_out)));
    }
}

// fill with uniform "Xavier" weight initializiation
// (by Xavier Glorot & Bengio) for tanh activation
template<typename T>
void Array<T>::fill_Xavier_uniform(int fan_in, int fan_out){
    for (int i=0;i<this->data_elements;i++){
        // get a random number from a uniform distribution between 0-1
        this->data[i] = Random<double>::uniform(0.0,1.0);
        // apply Xavier uniform formula
        this->data[i] *= sqrt(2/sqrt(double(fan_in+fan_out)));
    }
}

// fill with uniform "Xavier" weight initialization
// for sigmoid activation
template<typename T>
void Array<T>::fill_Xavier_sigmoid(int fan_in, int fan_out){
    for (int i=0;i<this->data_elements;i++){
        // get a random number from a uniform distribution between 0-1
        this->data[i] = Random<double>::uniform(0.0,1.0);
        // apply Xavier sigmoid formula
        this->data[i] *= 4*sqrt(6/(double(fan_in+fan_out)));
    }
}

// fill with "Kaiming He" normal weight initialization,
// used for ReLU activation
template<typename T>
void Array<T>::fill_He_ReLU(int fan_in){
    for (int i=0;i<this->data_elements;i++){
        // get a random number from a normal distribution with zero mean and variance one
        this->data[i] = Random<double>::gaussian(0.0,1.0);
        // apply He ReLU formula
        this->data[i] *= std::sqrt(2.0/fan_in);
    }
}

// fill with modified "Kaiming He" nornal weight initialization,
// used for ELU activation
template<typename T>
void Array<T>::fill_He_ELU(int fan_in){
    for (int i=0;i<this->data_elements;i++){
        // get a random value from a gaussian normal distribution with zero mean and variance one
        this->data[i] = Random<double>::gaussian(0.0,1.0);
        // apply He ELU formula
        this->data[i] *= std::sqrt(1.55/(double(fan_in)));
    }
}

// +=================================+   
// | Distribution Properties         |
// +=================================+

// returns the lowest value of the Array,
// across all dimensions
template<typename T>
T Array<T>::min() const {
    // check if Array has data
    if (this->data_elements==0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid useage of method 'Array<T>::min()': ",
            "not defined for empty array -> will return NAN");
        return T(NAN);
    }
    T result = this->data[0];
    for (int i=0;i<this->data_elements;i++){
        result = std::fmin(result, this->data[i]);
    }
    return result;
}

// returns the highest value of the Array,
// across all dimensions
template<typename T>
T Array<T>::max() const {
    // check if Array has data
    if (this->data_elements==0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid useage of method 'Array<T>::max()': ",
            "not defined for empty array -> will return NAN");
        return T(NAN);
    }
    T result = this->data[0];
    for (int i=0;i<this->data_elements;i++){
        result = std::fmax(result, this->data[i]);
    }
    return result;
}

// returns the value of the Array with the highest
// deviation from zero, across all dimensions
template<typename T>
T Array<T>::maxabs() const {
    // check if Array has data
    if (this->data_elements==0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'Array<T>::maxabs()': ",
            "not defined for empty array -> will return NAN");
        return T(NAN);
    }
    T result = this->data[0];
    for (int i=0;i<this->data_elements;i++){
        result = std::fabs(this->data[i]) > std::fabs(result) ? this->data[i] : result;
    }
    return result;
}

// find the 'mode', i.e. the item that occurs the most number of times
// among all elements of an Array
template<typename T>
T Array<T>::mode() const {
    // check if Array has data
    if (this->data_elements==0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid useage of method 'Array<T>::min()': ",
            "not defined for empty array -> will return NAN");
        return T(NAN);
    }    
    // Sort the array in ascending order
    auto sorted = this->sort();
    // Create an unordered map to store the frequency of each element
    std::unordered_map<T, size_t> freq_map;
    for (int i = 0; i < this->data_elements; i++) {
        freq_map[sorted.data[i]]++;
    }
    // Find the element(s) with the highest frequency
    T mode;
    std::vector<T> modes;
    size_t max_freq = 0;
    for (const auto& p : freq_map) {
        if (p.second > max_freq) {
            modes.clear();
            modes.push_back(p.first);
            max_freq = p.second;
        } else if (p.second == max_freq) {
            modes.push_back(p.first);
        }
    }
    // If there is only one mode, return it
    if (modes.size() == 1) {
        mode = modes[0];
    } else {
        // If there are multiple modes, return the first one
        mode = modes[0];
    }
    return mode;
}

// returns the arrithmetic mean of all values of the Array
template<typename T>
double Array<T>::mean() const {
    // check if Array has data
    if (this->data_elements==0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid useage of method 'Array<T>::min()': ",
            "not defined for empty array -> will return double(T(NAN))");
        return double(T(NAN));
    }    
    double sum=0;
    for (int n=0;n<this->data_elements;n++){
        sum+=this->data[n];
    }
    return sum/this->data_elements;
}

// returns the median of all values the Array
template<typename T>
double Array<T>::median() const {
    // check if Array has data
    if (this->data_elements==0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid useage of method 'Array<T>::min()': ",
            "not defined for empty array -> will return double(T(NAN))");
        return double(T(NAN));
    }    
    // Copy the data to a temporary array for sorting
    // note: .get() is used to retrieve a raw pointer from a std::unique_ptr
    T tmp[this->data_elements];
    std::copy(this->data.get(), this->data.get() + this->data_elements, tmp);

    // Sort the temporary array
    std::sort(tmp, tmp + this->data_elements);

    // Calculate the median based on the number of elements
    if (this->data_elements % 2 == 1) {
        // Odd number of elements: return the middle element
        return static_cast<double>(tmp[this->data_elements / 2]);
    } else {
        // Even number of elements: return the average of the two middle elements
        return static_cast<double>(tmp[this->data_elements / 2 - 1] + tmp[this->data_elements / 2]) / 2.0;
    }
}

// returns the variance of all values of a vector, matrix or array
// as a floating point number of type <double>
template<typename T>
double Array<T>::variance() const {
    // check if Array has data
    if (this->data_elements==0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid useage of method 'Array<T>::min()': ",
            "not defined for empty array -> will return double(T(NAN))");
        return double(T(NAN));
    }    
    double mean = this->mean();
    double sum_of_squares = 0.0;
    for (int i = 0; i < this->data_elements; i++) {
        double deviation = static_cast<double>(this->data[i]) - mean;
        sum_of_squares += deviation * deviation;
    }
    return sum_of_squares / static_cast<double>(this->data_elements);
}

// returns the standard deviation of all values a the vector, matrix array
// as a floating point number of type <double>
template<typename T>
double Array<T>::stddev() const {
    return std::sqrt(this->variance());
}

// returns the lowest values of Arrays nested inside an Array
// by elementwise comparison
template<typename T>
Array<double> Array<T>::nested_min() const {
    std::unique_ptr<Array<double>> result;
    // using a 'try' block in order to avoid a crash if the elements aren't nested Arrays
    try {
        result = std::make_unique<Array<double>>(this->data[0].get_shape());
        *result = this->data[0];
        for (int i=1;i<this->data_elements;i++){
            for (int j=0;j<result->get_elements();j++){
                result->data[j] = std::fmin(result->data[j], this->data[i][j]);
            }
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "improper usage of method 'Array<double> Array<T>::nested_min() const': ",
            "source must have type Array<Array<T>> but is ", this->get_typename());
    }
    return std::move(*result);
}

// returns the highest values of Arrays nested inside an Array
// by elementwise comparison
template<typename T>
Array<double> Array<T>::nested_max() const {
    std::unique_ptr<Array<double>> result;
    // using a 'try' block in order to avoid a crash if the elements aren't nested Arrays
    try {
        result = std::make_unique<Array<double>>(this->data[0].get_shape());
        *result = this->data[0];
        for (int i=1;i<this->data_elements;i++){
            for (int j=0;j<result->get_elements();j++){
                result->data[j] = std::fmax(result->data[j], this->data[i][j]);
            }
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "improper usage of method 'Array<double> Array<T>::nested_max() const': ",
            "source must have type Array<Array<T>> but is ", this->get_typename());
    }
    return std::move(*result);
}

// returns the elementwise arrithmetic mean across Arrays nested as Array elements
template<typename T>
Array<double> Array<T>::nested_mean() const {
    std::unique_ptr<Array<double>> result;
    // using a 'try' block in order to avoid a crash if the elements aren't nested Arrays
    try {
        result->fill_zeros();        
        for (int i=1;i<this->data_elements;i++){
            for (int j=0;j<result->get_elements();j++){
                result->data[j] += this->data[i][j];
            }
        }
        (*result)/=this->data_elements;
        return std::move(*result);
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "improper usage of method 'Array<double> Array<T>::nested_mean() const': ",
            "source must have type Array<Array<T>> but is ", this->get_typename());
    }
    return std::move(*result);
}

// returns the elementwise variance of an array of arrays
template<typename T>
Array<double> Array<T>::nested_variance() const {  
    // using a 'try' block in order to avoid a crash if the elements aren't nested Arrays
    try {
        Array<double> mean = this->nested_mean();
        Array<double> sum_of_squares = Array<double>(this->data[0].get_shape());        
        sum_of_squares.fill_zeros();
        for (int n = 0; n < this->data_elements; n++) {
            for (int i=0; i<sum_of_squares.get_elements(); i++){
                sum_of_squares[i] += std::pow(this->data[n][i] - mean[i], 2);
            }
        }
        return std::move(sum_of_squares/this->data_elements);  
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "improper usage of method 'Array<double> Array<T>::nested_variance() const': ",
            "source must have type Array<Array<T>> but is ", this->get_typename());
        std::unique_ptr<Array<double>> result;
        return std::move(*result);
    }
}

// returns the standard deviation of Arrays nested inside an Array
// by elementwise comparison
template<typename T>
Array<double> Array<T>::nested_stddev() const {
    return (this->nested_variance()).sqrt();
}

// returns the skewness of all data of the Array
// across all dimensions
template<typename T>
double Array<T>::skewness() const {
    double skewness = 0;
    for (int i = 0; i < this->data_elements; i++) {
        skewness += std::pow((this->data[i] - this->mean()) / this->stddev(), 3);
    }
    skewness /= this->data_elements;
    return skewness;
}

// returns the kurtosis of all data of the Array
// across all dimensions
template<typename T>
double Array<T>::kurtosis() const {
    // check if Array has data
    if (this->data_elements==0){
        Log::log(LOG_LEVEL_WARNING,
            "improper usage of method 'double Array<T>::kurtosis()' with empty Array",
            " --> result will be NAN");
        return NAN;
    }    
    double kurtosis = 0;
    for (int i=0; i<this->data_elements; i++) {
        kurtosis += std::pow((this->data[i] - this->mean()) / this->stddev(), 4);
    }
    kurtosis /= this->data_elements;
    kurtosis -= 3;
    return kurtosis;
}
// +=================================+   
// | Addition                        |
// +=================================+

// returns the sum of all array elements
template<typename T>
T Array<T>::sum() const {
    try {
        T result=this->data[0];
        for (int i=1; i<this->data_elements; i++){
            result+=this->data[i];
        }
        return result;
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'T Array<T>::sum()' has failed with type ",
            this->get_typename(), this->get_shapestring(), " -> returning NAN");
        return T(NAN);
    }
}

// elementwise addition of the specified value to all values of the array
template<typename T>
Array<T> Array<T>::operator+(const T value) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i <this->data_elements; i++){
            result->data[i]=this->data[i]+value;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'T Array<T>::operator+(const T value)' has failed with 'this' of type ",
            this->get_typename(), this->get_shapestring(), " adding value of '", value, "' -> returning Array<NAN>");
        result->fill_values(T(NAN));
    }    
    return std::move(*result);
}

// returns the resulting array of the elementwise addition of
// two array of equal dimensions;
// will return a NAN array if the dimensions don't match!
template<typename T>
Array<T> Array<T>::operator+(const Array<T>& other) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    if (!equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::operator+(const Array<T>& other)' has failed with 'this' of type ",
            this->get_typename(), this->get_shapestring(), " and 'other' of type ", 
            other.get_typename(), other.get_shapestring());        
    }
    else {
        try {
            for (int i=0; i<this->data_elements; i++){
                result->data[i]=this->data[i]+other.data[i];
            }
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'Array<T> Array<T>::operator+(const Array<T>& other)' has failed with 'this' of type ",
                this->get_typename(), this->get_shapestring(), " and 'other' of type ", 
                other.get_typename(), other.get_shapestring());            
        }
    }
    return std::move(*result);
}

// prefix increment operator;
// increments the values of the array by +1,
// returns a reference to the source array itself
template<typename T>
Array<T>& Array<T>::operator++() const {
    try {
        for (int i=0; i<this->data_elements; i++){
            this->data[i]+=1;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "prefix increment method 'Array<T>& Array<T>::operator++()' has failed with type ",
            this->get_typename(), this->get_shapestring(), " -> returning *this as unmodified");         
    }
    return *this;
}

// postfix increment operator;
// makes an internal copy of the array,
// then increments all values of the array by +1,
// then returns the temporary copy;
// note: more overhead then with the prefix increment
// because of extra copy!
template<typename T>
Array<T> Array<T>::operator++(int) const {
    std::unique_ptr<Array<T>> temp = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0; i<this->data_elements; i++){
            temp->data[i] = this->data[i];
            this->data[i]++;
        }
        return std::move(*temp);
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "postfix increment method 'Array<T>& Array<T>::operator++(int)' has failed with type ",
            this->get_typename(), this->get_shapestring(), " -> returning *this as unmodified");
        return *this;       
    }    
}

// elementwise addition of the specified
// value to the elements of the array
template<typename T>
void Array<T>::operator+=(const T value) {
    try {
        for (int i=0; i<this->data_elements; i++){
            this->data[i]+=value;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "elementwise addition with method 'void Array<T>::operator+=(const T value)' has failed with type ",
            this->get_typename(), this->get_shapestring(), " -> source Array will be unmodified");        
    }
}

// elementwise addition of the values of 'other'
// to the values of the corresponding elements of 'this';
// the dimensions of the Arrays must match!
// --> will otherwise result in Array<NAN>!
template<typename T>
void Array<T>::operator+=(const Array<T>& other){
    if (!equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "calling method 'void Array<T>::operator+=(const Array<T>& other)' with Arrays of non-matching shapes: this",
            this->get_shapestring(), " vs other", other.get_shapestring(), " -> result will be Array<NAN>");
        this->fill_values(T(NAN));
        return;
    }
    try {
        for (int i=0; i<this->data_elements; i++){
            this->data[i]+=other.data[i];
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "elementwise addition with method 'void Array<T>::operator+=(const Array<T>& other)' has failed; ",
            "'this' has type ", this->get_typename(), " with shape ", this->get_shapestring(),
            ", 'other' has type ", other.get_typename(), " with shape ", other.get_shapestring(),
            ", result will be Array<NAN>");
        this->fill_values(T(NAN));
        return;
    }    
}

// +=================================+   
// | Substraction                    |
// +=================================+

// elementwise substraction of the specified value from all values of the array
template<typename T>
Array<T> Array<T>::operator-(const T value) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i <this->data_elements; i++){
            result->data[i]=this->data[i]-value;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'T Array<T>::operator-(const T value)' has failed with type ",
            this->get_typename(), this->get_shapestring(), " -> returning Array<NAN>");
        result->fill_values(T(NAN));
    }    
    return std::move(*result);
}

// returns the resulting array of the elementwise substraction of
// two array of equal dimensions;
// will return a NAN array if the dimensions don't match!
template<typename T>
Array<T> Array<T>::operator-(const Array<T>& other) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    if (!equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::operator-(const Array<T>& other)' has failed with type ",
            this->get_typename(), this->get_shapestring(), " -> returning Array<NAN>");        
        result->fill_values(T(NAN));
    }
    else {
        try {
            for (int i=0; i<this->data_elements; i++){
                result->data[i]=this->data[i]-other.data[i];
            }
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'Array<T> Array<T>::operator-(const Array<T>& other)' has failed with type ",
                this->get_typename(), this->get_shapestring(), " -> returning Array<NAN>");        
            result->fill_values(T(NAN));            
        }
    }
    return std::move(*result);
}

// prefix decrement operator;
// decrements the values of the array by -1
template<typename T>
Array<T>& Array<T>::operator--() const {
    try {
        for (int i=0; i<this->data_elements; i++){
            this->data[i]-=1;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "prefix decrement method 'Array<T>& Array<T>::operator--()' has failed with type ",
            this->get_typename(), this->get_shapestring(), " -> returning *this as unmodified");         
    }
    return *this;
}

// postfix decrement operator;
// makes an internal copy of the array,
// then decrements all values of the array by -1,
// then returns the temporary copy;
// note: more overhead then with the prefix decrement
// because of extra copy!
template<typename T>
Array<T> Array<T>::operator--(int) const {
    std::unique_ptr<Array<T>> temp = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0; i<this->data_elements; i++){
            temp->data[i] = this->data[i];
            this->data[i]--;
        }
        return std::move(*temp);
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "postfix decrement method 'Array<T>& Array<T>::operator--(int)' has failed with type ",
            this->get_typename(), this->get_shapestring(), " -> returning *this as unmodified");
        return *this;       
    }    
}

// elementwise substraction of the specified
// value from the elements of the array
template<typename T>
void Array<T>::operator-=(const T value) {
    try {
        for (int i=0; i<this->data_elements; i++){
            this->data[i]-=value;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "elementwise substraction with method 'void Array<T>::operator-=(const T value)' has failed with type ",
            this->get_typename(), this->get_shapestring(), " -> source Array will be unmodified");        
    }
}

// elementwise substraction of the values of 'other'
// from the values of the corresponding elements of 'this';
// the dimensions of the Arrays must match!
// --> will otherwise result in Array<NAN>!
template<typename T>
void Array<T>::operator-=(const Array<T>& other){
    if (!equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "calling method 'void Array<T>::operator-=(const Array<T>& other)' with Arrays of non-matching shapes: this",
            this->get_shapestring(), " vs other", other.get_shapestring(), " -> result will be Array<NAN>");
        this->fill_values(T(NAN));
        return;
    }
    try {
        for (int i=0; i<this->data_elements; i++){
            this->data[i]-=other.data[i];
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "elementwise substraction with method 'void Array<T>::operator-=(const Array<T>& other)' has failed; ",
            "'this' has type ", this->get_typename(), " with shape ", this->get_shapestring(),
            ", 'other' has type ", other.get_typename(), " with shape ", other.get_shapestring(),
            ", result will be Array<NAN>");
        this->fill_values(T(NAN));
        return;
    }    
}

// +=================================+   
// | Multiplication                  |
// +=================================+

// returns the product reduction, i.e. the result
// of all individual elements of the array
template<typename T>
T Array<T>::product() const {
    // check if Array has data
    if (this->data_elements==0){
        Log::log(LOG_LEVEL_WARNING,
            "calling method 'T Array<T>::product()' on empty array -> result will be NAN");
        return NAN;
    }
    try {
        T result = this->data[0];
        for (int i=1; i<this->data_elements; i++){
            result*=this->data[i];
        }
        return result;
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'T Array<T>::product()' has failed with type ", this->get_typename(),
            this->get_shapestring(), " -> result will be NAN");        
        return T(NAN);
    }
}

// elementwise multiplication with a scalar
template<typename T>
Array<T> Array<T>::operator*(const T factor) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    // check if Array has data
    if (this->data_elements==0){
        Log::log(LOG_LEVEL_WARNING,
            "calling method 'Array<T> Array<T>::operator*(const T factor)' on empty >rray -> result will be empty, too");
    }
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i] = this->data[i] * factor;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method Array<T> Array<T>::operator*(const T factor)' has failed with type=",
            this->get_typename(), this->get_shapestring(), " and factor=", factor);
    }
    return std::move(*result);
}

// elementwise multiplication (*=) with a scalar
template<typename T>
void Array<T>::operator*=(const T factor) {
    // check if Array has data
    if (this->data_elements==0){
        Log::log(LOG_LEVEL_WARNING,
            "calling method 'void Array<T>::operator*=(const T factor)' on empty Array",
            " --> resulting in the unmodified empty source Array");
        return;
    }
    try {
        for (int i=0; i<this->data_elements; i++){
            this->data[i]*=factor;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'void Array<T>::operator*=(const T factor)' has failed with type=",
            this->get_typename(), this->get_shapestring(), " and factor=", factor);
    }
}

// elementwise multiplication of the values of the current
// array with the corresponding values of a second array,
// resulting in the 'Hadamard product';
// the dimensions of the two arrays must match!
// the function will otherwise return a NAN array!
template<typename T>
Array<T> Array<T>::Hadamard_product(const Array<T>& other) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    if(!equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "calling method 'Array<T> Array<T>::Hadamard_product(const Array<T>& other)'",
            " with arrays of non-matching shapes: this", this->get_shapestring(),
            " vs other", other.get_shapestring(), " -> result will be Array<NAN>");
        result->fill_values(T(NAN));
    }
    else {
        try {
            for (int i=0; i<this->data_elements; i++){
                result->data[i]=this->data[i]*other.data[i];
            }
        }
        catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::Hadamard_product(const Array<T>& other)' has failed",
            " with 'this' as type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") vs. 'other' as type ", other.get_typename(), " (shape: ", other.get_shapestring(),
            ") -> result will be Array<NAN>");
        result->fill_values(T(NAN));            
        }
    }
    return std::move(*result);
}

// Array tensor reduction ("tensor dotproduct")
template<typename T>
Array<T> Array<T>::tensordot(const Array<T>& other, const std::vector<int>& axes) const {
    // create a new tensor to hold the result
    std::unique_ptr<Array<T>> result;
    try {
        // compute the dimensions of the output tensor
        std::vector<int> out_dims;
        for (int i = 0; i < this->dimensions; i++) {
            if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
                out_dims.push_back(this->dim_size[i]);
            }
        }
        for (int i = 0; i < other.dimensions; i++) {
            if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
                out_dims.push_back(other.dim_size[i]);
            }
        }
        
        // set result shape
        result = std::make_unique<Array<T>>(out_dims);
        
        // perform the tensor contraction
        std::vector<int> index1(this->dimensions, 0);
        std::vector<int> index2(other.dimensions, 0);
        std::vector<int> index_out(out_dims.size(), 0);
        int contractionsize = 1;
        for (int i = 0; i < int(axes.size()); i++) {
            contractionsize *= this->dim_size[axes[i]];
        }
        for (int i = 0; i < result->dim_size[0]; i++) {
            index_out[0] = i;
            for (int j = 0; j < contractionsize; j++) {
                for (int k = 0; k < int(axes.size()); k++) {
                    index1[axes[k]] = j % this->dim_size[axes[k]];
                    index2[axes[k]] = j % other.dim_size[axes[k]];
                }
                for (int k = 1; k < int(out_dims.size()); k++) {
                    int size = k <= axes[0] ? this->dim_size[k - 1] : other.dim_size[k - int(axes.size()) - 1];
                    int val = (i / result->get_subspace(k)) % size;
                    if (k < axes[0] + 1) {
                        index1[k - 1] = val;
                    } else {
                        index2[k - int(axes.size()) - 1] = val;
                    }
                }
                result->set(index_out, result->get(index_out) + this->get(index1) * other.get(index2));
            }
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::tensordot(const Array<T>& other, const std::vector<int>& axes)'",
            " has failed with 'this' as type ", this->get_typename(), this->get_shapestring(),
            " vs. 'other' as type ", other.get_typename(), other.get_shapestring(), " -> result is undefined");
        return std::move(*result);        
    }
    
    return std::move(*result);
}

// Array tensordot (tensor contraction):
// overload for Arrays with 1 or 2 dimensions
template<typename T>
Array<T> Array<T>::tensordot(const Array<T>& other) const {
    // check valid dimensions
    if (this->dimensions!=other.dimensions || this->dimensions>2 || other.get_dimensions()>2){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'Array<T> Array<T>::tensordot(const Array<T>& other)': ",
            "this overload is meant to be used only if both Arrays are 1-dimensional or both are 2-dimensional; ",
            "failed with 'this' as type ", this->get_typename(), this->get_shapestring(),
            " vs. 'other' as type ", other.get_typename(), other.get_shapestring(),
            " -> the returned result assumes that the contraction axes are 0+1");
        return this->tensordot(other, {0,1});
    }
    // use case for 1d Arrays:
    if (this->dimensions==1){
        // TODO
    }
    else {
        // Create the resulting matrix
        std::initializer_list<int> result_shape = {this->get_size(0), other.get_size(1)};
        std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(result_shape);         
        // Check if the matrices can be multiplied
        if (this->get_size(1) != other.get_size(0)){
            Log::log(LOG_LEVEL_WARNING,
                "invalid usage of method 'Array<T> Array<T>::tensordot(const Array<T>& other)', ",
                "attempting Array 2d-tensordot with shapes that can't be multiplied: ",
                "failed with 'this' as type ", this->get_typename(), this->get_shapestring(),
                " vs. 'other' as type ", other.get_typename(), other.get_shapestring());
            return std::move(*result);
        }
        try {       
            // Compute the dot product
            for (int i = 0; i < this->get_size(0); i++) {
                for (int j = 0; j < other.get_size(1); j++) {
                    T sum = 0;
                    for (int k = 0; k < this->get_size(1); k++) {
                        sum += this->get(i, k) * other.get(k, j);
                    }
                    result->set(i, j, sum);
                }
            }
            return std::move(*result);
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'Array<T> Array<T>::tensordot(const Array<T>& other)' for 2d-tensordot has failed, ",
                "with 'this' as type ", this->get_typename(), this->get_shapestring(),
                " vs. 'other' as type ", other.get_typename(), other.get_shapestring(),
                " -> will return *this as unmodified");
            return std::move(*result);            
        }
    }
    // default return statement if all conditions above failed
    Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::tensordot(const Array<T>& other)' has failed, ",
        "with 'this' as type ", this->get_typename(), this->get_shapestring(),
        " vs. 'other' as type ", other.get_typename(), other.get_shapestring());
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    return std::move(*result);
}

// Array dotproduct, i.e. the scalar product
template<typename T>
T Array<T>::dotproduct(const Array<T>& other) const {
    // check for equal shape
    if (!this->equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "calling method 'T Array<T>::dotproduct(const Array<T>& other)'",
            " with arrays of non-matching shapes,",
            " with 'this' as type ", this->get_typename(), this->get_shapestring(),
            " vs. 'other' as type ", other.get_typename(), other.get_shapestring(),
            " -> result will be undefined");
        T result = T(NAN);
        return result;
    }
    try {
        T result = 0;
        for (int i = 0; i < this->data_elements; i++) {
            result += this->data[i] * other.data[i];
        }
        return result;
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'T Array<T>::dotproduct(const Array<T>& other)' has failed,",
            " with 'this' as type ", this->get_typename(), this->get_shapestring(),
            " vs. 'other' as type ", other.get_typename(), other.get_shapestring(),
            " -> result will be NAN");
        return T(NAN);        
    }
}

// Alias for tensordot matrix multiplication
template<typename T>
Array<T> Array<T>::operator*(const Array<T>& other) const {
    return this->tensordot(other);
}

// Alias for tensordot matrix multiplication
template<typename T>
void Array<T>::operator*=(const Array<T>& other) {
    *this = this->tensordot(other);
}

// +=================================+   
// | Division                        |
// +=================================+

// elementwise division by a scalar
template<typename T>
Array<T> Array<T>::operator/(const T quotient) const {
    if (quotient==0){
        Log::log(LOG_LEVEL_ERROR,
            "invalid call of method 'Array<T> Array<T>::operator/(const T quotient)' ",
            "with quotient=0 (zero division is undefined)");
    }
    return (*this)*(1/quotient);
}

// elementwise division (/=) by a scalar
template<typename T>
void Array<T>::operator/=(const T quotient){
    (*this)*=(1/quotient);
}

// elementwise division of the values of the current
// array by the corresponding values of a second Array,
// resulting in the 'Hadamard division';
// the dimensions of the two arrays must match!
// the function will otherwise return Array<NAN>!
template<typename T>
Array<T> Array<T>::Hadamard_division(const Array<T>& other) {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    if(!equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "calling method 'Array<T> Array<T>::Hadamard_division(const Array<T>& other)' ",
            "with arrays of non-matching shapes: ",
            "'this' as type ", this->get_typename(), this->get_shapestring(),
            " vs. 'other' as type ", other.get_typename(), other.get_shapestring(),
            " -> result will be Array<NAN>");
        result->fill_values(T(NAN));
    }
    else {
        try {
            for (int i=0; i<this->data_elements; i++){
                result->data[i]=this->data[i]/other.data[i];
            }
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'Array<T> Array<T>::Hadamard_division(const Array<T>& other)' has failed, with ",
                "'this' as type ", this->get_typename(), this->get_shapestring(),
                " vs. 'other' as type ", other.get_typename(), other.get_shapestring(),
                " -> result will be Array<NAN>");
            // check if failure was due to zero division
            try {
                for (int i=0;i<this->data_elements;i++){
                    if (this->data[i]==0){
                        std::cout << "reason: source Array has contains zero values (zero division is undefined)" << std::endl;
                        break;
                    }
                }
            }
            catch (...) {
                // empty
            }
            result->fill_values(T(NAN));            
        }
    }
    return std::move(*result);
}

// +=================================+   
// | Modulo                          |
// +=================================+

// elementwise modulo operation, converting the Array values
// to the remainders of their division by the specified number
template<typename T>
void Array<T>::operator%=(const double num){
    if (num==0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'Array<double> Array<T>::operator%(const double num)' ",
            "with num=0 (zero division is undefined) --> 'this' will remain unmodified");
        return;
    }    
    try {
        for (int i=0; i<this->data_elements; i++){
            this->data[i]%=num;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'void Array<T>::operator%=(const double num)' has failed ",
            "with type=", this->get_typename(), this->get_shapestring(),
            " and num=", num, " -> 'this' will remain unmodified");
    }
}

// elementwise modulo operation, resulting in an Array that
// contains the remainders of the division of the values of
// the original array by the specified number
template<typename T>
Array<double> Array<T>::operator%(const double num) const {
    std::unique_ptr<Array<double>> result = std::make_unique<Array<double>>(this->dim_size);
    if (num==0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'Array<double> Array<T>::operator%(const double num)' ",
            "with num=0 (zero division is undefined) --> returns Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);
    }
    try  {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]%num;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::operator%(const double num)' has failed ",
            "with type=", this->get_typename(), this->get_shapestring(),
            " and num=", num, " --> returns Array<NAN>");
        result->fill_values(T(NAN));       
    }
    return std::move(*result);
}

// +=================================+   
// | Exponentiation                  |
// +=================================+

// elementwise exponentiation to the power of
// the specified exponent
template<typename T>
Array<T> Array<T>::pow(const T exponent) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=std::pow(this->data[i], exponent);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::pow(const T exponent)' has failed ",
            "with type=", this->get_typename(), this->get_shapestring(),
            " and exponent=", exponent, " -> returns Array<NAN>"); 
        result->fill_values(T(NAN));          
    }
    return std::move(*result);
}

// elementwise exponentiation to the power of
// the corresponding values of the second array;
// the dimensions of the two array must match!
// the function will otherwise return a NAN array!
template<typename T>
Array<T> Array<T>::pow(const Array<T>& other) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    if(!equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "calling method 'Array<T> Array<T>::pow(const Array<T>& other)' ",
            "with arrays of non-matching shapes: ",
            "'this' as type ", this->get_typename(), this->get_shapestring(),
            " vs. 'other' as type ", other.get_typename(), other.get_shapestring(),
            " -> result will be Array<NAN>");
        result->fill_values(T(NAN));
    }
    else {
        try { 
            for (int i=0; i<this->data_elements; i++){
                result->data[i]=std::pow(this->data[i], other.data[i]);
            }
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'Array<T> Array<T>::pow(const Array<T>& other)' has failed, with ",
                "'this' as type ", this->get_typename(), this->get_shapestring(),
                " vs. 'other' as type ", other.get_typename(), other.get_shapestring(),
                " -> result will be Array<NAN>");
            result->fill_values(T(NAN));            
        }
    }
    return std::move(*result);
}

// converts the individual values of the array
// elementwise to their square root
template<typename T>
Array<T> Array<T>::sqrt() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=std::sqrt(this->data[i]);
        }
    }
    catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'Array<T> Array<T>::sqrt()' has failed, with type ",
                this->get_typename(), this->get_shapestring(),
                " -> result will be Array<NAN>");
            try {
                for (int i=0;i<this->data_elements;i++){
                    if (this->data[i]<0){
                        std::cout << "reason: source Array has negative values" << std::endl;
                        break;
                    }
                }
            }
            catch (...) {
                // empty
            }
            result->fill_values(T(NAN));         
    }
    return std::move(*result);
}

// converts the individual values of the array
// elementwise to their natrual logarithm
template<typename T>
Array<T> Array<T>::log() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=std::log(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::log()' has failed, with type=",
            this->get_typename(), this->get_shapestring(),
            ") -> result will be Array<NAN>"); 
        // check if failure was due to negative or zero values
        try {
            for (int i=0;i<this->data_elements;i++){
                if (this->data[i]<0){
                    std::cout << "reason: source Array has negative values (log(x) is undefined for x<=0)" << std::endl;
                    break;
                }
                if (this->data[i]==0){
                    std::cout << "reason: source Array contains zero values (log(x) is undefined for x<=0)" << std::endl;
                    break;
                }                    
            }
        }
        catch (...) {
            // empty
        }            
        result->fill_values(T(NAN)); 
    }
    return std::move(*result);
}

// converts the individual values of the array
// elementwise to their base-10 logarithm
template<typename T>
Array<T> Array<T>::log10() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=std::log10(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::log10()' has failed, with type ",
            this->get_typename(), this->get_shapestring(),
            " -> result will be Array<NAN>"); 
        // check if failure was due to negative or zero values
        try {
            for (int i=0;i<this->data_elements;i++){
                if (this->data[i]<0){
                    std::cout << "reason: source Array has negative values (log10(x) is undefined for x<=0)" << std::endl;
                    break;
                }
                if (this->data[i]==0){
                    std::cout << "reason: source Array contains zero values (log10(x) is undefined for x<=0)" << std::endl;
                    break;
                }                    
            }
        }   
        catch (...) {
            // empty
        }
        result->fill_values(T(NAN)); 
    }
    return std::move(*result);
}

// +=================================+   
// | Rounding                        |
// +=================================+

// rounds the values of the array elementwise
// to their nearest integers
template<typename T>
Array<T> Array<T>::round() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::round(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::round()' has failed, with type ",
            this->get_typename(), this->get_shapestring(),
            " -> result will be Array<NAN>");
        result->fill_values(T(NAN));
    }
    return std::move(*result);
}

// rounds the values of the array elementwise
// to their next lower integers
template<typename T>
Array<T> Array<T>::floor() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::floor(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::floor()' has failed, with type ",
            this->get_typename(), this->get_shapestring(),
            " -> result will be Array<NAN>");
        result->fill_values(T(NAN));        
    }
    return std::move(*result);
}

// returns a copy of the array that stores the values as rounded
// to their next higher integers
template<typename T>
Array<T> Array<T>::ceil() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::ceil(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::ceil()' has failed, with type ",
            this->get_typename(), this->get_shapestring(),
            " -> result will be Array<NAN>");
        result->fill_values(T(NAN));          
    }
    return std::move(*result);
}

// returns a copy of the array that stores the
// absolute values of the source array
template<typename T>
Array<T> Array<T>::abs() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::fabs(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::abs()' has failed, with type ",
            this->get_typename(), this->get_shapestring(),
            " -> result will be Array<NAN>");
        result->fill_values(T(NAN));          
    }
    return std::move(*result);
}

// +=================================+   
// | Min, Max                        |
// +=================================+

// elementwise minimum of the specified value
// and the data elements of the Array
template<typename T>
Array<T> Array<T>::min(const T value) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::fmin(this->data[i], value);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::min(const T value)' has failed, with type ",
            this->get_typename(), this->get_shapestring(),
            " with value of type ", typeid(T).name(), " -> result will be Array<NAN>");
        result->fill_values(T(NAN));          
    }
    return std::move(*result);
}

// elementwise maximum of the specified value
// and the data elements of the Array
template<typename T>
Array<T> Array<T>::max(const T value) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::fmax(this->data[i], value);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::max(const T value)' has failed, with type ",
            this->get_typename(), " (shape: ", this->get_shapestring(),
            ") with value of type ", typeid(T).name(), " --> result will be Array<NAN>");
        result->fill_values(T(NAN));          
    }
    return std::move(*result);
}

// returns the result of elementwise min() comparison
// of 'this' vs 'other'
template<typename T>
Array<T> Array<T>::min(const Array<T>& other) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    if (!equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::min(const Array<T>& other)' has failed with ",
            "arrays of non-matching shapes: ",
            "this", this->get_shapestring(), " vs other", other.get_shapestring(),
            " --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);
    }
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::fmin(this->data[i], other.data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::min(const Array<T>& other)' has failed, ",
            "with 'this' of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") vs. 'other' of type ", other.get_typename(), " (shape: ", other.get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));          
    }
    return std::move(*result);
}

// returns the result of elementwise min() comparison
// of 'this' vs 'other'
template<typename T>
Array<T> Array<T>::max(const Array<T>& other) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    if (!equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::max(const Array<T>& other)' has failed with ",
            "arrays of non-matching shapes: ",
            "this", this->get_shapestring(), " vs other", other.get_shapestring(),
            " --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);
    }
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::fmax(this->data[i], other.data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::max(const Array<T>& other)' has failed, ",
            "with 'this' of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") vs. 'other' of type ", other.get_typename(), " (shape: ", other.get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));          
    }
    return std::move(*result);
}

// +=================================+   
// | Trigonometric Functions         |
// +=================================+

// elementwise application of the cos() function
template<typename T>
Array<T> Array<T>::cos() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::cos(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::cos()' has failed, with type ",
            this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));         
    }
    return std::move(*result);
}

// elementwise application of the sin() function
template<typename T>
Array<T> Array<T>::sin() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::sin(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::sin()' has failed, with type ",
            this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));         
    }
    return std::move(*result);
}

// elementwise application of the tan() function
template<typename T>
Array<T> Array<T>::tan() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::tan(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::tan()' has failed, with type ",
            this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));         
    }
    return std::move(*result);
}

// elementwise application of the acos() function
template<typename T>
Array<T> Array<T>::acos() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::acos(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::acos()' has failed, with type ",
            this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));         
    }
    return std::move(*result);
}

// elementwise application of the asin() function
template<typename T>
Array<T> Array<T>::asin() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::asin(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::asin()' has failed, with type ",
            this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));         
    }
    return std::move(*result);
}


// elementwise application of the atan() function
template<typename T>
Array<T> Array<T>::atan() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::atan(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::atan()' has failed, with type ",
            this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));         
    }
    return std::move(*result);
}

// +=================================+   
// | Hyperbolic Functions            |
// +=================================+

// elementwise application of the cosh() function
template<typename T>
Array<T> Array<T>::cosh() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::cosh(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::cosh()' has failed, with type ",
            this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));         
    }
    return std::move(*result);
}

// elementwise application of the sinh() function
template<typename T>
Array<T> Array<T>::sinh() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::sinh(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::sinh()' has failed, with type ",
            this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));         
    }
    return std::move(*result);
}

// elementwise application of the tanh() function
template<typename T>
Array<T> Array<T>::tanh() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::tanh(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::tanh()' has failed, with type ",
            this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));         
    }
    return std::move(*result);
}

// elementwise application of the acosh() function
template<typename T>
Array<T> Array<T>::acosh() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::acosh(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::acosh()' has failed, with type ",
            this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));         
    }
    return std::move(*result);
}

// elementwise application of the asinh() function
template<typename T>
Array<T> Array<T>::asinh() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::asinh(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::asinh()' has failed, with type ",
            this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));         
    }
    return std::move(*result);
}

// elementwise application of the atanh() function
template<typename T>
Array<T> Array<T>::atanh() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0;i<this->data_elements;i++){
            result->data[i]=std::atanh(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::atanh()' has failed, with type ",
            this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));         
    }
    return std::move(*result);
}

// +=================================+   
// | Find, Replace                   |
// +=================================+

// returns the number of occurrences of the specified value
template<typename T>
int Array<T>::find(const T value) const {
    try {
        int counter=0;
        for (int i=0; i<this->data_elements; i++){
            counter+=(this->data[i]==value);
        }
        return counter;
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "invalid usage of method 'int Array<T>::find(const T value)' with type ",
            this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and value of type ", typeid(value).name(), " --> returns 0");
        return 0;
    }
}

// replace all findings of given value by specified new value
template<typename T>
Array<T> Array<T>::replace(const T old_value, const T new_value) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i] = this->data[i]==old_value ? new_value : this->data[i];
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::replace(const T old_value, const T new_value)' has failed, ",
        "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
        "), query type 'old_value' as type ", typeid(old_value).name(),
        " and replacement type 'new_value' as type ", typeid(new_value).name(),
        " --> result will be Array<NAN>");
        result->fill_values(T(NAN));         
    }
    return std::move(*result);
}

// returns 1 for all positive elements and -1 for all negative elements
template<typename T>
Array<char> Array<T>::sign() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i] = this->data[i]>=0 ? 1 : -1;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
        "method 'Array<T> Array<T>::sign()' has failed, ",
        "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
        ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));         
    }
    return std::move(*result);
}

// +=================================+   
// | Scaling                         |
// +=================================+

// minmax scaling
template<typename T>
Array<double> Array<T>::scale_minmax(T min,T max) const {
    std::unique_ptr<Array<double>> result = std::make_unique<Array<double>>(this->dim_size);
    try {
        T data_min = this->data.min();
        T data_max = this->data.max();
        // check NAN
        try {
            if (data_min!=data_min || data_max!=data_max){
                Log::log(LOG_LEVEL_WARNING,
                    "method 'Array<double> Array<T>::scale_minmax(T min,T max)' has failed, ",
                    "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
                    ") and min value of type ",typeid(min).name, " / max value of type ",
                    typeid(max).name(), ", reason: checking for data_min and data_max resulted in NAN",
                    " --> result will be Array<NAN>");
                result->fill_values(T(NAN));
                return std::move(*result);
            }
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'Array<double> Array<T>::scale_minmax(T min,T max)' has failed, ",
                "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
                ") and min value of type ",typeid(min).name, " / max value of type ",
                typeid(max).name(), ", reason: NAN check of data_min and data_max failed ",
                "--> result will be Array<NAN>");
            result->fill_values(T(NAN));
            return std::move(*result);
        }
        double factor = (max-min) / (data_max-data_min);
        for (int i=0; i<this->get_elements(); i++){
            result->data[i] = (this->data[i] - data_min) * factor + min;
        }
        return std::move(*result);
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<double> Array<T>::scale_minmax(T min,T max)' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and min value of type ",typeid(min).name, " / max value of type ",
            typeid(max).name(), " --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);
    }
}

// mean scaling
template<typename T>
Array<double> Array<T>::scale_mean() const {
    std::unique_ptr<Array<double>> result = std::make_unique<Array<double>>(this->dim_size);
    try {
        T data_min = this->data.min();
        T data_max = this->data.max();
        T range = data_max - data_min;
        // check NAN
        try {
            if (data_min!=data_min || data_max!=data_max){
                Log::log(LOG_LEVEL_WARNING,
                    "method 'Array<double> Array<T>::scale_mean()' has failed, ",
                    "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
                    "), reason: checking for data_min and data_max resulted in NAN",
                    " --> result will be Array<NAN>");
                result->fill_values(T(NAN));
                return std::move(*result);
            }
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'Array<double> Array<T>::scale_mean()' has failed, ",
                "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
                "), reason: NAN check of data_min and data_max failed ",
                "--> result will be Array<NAN>");
            result->fill_values(T(NAN));
            return std::move(*result);
        }        
        double mean = this->mean();
        for (int i=0; i<this->get_elements(); i++){
            result->data[i] = (this->data[i] - mean) / range;
        }
        return std::move(*result);
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<double> Array<T>::scale_mean()' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);        
    }
}

// scaling with zero mean and variance 1
template<typename T>
Array<double> Array<T>::scale_standardized() const {
    std::unique_ptr<Array<double>> result = std::make_unique<Array<double>>(this->dim_size);
    double mean = this->mean();
    double variance = this->variance();
    // check NAN
    try {
        if (mean!=mean || variance!=variance){
            Log::log(LOG_LEVEL_WARNING,
                "method 'Array<double> Array<T>::scale_standardized()' has failed, ",
                "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
                "), reason: checking for mean and/or variance resulted in NAN",
                " --> result will be Array<NAN>");
            result->fill_values(T(NAN));
            return std::move(*result);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<double> Array<T>::scale_standardized()' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            "), reason: NAN check of mean and/or varience failed ",
            "--> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);
    }       
    try {
        for (int i=0; i<this->get_elements(); i++){
            result->data[i] = (this->data[i] - mean) / variance;
        }
        return std::move(*result);
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<double> Array<T>::scale_standardized()' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result); 
    }
}

template<typename T>
Array<double> Array<T>::scale_unit_length() const {
    std::unique_ptr<Array<double>> result = std::make_unique<Array<double>>(this->dim_size);
    // calculate the Euclidean norm of the data array
    T norm = 0;
    int elements = this->get_elements();
    for (int i = 0; i < elements; i++) {
        norm += std::pow(this->data[i], 2);
    }
    if (norm==0 || norm!=norm){
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<double> Array<T>::scale_unit_length()' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            "), reason: Euclidian norm is 0 or NAN --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result); 
    }
    try {
        norm = std::sqrt(norm);
        // scale the data array to unit length
        for (int i = 0; i < elements; i++) {
            result->data[i] = this->data[i]/norm;
        }
        return std::move(*result);
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<double> Array<T>::scale_unit_length()' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);        
    }
}

// +=================================+   
// | Activation Functions            |
// +=================================+

// neural network activation functions
template<typename T>
Array<T> Array<T>::activation(ActFunc activation_function) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        double alpha = 0.01;
        switch (activation_function){
            case RELU: {
                for (int i=0;i<this->data_elements;i++){
                    result->data[i] = this->data[i] * this->data[i]>0;
                } 
            } break;
            case LRELU: {
                for (int i=0;i<this->data_elements;i++){
                    result->data[i] = this->data[i]>0 ? this->data[i] : this->data[i]*alpha;
                }
            } break;
            case ELU: {
                for (int i=0;i<this->data_elements;i++){
                    result->data[i] = this->data[i]>0 ? this->data[i] : alpha*(std::exp(this->data[i])-1); 
                } 
            } break;
            case SIGMOID: {
                for (int i=0;i<this->data_elements;i++){
                    result->data[i] = 1/(1+std::exp(-this->data[i])); 
                } 
            } break;   
            case TANH: {
                for (int i=0;i<this->data_elements;i++){
                    result->data[i] = std::tanh(this->data[i]);
                } 
            } break;
            case SOFTMAX: {
                // TODO
            } break;
            case IDENT: {
                // do nothing   
            } break;
            default: /* do nothing */ break;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::activation(ActFunc activation_function)' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);        
    }
    return std::move(*result);
}

// derivatives of neural network activation functions
template<typename T>
Array<T> Array<T>::derivative(ActFunc activation_function) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        double alpha = 0.01;
        switch (activation_function){
            case RELU: {
                for (int i=0;i<this->data_elements;i++){
                    result->data[i] = this->data[i]>0 ? 1 : 0;
                } 
            } break;
            case LRELU: {
                for (int i=0;i<this->data_elements;i++){
                    result->data[i] = this->data[i]>0 ? 1 : alpha;
                }
            } break;
            case ELU: {
                for (int i=0;i<this->data_elements;i++){
                    result->data[i] = this->data[i]>0 ? 1 : alpha*std::exp(this->data[i]);
                } 
            } break;
            case SIGMOID: {
                for (int i=0;i<this->data_elements;i++){
                    result->data[i] = std::exp(this->data[i])/std::pow(std::exp(this->data[i])+1,2); 
                } 
            } break;   
            case TANH: {
                for (int i=0;i<this->data_elements;i++){
                    result->data[i] = 1-std::pow(std::tanh(this->data[i]),2);
                } 
            } break;
            case SOFTMAX: {
                // TODO
            } break;
            case IDENT: {
                result->fill_values(1);  
            } break;
            default: /* do nothing */ break;
        }
    }
    catch (...)  {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::derivative(ActFunc activation_function)' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);        
    }
    return std::move(*result);
}

// +=================================+   
// | Custom Functions                |
// +=================================+

// modifies the given vector, matrix or array by applying
// the referred function to all its values
// (the referred function should take a single argument of type <T>)
template<typename T>
Array<T> Array<T>::function(const T (*pointer_to_function)(T)) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i] = pointer_to_function(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::function(const T (*pointer_to_function)(T))' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);            
    }
    return std::move(*result);
}

// +=================================+   
// | Outliers Treatment              |
// +=================================+

// truncate outliers by z-score
template<typename T>
Array<T> Array<T>::outliers_truncate(double z_score) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    double mean = this->mean();
    double stddev = this->stddev();
    // NAN check
    if (mean!=mean || stddev!=stddev){
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::outliers_truncate(double z_score)' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            "), reason: retrieving mean and/or stddev resulted in NAN --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);         
    }
    try {
        double lower_margin = mean - z_score*stddev;
        double upper_margin = mean + z_score*stddev;
        for (int i=0;i<this->_elements;i++){
            result->data[i] = this->data[i]>upper_margin ? upper_margin : this->data[i];
            result->data[i] = this->data[i]<lower_margin ? lower_margin : this->data[i];
        }
        return std::move(*result);
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::outliers_truncate(double z_score)' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);         
    }
}

// truncate outliers by z-score winsoring
template<typename T>
Array<T> Array<T>::outliers_winsoring(double z_score) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    double mean = this->mean();
    double stddev = this->stddev();
    // NAN check
    if (mean!=mean || stddev!=stddev){
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::outliers_winsoring(double z_score)' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            "), reason: retrieving mean and/or stddev resulted in NAN --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);         
    }
    try {
        double lower_margin = mean - z_score*stddev;
        double upper_margin = mean + z_score*stddev;
        T highest_valid = mean;
        T lowest_valid = mean;
        for (int i=0;i<this->_elements;i++){
            if (this->_data[i] < upper_margin && this->_data[i] > lower_margin){
                highest_valid = std::fmax(highest_valid, this->_data[i]);
                lowest_valid = std::fmin(lowest_valid, this->_data[i]);
            }
        }    
        for (int i=0;i<this->_elements;i++){
            result->data[i] = this->data[i]>upper_margin ? highest_valid : this->data[i];
            result->data[i] = this->data[i]<lower_margin ? lowest_valid : this->data[i];
        }
        return std::move(*result);
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::outliers_winsoring(double z_score)' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            "), --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);         
    }
}

template<typename T>
Array<T> Array<T>::outliers_mean_imputation(double z_score) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    double mean = this->mean();
    double stddev = this->stddev();
    // NAN check
    if (mean!=mean || stddev!=stddev){
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::outliers_mean_imputation(double z_score)' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            "), reason: retrieving mean and/or stddev resulted in NAN --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);         
    }
    try {
        double lower_margin = mean - z_score*stddev;
        double upper_margin = mean + z_score*stddev;
        for (int i=0;i<this->_elements;i++){
            result->data[i] = this->data[i];
            if (this->_data[i] > upper_margin || this->_data[i] < lower_margin){
                result->_data[i] = mean;
            }
        }
        return std::move(*result);
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::outliers_mean_imputation(double z_score)' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);
    }
}

template<typename T>
Array<T> Array<T>::outliers_median_imputation(double z_score) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    double median = this->median();
    double mean = this->mean();
    double stddev = this->stddev();
    // NAN check
    if (mean!=mean || stddev!=stddev || stddev!=stddev){
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::outliers_median_imputation(double z_score)' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            "), reason: retrieving mean and/or stddev and/or median resulted in NAN --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);         
    }
    try {
        double lower_margin = mean - z_score*stddev;
        double upper_margin = mean + z_score*stddev;
        for (int i=0;i<this->_elements;i++){
            if (this->_data[i] > upper_margin || this->_data[i] < lower_margin){
                this->_data[i] = median;
            }
        }
        return std::move(*result);
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::outliers_median_imputation(double z_score)' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);        
    }
}

template<typename T>
Array<T> Array<T>::outliers_value_imputation(T value, double z_score) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    double mean = this->mean();
    double stddev = this->stddev();
    if (mean!=mean || stddev!=stddev){
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::outliers_value_imputation(T value, double z_score)' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            "), reason: retrieving mean and/or stddev resulted in NAN --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);         
    }
    try {
        double lower_margin = mean - z_score*stddev;
        double upper_margin = mean + z_score*stddev;
        for (int i=0;i<this->_elements;i++){
            if (this->_data[i] > upper_margin || this->_data[i] < lower_margin){
                this->_data[i] = value;
            }
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::outliers_value_imputation(T value, double z_score)' has failed, ",
            "with Array of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") --> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);  
    }
}

// +=================================+   
// | Assignment                      |
// +=================================+

// copy assignment operator with second Array as argument:
// copies the values from a second array into the values of
// the current array;
template<typename T>
Array<T>& Array<T>::operator=(const Array<T>& other) {
    if (!equalsize(other)){
        try {
            // Allocate new memory for the array
            std::unique_ptr<T[]> newdata = std::make_unique<T[]>(other.get_capacity());
            // Copy the elements from the other array to the new array
            std::copy(other.data.get(), other.data.get() + other.get_elements(), newdata.get());
            // Assign the new data to this object
            this->data = std::move(newdata);
            this->data_elements = other.get_elements();
            this->dim_size = other.dim_size;
            this->subspace_size = other.subspace_size;
            this->capacity = other.get_capacity();
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'Array<T>& Array<T>::operator=(const Array<T>& other)' has failed, "
                "with 'this' of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
                ") and 'other' of type ", other.get_typename(), " (shape: ", other.get_shapestring(),")");
        }
    }
    else {
        try {
            std::copy(other.data.get(), other.data.get() + other.get_elements(), this->data.get());
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'Array<T>& Array<T>::operator=(const Array<T>& other)' has failed, "
                "with 'this' of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
                ") and 'other' of type ", other.get_typename(), " (shape: ", other.get_shapestring(),")");                
        }
    }
    return *this;
}

// Array move assignment
template<typename T>
Array<T>& Array<T>::operator=(Array<T>&& other) noexcept {
    this->resize(other.get_elements());
    this->data_elements = other.get_elements();
    this->data = std::move(other.data);
    this->dim_size = std::move(other.dim_size);
    this->subspace_size = std::move(other.subspace_size);
    this->capacity = other.get_capacity();
    other.data.reset();
    return *this;
}

// copy assignment operator with C-style array as argument
template<typename T>
Array<T>& Array<T>::operator=(const T (&arr)[]){
    // check if arr is initialized
    int arr_elements = sizeof(arr) / sizeof(arr[0]);
    if (arr_elements != this->data_elements){
        try {
            resize_array(this->data, this->data_elements, arr_elements);
            // Copy the elements from array argument to 'this'
            std::copy(arr.get(), arr.get() + arr_elements, this->data.get());
            this->data_elements = arr_elements;
            this->dim_size = {arr_elements};
            this->subspace_size = {arr_elements};
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'Array<T>& Array<T>::operator=(const T[]& arr)' has failed, "
                "with 'this' of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
                ") and 'arr' with ", arr_elements, " elements");
        }
    }
    else {
        try {
            std::copy(arr.get(), arr.get() + arr_elements, this->data.get());
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'Array<T>& Array<T>::operator=(const T[]& arr)' has failed, "
                "with 'this' of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
                ") and 'arr' with ", arr_elements, " elements");             
        }
    }
    return *this;
}

// move assignment operator with C-style array as argument
template<typename T>
Array<T>& Array<T>::operator=(T (&&arr)[]) noexcept {
    int arr_elements = sizeof(arr) / sizeof(arr[0]);
    int oldSize = this->data_elements;
    this->data_elements = arr_elements;
    if (this->data_elements!=arr_elements){
        resize_array(this->data, oldSize, arr_elements);
    }
    this->data = std::move(arr);
    this->dim_size = {arr_elements};
    this->subspace_size = {arr_elements};
    this->capacity = arr_elements;
    arr.reset();
    return *this;
}
// copy assignment operator with std::vector as argument
template<typename T>
Array<T> Array<T>::operator=(std::vector<T> vector){
    int elements = vector.size();
    Array<T> result = Array<T>({elements});
    for (int i=0;i<elements;i++){
        result[i] =vector[i];
    }
    return result;
}

// indexing operator [] for reading
template<typename T>
T& Array<T>::operator[](const int index) const {
    if (index>=this->data_elements){
        Log::log(LOG_LEVEL_WARNING,
        "invalid usage of method 'T& Array<T>::operator[](const int index)', ",
        "index out of range: index is ", index, ", but the Array has only ", this->data_elements,
        " elements; indexing starts from zero, therefore ", this->data_elements-1,
        " is the highest allowed argument -> will be clipped to match this range");
    }
    if (index<0){
        Log::log(LOG_LEVEL_WARNING,
        "invalid usage of method 'T& Array<T>::operator[](const int index)', ",
        "index out of range: index is ", index,
        " but must be positive -> will be clipped to 0 to match the allowed range");
    }    
    int valid_index = std::max(std::min(index, this->data_elements-1),0);
    return this->data[valid_index];
}

// indexing operator [] for writing
template<typename T>
T& Array<T>::operator[](const int index) {
    if (index>=this->data_elements){
        Log::log(LOG_LEVEL_WARNING,
        "invalid usage of method 'T& Array<T>::operator[](const int index)', ",
        "index out of range: index is ", index, ", but the Array has only ", this->data_elements,
        " elements; indexing starts from zero, therefore ", this->data_elements-1,
        " is the highest allowed argument -> will be clipped to match this range");
    }
    if (index<0){
        Log::log(LOG_LEVEL_WARNING,
        "invalid usage of method 'T& Array<T>::operator[](const int index)', ",
        "index out of range: index is ", index,
        " but must be positive -> will be clipped to 0 to match the allowed range");
    }    
    int valid_index = std::max(std::min(index, this->data_elements-1),0);
    return this->data[valid_index];
}

// +=================================+   
// | Elementwise Comparison by       |
// | Single Value                    |
// +=================================+

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are greater than
// the specified value
template<typename T>
Array<bool> Array<T>::operator>(const T value) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]>value;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<bool> Array<T>::operator>(const T value)' has failed, "
            "with type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and value of type ", typeid(value).name());
    }
    return std::move(*result);
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are greater than or equal
// to the specified argument value
template<typename T>
Array<bool> Array<T>::operator>=(const T value) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]>=value;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<bool> Array<T>::operator>=(const T value)' has failed, "
            "with type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and value of type ", typeid(value).name());
    }
    return std::move(*result);
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are equal to the specified
// argument value
template<typename T>
Array<bool> Array<T>::operator==(const T value) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]==value;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<bool> Array<T>::operator==(const T value)' has failed, "
            "with type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and value of type ", typeid(value).name());        
    }
    return std::move(*result);
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are unequal to the specified
// argument value
template<typename T>
Array<bool> Array<T>::operator!=(const T value) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]!=value;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<bool> Array<T>::operator!=(const T value)' has failed, "
            "with type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and value of type ", typeid(value).name());        
    }
    return std::move(*result);
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are less than the
// specified argument value
template<typename T>
Array<bool> Array<T>::operator<(const T value) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]<value;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<bool> Array<T>::operator<(const T value)' has failed, "
            "with type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and value of type ", typeid(value).name());        
    }
    return std::move(*result);
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are less than or equal to
// the specified argument value
template<typename T>
Array<bool> Array<T>::operator<=(const T value) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]<=value;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<bool> Array<T>::operator<=(const T value)' has failed, "
            "with type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and value of type ", typeid(value).name());        
    }
    return std::move(*result);
}

// +=================================+   
// | Elementwise Comparison by       |
// | Second Array                    |
// +=================================+

// returns a boolean array of equal dimensions
// to the source array, with its values indicating
// whether the values of the source array are greater
// than the corresponding values of the second array,
// i.e. by elementwise comparison;
// make sure the the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
Array<bool> Array<T>::operator>(const Array<T>& other) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    if (!this->equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'Array<bool> Array<T>::operator>(const Array<T>& other)' ",
            "with Arrays of non-matching shapes: ",
            "this", this->get_shapestring(), " vs other", other.get_shapestring(),
            " -> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);
    }
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]>other.data[i];
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<bool> Array<T>::operator>(const Array<T>& other)' has failed, ",
            "with 'this'  of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and 'other' of type ", other.get_typename(), " (shape: ", other.get_shapestring(),
            ") -> result will be Array<NAN>");
        result->fill_values(T(NAN));
    }
    return std::move(*result);
}

// returns a boolean array of equal dimensions
// to the source array, with its values indicating
// whether the values of the source array are greater
// than or equal to the corresponding values of the
// second array, i.e. by elementwise comparison;
// make sure the the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
Array<bool> Array<T>::operator>=(const Array<T>& other) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    if (!this->equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'Array<bool> Array<T>::operator>=(const Array<T>& other)' ",
            "with Arrays of non-matching shapes: ",
            "this", this->get_shapestring(), " vs other", other.get_shapestring(),
            " -> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);
    }
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]>=other.data[i];
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<bool> Array<T>::operator>=(const Array<T>& other)' has failed, ",
            "with 'this'  of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and 'other' of type ", other.get_typename(), " (shape: ", other.get_shapestring(),
            ") -> result will be Array<NAN>");
        result->fill_values(T(NAN));
    }
    return std::move(*result);
}

// returns a boolean array of equal dimensions
// to the source array, with its values indicating
// whether the values of the source array are equal
// to the corresponding values of the second array,
// i.e. by elementwise comparison;
// make sure the the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
Array<bool> Array<T>::operator==(const Array<T>& other) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    if (!this->equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'Array<bool> Array<T>::operator==(const Array<T>& other)' ",
            "with Arrays of non-matching shapes: ",
            "this", this->get_shapestring(), " vs other", other.get_shapestring(),
            " -> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);
    }
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]==other.data[i];
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<bool> Array<T>::operator==(const Array<T>& other)' has failed, ",
            "with 'this'  of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and 'other' of type ", other.get_typename(), " (shape: ", other.get_shapestring(),
            ") -> result will be Array<NAN>");
        result->fill_values(T(NAN));
    }
    return std::move(*result);
}

// returns a boolean array of equal dimensions
// to the source array, with its values indicating
// whether the values of the source array are unequal
// to the corresponding values of the second array,
// i.e. by elementwise comparison;
// make sure the the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
Array<bool> Array<T>::operator!=(const Array<T>& other) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    if (!this->equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'Array<bool> Array<T>::operator!=(const Array<T>& other)' ",
            "with Arrays of non-matching shapes: ",
            "this", this->get_shapestring(), " vs other", other.get_shapestring(),
            " -> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);
    }
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]!=other.data[i];
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<bool> Array<T>::operator!=(const Array<T>& other)' has failed, ",
            "with 'this'  of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and 'other' of type ", other.get_typename(), " (shape: ", other.get_shapestring(),
            ") -> result will be Array<NAN>");
        result->fill_values(T(NAN));
    }
    return std::move(*result);
}

// returns a boolean array of equal dimensions
// to the source array, with its values indicating
// whether the values of the source array are less
// than the corresponding values of the second array,
// i.e. by elementwise comparison;
// make sure the the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
Array<bool> Array<T>::operator<(const Array<T>& other) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    if (!this->equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'Array<bool> Array<T>::operator<(const Array<T>& other)' ",
            "with Arrays of non-matching shapes: ",
            "this", this->get_shapestring(), " vs other", other.get_shapestring(),
            " -> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);
    }
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]<other.data[i];
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<bool> Array<T>::operator<(const Array<T>& other)' has failed, ",
            "with 'this'  of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and 'other' of type ", other.get_typename(), " (shape: ", other.get_shapestring(),
            ") -> result will be Array<NAN>");
        result->fill_values(T(NAN));
    }
    return std::move(*result);
}

// returns a boolean array of equal dimensions
// to the source array, with its values indicating
// whether the values of the source array are less than
// or equal to the corresponding values of the second array,
// i.e. by elementwise comparison;
// make sure the the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
Array<bool> Array<T>::operator<=(const Array<T>& other) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    if (!this->equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'Array<bool> Array<T>::operator<=(const Array<T>& other)' ",
            "with Arrays of non-matching shapes: ",
            "this", this->get_shapestring(), " vs other", other.get_shapestring(),
            " -> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);
    }
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]<=other.data[i];
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<bool> Array<T>::operator<=(const Array<T>& other)' has failed, ",
            "with 'this'  of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and 'other' of type ", other.get_typename(), " (shape: ", other.get_shapestring(),
            ") -> result will be Array<NAN>");
        result->fill_values(T(NAN));
    }
    return std::move(*result);
}

// +=================================+   
// | Elementwise Logical Operations  |
// +=================================+

// returns a boolean array as the result of the
// logical AND of the source array and the specified
// boolean argument value
template<typename T>
Array<bool> Array<T>::operator&&(const bool value) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]&&value;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<bool> Array<T>::operator&&(const bool value)' has failed, ",
            "with type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and comparison value of type ", typeid(value).name(),
            ") -> result will be Array<NAN>");
        result->fill_values(T(NAN));
    }
    return std::move(*result);
}

// returns a boolean array as the result of the
// logical OR of the source array and the specified
// boolean argument value
template<typename T>
Array<bool> Array<T>::operator||(const bool value) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]||value;
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<bool> Array<T>::operator||(const bool value)' has failed, ",
            "with type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and comparison value of type ", typeid(value).name(),
            ") -> result will be Array<NAN>");
        result->fill_values(T(NAN));
    }
    return std::move(*result);
}

// returns a boolean array as the result of the
// logical NOT of the source array
template<typename T>
Array<bool> Array<T>::operator!() const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=!this->data[i];
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<bool> Array<T>::operator!()' has failed, ",
            "with type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") -> result will be Array<NAN>");
        result->fill_values(T(NAN));
    }
    return std::move(*result);
}

// returns a boolean array as the result of the
// elementwise logical AND operation between the
// source array and the corresponding values of the
// second array;
// make sure that the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
Array<bool> Array<T>::operator&&(const Array<T>& other) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    if (!this->equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'Array<bool> Array<T>::operator&&(const Array<T>& other)' ",
            "with Arrays of non-matching shapes: ",
            "this", this->get_shapestring(), " vs other", other.get_shapestring(),
            " -> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);
    }
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]&&other.data[i];
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<bool> Array<T>::operator&&(const Array<T>& other)' has failed, ",
            "with 'this'  of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and 'other' of type ", other.get_typename(), " (shape: ", other.get_shapestring(),
            ") -> result will be Array<NAN>");
        result->fill_values(T(NAN));
    }
    return std::move(*result);
}

// returns a boolean array as the result of the
// elementwise logical OR operation between the
// source array and the corresponding values of the
// second array;
// make sure that the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
Array<bool> Array<T>::operator||(const Array<T>& other) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    if (!this->equalsize(other)){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'Array<bool> Array<T>::operator||(const Array<T>& other)' ",
            "with Arrays of non-matching shapes: ",
            "this", this->get_shapestring(), " vs other", other.get_shapestring(),
            " -> result will be Array<NAN>");
        result->fill_values(T(NAN));
        return std::move(*result);
    }
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]||other.data[i];
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<bool> Array<T>::operator||(const Array<T>& other)' has failed, ",
            "with 'this'  of type ", this->get_typename(), " (shape: ", this->get_shapestring(),
            ") and 'other' of type ", other.get_typename(), " (shape: ", other.get_shapestring(),
            ") -> result will be Array<NAN>");
        result->fill_values(T(NAN));
    }
    return std::move(*result);
}

// +=================================+   
// | Type Casting                    |
// +=================================+

// type casting (explicit by default)
template<typename T>
template<typename C>
Array<T>::operator Array<C>(){
    std::unique_ptr<Array<C>> result = std::make_unique<Array<C>>(this->dim_size);
    try {
        for (int i=0; i<this->data_elements; i++){
            result->data[i] = static_cast<C>(this->data[i]);
        }
    }
    catch (...) {
        Log::log(LOG_LEVEL_WARNING,
            "typecasting with method 'Array<T>::operator Array<C>()' has failed; ",
            "source type T is ", typeid(T).name(), ", target type C is ",
            typeid(C).name());
    }
    return std::move(*result);
}

// +=================================+   
// | pointer                         |
// +=================================+

// dereference operator
template<typename T>
Array<typename std::remove_pointer<T>::type> Array<T>::operator*() const {
    using InnerType = typename std::remove_pointer<T>::type;
    Array<InnerType> result(this->dim_size);
    for (int i = 0; i < this->data_elements; i++) {
        result.data[i] = *(this->data[i]);
    }
    return result;
}



// 'address-of' operator
template<typename T>
Array<T*> Array<T>::operator&() const {
    Array<T*> result(this->dim_size);
    for (int i = 0; i < this->data_elements; i++) {
        result.data[i] = &(this->data[i]);
    }
    return result;
}

// +=================================+   
// | Array Conversion                |
// +=================================+

// flattens the Array into a one-dimensional vector
template<typename T>
Array<T> Array<T>::flatten() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->data_elements);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=this->data[i];
    }
    return std::move(*result);
}

// reshapes the Array; any surplus elements will
// be set to the specified value;
template<typename T>
void Array<T>::reshape(std::vector<int> shape, const T init_value){
    Array<T> temp=*this;
    std::unique_ptr<Array<T>> new_array = std::make_unique<Array<T>>(shape);
    new_array->fill_values(init_value);
    std::vector<int> index(new_array.get_dimensions());
    for (int i=0;i<temp.get_elements();i++){
        bool index_ok=true;
        for (int d=0;d<new_array.get_dimensions();d++){
            if (temp.get_size(d)>new_array.get_size(d)){
                index_ok=false;
                break;
            }
            index[d]=temp.get_index(i)[d];
        }
        if (index_ok){
            new_array.set(index,temp[i]);
        }
    }
}

// reshapes the Array; any surplus elements will
// be set to the specified value;
template<typename T>
void Array<T>::reshape(std::initializer_list<int> shape, const T init_value){
    Array<T> temp=*this;
    std::unique_ptr<Array<T>> new_array = std::make_unique<Array<T>>(shape);
    new_array->fill_values(init_value);
    std::vector<int> index(new_array.get_dimensions());
    for (int i=0;i<temp.get_elements();i++){
        bool index_ok=true;
        for (int d=0;d<new_array.get_dimensions();d++){
            if (temp.get_size(d)>new_array.get_size(d)){
                index_ok=false;
                break;
            }
            index[d]=temp.get_index(i)[d];
        }
        if (index_ok){
            new_array.set(index,temp[i]);
        }
    }
}

// reshapes the Array; any surplus elements will
// be set to the specified value;
template<typename T>
void Array<T>::reshape(Array<int> shape, const T init_value){
    Array<T> temp=*this;
    std::unique_ptr<Array<T>> new_array = std::make_unique<Array<T>>(shape);
    new_array->fill_values(init_value);
    std::vector<int> index(new_array.get_dimensions());
    for (int i=0;i<temp.get_elements();i++){
        bool index_ok=true;
        for (int d=0;d<new_array.get_dimensions();d++){
            if (temp.get_size(d)>new_array.get_size(d)){
                index_ok=false;
                break;
            }
            index[d]=temp.get_index(i)[d];
        }
        if (index_ok){
            new_array.set(index,temp[i]);
        }
    }
}

// concatenate 'this' Array and 'other' Array
// by stacking along the specified axis
template<typename T>
Array<T> Array<T>::concatenate(const Array<T>& other, const int axis) const {

    // check if both arrays have the same number of dimensions
    if (this->dimensions != other.get_dimensions()){
        throw std::invalid_argument("can't concatenate arrays with unequal number of dimensions");
    }

    // deal with the special case of 1d Arrays
    // (note: axis argument has no effect because there's only one axis anyway)
    if (other.get_dimensions()==1) {
        std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>({this->data_elements + other.get_elements()});
        // copy the elements of the current array to the equivalent index of the result array
        for (int i=0;i<this->data_elements;i++){
            result->data[i] = this->data[i];
        }
        // stitch in the elements of the second array
        for (int i=0;i<other.data_elements;i++){
            result->data[this->data_elements+i] = other.data[i];
        }
        // return result
        return std::move(*result);
    }
    else {
        Log::log(LOG_LEVEL_WARNING,
            "method 'Array<T> Array<T>::concatenate(const Array<T>& other)' has failed: ",
            "'other' must be 1-dimensional but is ", other.get_dimensions(), "d --> returning 'this'");
        return *this;
    }

    // check if all dimensions except for the concatenation axis match
    for (int d=0; d<this->dimensions; d++){
        if (d!=axis && this->dim_size[d] != other.get_size(d)){
            throw std::invalid_argument("can't concatenate arrays with unequal dimension sizes along any axis other than the concatenation axis");
        }
    }
    // check for valid concatenation axis
    if (axis<0 || axis>=this->dimensions){
        throw std::invalid_argument("invalid concatenation axis: must fit withing the number of dimensions of the arrays to be concatenated");
    }

    // create a result Array of the concatenated size
    std::vector<int> shape = this->dim_size;
    shape[axis]+=other.get_size(axis);
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(shape);
    // copy the elements of the current array to the equivalent index of the result array
    std::vector<int> index;
    for (int i=0;i<this->data_elements;i++){
        index = this->get_index(i);
        result->data[result->get_element(index)] = this->data[i];
    }
    // stitch in the elements of the second array
    for (int i=0;i<other.data_elements;i++){
        index = other.get_index(i);
        index[axis]+=this->dim_size[axis];
        result->data[result->get_element(index)] = other.data[i];
    }
    // return result
    return std::move(*result);
}

// adds a dimension (the index of the new dimension
// will be the last one, i.e. at the end of the index vector)
template<typename T>
Array<T> Array<T>::add_dimension(int size, T init_value) const {
    std::vector<int> new_shape = this->dim_size;
    new_shape.push_back(size);
    std::unique_ptr<Array<T>> result = Array<T>(new_shape);
    result->fill_values(init_value);
    for (int i=0;i<this->data_elements;i++){
        result->data[i] = this->data[i];
    }
    return std::move(*result);
}

// stacks a nested Array of Arrays into a single combined array
template<typename T>
T Array<T>::stack() const {
    std::unique_ptr<T> result = std::make_unique<T>(this->get_stacked_shape());
    std::vector<int> index(this->get_dimensions()+1);
    for (int n = 0; n < this->data_elements; n++){
        for (int i=0; i<this->data[n].get_elements(); i++){
            index = this->data[n].get_index(i);
            index.push_back(n);
            result->set(index,this->data[n][i]);
        }
    }
    return std::move(*result);
}

// returns the shape that result from stacking
// a nested Array of Arrays into a single combined Array
template<typename T>
std::vector<int> Array<T>::get_stacked_shape() const {
    std::vector<int> result = this->data[0].get_shape();
    result.push_back(this->data_elements);
    return result;
}

// calculates correlation metrics (Pearson, Spearman, ANOVA)
// of 'this' as x_data and 'other' as y_data
template<typename T>
CorrelationResult<T> Array<T>::correlation(const Array<T>& other) const {
    if (this->data_elements != other.data_elements) {
        Log::log(LOG_LEVEL_WARNING,
            "Invalid use of method 'CorrelationResult<T> Array<T>::correlation(const Array<T>& other)': ",
            "both Arrays should have the same number of elements ",
            " --> the surplus elements of the larger array will be ignored");
    }
    int elements=std::min(this->data_elements, other.get_elements());    
    std::unique_ptr<CorrelationResult<T>> result = std::make_unique<CorrelationResult<T>>(elements);

    // get empirical vector autocorrelation (Pearson coefficient R), assumimg linear dependence
    result->x_mean=this->mean();
    result->y_mean=other.mean();
    result->covariance=0;
    for (int i=0;i<elements;i++){
        result->covariance+=(this->_data[i] - result->x_mean) * (other._data[i] - result->y_mean);
    }
    result->x_stddev=this->stddev();
    result->y_stddev=other.stddev();
    result->Pearson_R = result->covariance / (result->x_stddev * result->y_stddev);   

    // get r_squared (coefficient of determination) assuming linear dependence
    double x_mdev2_sum=0,y_mdev2_sum=0,slope_numerator=0;
    for (int i=0;i<elements;i++){
        x_mdev2_sum += std::pow(this->_data[i] - result->x_mean, 2); //=slope denominator
        y_mdev2_sum += std::pow(other._data[i] - result->y_mean, 2); //=SST
        slope_numerator += (this->_data[i] - result->x_mean) * (other._data[i] - result->y_mean);
    }
    result->SST = y_mdev2_sum;
    result->slope = slope_numerator / x_mdev2_sum;
    result->y_intercept = result->y_mean - result->slope * result->x_mean;

    // get regression line values
    for (int i=0; i<elements; i++){
        result->y_predict[i] = result->y_intercept + result->slope * this->_data[i];
        // get sum of squared (y-^y) //=SSE   
        result->SSE += std::pow(result->y_predict[i] - result->y_mean, 2);
        result->SSR += std::pow(other._data[i] - result->y_predict[i], 2);
    };
    result->r_squared = result->SSE/(std::fmax(y_mdev2_sum,__DBL_MIN__)); //=SSE/SST, equal to 1-SSR/SST
    
    // ANOVA
    double df_error = elements - 2;
    result->MSE = result->SSE / df_error;
    double df_regression = 1;
    result->MSR = result->SSR / df_regression;
    result->ANOVA_F = result->MSR / result->MSE;
    result->ANOVA_p = 1 - cdf<double>::F_distribution(result->ANOVA_F, df_regression, df_error);

    // Spearman correlation, assuming non-linear monotonic dependence
    auto rank_x = this->ranking();
    auto rank_y = other.ranking();
    double numerator=0;
    for (int i=0;i<elements;i++){
        numerator+=6*std::pow(rank_x._data[i] - rank_y._data[i],2);
    }
    result->Spearman_Rho=1-numerator/(elements*(std::pow(elements,2)-1));
    // test significance against null hypothesis
    double fisher_transform=0.5*std::log( (1 + result->Spearman_Rho) / (1 - result->Spearman_Rho) );
    result->z_score = sqrt((elements-3)/1.06)*fisher_transform;
    result->t_score = result->Spearman_Rho * std::sqrt((elements-2)/(1-std::pow(result->Spearman_Rho,2)));
    
    return std::move(*result);    
}

// performs linear regression with the source Aector as
// x_data and a second Array as corresponding the y_data;
// the results will be stored in a custom struct;
// make sure that both Arrays have the same number of
// elements (otherwise the surplus elements of the
// larger vector will be discarded)
template<typename T>
LinRegResult<T> Array<T>::regression_linear(const Array<T>& other) const {
    // create result struct
    int elements = std::min(this->_elements, other.get_elements());
    std::unique_ptr<LinRegResult<T>> result = std::make_unique<LinRegResult<T>>(elements);
    // get mean for x and y values
    for (int i = 0; i < elements; i++){
        result->x_mean += this->_data[i];
        result->y_mean += other._data[i];
    }
    result->x_mean /= elements;
    result->y_mean /= elements;
    // get sum of squared mean deviations
    double x_mdev2_sum = 0, y_mdev2_sum = 0, slope_num = 0;
    for (int n = 0; n < elements; n++){
        double x_mdev = this->_data[n] - result->x_mean;
        double y_mdev = other._data[n] - result->y_mean;
        x_mdev2_sum += x_mdev * x_mdev;
        y_mdev2_sum += y_mdev * y_mdev;
        slope_num += x_mdev * y_mdev;
        result->_y_regression[n] = result->y_intercept + result->_slope * this->_data[n];
        result->_residuals[n] = other._data[n] - result->_y_regression[n];
        result->SSR += result->_residuals[n] * result->_residuals[n];
    }
    // get slope
    result->_slope = slope_num / (x_mdev2_sum + std::numeric_limits<T>::min());
    // get y intercept
    result->y_intercept = result->y_mean - result->_slope * result->x_mean;
    // get r_squared
    result->SST = y_mdev2_sum;
    result->r_squared = 1 - result->SSR / (result->SST + std::numeric_limits<T>::min());

    return std::move(*result);
}


// performs polynomial regression (to the specified power)
// with the source Array as x_data and a second Array
// as the corresponding y_data;
// make sure that both vectors have the same number of
// elements (y_datawise the surplus elements of the
// larger vector will be discarded)
template<typename T>
PolyRegResult<T> Array<T>::regression_polynomial(const Array<T>& other, const int power) const {
    // create result struct
    int elements=std::min(this->get_elements(), other.get_elements());
    std::unique_ptr<PolyRegResult<T>> result = std::make_unique<PolyRegResult<T>>(elements, power);

    // Create matrix of x values raised to different powers
    Array<T> X = std::make_unique<Array<T>>({elements, power + 1});
    for (int i = 0; i < elements; i++) {
        for (int p = 1; p <= power; p++) {
            X->set(i,p,std::pow(this->_data[i],p));
        }
    }

    // Perform normal equation
    for (int i = 0; i <= power; i++) {
        for (int j = 0; j <= power; j++) {
            T sum = 0;
            for (int k = 0; k < elements; k++) {
                sum += X->get(k,i) * X->get(k,j);
            }
            X->set(i,j,sum);
        }
        result->coefficient[i] = 0;
        for (int k = 0; k < elements; k++) {
            result->coefficient[i] += other._data[k] * X->get(k,i);
        }
    }
    // Get R-squared value and other statistics
    result->y_mean = std::accumulate(other._data.begin(), other._data.end(), 0.0) / elements;
    result->x_mean = std::accumulate(this->_data.begin(), this->_data.end(), 0.0) / elements;
    for (int i = 0; i < elements; i++) {
        double y_pred = 0;
        for (int j = 0; j <= power; j++) {
            y_pred += result->coefficient[j] * pow(this->_data[i], j);
        }
        result->SS_res += std::pow(other._data[i] - y_pred, 2);
        result->SS_tot += std::pow(other._data[i] - result->y_mean, 2);
    }
    result->r_squared = 1 - result->SS_res / result->SS_tot;
    result->RSS = std::sqrt(result->SS_res / (elements - power - 1));
    result->MSE = result->RSS/elements;

    return std::move(*result);
}

// returns a histogram of the source Array raw data
// with the specified number of bars and returns the 
// result as type struct Histogram<T>';
// this function technically works with multidimensional
// Arrays but is mainly intended to be used with
// 1d Arrays ('vectors')
template <typename T>
HistogramResult<T> Array<T>::histogram(int bars) const {
    std::unique_ptr<HistogramResult<T>> histogram = std::make_unique<HistogramResult<T>>(bars);
    // get min and max value from sample
    histogram->min = this->data[0];
    histogram->max = this->data[0];
    for (int i=0;i<this->_elements;i++){
        histogram->min=std::fmin(histogram->min, this->data[i]);
        histogram->max=std::fmax(histogram->max, this->data[i]);
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
    for (int i=0;i<this->_elements;i++){
        histogram->bar[int((this->data[i]-histogram->min)/histogram->bar_width)].abs_count++;
    }

    // convert to relative values
    for (int i=0;i<bars;i++){
        histogram->bar[i].rel_count=histogram->bar[i].abs_count/this->_elements;
    }
    return std::move(*histogram);
}

// adds padding in all directions
template<typename T>
Array<T> Array<T>::padding(const int amount, const T value) const {
    std::vector<int> target_shape = this->dim_size;
    for (int d=0; d<this->dimensions; d++){
        target_shape[d] = this->dim_size[d] + 2*amount;
    }
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(target_shape);
    result->fill_values(value);
    std::vector<int> index;
    for (int i=0; i<this->data_elements; i++){
        index = this->get_index(i);
        for (int ii=0; ii<index.size(); ii++){
            index[ii] += amount;
        }
        result->set(index,this->data[i]);
    }
    return std::move(*result);
}

// adds padding to the beginning of each dimension
template<typename T>
Array<T> Array<T>::padding_pre(const int amount, const T value) const {
    std::vector<int> target_shape = this->dim_size;
    for (int d=0; d<this->dimensions; d++){
        target_shape[d] = this->dim_size[d] + amount;
    }
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(target_shape);
    result->fill_values(value);
    std::vector<int> index;
    for (int i=0; i<this->data_elements; i++){
        index = this->get_index(i);
        for (int ii=0; ii<index.size(); ii++){
            index[ii] += amount;
        }
        result->set(index,this->data[i]);
    }
    return std::move(*result);
}

// adds padding to the end of each dimension
template<typename T>
Array<T> Array<T>::padding_post(const int amount, const T value) const {
    std::vector<int> target_shape = this->dim_size;
    for (int d=0; d<this->dimensions; d++){
        target_shape[d] = this->dim_size[d] + amount;
    }
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(target_shape);
    result->fill_values(value);
    std::vector<int> index;
    for (int i=0; i<this->data_elements; i++){
        index = this->get_index(i);
        result->set(this->get_index(i),this->data[i]);
    }
    return std::move(*result);
}

// dissects the specified axis, resulting in a stack
// of sub-Array slices
template<typename T>
Array<Array<T>> Array<T>::dissect(int axis) const {
    int result_slices = this->dim_size[axis];
    Array<Array<T>> result = Array<Array<T>>({result_slices});
    // get slice shape and instantiate slices
    std::vector<int> slice_shape = this->dim_size;
    slice_shape.erase(slice_shape.begin()+axis);
    for (int i=0;i<result_slices;i++){
        result.data[i] = Array<T>(slice_shape);
    }
    // copy source data to corresponding slices
    for (int i=0;i<this->data_elements;i++){
        std::vector<int> index = this->get_index(i);
        index.erase(index.begin()+axis);
        result[this->get_index(i)[axis]].set(index, this->data[i]);
    }
    return result;
}

// moves a box of shape 'slider_shape' across the Array, with the step lengths
// specified as 'stride_shape'; from each slider position the pooled result of
// the slider box gets assigned to a corresponding element of the result array
template<typename T>
Array<T> Array<T>::pool(PoolMethod method, const std::initializer_list<int> slider_shape, const std::initializer_list<int> stride_shape) const {
    std::vector<int> slider_shape_vec = initlist_to_vector(slider_shape);
    std::vector<int> stride_shape_vec = initlist_to_vector(stride_shape);
    // confirm valid slider shape
    if (int(slider_shape.size()) != this->dimensions){
        Log::log(LOG_LEVEL_WARNING,
            "slider shape for pooling operation must have same number of dimensions ",
            "as the array it is acting upon -> auto-adjusting slider shape to fit");
        while (int(slider_shape_vec.size()) < this->dimensions){
            slider_shape_vec.push_back(1);
        }
        while (int(slider_shape_vec.size()) > this->dimensions){
            slider_shape_vec.pop_back();
        }
    }
    // confirm valid stride shape
    if (int(stride_shape.size()) != this->dimensions){
        Log::log(LOG_LEVEL_WARNING,
            "stride shape for pooling operation must have same number of dimensions ",
            "as the array it is acting upon -> auto-adjusting stride shape to fit");
        while (int(stride_shape_vec.size()) < this->dimensions){
            stride_shape_vec.push_back(1);
        }
        while (int(stride_shape_vec.size()) > this->dimensions){
            stride_shape_vec.pop_back();
        }            
    }
    // create source index
    std::vector<int> index_source(this->dimensions);
    // get result shape
    std::vector<int> result_shape(this->dimensions,1);
    for (int d=0;d<this->dimensions;d++){
        result_shape[d] = this->dim_size[d] / std::max(1,stride_shape_vec[d]);
    }
    // create result array
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(result_shape);
    std::vector<int> index_result(result->get_dimensions());
    // create a sliding box for pooling
    Array<double> slider = Array<double>(slider_shape_vec);
    std::vector<int> index_slider(slider.get_dimensions());
    std::vector<int> index_combined(this->dimensions);
    // iterate over result
    int result_elements = result->get_elements();
    for (int j=0;j<result_elements;j++){
        // get associated result index
        index_result = result->get_index(j);
        // get corresponding source index at slider position
        for (int d=0;d<this->dimensions;d++){
            index_source[d] = index_result[d] * stride_shape_vec[d];
        }
        // iterate over elements of the slider
        for (int n=0;n<slider.get_elements();n++){
            // update multidimensional index of the slider element
            index_slider = slider.get_index(n);
            // get combined index and check if it fits within source boundaries            
            bool index_ok=true;
            for (int d=0;d<int(index_combined.size());d++){
                index_combined[d] = index_source[d] + index_slider[d];
                if (index_combined[d]>=this->dim_size[d]){
                    index_ok=false;
                    break;
                }                
            }
            // assing slider value from the element with the index of the sum of index_i+index_slider
            if (index_ok){
                slider.set(n, this->get(index_combined));
            }
        }
        switch (method) {
            case PoolMethod::MAX: result->set(j,slider.max()); break;
            case PoolMethod::MAXABS: result->set(j,slider.maxabs()); break;
            case PoolMethod::MEAN: result->set(j,slider.mean()); break;
            case PoolMethod::MIN: result->set(j,slider.min()); break;
            case PoolMethod::MEDIAN: result->set(j,slider.median()); break;
            case PoolMethod::MODE: result->set(j,slider.mode()); break;
        }
    }
    return std::move(*result);
}

// moves a box of shape 'slider_shape' across the Array, with the step lengths
// specified as 'stride_shape'; from each slider position the pooled result of
// the slider box gets assigned to a corresponding element of the result array
template<typename T>
Array<T> Array<T>::pool(PoolMethod method, const std::vector<int> slider_shape, const std::vector<int> stride_shape) const {
    std::vector<int> slider_shape_vec = slider_shape;
    std::vector<int> stride_shape_vec = stride_shape;
    // confirm valid slider shape
    if (int(slider_shape.size()) != this->dimensions){
        Log::log(LOG_LEVEL_WARNING,
            "slider shape for pooling operation must have same number of dimensions ",
            "as the array it is acting upon -> auto-adjusting slider shape to fit");
        while (int(slider_shape_vec.size()) < this->dimensions){
            slider_shape_vec.push_back(1);
        }
        while (int(slider_shape_vec.size()) > this->dimensions){
            slider_shape_vec.pop_back();
        }
    }
    // confirm valid stride shape
    if (int(stride_shape.size()) != this->dimensions){
        Log::log(LOG_LEVEL_WARNING,
            "stride shape for pooling operation must have same number of dimensions ",
            "as the array it is acting upon -> auto-adjusting stride shape to fit");
        while (int(stride_shape_vec.size()) < this->dimensions){
            stride_shape_vec.push_back(1);
        }
        while (int(stride_shape_vec.size()) > this->dimensions){
            stride_shape_vec.pop_back();
        }            
    }
    // create source index
    std::vector<int> index_source(this->dimensions);
    // get result shape
    std::vector<int> result_shape(this->dimensions,1);
    for (int d=0;d<this->dimensions;d++){
        result_shape[d] = this->dim_size[d] / std::max(1,stride_shape_vec[d]);
    }
    // create result array
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(result_shape);
    std::vector<int> index_result(result->get_dimensions());
    // create a sliding box for pooling
    Array<double> slider = Array<double>(slider_shape_vec);
    std::vector<int> index_slider(slider.get_dimensions());
    std::vector<int> index_combined(this->dimensions);
    // iterate over result
    int result_elements = result->get_elements();
    for (int j=0;j<result_elements;j++){
        // get associated result index
        index_result = result->get_index(j);
        // get corresponding source index at slider position
        for (int d=0;d<this->dimensions;d++){
            index_source[d] = index_result[d] * stride_shape_vec[d];
        }
        // iterate over elements of the slider
        for (int n=0;n<slider.get_elements();n++){
            // update multidimensional index of the slider element
            index_slider = slider.get_index(n);
            // get combined index and check if it fits within source boundaries            
            bool index_ok=true;
            for (int d=0;d<int(index_combined.size());d++){
                index_combined[d] = index_source[d] + index_slider[d];
                if (index_combined[d]>=this->dim_size[d]){
                    index_ok=false;
                    break;
                }                
            }
            // assing slider value from the element with the index of the sum of index_i+index_slider
            if (index_ok){
                slider.set(n, this->get(index_combined));
            }
        }
        switch (method) {
            case PoolMethod::MAX: result->set(j,slider.max()); break;
            case PoolMethod::MAXABS: result->set(j,slider.maxabs()); break;
            case PoolMethod::MEAN: result->set(j,slider.mean()); break;
            case PoolMethod::MIN: result->set(j,slider.min()); break;
            case PoolMethod::MEDIAN: result->set(j,slider.median()); break;
            case PoolMethod::MODE: result->set(j,slider.mode()); break;
        }
    }
    return std::move(*result);
}

template<typename T>
Array<T> Array<T>::convolution(const Array<T>& filter, bool padding) const {
    // declare result Array
    std::unique_ptr<Array<T>> result;

    // check valid filter dimensions
    if (filter.dimensions>this->dimensions){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'Array<T> Array<T>::convolution(const Array<T>& filter)': ",
            "filter can't have more dimensions then the array; filter has shape ", filter.get_shapestring(),
            ", source array has shape ", this->get_shapestring());
    }
    for (int d=0;d<filter.dimensions;d++){
        if (filter.get_size(d)>this->dim_size[d]){
            Log::log(LOG_LEVEL_WARNING,
                "invalid usage of method 'Array<T> Array<T>::convolution(const Array<T>& filter)': ",
                "the source Array has shape ", this->get_shapestring(), " whilst the filter has shape ",
                filter.get_shapestring(), ", therefore the filter has size ", filter.get_size(d),
                " in dimension ", d, ", but the Array has only size ", this->dim_size[d],
                " in dimension ", d);
        }
    }
    if (filter.dimensions<this->dimensions-1){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'Array<T> Array<T>::convolution(const Array<T>& filter)': ",
            "Array is ", this->dimensions, "-dimensional, therefore only filters with ",
            this->dimensions-1, " or ", this->dimensions, " are allowed");
    }
    // 1d convolution for vectors
    if (this->dimensions == 1){
        try {
            int filter_width = filter.get_size(0);
            result = std::make_unique<Array<T>>(this->data_elements - ((filter_width-1)*!padding));
            result->fill_zeros();
            for (int i=0; i<result->get_elements(); i++){
                for (int ii=0; ii<filter_width; ii++){
                    if (i+ii>=this->data_elements){break;}
                    result->data[i] += this->data[i+ii] * filter.data[ii];
                }
            }
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'Array<T> Array<T>::convolution(const Array<T>& filter)' has failed ",
                "to calculate convolution operation for the 1d source Array");
        }
    }
    // 2d convolution for matrices
    if (this->dimensions==2 && filter.dimensions==2){
        try {
            int filter_height = filter.get_size(0);
            int filter_width = filter.get_size(1);
            std::initializer_list<int> result_shape = {this->dim_size[0] - (filter_height-1)*!padding,
                                                       this->dim_size[1] - (filter_width-1)*!padding};
            result = std::make_unique<Array<T>>(result_shape);
            result->fill_zeros();
            for (int row=0;row<result->get_size(0);row++){
                for (int col=0; col<result->get_size(1);col++){
                    for (int filter_row=0;filter_row<filter_height;filter_row++){
                        if (row+filter_row>=this->dim_size[0]){break;}
                        for (int filter_col=0;filter_col<filter_width;filter_col++){
                            if (col+filter_col>=this->dim_size[1]){break;}
                            result->data[this->get_element({row,col})] += this->data[this->get_element({row+filter_row,col+filter_col})] * filter.get({filter_row,filter_col});
                        }
                    }
                }
            }
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'Array<T> Array<T>::convolution(const Array<T>& filter)' has failed ",
                "to calculate convolution operation for the 2d source Array with 2d filter");            
        }
    }
    // 2d convoltion for 3d arrays (with 3d filters)
    if (this->dimensions==3 && filter.dimensions==3){
        try {
            int filter_height = filter.get_size(0);
            int filter_width = filter.get_size(1);
            int filter_depth = filter.get_size(2);            
            std::initializer_list<int> result_shape = {this->dim_size[0] - (filter_height-1)*!padding,
                                                       this->dim_size[1] - (filter_width-1)*!padding};
            result= std::make_unique<Array<T>>(result_shape);
            result->fill_zeros();            
            for (int row=0;row<result->get_size(0);row++){
                for (int col=0; col<result->get_size(1);col++){
                    for (int filter_row=0;filter_row<filter_height;filter_row++){
                        if (row+filter_row>=this->dim_size[0]){break;}
                        for (int filter_col=0;filter_col<filter_width;filter_col++){
                            if (col+filter_col>=this->dim_size[1]){break;}
                            for (int filter_channel=0;filter_channel<filter_depth && filter_channel<this->dim_size[2];filter_channel++){
                                result->data[this->get_element({row,col})] += this->data[this->get_element({row+filter_row,col+filter_col,filter_channel})] * filter.get({filter_row,filter_col,filter_channel});
                            }
                        }
                    }
                }
            }
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'Array<T> Array<T>::convolution(const Array<T>& filter)' has failed ",
                "to calculate convolution operation for the 3d source Array with 3d filter");             
        }        
    }

    // nd convolution
    if (filter.dimensions == this->dimensions-1){
        try {
            std::vector<int> result_shape;
            for (int d=0;d<this->dimensions-1;d++){
                result_shape.push_back(this->dim_size[d]- (filter.get_size(d)-1)*padding);
            }
            result_shape.push_back(filter.get_size(filter.dimensions-1));
            result = std::make_unique<Array<T>>(result_shape);
            std::vector<int> source_index(this->dimensions);
            std::vector<int> filter_index(filter.dimensions);
            std::vector<int> combined_index(this->dimensions);
            for (int i=0;i<result->get_elements();i++){
                source_index = this->get_index(i);
                combined_index = source_index;
                // assign dotproduct of filter and corresponding array elements to result array
                for (int ii=0;ii<filter.data_elements;ii++){
                    filter_index = filter.get_index(ii);
                    bool index_okay=true;
                    for (int d=0;d<filter.dimensions;d++){
                        combined_index[d] += filter_index[d];
                        if (combined_index[d]>=result->get_size(d)){
                            index_okay=false;
                            break;
                        }
                    }
                    if (index_okay){
                        result->data[i] += this->get(combined_index) * filter.get(filter_index);
                    }
                }
            }
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'Array<T> Array<T>::convolution(const Array<T>& filter)' has failed ",
                "to calculate convolution operation for the ", this->dimensions, "d source ",
                "Array with a ", filter.get_dimensions(), "d filter");                
        }
    }

    return std::move(*result);
}

template<typename T>
Array<int> Array<T>::get_convolution_shape(Array<int>& filter_shape, const bool padding) const {

    // check valid filter dimensions
    if (filter_shape.get_dimensions()>this->dimensions){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method ",
            "'std::vector<int> Array<T>::get_convolution_shape(std::vector<int>& filter_shape, const bool padding=false)': ",
            "filter can't have more dimensions then the array; filter has shape ", filter_shape.get_shapestring(),
            ", source array has shape ", this->get_shapestring());
    }
    for (int d=0;d<filter_shape.get_dimensions();d++){
        if (filter_shape.get_size(d)>this->get_size(d)){
            Log::log(LOG_LEVEL_WARNING,
                "invalid usage of method ",
                "'std::vector<int> Array<T>::get_convolution_shape(std::vector<int>& filter_shape, const bool padding=false)': ",
                "the source Array has shape ", this->get_shapestring(), " whilst the filter has shape ",
                filter_shape.get_shapestring(), ", therefore the filter has size ", filter_shape.get_size(d),
                " in dimension ", d, ", but the Array has only size ", this->get_size(d),
                " in dimension ", d);
        }
    }
    if (filter_shape.get_dimensions()<this->dimensions-1){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method ",
            "'std::vector<int> Array<T>::get_convolution_shape(std::vector<int>& filter_shape, const bool padding=false)': ",
            "Array is ", this->dimensions, "-dimensional, therefore only filters with ",
            this->dimensions-1, " or ", this->dimensions, " are allowed");
    }

    // return original shape if padding applies
    if (padding) {
        Array<int> result(this->dimensions);
        for (int d=0; d<this->dimensions; d++) {
            result[d] = this->get_size(d);
        }
        return result;
    }

    // 1d convolution for 1d arrays
    if (this->dimensions == 1){
        Array<int> result(1);
        try {
            int filter_width = filter_shape.get_size(0);
            result[0] = this->data_elements - filter_width;
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'std::vector<int> Array<T>::get_convolution_shape(std::vector<int>& filter_shape, const bool padding=false)' ",
                "has failed with source Array shape ", this->get_shapestring(), " and filter shape ", filter_shape.get_shapestring());
        }
        return result;
    }
    // 2d convolution for 2d arrays
    if (this->dimensions==2 && filter_shape.get_dimensions()==2){
        Array<int> result(2);
        try {
            result[0] = this->dim_size[0]-filter_shape.get_size(0);
            result[1] = this->dim_size[1]-filter_shape.get_size(1);
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'std::vector<int> Array<T>::get_convolution_shape(std::vector<int>& filter_shape, const bool padding=false)' ",
                "has failed with source Array shape ", this->get_shapestring(), " and filter shape ", filter_shape.get_shapestring());            
        }
        return result;
    }
    // 2d convoltion for 3d arrays
    if (this->dimensions==3 && filter_shape.get_dimensions()==3){
        Array<int> result(2);
        try {
            result[0] = this->dim_size[0]-filter_shape.get_size(0);
            result[1] = this->dim_size[1]-filter_shape.get_size(1);
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'std::vector<int> Array<T>::get_convolution_shape(std::vector<int>& filter_shape, const bool padding=false)' ",
                "has failed with source Array shape ", this->get_shapestring(), " and filter shape ", filter_shape.get_shapestring());                        
        }     
        return result;   
    }

    // nd convolution
    if (filter_shape.get_dimensions() == this->dimensions-1){
        Array<int> result;
        try {
            for (int d=0;d<this->dimensions-1;d++){
                result.push_back(this->dim_size[d]-filter_shape.get_size(d));
            }
            result.push_back(filter_shape.get_size(filter_shape.get_dimensions()-1));
            return result;
        }
        catch (...) {
            Log::log(LOG_LEVEL_WARNING,
                "method 'std::vector<int> Array<T>::get_convolution_shape(std::vector<int>& filter_shape, const bool padding=false)' ",
                "has failed with source Array shape ", this->get_shapestring(), " and filter shape ", filter_shape.get_shapestring());
        }
        return result;
    }
    // default return statement if the conditions above all fail
    return this->get_shape();
}

template<typename T>
std::vector<int> Array<T>::get_convolution_shape(std::vector<int>& filter_shape, const bool padding) const {
    // convert filter shape from type std::vector to Array
    Array<int> filter_shape_Array(filter_shape.size());
    for (int d=0; d<int(filter_shape.size()); d++){
        filter_shape_Array[d] = filter_shape[d];
    }
    // get result
    Array<int> result_Array = get_convolution_shape(filter_shape_Array, padding);
    // convert back into std::vector
    std::vector<int> result_vector(result_Array.get_dimensions());
    for (int d=0; d<result_Array.get_dimensions(); d++){
        result_vector[d] = result_Array[d];
    }
    // return result
    return result_vector;
}

// +=================================+   
// | Constructors & Destructors      |
// +=================================+

// constructor for multi-dimensional array:
// pass dimension size (elements per dimension)
// as an initializer_list, e.g. {3,4,4}
template<typename T>
Array<T>::Array(const std::initializer_list<int>& shape) {
    // set dimensions
    this->dimensions = (int)shape.size();
    // check if init_list empty
    if (this->dimensions==0){
        this->data_elements=0;
        this->capacity=0;
        return;
    }
    // store size of individual dimensions in std::vector<int> size member variable
    auto iterator=shape.begin();
    int n=0;
    this->dim_size.resize(this->dimensions);
    for (; iterator!=shape.end();n++, iterator++){
        this->dim_size[n]=*iterator;
    }
    // calculate the subspace size for each dimension
    int totalsize = 1;
    this->subspace_size.resize(this->dimensions);
    for (int i = 0; i < this->dimensions; i++) {
        totalsize *= this->dim_size[i];
        this->subspace_size[i] = totalsize;
    }
    this->data_elements = totalsize;
    // set reserve capacity for 1d Arrays
    this->capacity = this->data_elements;
    if (this->dimensions==1){
        this->capacity = (1.0f+this->_reserve) * this->data_elements;
    }
    // initialize data buffer
    this->data = std::make_unique<T[]>(this->capacity);
};

// constructor for multidimensional array:
// pass dimension size (elements per dimension)
// as type std::vector<int>
template<typename T>
Array<T>::Array(const std::vector<int>& shape) {
    // check if init_list empty
    if (shape.size()==0){return;}
    // set dimensions
    this->dimensions = shape.size();
    // calculate subspace size for each dimension
    int totalsize = 1;
    this->subspace_size.resize(this->dimensions);
    this->dim_size.resize(this->dimensions);
    for (int i = 0; i < this->dimensions; i++) {
        this->subspace_size[i] = totalsize;
        this->dim_size[i]=shape[i];
        totalsize *= this->dim_size[i];
    }
    this->data_elements = totalsize;
    // set reserve capacity for 1d Arrays
    this->capacity = this->data_elements;
    if (this->dimensions==1){
        this->capacity = (1.0f+this->_reserve) * this->data_elements;
    }
    // initialize data buffer
    this->data = std::make_unique<T[]>(this->capacity);
}

// constructor for 1d Arrays
template<typename T>
Array<T>::Array(const int elements) {
    Array({elements});
}

// Array move constructor
template<typename T>
Array<T>::Array(Array&& other) noexcept {
    this->data_elements = other.get_elements();
    this->capacity = other.get_capacity();
    this->dimensions = other.dimensions;    
    this->data = std::move(other.data);
    this->dim_size = std::move(other.dim_size);
    this->subspace_size = std::move(other.subspace_size);
    other.data.reset();
}

// Array copy constructor
template<typename T>
Array<T>::Array(Array& other) {
    this->data_elements = other.get_elements();
    this->capacity = other.get_capacity();    
    this->dimensions = other.dimensions;
    this->dim_size = other.dim_size;
    this->subspace_size = other.subspace_size;
    this->data = std::make_unique<T[]>(data_elements);
    std::copy(other.data.get(), other.data.get() + other.get_elements(), this->data.get());
}  

// virtual destructor
template<typename T>
Array<T>::~Array(){
    // empty
}

// +=================================+   
// | Private Member Functions        |
// +=================================+

// check whether this array and a second array
// match with regard to their number of dimensions
// and their size per individual dimensions
template<typename T>
bool Array<T>::equalsize(const Array<T>& other) const {
    if (this->dimensions!=other.get_dimensions()){
        return false;
    }
    for (int n=0; n<this->dimensions; n++){
        if (this->dim_size[n]!=other.get_size(n)){
            return false;
        }
    }
    return true;
}    

// private helper function:
// change the size of a simple C-style array via its std::unique_ptr<T[]>
// by allocating new memory and copying the previous data to the new location
template<typename T>
void Array<T>::resize_array(std::unique_ptr<T[]>& arr, const int oldSize, const int newSize, T init_value) {
    // Create a new array with the desired size
    std::unique_ptr<T[]> newArr = std::make_unique<T[]>(newSize);
    // Copy the elements from the old array to the new array
    for (int i = 0; i < newSize; i++) {
        newArr[i] = i<oldSize ? arr[i] : init_value;
    }
    // Assign the new array to the old array variable
    arr = std::move(newArr);
}

// +=================================+   
// | Dynamic Handling of Arrays      |
// +=================================+

// push back 1 element into the 1d Array
// returns the resulting total number of elements
template<typename T>
int Array<T>::push_back(const T init_value){
    if (this->dimensions != 1){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'int Array<T>::push_back(const T value)': ",
            "can only be used with 1d Arrays but this Array is ", this->dimensions, "d!");
        return this->data_elements;
    }
    this->data_elements++;
    this->dim_size[0]++;
    if (this->data_elements>this->capacity){
        this->capacity=int(this->data_elements*(1.0+this->_reserve));
        resize_array(this->data, this->data_elements-1, this->capacity);
    }
    this->data[this->data_elements-1]=init_value;
    return this->data_elements;
}

// resize the array in the specified dimension (default: dimension 0)
template<typename T>
void Array<T>::resize(const int newsize, int dimension, T init_value){
    int dimensions_old = this->dimensions;
    int dimensions_new = std::max(dimensions_old, dimension+1);
    // make a copy of the original Array
    Array<T> temp_copy = *this;
    // set up the new dimensions
    this->dim_size.resize(dimensions_new);
    for (int d=0; d<dimensions_new; d++){
        if (d==dimensions){
            this->dim_size[d] = newsize;
        }
        else if (d>=dimensions_old && d!=dimension){
            this->dim_size[d] = 1;
        }
        else {
            // keep the size of the given dimension as it is
        }
    }
    this->dimensions = dimensions_new;
    // get number of data_elements with new size
    int oldSize = this->data_elements;
    this->data_elements = this->dim_size[0];
    for (int d=1; d<this->dimensions; d++){
        this->data_elements *= this->dim_size[d];
    }    
    // reserve memory for the new elements
    if (this->data_elements>this->capacity){
        this->capacity=int(this->data_elements*(1.0+this->_reserve));
        resize_array(this->data, oldSize, this->capacity);
    }    
    // transfer data from the temporary copy to the updated array
    std::vector<int> index(dimensions_new);
    for (int i=0; i<this->data_elements; i++){
        index = this->get_index(i);
        if (temp_copy.index_isvalid(index)){
            this->data[i] = temp_copy.get(index);
        }
        else {
            this->data[i] = init_value;
        }
    }    
}

// grows the Array size of the specified dimension (default: dimension index 0)
// by the specified number of additional(!) elements and initializes these new elements
// to the specified value (default=0);
// will only re-allocate memory if the new size exceeds
// the capacity; returns the new total number of elements
template<typename T>
int Array<T>::grow(const int additional_elements, int dimension, T init_value){
    this->resize(this->dim_size[dimension]+additional_elements, dimension, init_value);
    return this->data_elements;
}

// shrins the Array size of the specified dimension (default: dimension index 0)
// by the specified number of elements;
// returns the new total number of elements
template<typename T>
int Array<T>::shrink(const int remove_amount, int dimension){
    this->resize(this->dim_size[dimension]-remove_amount, dimension);
    return this->data_elements;
}

// pop 1 element from the end of the Array
template<typename T>
T Array<T>::pop_last(){
    if (this->dimensions != 1){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'T Array<T>::pop_last()': ",
            "is meant to be used only for 1d Arrays but this Array is ", this->dimensions, "d!",
            " -> will return the last element of the flattend array without removing anything");
        return this->data[this->data_elements-1];
    }
    this->data_elements--;
    this->dim_size[0]--;
    return this->data[this->data_elements];
}

// pop 1 element from the beginning of the Array
template<typename T>
T Array<T>::pop_first(){
    if (this->dimensions != 1){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'T Array<T>::pop_first()': ",
            "is meant to be used only for 1d Arrays but this Array is ", this->dimensions, "d!",
            " -> will return the first element of the flattend array without removing anything");
        return this->data[0];
    }    
    T temp = this->data[0];
    // reassign pointer to position of the raw pointer to the element at index 1
    this->data = std::unique_ptr<T[]>(this->data.release() + 1, std::default_delete<T[]>());
    this->data_elements--;
    this->dim_size[0]--;
    return temp;
}

// removes the element of the given index and returns its value
template<typename T>
T Array<T>::erase(const int index){
    T result = this->data[index];
    for (int i=index;i<this->data_elements-1;i++){
        this->data[i] = this->data[i+1];
    }
    this->data_elements--;
    this->dim_size[0]--;
    return result;
}

// helper method to confirm valid multidimensional index
template<typename T>
bool Array<T>::index_isvalid(const std::vector<int>& index) const {
    if ((int)index.size() != this->dimensions) return false;
    for (int d=0; d<this->dimensions; d++){
        if (index[d]<0 || index[d]>=this->dim_size[d]) return false;
    }
    return true;
}

// helper method to confirm valid 1d index
template<typename T>
bool Array<T>::index_isvalid(const int index) const {
    return this->dimensions==1 && index>0 && index<this->data_elements ? true : false;
}

// +=================================+   
// | 1d Array Conversion             |
// +=================================+

// returns the vector as a single column matrix, 
// i.e. as transposition with data in rows (single column)
template<typename T>
Array<T> Array<T>::transpose() const {
    if (this->dimensions==1){
        std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>({this->data_elements,1});
        for (int i=0; i<this->data_elements; i++){
            result->set({i,0}, this->data[i]);
        }
        return std::move(*result);
    }
    else {
        // create a new 2d Array with swapped dimensions
        std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>({this->dim_size[1], this->dim_size[0]});

        for(int i = 0; i < this->dim_size[0]; i++){
            for(int j = 0; j < this->dim_size[1]; j++){
                // swap indices and copy element to result
                result->set({j,i}, this->get({i,j}));
            }
        }
        return std::move(*result);
    }    
}

// returns a reverse order copy of the
// original 1d Array<T>
template<typename T>
Array<T> Array<T>::reverse() const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>({this->data_elements});
    for (int i=0;i<this->data_elements;i++){
        result->data[i] = this->data[this->data_elements-1-i];
    }
    return std::move(*result);
}


// +=================================+   
// | 1d Array Sample Analysis         |
// +=================================+

// returns a vector of integers that represent
// a ranking of the source vector via bubble sorting
// the ranks
template<typename T>
Array<int> Array<T>::ranking() const {
    // initialize ranks
    std::unique_ptr<Array<int>> rank = std::make_unique<Array<int>>(this->data_elements);
    rank->fill_range(0,1);
    // ranking loop
    bool ranking_completed=false;
    while (!ranking_completed){
        ranking_completed=true; //=let's assume this until a wrong order is found
        for (int i=0;i<this->data_elements-1;i++){
            // pairwise comparison:
            if ((this->data[i]>this->data[i+1] && rank->data[i]<rank->data[i+1]) ||
                (this->data[i]<this->data[i+1] && rank->data[i]>rank->data[i+1])){
                ranking_completed=false;
                // swap ranks
                std::swap(rank->data[i+1], rank->data[i]);
            }           
        }
    }  
    return std::move(*rank);
}

// returns an exponentially smoothed copy of the source vector,
// e.g. for time series
template<typename T>
Array<T> Array<T>::exponential_smoothing(bool as_series) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->data_elements);
    double alpha=2/(this->data_elements);

    if (as_series){
        result->set(this->data_elements-1, this->mean());
        for (int n=this->data_elements-2; n>=0; n--){
            result->set(n, alpha*(this->data[n] - result->get(n+1)) + result->get(n+1));
        }     
    }
    else{
        result->set(0, this->mean());
        for (int n=1; n<this->data_elements; n++){
            result->set(n, alpha*(this->data[n]-result->get(n-1)) + result->get(n-1));
        }
    }
    return std::move(*result);
}

// returns the weighted average of a sample Array,
// e.g. for time series data
template <typename T>
double Array<T>::weighted_average(bool as_series) const {
    double weight=0, weight_sum=0, sum=0;
    if (!as_series){ //=indexing from zero, lower index means lower attributed weight
        for (int n=0;n<this->data_elements;n++){
            weight++;
            weight_sum+=weight;
            sum+=weight*this->data[n];
        }
        return sum/(this->data_elements*weight_sum);
    }
    else {
        for (int n=this->data_elements-2;n>=0;n--) {
            weight++;
            weight_sum+=weight;
            sum+=weight*this->data[n];
        }
        return sum/(this->data_elements*weight_sum);
    }   
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
// The method for differencing is set to first order integer by default,
// by can be changed to other methods via the arguments
template<typename T>
double Array<T>::Dickey_Fuller(DIFFERENCING method, double degree, double fract_exponent) const {
    // make two copies
    Array<T> data_copy(this->data_elements);
    Array<T> stat_copy(this->data_elements);    
    for (int i=0;i<this->data_elements;i++){
        data_copy.data[i] = this->data[i];
        stat_copy.data[i] = this->data[i];
    }
    // make one of the copies stationary
    stat_copy.stationary(method, degree, fract_exponent);
    // pop the first element of the other copy to make their size match again
    data_copy.pop_first();
    // correlate the raw copy with the corresponding stationary transformation
    double Pearson_R = data_copy.correlation(stat_copy)->Pearson_R;
    // calculate result
    return Pearson_R*std::sqrt((double)(this->data_elements-1)/(1-std::pow(Pearson_R,2)));  
}

// takes the source vector and another vector (passed as parameter) and
// performs an Engle-Granger test in order to test the given numeric sample
// for cointegration, i.e. checking series data for a long-term relationship.
// The test was proposed by Clive Granger and Robert Engle in 1987.
// If the returned p-value is less than a chosen significance level (typically 0.05),
// it suggests that the two time series are cointegrated and have a long-term relationship.
// Make sure that both Array<T> have the same number of elements! Otherwise the surplus
// elements of the larger vector will be clipped and the result isn't meaningful;
template<typename T>
double Array<T>::Engle_Granger(const Array<T>& other) const {
    // make copies of the x+y source data
    int elements = std::fmin(this->data_elements, other.get_elements());
    Array<T> xdata(elements);
    Array<T> ydata(elements);
    for (int i=0;i<elements;i++){
        xdata.data[i] = this->data[i];
        ydata.data[i] = other.data[i];
    }
    // make the data stationary
    std::unique_ptr<Array<T>> x_stat = xdata.stationary();
    std::unique_ptr<Array<T>> y_stat = ydata.stationary();
    // perform linear regression on x versus y
    std::unique_ptr<typename Array<T>::LinReg> regr_result = x_stat->linear_regression(y_stat);
    // perform a Dickey_Fuller test on the residuals
    Array<double> residuals(elements);
    residuals.data = regr_result->residuals;
    return residuals.Dickey_Fuller();
}      

// returns a stationary transformation of the vector data,
// e.g. for time series data;
// available differencing methods:
//    integer=1,
//    logreturn=2,
//    fractional=3,
//    deltamean=4,
//    original=5
template<typename T>
Array<T> Array<T>::stationary(DIFFERENCING method, double degree, double fract_exponent) const {
    // make a copy
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->data_elements);
    for (int i = 0; i < this->data_elements; i++) {
        result->data[i] = this->data[i];
    }
    if (method == integer) {
        for (int d = 1; d <= (int)degree; d++) { //=loop allows for higher order differencing
            for (int t = this->data_elements - 1; t > 0; t--) {
                result->data[t] -= result->data[t - 1];
            }
            // Remove the first element from the unique_ptr
            std::unique_ptr<T[]> newdata = std::make_unique<T[]>(this->data_elements - 1);
            for (int i = 0; i < this->data_elements - 1; i++) {
                newdata[i] = result->data[i + 1];
            }
            result->data = std::move(newdata);
            result->_elements--;
        }
    }
    if (method == logreturn) {
        for (int d = 1; d <= round(degree); d++) { //=loop allows for higher order differencing
            for (int t = this->data_elements - 1; t > 0; t--) {
                if (result->data[t - 1] != 0) {
                    result->data[t] = log(std::numeric_limits<T>::min() + std::fabs(result->data[t] / (result->data[t - 1] + std::numeric_limits<T>::min())));
                }
            }
            // for each "degree":
            // pop the first element from the unique_ptr
            std::unique_ptr<T[]> newdata = std::make_unique<T[]>(this->data_elements - 1);
            for (int i = 0; i < this->data_elements - 1; i++) {
                newdata[i] = result->data[i + 1];
            }
            result->data = std::move(newdata);
            result->_elements--;
        }
    }
    if (method == fractional) {
        for (int t = result->get_size() - 1; t > 0; t--) {
            if (result->data[t - 1] != 0) {
                double stat = log(std::numeric_limits<T>::min() + fabs(this->data[t] / this->data[t - 1])); //note: DBL_MIN and fabs are used to avoid log(x<=0)
                double non_stat = log(fabs(this->data[t]) + std::numeric_limits<T>::min());
                result->data[t] = degree * stat + pow((1 - degree), fract_exponent) * non_stat;
            }
        }
        // Remove the first element from the unique_ptr
        std::unique_ptr<T[]> newdata(new T[this->data_elements - 1]);
        for (int i = 0; i < this->data_elements - 1; i++) {
            newdata[i] = result->data[i + 1];
        }
        result->data = std::move(newdata);
        result->_elements--;
    }
    if (method==deltamean){
        double sum=0;
        for (int i=0;i<this->data_elements;i++){
            sum+=this->data[i];
        }
        double x_mean=sum/this->data_elements;
        for (int t=this->data_elements-1;t>0;t--){
            result->data[t]-=x_mean;
        }
        result->_elements--;
        for (int i = 0; i < result->_elements; i++) {
            result->data[i] = result->data[i + 1];
        }
    }
    return std::move(*result);
}


// sorts the values of the vector via pairwise comparison
// default: ascending order;
// set 'false' flag for sorting in reverse order
template<typename T>
Array<T> Array<T>::sort(bool ascending) const {
    // make a copy
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i] = this->data[i];
    }
    bool completed=false;
    while (!completed){
        completed=true; //let's assume this until proven otherwise
        for (int i=0;i<this->data_elements-1;i++){
            if(ascending){
                if (result->data[i] > result->data[i+1]){
                    completed=false;
                    double temp=result->data[i];
                    result->data[i] = result->data[i+1];
                    result->data[i+1] = temp;
                }
            }
            else{
                if (result->data[i] < result->data[i+1]){
                    completed=false;
                    double temp=result->data[i];
                    result->data[i] = result->data[i+1];
                    result->data[i+1] = temp;
                }
            }
        }
    }
    return std::move(*result);
}

// returns a randomly shuffled copy of the vector
template<typename T>
Array<T> Array<T>::shuffle() const {
    // make a copy
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->data_elements);
    for (int i=0;i<this->data_elements;i++){
        result->data[i] = this->data[i];
    }
    // iterate over vector elements and find a random second element to swap places
    for (int i=0;i<this->data_elements;i++){
        int new_position=std::floor(Random<double>::uniform()*this->data_elements);
        T temp=this->data[new_position];
        result->data[new_position] = result->data[i];
        result->data[i] = temp;
    }
    return std::move(*result);
}

// return the covariance of the linear relation
// of the source vector versus a second vector
template<typename T>
double Array<T>::covariance(const Array<T>& other) const {
    if (this->data_elements != other.get_elements()) {
        Log::log(LOG_LEVEL_WARNING, "Invalid use of method Array<T>::covariance(); both vectors should have the same number of elements");
    }
    int elements=std::min(this->data_elements, other.get_elements());
    double mean_this = this->mean();
    double mean_other = other.mean();
    double cov = 0;
    for (int i = 0; i < elements; i++) {
        cov += (this->data[i] - mean_this) * (other.data[i] - mean_other);
    }
    cov /= elements;
    return cov;
}



// Array binning, i.e. quantisizing into n bins,
// each represented by the mean of the values inside that bin;
// returning the result as pointer to a new Array of size n
template<typename T>
Array<T> Array<T>::binning(const int bins) const {
    if (this->data_elements == 0) {
        Log::log(LOG_LEVEL_WARNING, "Cannot bin an empty vector.");
        return *this;
    }
    if (bins <= 0) {
        Log::log(LOG_LEVEL_WARNING, "invalid parameter for number of bins: is ", bins, " but must be >1");
        return *this;
    }
    if (bins >= this->data_elements) {
        Log::log(LOG_LEVEL_WARNING, "invalid use of binning method: number of bins must be less than the number of elements in the vector.");
        return *this;
    }
    // prepare the data structure to put the results
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(bins);    
    result->fill_zeros();
    // get a sorted copy of the original data (ascending order)
    auto sorted = this->sort();
    // calculate bin size
    T min = this->min();
    T max = this->max();
    if (min == max) {
        // There's only one unique value in the vector, so we can't bin it
        Log::log(LOG_LEVEL_WARNING, "can't bin a vector with only one unique value.");
        return *this;
    }
    T bin_size = (max - min) / bins;
    int bin = 0;
    int bin_items = 0;
    int i = 0;
    while (bin < bins && i < this->data_elements) {
        // until bin is full
        while (sorted->data[i] <= min + bin_size * (bin + 1)) {
            result->data[bin] += this->data[i];
            bin_items++;
            i++;
            if (i == this->data_elements) {
                break;
            }
        }
        // calculate mean and move to next bin
        result->data[bin] /= bin_items;
        bin++;
        bin_items = 0;
    }
    return std::move(*result);
}

// +=================================+   
// | Output                          |
// +=================================+

// prints the Array to the console
template<typename T>
void Array<T>::print(std::string comment, std::string delimiter, std::string line_break, bool with_indices) const {
    if (comment!=""){
        std::cout << comment << std::endl;
    }

    if (this->dimensions==1){
        // iterate over elements
        for (int i=0;i<this->data_elements;i++){
            // add indices
            if (with_indices){
                std::cout << "[" << i << "]=";
            }
            // add value
            std::cout << this->data[i];
            // add delimiter between elements (except after last value in row)
            if (i != this->data_elements-1) {
                std::cout << delimiter;
            }         
        }
        // add line break character(s) to end of the row
        std::cout << line_break;        
        return;       
    }

    // create a vector for temporary storage of the current index (needed for indexing dimensions >=2);
    std::vector<int> index(this->dimensions);
    std::fill(index.begin(),index.end(),0);   

    if (this->dimensions==2){
        // iterate over rows
        for (int row=0; row < (this->dimensions==1 ? 1 : this->dim_size[0]); row++) {
            // iterate over columns
            for (int col=0; col < (this->dimensions==1 ? this->dim_size[0] : this->dim_size[1]); col++) {                
                // add indices
                if (with_indices) {
                    std::cout << "[" << row << "]" << "[" << col << "]=";
                }
                // add value
                index[0]=row; index[1]=col;
                std::cout << this->get(index);
                // add delimiter between columns (except after last value in row)
                if (col != int(this->dim_size[1]-1)) {
                    std::cout << delimiter;
                }
            }
            // add line break character(s) to end of current row
            std::cout << line_break;            
        }
    }

    else { //=dimensions >=2
        // iterate over rows
        for (int row = 0; row < this->dim_size[0]; row++) {
            index[0] = row;
            // iterate over columns
            for (int col = 0; col < this->dim_size[1]; col++) {
                index[1] = col;
                // add opening brace for column
                std::cout << "{";
                // iterate over higher dimensions
                for (int d = 2; d < this->dimensions; d++) {
                    // add opening brace for dimension
                    std::cout << "{";
                    // iterate over entries in the current dimension
                    for (int i = 0; i < this->dim_size[d]; i++) {
                        // update index
                        index[d] = i;
                        // add indices
                        if (with_indices) {
                            for (int dd = 0; dd < this->dimensions; dd++) {
                                std::cout << "[" << index[dd] << "]";
                            }
                            std::cout << "=";
                        }
                        // add value
                        std::cout << this->get(index);
                        // add delimiter between values
                        if (i != this->dim_size[d] - 1) {
                            std::cout << delimiter;
                        }
                    }
                    // add closing brace for the current dimension
                    std::cout << "}";
                    // add delimiter between dimensions
                    if (d != this->dimensions - 1) {
                        std::cout << delimiter;
                    }
                }
                // add closing brace for column
                std::cout << "}";
                // add delimiter between columns
                if (col != this->dim_size[1] - 1) {
                    std::cout << delimiter;
                }
            }
            // add line break character(s) to end of current row
            std::cout << line_break;
        }
    }
}