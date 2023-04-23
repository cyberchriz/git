#include "../headers/array.h"

// +=================================+   
// | getters & setters               |
// +=================================+

// assigns a value to an array element via
// a std::initializer_list<int> index
template<typename T>
void Array<T>::set(const std::initializer_list<int>& index, const T value){
    this->_data[get_element(index)] = value;
}

// assigns a value to an array element, with index parameter
// as const std::vector<int>&
template<typename T>
void Array<T>::set(const std::vector<int>& index, const T value){
    this->_data[get_element(index)] = value;
}

// assigns a value to an element of a one-dimensional vector
// via its index (with index parameter as type const int)
template<typename T>
void Vector<T>::set(const int index, const T value){
    this->_data[index] = value;
};

// assigns a value to an element of a two-dimensional
// matrix via its index (index parameters as const int)
template<typename T>
void Matrix<T>::set(const int row, const int col, const T value){
    this->_data[std::fmin(this->_elements-1,col + row*this->_size[1])] = value;
}

// returns the value of an array element via its index
template<typename T>
T Array<T>::get(const std::initializer_list<int>& index) const {
    int element=get_element(index);
    return this->_data[element];
}

// returns the value of an array element via
// its index (as type const std::vector<int>&)
template<typename T>
T Array<T>::get(const std::vector<int>& index) const {
    int element=get_element(index);
    return this->_data[element];
}

// returns the value of a vector element via its index
// (as a const int value)
template<typename T>
T Vector<T>::get(const int index) const {
    return this->_data[std::fmin(index,this->_elements-1)];
}

// returns the value of a 2d matrix element via its index
template<typename T>
T Matrix<T>::get(const int row, const int col) const {
    return this->_data[std::fmin(this->_elements-1,col + row*this->_size[1])];
}

// returns the number of dimensions of the array
template<typename T>
int Array<T>::get_dimensions() const {
    return this->_dimensions;
}

// returns the number of elements of the specified array dimension
template<typename T>
int Array<T>::get_size(int dimension) const {
    return this->_size[dimension];
}

// returns the total number of elements across the array
// for all dimensions
template<typename T>
int Array<T>::get_elements() const {
    return this->_elements;
}

// converts a multidimensional index (represented by the values of
// a std::initializer_list<int>) to a one-dimensional index (=as a scalar)
template<typename T>
int Array<T>::get_element(const std::initializer_list<int>& index) const {
    // check for invalid index dimensions
    if (index.size()>this->_dimensions){
        std::cout << "WARNING: The method 'get_element() has been used with an invalid index:";
        // print the invalid index to the console
        auto iterator = index.begin();
        for (;iterator!=index.end();iterator++){
            std::cout << " " << *iterator;
        }
        std::cout << "\nThe corresponding array has the following dimensions:";
        for (int i=0;i<this->_dimensions;i++){
            std::cout << " " << i;
        }
        std::cout << std::endl;
        return -1;
    }
    // deal with the special case of single dimension arrays ("Vector<T>")
    if (this->_dimensions == 1){
        return *(index.begin());
    }
    // initialize result to number of elements belonging to last dimension
    int result = *(index.end()-1);
    // initialize iterator to counter of second last dimension
    auto iterator = index.end()-2;
    // initialize dimension index to second last dimension
    int i = this->_dimensions-2;
    // decrement iterator down to first dimension
    for (; iterator >= index.begin(); i--, iterator--){
        // initialize amount to add to count in dimension i
        int add = *iterator;
        // multiply by product of sizes of dimensions higher than i
        int s=this->_dimensions-1;
        for(; s >i; s--){
            add *= this->_size[s];
        }
        // add product to result 
        result += add;
    }
    // limit to valid boundaries,
    // i.e. double-check that the result isn't higher than the total
    // number of elements in the corresponding data buffer (this->_data[])
    result=std::fmin(this->_elements-1,result);
    return result;
}

// converts a multidimensional index (represented by C-style
// integer array) to a one-dimensional index (=as a scalar)
template<typename T>
int Array<T>::get_element(const std::vector<int>& index)  const {
    // initialize result to number of elements belonging to last dimension
    int result = index[this->_dimensions-1];
    // initialize dimension index to second last dimension
    int i = this->_dimensions-2;
    for (; i>=0;i--){
        // initialize amount to add to count in dimension i
        int add = index[i];
        // multiply by product of sizes of dimensions higher than i
        int s=this->_dimensions-1;
        for(; s > i; s--){
            add *= this->_size[s];
        }
        // add product to result;
        result += add;
    }
    // limit to valid boundaries
    result=std::fmin(this->_elements-1,result);
    return result;
}

// Return the subspace size
template<typename T>
int Array<T>::get_subspace(int dimension) const {
    return this->_subspace_size[dimension];
}

// +=================================+   
// | fill, initialize                |
// +=================================+

// fill entire array with given value
template<typename T>
void Array<T>::Fill::values(const T value){
    for (int i=0;i<arr._elements;i++){
        arr._data[i]=value;
    }
}

// initialize all values of the array with zeros
template<typename T>
void Array<T>::Fill::zeros(){
    values(0);
}

// fill with identity matrix
template<typename T>
void Array<T>::Fill::identity(){
    // initialize with zeros
    values(0);
    // get size of smallest dimension
    int max_index=arr._size[0];
    for (int i=1; i<arr._dimensions; i++){
        max_index=std::min(max_index,arr._size[i]);
    }
    std::vector<int> index(arr._dimensions);
    // add 'ones' of identity matrix
    for (int i=0;i<max_index;i++){
        for (int d=0;d<arr._dimensions;d++){
            index[d]=i;
        }
        arr.set(index,1);
    }
}

// fill with values from a random normal distribution
template<typename T>
void Array<T>::Fill::random_gaussian(const T mu, const T sigma){
    for (int i=0; i<arr._elements; i++){
        arr._data[i] = Random<T>::gaussian(mu,sigma);
    }           
}

// fill with values from a random uniform distribution
template<typename T>
void Array<T>::Fill::random_uniform(const T min, const T max){
    for (int i=0; i<arr._elements;i++){
        arr._data[i] = Random<T>::uniform(min,max);
    }
}
// fills the array with a continuous
// range of numbers (with specified start parameter
// referring to the zero position and a step parameter)
// in all dimensions
template<typename T>
void Array<T>::Fill::range(const T start, const T step){
    if (arr._dimensions==1){
        for (int i=0;i<arr._elements;i++){
            arr._data[i]=start+i*step;
        }
    }
    else {
        std::vector<int> index(arr._dimensions);
        std::fill(index.begin(),index.end(),0);
        for (int d=0;d<arr._dimensions;d++){
            for (int i=0;i<arr._size[d];i++){
                index[d]=i;
                arr.set(index,start+i*step);
            }
        }
    }
}

// randomly sets a specified fraction of the values to zero
// and retains the rest
template<typename T>
void Array<T>::Fill::dropout(double ratio){
    for (int i=0;i<arr._elements;i++){
        arr._data[i] *= Random<double>::uniform() > ratio;
    }
}

// randomly sets the specified fraction of the values to zero
// and the rest to 1 (default: 0.5, i.e. 50%)
template<typename T>
void Array<T>::Fill::binary(double ratio){
    for (int i=0;i<arr._elements;i++){
        arr._data[i] = Random<double>::uniform() > ratio;
    }
}

// +=================================+   
// | Distribution Properties         |
// +=================================+

// returns the lowest value of all values of a vector, matrix or array
template<typename T>
T Array<T>::min() const {
    if (this->_elements==0){
        std::cout << "WARNING: improper use of method Array<T>::min(): not defined for empty array!\n";
        return T(NAN);
    }
    T result = this->_data[0];
    for (int i=0;i<this->_elements;i++){
        result = std::fmin(result, this->_data[i]);
    }
    return result;
}

// returns the highest value of all values of a vector, matrix or array
template<typename T>
T Array<T>::max() const {
    if (this->_elements==0){
        std::cout << "WARNING: improper use of method Array<T>::min(): not defined for empty array!\n";
        return T(NAN);
    }
    T result = this->_data[0];
    for (int i=0;i<this->_elements;i++){
        result = std::fmax(result, this->_data[i]);
    }
    return result;
}

// returns the arrithmetic mean of all values a vector, matrix or array
template<typename T>
double Array<T>::mean() const {
    double sum=0;
    for (int n=0;n<this->_elements;n++){
        sum+=this->_data[n];
    }
    return sum/this->_elements;
}

// returns the median of all values of a vector, martrix or array
template<typename T>
double Array<T>::median() const {
    // Copy the data to a temporary array for sorting
    // note: .get() is used to retrieve a raw pointer from a std::unique_ptr
    T tmp[this->_elements];
    std::copy(this->_data.get(), this->_data.get() + this->_elements, tmp);

    // Sort the temporary array
    std::sort(tmp, tmp + this->_elements);

    // Calculate the median based on the number of elements
    if (this->_elements % 2 == 1) {
        // Odd number of elements: return the middle element
        return static_cast<double>(tmp[this->_elements / 2]);
    } else {
        // Even number of elements: return the average of the two middle elements
        return static_cast<double>(tmp[this->_elements / 2 - 1] + tmp[this->_elements / 2]) / 2.0;
    }
}

// find the 'mode', i.e. the item that occurs the most number of times
template<typename T>
T Array<T>::mode() const {
    // Sort the array in ascending order
    auto sorted = this->sort();
    // Create an unordered map to store the frequency of each element
    std::unordered_map<T, size_t> freq_map;
    for (size_t i = 0; i < this->_elements; i++) {
        freq_map[sorted->_data[i]]++;
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



// returns the variance of all values of a vector, matrix or array
template<typename T>
double Array<T>::variance() const {
    double mean = this->mean();
    double sum_of_squares = 0.0;
    for (int i = 0; i < this->_elements; i++) {
        double deviation = static_cast<double>(this->_data[i]) - mean;
        sum_of_squares += deviation * deviation;
    }
    return sum_of_squares / static_cast<double>(this->_elements);
}

// returns the standard deviation of all values a the vector, matrix array
template<typename T>
double Array<T>::stddev()  const {
    return std::sqrt(this->variance());
}

// returns the skewness of all data of the vector/matrix/array
template<typename T>
double Array<T>::skewness() const {
    double skewness = 0;
    for (size_t i = 0; i < this->_elements; i++) {
        skewness += std::pow((this->_data[i] - this->mean()) / this->stddev(), 3);
    }
    skewness /= this->_elements;
    return skewness;
}

// returns the kurtosis of all data of the vector/matrix/array
template<typename T>
double Array<T>::kurtosis() const {
    double kurtosis = 0;
    for (int i=0; i<this->_elements; i++) {
        kurtosis += std::pow((this->_data[i] - this->mean()) / this->stddev(), 4);
    }
    kurtosis /= this->_elements;
    kurtosis -= 3;
    return kurtosis;
}
// +=================================+   
// | Addition                        |
// +=================================+

// returns the sum of all array elements
template<typename T>
T Array<T>::sum() const {
    T result=0;
    for (int i=0; i<this->_elements; i++){
        result+=this->_data[i];
    }
    return result;
}

// elementwise addition of the specified value to all values of the array
template<typename T>
std::unique_ptr<Array<T>> Array<T>::operator+(const T value) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->_size);
    for (int i=0;i <this->_elements; i++){
        result->_data[i]=this->_data[i]+value;
    }
    return result;
}

// returns the resulting array of the elementwise addition of
// two array of equal dimensions;
// will return a NAN array if the dimensions don't match!
template<typename T>
std::unique_ptr<Array<T>> Array<T>::operator+(const Array<T>& other) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->_size);
    if (!equal_size(other)){
        result->fill->values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]+other._data[i];
        }
    }
    return result;
}

// prefix increment operator;
// increments the values of the array by +1,
// returns a reference to the source array itself
template<typename T>
Array<T>& Array<T>::operator++(){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]+=1;
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
std::unique_ptr<Array<T>> Array<T>::operator++(int) const {
    std::unique_ptr<Array<T>> temp = std::make_unique<Array<T>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        temp->_data[i] = this->_data[i];
        this->_data[i]++;
    }
    return temp;
}

// elementwise addition of the specified
// value to the elements of the array
template<typename T>
void Array<T>::operator+=(const T value) {
    for (int i=0; i<this->_elements; i++){
        this->_data[i]+=value;
    }
}

// elementwise addition of the values of the second
// array to the corresponding values of the current array;
// the dimensions of the arrays must match!
// the function will otherwise turn the source array into
// a NAN array!
template<typename T>
void Array<T>::operator+=(const Array<T>& other){
    if (!equal_size(other)){
        this->fill->values(T(NAN));
        return;
    }
    for (int i=0; i<this->_elements; i++){
        this->_data[i]+=other._data[i];
    }
}

// +=================================+   
// | Substraction                    |
// +=================================+

// elementwise substraction of the specified value from all values of the array
template<typename T>
std::unique_ptr<Array<T>> Array<T>::operator-(const T value)  const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->_size);
    for (int i=0;i <this->_elements; i++){
        result->_data[i]=this->_data[i]-value;
    }
    return result;
}

// returns the resulting array of the elementwise substraction of
// two array of equal dimensions (this minus other);
// will return a NAN array if the dimensions don't match!
template<typename T>
std::unique_ptr<Array<T>> Array<T>::operator-(const Array<T>& other) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->_size);
    if (!equal_size(other)){
        result->fill->values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]-other._data[i];
        }
    }
    return result;
}

// prefix decrement operator;
// first decrements the values of the array by -1,
// then returns the modified array
template<typename T>
Array<T>& Array<T>::operator--(){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]-=1;
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
std::unique_ptr<Array<T>> Array<T>::operator--(int) const {
    auto temp = this->copy();
    for (int i=0; i<this->_elements; i++){
        this->_data[i]-=1;
    }
    return temp;
}

// elementwise substraction of the specified
// value from the elements of the array
template<typename T>
void Array<T>::operator-=(const T value) {
    for (int i=0; i<this->_elements; i++){
        this->_data[i]-=value;
    }
}

// elementwise substraction of the values of the second
// array from the corresponding values of the current array,
// i.e. this minus other;
// the dimensions of the arrays must match!
// the function will otherwise turn the source array into
// a NAN array!
template<typename T>
void Array<T>::operator-=(const Array<T>& other){
    if (!equal_size(other)){
        this->fill->values(T(NAN));
        return;
    }
    for (int i=0; i<this->_elements; i++){
        this->_data[i]-=other._data[i];
    }
}

// +=================================+   
// | Multiplication                  |
// +=================================+

// returns the product reduction, i.e. the result
// of all individual elements of the array
template<typename T>
T Array<T>::product() const {
    if (this->_elements==0){
        return T(NAN);
    }
    T result = this->_data[0];
    for (int i=1; i<this->_elements; i++){
        result*=this->_data[i];
    }
    return result;
}

// elementwise multiplication with a scalar
template<typename T>
std::unique_ptr<Array<T>> Array<T>::operator*(const T factor) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]*factor;
    }
    return result;
}

// elementwise multiplication (*=) with a scalar
template<typename T>
void Array<T>::operator*=(const T factor){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]*=factor;
    }
}

// elementwise multiplication of the values of the current
// array with the corresponding values of a second array,
// resulting in the 'Hadamard product';
// the dimensions of the two arrays must match!
// the function will otherwise return a NAN array!
template<typename T>
std::unique_ptr<Array<T>> Array<T>::Hadamard(const Array<T>& other)  const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->_size);
    if(!equal_size(other)){
        result->fill->values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]*other._data[i];
        }
    }
    return result;
}

// Array tensor reduction ("tensor dotproduct")
template<typename T>
std::unique_ptr<Array<T>> Array<T>::tensordot(const Array<T>& other, const std::vector<int>& axes) const {
    // check that the number of axes to contract is the same for both tensors
    if (axes.size() != other._dimensions) {
        std::cout << "WARNING: Invalid use of function Array<T>::tensordot(). Number of contraction axes must be the same for both tensors." << std::endl;
        return nullptr;
    }
    
    // check that the sizes of the axes to contract are the same for both tensors
    for (int i = 0; i < axes.size(); i++) {
        if (this->_size[axes[i]] != other._size[axes[i]]) {
        std::cout << "WARNING: Invalid use of function Array<T>::tensordot(). Sizes of contraction axes must be the same for both tensors." << std::endl;
        return nullptr;
        }
    }
    
    // compute the dimensions of the output tensor
    std::vector<int> out_dims;
    for (int i = 0; i < this->_dimensions; i++) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
            out_dims.push_back(this->_size[i]);
        }
    }
    for (int i = 0; i < other._dimensions; i++) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
            out_dims.push_back(other._size[i]);
        }
    }
    
    // create a new tensor to hold the result
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(out_dims);
    
    // perform the tensor contraction
    std::vector<int> index1(this->_dimensions, 0);
    std::vector<int> index2(other._dimensions, 0);
    std::vector<int> index_out(out_dims.size(), 0);
    int contraction_size = 1;
    for (int i = 0; i < axes.size(); i++) {
        contraction_size *= this->_size[axes[i]];
    }
    for (int i = 0; i < result->_size[0]; i++) {
        index_out[0] = i;
        for (int j = 0; j < contraction_size; j++) {
            for (int k = 0; k < axes.size(); k++) {
                index1[axes[k]] = j % this->_size[axes[k]];
                index2[axes[k]] = j % other._size[axes[k]];
            }
            for (int k = 1; k < out_dims.size(); k++) {
                int size = k <= axes[0] ? this->_size[k - 1] : other._size[k - axes.size() - 1];
                int val = (i / result->get_subspace(k)) % size;
                if (k < axes[0] + 1) {
                    index1[k - 1] = val;
                } else {
                    index2[k - axes.size() - 1] = val;
                }
            }
            result->set(index_out, result->get(index_out) + this->get(index1) * other.get(index2));
        }
    }
    
    return result;
}



// Alias for the dotproduct (=scalar product)
template<typename T>
T Array<T>::operator*(const Array<T>& other) const {
    return this->dotproduct(other);
}

// Array dotproduct, i.e. the scalar product
template<typename T>
T Array<T>::dotproduct(const Array<T>& other) const {
    // check for equal shape
    if (!this->equal_size(other)){
        std::cout << "WARNING: Invalid usage. The method 'Array<T>::dotproduct() has been used with unequal array shapes:" << std::endl;
        std::cout << "- source array shape:";
        for (int i=0;i<this->_dimensions;i++){
            std::cout << " " << this->_size[i];
        }
        std::cout << std::endl;
        std::cout << "- second array shape:";
        for (int i=0;i<other._dimensions;i++){
            std::cout << " " << other._size[i];
        }
        std::cout << std::endl;        
        return -1;
    }
    T result = 0;
    for (int i = 0; i < this->_elements; i++) {
        result += this->_data[i] * other._data[i];
    }
    return result;
}


// vector dotproduct ("scalar product")
template<typename T>
T Vector<T>::dotproduct(const Vector<T>& other) const {
    if (this->_elements != other._elements){
        return T(NAN);
    }
    T result = 0;
    for (int i = 0; i < this->_elements; i++){
        result += this->_data[i] * other._data[i];
    }
    return result;
}

// operator* as alias for vector dotproduct
template<typename T>
T Vector<T>::operator*(const Vector<T>& other) const {
    return this->dotproduct(other);
}

// returns the product of two 2d matrices
// as their tensor contraction
template<typename T>
std::unique_ptr<Matrix<T>> Matrix<T>::tensordot(const Matrix<T>& other) const {
    // Create the resulting matrix
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(this->_size[0], other._size[1]);
    // Check if the matrices can be multiplied
    if (this->_size[1] != other._size[0]){
        result->fill->values(T(NAN));
        return result;
    }
    // Compute the dot product
    for (int i = 0; i < this->_size[0]; i++) {
        for (int j = 0; j < other._size[1]; j++) {
            T sum = 0;
            for (int k = 0; k < this->_size[1]; k++) {
                sum += this->get(i, k) * other.get(k, j);
            }
            result->set(i, j, sum);
        }
    }
    return result;
}

// operator* as alias for the dotproduct (=scalar product)
template<typename T>
T Matrix<T>::operator*(const Matrix<T>& other) const {
    return this->tensordot(other);
}

// +=================================+   
// | Division                        |
// +=================================+

// elementwise division by a scalar
template<typename T>
std::unique_ptr<Array<T>> Array<T>::operator/(const T quotient) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]/quotient;
    }
    return result;
}

// elementwise division (/=) by a scalar
template<typename T>
void Array<T>::operator/=(const T quotient){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]/=quotient;
    }
}

// +=================================+   
// | Modulo                          |
// +=================================+

// elementwise modulo operation, converting the array values
// to the remainders of their division by the specified number
template<typename T>
void Array<T>::operator%=(const double num){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]%=num;
    }
}

// elementwise modulo operation, resulting in an array that
// contains the remainders of the division of the values of
// the original array by the specified number
template<typename T>
std::unique_ptr<Array<double>> Array<T>::operator%(const double num) const {
    std::unique_ptr<Array<double>> result = std::make_unique<Array<double>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]%num;
    }
    return result;
}

// +=================================+   
// | Exponentiation                  |
// +=================================+

// elementwise exponentiation to the power of
// the specified exponent
template<typename T>
void Array<T>::pow(const T exponent){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]=std::pow(this->_data[i], exponent);
    }
}

// elementwise exponentiation to the power of
// the corresponding values of the second array;
// the dimensions of the two array must match!
// the function will otherwise return a NAN array!
template<typename T>
void Array<T>::pow(const Array<T>& other){
    if (!equal_size(other)){
        this->fill->values(T(NAN));
        return;
    }
    else {
        for (int i=0; i<this->_elements; i++){
            this->_data[i]=std::pow(this->_data[i], other._data[i]);
        }
    }
}

// converts the individual values of the array
// elementwise to their square root
template<typename T>
void Array<T>::sqrt(){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]=std::sqrt(this->_data[i]);
    }
}

// converts the individual values of the array
// elementwise to their natrual logarithm
template<typename T>
void Array<T>::log(){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]=std::log(this->_data[i]);
    }
}

// converts the individual values of the array
// elementwise to their base-10 logarithm
template<typename T>
void Array<T>::log10(){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]=std::log10(this->_data[i]);
    }
}

// +=================================+   
// | Rounding                        |
// +=================================+

// rounds the values of the array elementwise
// to their nearest integers
template<typename T>
void Array<T>::round(){
    for (int i=0;i<this->_elements;i++){
        this->_data[i]=std::round(this->_data[i]);
    }
}

// rounds the values of the array elementwise
// to their next lower integers
template<typename T>
void Array<T>::floor(){
    for (int i=0;i<this->_elements;i++){
        this->_data[i]=std::floor(this->_data[i]);
    }
}

// rounds the values of the array elementwise
// to their next higher integers
template<typename T>
void Array<T>::ceil(){
    for (int i=0;i<this->_elements;i++){
        this->_data[i]=std::ceil(this->_data[i]);
    }
}

// +=================================+   
// | Find, Replace                   |
// +=================================+

// returns the number of occurrences of the specified value
template<typename T>
int Array<T>::find(const T value) const {
    int counter=0;
    for (int i=0; i<this->_elements; i++){
        counter+=(this->_data[i]==value);
    }
    return counter;
}

// replace all findings of given value by specified new value
template<typename T>
void Array<T>::replace(const T old_value, const T new_value){
    for (int i=0; i<this->_elements; i++){
        if (this->_data[i]==old_value){
            this->_data[i]=new_value;
        }
    }
}

// +=================================+   
// | Custom Functions                |
// +=================================+

// modifies the given vector, matrix or array by applying
// the referred function to all its values
// (the referred function should take a single argument of type <T>)
template<typename T>
void Array<T>::function(const T (*pointer_to_function)(T)){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]=pointer_to_function(this->_data[i]);
    }
}

// +=================================+   
// | Assignment                      |
// +=================================+

// copy assignment operator with second Array as argument:
// copies the values from a second array into the values of
// the current array; the _dimensions of target and source must match!
template<typename T>
Array<T>& Array<T>::operator=(const Array<T>& other) {
    // Check for self-assignment
    if (this != &other) {
        // Allocate new memory for the array
        std::unique_ptr<T[]> new_data = std::make_unique<T[]>(other._elements);
        // Copy the elements from the other array to the new array
        std::copy(other._data.get(), other._data.get() + other._elements, new_data.get());
        // Assign the new data to this object
        this->_data = std::move(new_data);
        this->_elements = other._elements;
        this->_size = other._size;
    }
    return *this;
}

// copy assignment operator with second Matrix as argument:
// copies the values from a second Matrix into the values of
// the current Matrix
template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other) {
    // Check for self-assignment
    if (this != &other) {
        // Allocate new memory for the array
        std::unique_ptr<T[]> new_data = std::make_unique<T[]>(other._elements);
        // Copy the elements from the other array to the new array
        std::copy(other._data.get(), other._data.get() + other._elements, new_data.get());
        // Assign the new data to this object
        this->_data = std::move(new_data);
        this->_elements = other._elements;
        this->_size = other._size;
    }
    return *this;
}

// copy assignment operator with second Vector as argument:
// copies the values from a second Vector into the values of
// the current Vector
template<typename T>
Vector<T>& Vector<T>::operator=(const Vector<T>& other) {
    // Check for self-assignment
    if (this != &other) {
        // Allocate new memory for the array
        std::unique_ptr<T[]> new_data = std::make_unique<T[]>(other._elements);
        // Copy the elements from the other array to the new array
        std::copy(other._data.get(), other._data.get() + other._elements, new_data.get());
        // Assign the new data to this object
        this->_data = std::move(new_data);
        this->_elements = other._elements;
        this->_size = other._size;
    }
    return *this;
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
std::unique_ptr<Array<bool>> Array<T>::operator>(const T value) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]>value;
    }
    return result;
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are greater than or equal
// to the specified argument value
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator>=(const T value) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]>=value;
    }
    return result;
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are equal to the specified
// argument value
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator==(const T value) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]==value;
    }
    return result;
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are unequal to the specified
// argument value
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator!=(const T value) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]!=value;
    }
    return result;
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are less than the
// specified argument value
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator<(const T value) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]<value;
    }
    return result;
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are less than or equal to
// the specified argument value
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator<=(const T value) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]<=value;
    }
    return result;
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
std::unique_ptr<Array<bool>> Array<T>::operator>(const Array<T>& other) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    if (!this->equal_size(other)){
        result->fill->values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]>other._data[i];
        }
    }
    return result;
}

// returns a boolean array of equal dimensions
// to the source array, with its values indicating
// whether the values of the source array are greater
// than or equal to the corresponding values of the
// second array, i.e. by elementwise comparison;
// make sure the the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator>=(const Array<T>& other) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    if (!this->equal_size(other)){
        result->fill->values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]>=other._data[i];
        }
    }
    return result;
}

// returns a boolean array of equal dimensions
// to the source array, with its values indicating
// whether the values of the source array are equal
// to the corresponding values of the second array,
// i.e. by elementwise comparison;
// make sure the the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator==(const Array<T>& other) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    if (!this->equal_size(other)){
        result->fill->values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]==other._data[i];
        }
    }
    return result;
}

// returns a boolean array of equal dimensions
// to the source array, with its values indicating
// whether the values of the source array are unequal
// to the corresponding values of the second array,
// i.e. by elementwise comparison;
// make sure the the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator!=(const Array<T>& other) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    if (!this->equal_size(other)){
        result->fill->values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]!=other._data[i];
        }
    }
    return result;
}

// returns a boolean array of equal dimensions
// to the source array, with its values indicating
// whether the values of the source array are less
// than the corresponding values of the second array,
// i.e. by elementwise comparison;
// make sure the the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator<(const Array<T>& other) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    if (!this->equal_size(other)){
        result->fill->values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]<other._data[i];
        }
    }
    return result;
}

// returns a boolean array of equal dimensions
// to the source array, with its values indicating
// whether the values of the source array are less than
// or equal to the corresponding values of the second array,
// i.e. by elementwise comparison;
// make sure the the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator<=(const Array<T>& other) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    if (!this->equal_size(other)){
        result->fill->values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]<=other._data[i];
        }
    }
    return result;
}
// +=================================+   
// | Elementwise Logical Operations  |
// +=================================+

// returns a boolean array as the result of the
// logical AND of the source array and the specified
// boolean argument value
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator&&(const bool value) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]&&value;
    }
    return result;
}

// returns a boolean array as the result of the
// logical OR of the source array and the specified
// boolean argument value
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator||(const bool value) const {
    std::unique_ptr<Array<bool>> result(new Array(this->_size));
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]||value;
    }
    return result;
}

// returns a boolean array as the result of the
// logical NOT of the source array
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator!() const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=!this->_data[i];
    }
    return result;
}

// returns a boolean array as the result of the
// elementwise logical AND operation between the
// source array and the corresponding values of the
// second array;
// make sure that the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator&&(const Array<T>& other) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    if (!this->equal_size(other)){
        result->fill->values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]&&other._data[i];
        }
    }
    return result;
}

// returns a boolean array as the result of the
// elementwise logical OR operation between the
// source array and the corresponding values of the
// second array;
// make sure that the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator||(const Array<T>& other) const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    if (!this->equal_size(other)){
        result->fill->values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]||other._data[i];
        }
    }
    return result;
}

// +=================================+   
// | Type Casting                    |
// +=================================+

// type casting (explicit by default)
template<typename T>
template<typename C>
Array<T>::operator Array<C>(){
    Array<C> result = std::make_unique<Array<C>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=C(this->_data[i]);
    }
    return result;
}

// +=================================+   
// | Class Conversion                |
// +=================================+

// flattens an array or matrix into a one-dimensional vector
template<typename T>
std::unique_ptr<Vector<T>> Array<T>::flatten() const {
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->_elements);
    for (int i=0;i<this->_elements;i++){
        result->_data[i]=this->_data[i];
    }
    return result;
}

// converts an array into a 2d matrix of the specified size;
// if the new matrix has less elements in any of the dimensions,
// the surplus elements of the source array will be ignored;
// if the new matrix has more elements than the source array, these
// additional elements will be initialized with the specified value
template<typename T>
std::unique_ptr<Matrix<T>> Array<T>::asMatrix(const int rows, const int cols, T init_value) const {
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(rows,cols);
    result->fill->values(init_value);
    std::vector<int> index(this->_dimensions);
    // reset the indices of higher dimensions to all zeros
    for (int d=0;d<this->_dimensions;d++){
        index[d]=0;
    }
    // iterate over first and second dimension, i.e. keeping
    // the higher dimensions at index zero
    for (int row=0;row<this->_size[0];row++){
        index[0]=row;
        for (int col=0;col<this->_size[1];col++){
            index[1]=col;
            result->set(row,col,this->get(index));
        }
    }
    return result;
}

// converts a matrix into another 2d matrix of different size;
// if the new matrix has less elements in any of the two dimensions,
// the surplus elements of the source will be ignored; if the new matrix
// has more elements, these additional elements will be initialized with
// the specified value
template<typename T>
std::unique_ptr<Matrix<T>> Matrix<T>::asMatrix(const int rows, const int cols, T init_value) const {
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(rows,cols);
    result->fill->values(init_value);
    for (int r=0;r<std::fmin(rows,this->_size[0]);r++){
        for (int c=0;c<std::fmin(cols,this->_size[1]);c++){
            result->set(r,c,this->get(r,c));
        }
    }
    return result;
}

// converts a vector into a 2d matrix of the specified size;
// the extra rows will be initialized with zeros
// if the new matrix has less columns than the source vector
// has elements, the surplus elements (for row 0) of the source
// will be ignored; if the new matrix has more columns than the
// source vector has elements, these additional elements will
// be initialized with the specified init value;
template<typename T>
std::unique_ptr<Matrix<T>> Vector<T>::asMatrix(const int rows, const int cols, T init_value) const {
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(rows,cols);
    result->fill->values(init_value);
    for (int i=0;i<std::fmin(this->_elements,cols);i++){
        result->set(0,i, this->_data[i]);
    }
    return result;
}

// converts an array, matrix or vector into a 2d matrix;
// the exact behavior will depend on the source dimensions:
// 1. if the source array already is 2-dimensional, the total size
// and the size per dimension will remain unchanged, only the
// datatype of the returned object is now 'Matrix<T>'
// 2. if the source has more than 2 dimensions, only values from
// index 0 of the higher dimensions will be copied into the
// returned result
template<typename T>
std::unique_ptr<Matrix<T>> Array<T>::asMatrix() const {
    std::unique_ptr<Matrix<T>> result;;
    if (this->_dimensions==2){
        result=std::make_unique<Matrix<T>>(this->_size[0],this->_size[1]);
        int index;
        for (int row=0;row<this->_size[0];row++){
            for (int col=0;col<this->_size[1];col++){
                index = col +row*this->_size[1];
                result->_data[index] = this->_data[index];
            }
        }
    }
    else {
        result=std::make_unique<Matrix<T>>(this->_size[0],this->_size[1]);
        std::vector<int> index(this->_dimensions);
        // reset dimension indices to all zeros
        std::fill(index.begin(),index.end(),0);
        // iterate over first and second dimension, i.e. keeping
        // the higher dimensions at index zero
        for (int row=0;row<this->_size[0];row++){
            index[0]=row;
            for (int col=0;col<this->_size[1];col++){
                index[1]=col;
                // assign matching elements with values from source
                result->set(row,col,this->get(index));
            }
        }
    }
    return result;
}

// converts a vector into a single row matrix with
// unchanged number of elements
template<typename T>
std::unique_ptr<Matrix<T>> Vector<T>::asMatrix() const {
    std::unique_ptr<Matrix<T>> result;;
    result=std::make_unique<Matrix<T>>(1,this->_elements);
    for (int i=0;i<this->_elements;i++){
        result->_data[i]=this->_data[i];
    }
    return result;
}

// converts a vector or matrix into an array
// or converts a preexisting array into an array of
// the specified new size;
// surplus elements of the source that go beyond the
// limits of the target will be cut off; if the target
// is bigger, the surplus target elements that have no
// corresponding index at the source will be initialized
// with the specified init_value
template<typename T>
std::unique_ptr<Array<T>> Array<T>::asArray(const std::initializer_list<int>& init_list, T init_value) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(init_list);
    result->fill->values(init_value);
    // reset result index to all zeros
    std::vector<int> result_index(result->get_dimensions());
    std::fill(result_index.begin(),result_index.end(),0);
    // reset source index to all zeros
    std::vector<int> source_index(this->_dimensions);
    std::fill(source_index.begin(),source_index.end(),0);
    // iterate over source dimensions
    for (int d=0;d<std::fmin(this->_dimensions,result->get_dimensions());d++){
        // iterate over elements of given dimension
        for (int i=0;i<std::fmin(this->_size[d],result->get_size(d));i++){
            result_index[d]=i;
            source_index[d]=i;
            result->set(result_index,this->_data[get_element(source_index)]);
        }
    }
    return result;
}

// +=================================+   
// | Constructors & Destructors      |
// +=================================+

// constructor for multi-dimensional array:
// pass dimension size (elements per dimension)
// as an initializer_list, e.g. {3,4,4}
template<typename T>
Array<T>::Array(const std::initializer_list<int>& init_list) {
    // set dimensions + check if init_list empty
    this->_dimensions = (int)init_list.size();
    if (this->_dimensions==0){
        return;
    }
    // store size of individual dimensions in std::vector<int> _size member variable
    auto iterator=init_list.begin();
    int n=0;
    this->_size.resize(this->_dimensions);
    for (; iterator!=init_list.end();n++, iterator++){
        this->_size[n]=*iterator;
    }
    // calculate the size of each subspace for each dimension
    int total_size = 1;
    this->_subspace_size.resize(this->_dimensions);
    for (int i = 0; i < this->_dimensions; i++) {
        this->_subspace_size[i] = total_size;
        total_size *= this->_size[i];
    }    
    // count total number of elements
    this->_elements=1;
    for (int d=0;d<this->_dimensions;d++){
        this->_elements*=this->_size[d];
    }
    this->init();
};

// constructor for multidimensional array:
// pass dimension size (elements per dimension)
// as type std::vector<int>
template<typename T>
Array<T>::Array(const std::vector<int>& dimensions){
    // set dimensions + check if init_list empty
    this->_dimensions = dimensions.size();
    if (this->_dimensions==0){
        return;
    }
    // store size of individual dimensions in std::vector<int> _size member variable
    // and count total number of elements
    this->_elements=1;
    this->_size.resize(this->_dimensions);
    for (int i=0;i<this->_dimensions;i++){
        this->_size[i]=dimensions[i];
        this->_elements*=dimensions[i];
    }
    // calculate the size of each subspace for each dimension
    int total_size = 1;
    this->_subspace_size.resize(this->_dimensions);
    for (int i = 0; i < this->_dimensions; i++) {
        this->_subspace_size[i] = total_size;
        total_size *= this->_size[i];
    }       
    this->init();    
}

// virtual destructor for parent class
template<typename T>
Array<T>::~Array(){
    // empty
}

// override destructor for Matrix<T>
template<typename T>
Matrix<T>::~Matrix(){
    // empty
}

// override destructor for Matrix<T>
template<typename T>
Vector<T>::~Vector(){
    // empty
}

// constructor for a one-dimensional vector
template<typename T>
Vector<T>::Vector(const int elements) {
    this->_size.resize(1);
    this->_subspace_size.push_back(elements);
    this->_size[0]=elements;
    this->_elements = elements;
    this->_capacity = (1.0f+this->_reserve)*elements;
    this->_dimensions = 1;
    this->init();  
}

// constructor for 2d matrix
template<typename T>
Matrix<T>::Matrix(const int rows, const int cols) {
    this->_elements = rows * cols;
    this->_dimensions = 2;
    this->_size.resize(this->_dimensions);
    this->_size[0]=rows;
    this->_size[1]=cols;
    // calculate the size of each subspace for each dimension
    int total_size = 1;
    this->_subspace_size.resize(2);
    this->_subspace_size[0]=rows;
    this->_subspace_size[1]=rows*cols;
    this->init();    
}

// protected helper function for constructors
template<typename T>
void Array<T>::init(){
    // set up data buffer
    this->_data = std::make_unique<T[]>(this->_elements);
    // instantiate helper structs
    this->scale = std::make_unique<typename Array<T>::Scaling>(*this);
    this->fill = std::make_unique<typename Array<T>::Fill>(*this);
    this->activation = std::make_unique<typename Array<T>::Activation>(*this);
    this->outliers = std::make_unique<Outliers>(*this);    
}

// +=================================+   
// | Private Member Functions        |
// +=================================+

// check whether this array and a second array
// match with regard to their number of dimensions
// and their size per individual dimensions
template<typename T>
bool Array<T>::equal_size(const Array<T>& other) const {
    if (this->_dimensions!=other.get_dimensions()){
        return false;
    }
    for (int n=0; n<this->_dimensions; n++){
        if (this->_size[n]!=other.get_size(n)){
            return false;
        }
    }
    return true;
}    

// change the size of a simple C-style array via its std::unique_ptr<T[]>
// by allocating new memory and copying the previous data to the new location
template<typename T>
void Array<T>::resizeArray(std::unique_ptr<T[]>& arr, const int newSize) {
    // Create a new array with the desired size
    std::unique_ptr<T[]> newArr(new T[newSize]);
    // Copy the elements from the old array to the new array
    for (int i = 0; i < newSize; i++) {
        if (i < this->__elements) {
            newArr[i] = arr[i];
        } else {
            newArr[i] = 0;
        }
    }
    // Assign the new array to the old array variable
    arr = std::move(newArr);
    // Update the number of elements in the array
    this->_elements = newSize;
}

// +=================================+   
// | Dynamic Vector Handling         |
// +=================================+

// push back 1 element into the Vector
template<typename T>
int Vector<T>::push_back(const T value){
    this->_elements++;
    if (this->_elements>this->_capacity){
        this->_capacity=int(this->_elements*(1.0+this->_reserve));
        resizeArray(this->_data, this->_capacity);
    }
    this->_data[this->_elements-1]=value;
    return this->_elements;
}

// resize the vector to a new number of elements
template<typename T>
void Vector<T>::resize(const int new_size){
    this->_elements=new_size;
    if (this->_elements>this->_capacity){
        this->_capacity=int(this->_elements*(1.0+this->_reserve));
        resizeArray(this->_data, this->_capacity);
    }    
}
// grows the vector size by the specified number of
// additional elements and initializes these new elements
// to the specified value (default=0);
// will only re-allocate memory if the new size exceeds
// the capacity; returns the new total number of elements
template<typename T>
int Vector<T>::grow(const int additional_elements,T value){
    if (additional_elements<1){return 0;}
    int new_size=this->_elements+additional_elements;
    // re-allocate memory if the new size exceeds the capacity
    if (new_size>this->_capacity){
        this->capacity=int(this->_elements*(1.0+this->_reserve));
        resizeArray(&this->_data, this->_capacity);
    }
    // initialize the new elements
    for (int i=this->_elements;i<new_size;i++){
        this->_data[i]=value;
    }
    this->_elements=new_size;
    return new_size;
}

// shrinks the vector size by the specified number of
// elements and returns the resulting new number of
// remaining total elements
template<typename T>
int Vector<T>::shrink(const int remove_amount){
    int new_size=std::fmax(0,this->_elements-remove_amount);
    this->_elements=new_size;
    return new_size;
}

// pop 1 element from the end the Vector
template<typename T>
T Vector<T>::pop(){
    this->_elements--;
    return this->_data[this->_elements];
}

// removes the element of the given index and returns its value
template<typename T>
T Vector<T>::erase(const int index){
    T result = this->_data[index];
    for (int i=index;i<this->_elements-1;i++){
        this->_data[i] = this->_data[i+1];
    }
    this->_elements--;
    return result;
}

// returns the available total capacity of a vector
// without re-allocating memory
template<typename T>
int Vector<T>::get_capacity() const {
    return this->_capacity;
}

// returns the current size (=number of elements)
// of the vector; equivalent to .get_elements()
// or .get_size(0);
template<typename T>
int Vector<T>::size() const {
    return this->_elements;
}

// +=================================+   
// | Vector Conversion               |
// +=================================+

// returns the vector as a single column matrix, 
// i.e. as transposition with data in rows (single column)
template<typename T>
std::unique_ptr<Matrix<T>> Vector<T>::transpose() const {
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(this->_elements, 1);
    for (int i=0; i<this->_elements; i++){
        result->set(i, 0, this->_data[i]);
    }
    return result;
}

// returns a pointer to a reverse order copy of the
// original Vector<T>
template<typename T>
std::unique_ptr<Vector<T>> Vector<T>::reverse() const {
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->_elements);
    for (int i=0;i<this->_elements;i++){
        result->_data[i] = this->_data[this->_elements-1-i];
    }
    return result;
}


// +=================================+   
// | Vector Sample Analysis          |
// +=================================+

// returns a vector of integers that represent
// a ranking of the source vector via bubble sorting
// the ranks
template<typename T>
std::unique_ptr<Vector<int>> Vector<T>::ranking() const {
    // initialize ranks
    std::unique_ptr<Vector<int>> rank = std::make_unique<Vector<int>>(this->_elements);
    rank->fill->range(0,1);
    // ranking loop
    bool ranking_completed=false;
    while (!ranking_completed){
        ranking_completed=true; //=let's assume this until a wrong order is found
        for (int i=0;i<this->_elements-1;i++){
            // pairwise comparison:
            if ((this->_data[i]>this->_data[i+1] && rank->_data[i]<rank->_data[i+1]) ||
                (this->_data[i]<this->_data[i+1] && rank->_data[i]>rank->_data[i+1])){
                ranking_completed=false;
                // swap ranks
                std::swap(rank->_data[i+1], rank->_data[i]);
            }           
        }
    }  
    return rank;
}

// returns an exponentially smoothed copy of the source vector,
// e.g. for time series
template<typename T>
std::unique_ptr<Vector<T>> Vector<T>::exponential_smoothing(bool as_series) const {
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->_elements);
    double alpha=2/(this->_elements);

    if (as_series){
        result->set(this->_elements-1, this->mean());
        for (int n=this->_elements-2; n>=0; n--){
            result->set(n, alpha*(this->_data[n] - result->get(n+1)) + result->get(n+1));
        }     
    }
    else{
        result->set(0, this->mean());
        for (int n=1; n<this->_elements; n++){
            result->set(n, alpha*(this->_data[n]-result->get(n-1)) + result->get(n-1));
        }
    }
    return result;
}

// returns the weighted average of a sample Vector,
// e.g. for time series data
template <typename T>
double Vector<T>::weighted_average(bool as_series) const {
    double weight=0, weight_sum=0, sum=0;
    if (!as_series){ //=indexing from zero, lower index means lower attributed weight
        for (int n=0;n<this->_elements;n++){
            weight++;
            weight_sum+=weight;
            sum+=weight*this->_data[n];
        }
        return sum/(this->_elements*weight_sum);
    }
    else {
        for (int n=this->_elements-2;n>=0;n--) {
            weight++;
            weight_sum+=weight;
            sum+=weight*this->_data[n];
        }
        return sum/(this->_elements*weight_sum);
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
double Vector<T>::Dickey_Fuller(DIFFERENCING method, double degree, double fract_exponent) const {
    // make two copies
    Vector<T> data_copy(this->_elements);
    Vector<T> stat_copy(this->_elements);    
    for (int i=0;i<this->_elements;i++){
        data_copy._data[i] = this->_data[i];
        stat_copy._data[i] = this->_data[i];
    }
    // make one of the copies stationary
    stat_copy.stationary(method, degree, fract_exponent);
    // pop the first element of the other copy to make their size match again
    data_copy.pop_first();
    // correlate the raw copy with the corresponding stationary transformation
    double Pearson_R = data_copy.correlation(stat_copy)->Pearson_R;
    // calculate result
    return Pearson_R*std::sqrt((double)(this->_elements-1)/(1-std::pow(Pearson_R,2)));  
}

// takes the source vector and another vector (passed as parameter) and
// performs an Engle-Granger test in order to test the given numeric sample
// for cointegration, i.e. checking series data for a long-term relationship.
// The test was proposed by Clive Granger and Robert Engle in 1987.
// If the returned p-value is less than a chosen significance level (typically 0.05),
// it suggests that the two time series are cointegrated and have a long-term relationship.
// Make sure that both Vector<T> have the same number of elements! Otherwise the surplus
// elements of the larger vector will be clipped and the result isn't meaningful;
template<typename T>
double Vector<T>::Engle_Granger(const Vector<T>& other) const {
    // make copies of the x+y source data
    int elements = std::fmin(this->_elements, other._elements);
    Vector<T> x_data(elements);
    Vector<T> y_data(elements);
    for (int i=0;i<elements;i++){
        x_data._data[i] = this->_data[i];
        y_data._data[i] = other._data[i];
    }
    // make the data stationary
    std::unique_ptr<Vector<T>> x_stat = x_data.stationary();
    std::unique_ptr<Vector<T>> y_stat = y_data.stationary();
    // perform linear regression on x versus y
    std::unique_ptr<LinReg<T>> regr_result = x_stat->linear_regression(y_stat);
    // perform a Dickey_Fuller test on the residuals
    Vector<double> residuals(elements);
    residuals._data = regr_result->residuals;
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
std::unique_ptr<Vector<T>> Vector<T>::stationary(DIFFERENCING method, double degree, double fract_exponent)  const {
    // make a copy
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->_elements);
    for (int i = 0; i < this->_elements; i++) {
        result->_data[i] = this->_data[i];
    }
    if (method == integer) {
        for (int d = 1; d <= (int)degree; d++) { //=loop allows for higher order differencing
            for (int t = this->_elements - 1; t > 0; t--) {
                result->_data[t] -= result->_data[t - 1];
            }
            // Remove the first element from the unique_ptr
            std::unique_ptr<T[]> new_data = std::make_unique<T[]>(this->_elements - 1);
            for (int i = 0; i < this->_elements - 1; i++) {
                new_data[i] = result->_data[i + 1];
            }
            result->_data = std::move(new_data);
            result->_elements--;
        }
    }
    if (method == logreturn) {
        for (int d = 1; d <= round(degree); d++) { //=loop allows for higher order differencing
            for (int t = this->_elements - 1; t > 0; t--) {
                if (result->_data[t - 1] != 0) {
                    result->_data[t] = log(std::numeric_limits<T>::min() + std::fabs(result->_data[t] / (result->_data[t - 1] + std::numeric_limits<T>::min())));
                }
            }
            // for each "degree":
            // pop the first element from the unique_ptr
            std::unique_ptr<T[]> new_data = std::make_unique<T[]>(this->_elements - 1);
            for (int i = 0; i < this->_elements - 1; i++) {
                new_data[i] = result->_data[i + 1];
            }
            result->_data = std::move(new_data);
            result->_elements--;
        }
    }
    if (method == fractional) {
        for (int t = result->size() - 1; t > 0; t--) {
            if (result->_data[t - 1] != 0) {
                double stat = log(std::numeric_limits<T>::min() + fabs(this->_data[t] / this->_data[t - 1])); //note: DBL_MIN and fabs are used to avoid log(x<=0)
                double non_stat = log(fabs(this->_data[t]) + std::numeric_limits<T>::min());
                result->_data[t] = degree * stat + pow((1 - degree), fract_exponent) * non_stat;
            }
        }
        // Remove the first element from the unique_ptr
        std::unique_ptr<T[]> new_data(new T[this->_elements - 1]);
        for (int i = 0; i < this->_elements - 1; i++) {
            new_data[i] = result->_data[i + 1];
        }
        result->_data = std::move(new_data);
        result->_elements--;
    }
    if (method==deltamean){
        double sum=0;
        for (int i=0;i<this->_elements;i++){
            sum+=this->_data[i];
        }
        double x_mean=sum/this->_elements;
        for (int t=this->_elements-1;t>0;t--){
            result->_data[t]-=x_mean;
        }
        result->_elements--;
        for (int i = 0; i < result->_elements; i++) {
            result->_data[i] = result->_data[i + 1];
        }
    }
    return result;
}


// sorts the values of the vector via pairwise comparison
// default: ascending order;
// set 'false' flag for sorting in reverse order
template<typename T>
std::unique_ptr<Vector<T>> Vector<T>::sort(bool ascending) const {
    // make a copy
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->_elements);
    for (int i=0;i<this->_elements;i++){
        result->_data[i] = this->_data[i];
    }
    bool completed=false;
    while (!completed){
        completed=true; //let's assume this until proven otherwise
        for (int i=0;i<this->_elements-1;i++){
            if(ascending){
                if (result->_data[i] > result->_data[i+1]){
                    completed=false;
                    double temp=result->_data[i];
                    result->_data[i] = result->_data[i+1];
                    result->_data[i+1] = temp;
                }
            }
            else{
                if (result->_data[i] < result->_data[i+1]){
                    completed=false;
                    double temp=result->_data[i];
                    result->_data[i] = result->_data[i+1];
                    result->_data[i+1] = temp;
                }
            }
        }
    }
    return result;
}

// returns a randomly shuffled copy of the vector
template<typename T>
std::unique_ptr<Vector<T>> Vector<T>::shuffle() const {
    // make a copy
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->_elements);
    for (int i=0;i<this->_elements;i++){
        result->_data[i] = this->_data[i];
    }
    // iterate over vector elements and find a random second element to swap places
    for (int i=0;i<this->_elements;i++){
        int new_position=std::floor(Random<double>::uniform()*this->_elements);
        T temp=this->_data[new_position];
        result->_data[new_position] = result->_data[i];
        result->_data[i] = temp;
    }
    return result;
}

// performs linear regression with the source vector as
// x_data and a second vector as corresponding the y_data;
// the results will be stored in a struct;
// make sure that both vectors have the same number of
// elements (otherwise the surplus elements of the
// larger vector will be discarded)
template <typename T>
std::unique_ptr<LinReg<T>> Vector<T>::linear_regression(const Vector<T> &other) const {
    // create result struct
    int elements = std::min(this->_elements, other.get_elements());
    std::unique_ptr<LinReg<T>> result = std::make_unique<LinReg<T>>(elements);
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
        result->y_regression[n] = result->y_intercept + result->slope * this->_data[n];
        result->residuals[n] = other._data[n] - result->y_regression[n];
        result->SSR += result->residuals[n] * result->residuals[n];
    }
    // get slope
    result->slope = slope_num / (x_mdev2_sum + std::numeric_limits<T>::min());
    // get y intercept
    result->y_intercept = result->y_mean - result->slope * result->x_mean;
    // get r_squared
    result->SST = y_mdev2_sum;
    result->r_squared = 1 - result->SSR / (result->SST + std::numeric_limits<T>::min());

    return result;
}


// performs polynomial regression (to the specified power)
// with the source vector as the x data and a second vector
// as the corresponding y data;
// make sure that both vectors have the same number of
// elements (otherwise the surplus elements of the
// larger vector will be discarded)
template <typename T>
std::unique_ptr<PolyReg<T>> Vector<T>::polynomial_regression(const Vector<T> &other, const int power) const {
    // create result struct
    int elements=std::min(this->_elements, other.get_elements());
    std::unique_ptr<PolyReg<T>> result = std::make_unique<PolyReg<T>>(elements,power);

    // Create matrix of x values raised to different powers
    std::unique_ptr<Matrix<T>> X = std::make_unique<Matrix<T>>(elements, power + 1);
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

    return result;
}

// Performs correlation analysis on the source vector as x_data versus
// a second Vector<T> as y_data (passed in as a parameter)
// and returns the results in a struct std::unique_ptr<Correlation<T>>
template<typename T>
std::unique_ptr<Correlation<T>> Vector<T>::correlation(const Vector<T>& other) const {
    if (this->_elements != other._elements) {
        std::cout << "WARNING: Invalid use of method Vector<T>::correlation(); both vectors should have the same number of elements" << std::endl;
    }
    int elements=std::min(this->_elements, other._elements);    
    std::unique_ptr<Correlation<T>> result = std::make_unique<Correlation<T>>(elements);

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

    // Spearman correlation, assuming non-linear monotonic dependence
    std::unique_ptr<Vector<int>> rank_x = this->ranking();
    std::unique_ptr<Vector<int>> rank_y = other.ranking();
    double numerator=0;
    for (int i=0;i<elements;i++){
        numerator+=6*std::pow(rank_x->_data[i]-rank_y->_data[i],2);
    }
    result->Spearman_Rho=1-numerator/(elements*(std::pow(elements,2)-1));
    // test significance against null hypothesis
    double fisher_transform=0.5*std::log( (1 + result->Spearman_Rho) / (1 - result->Spearman_Rho) );
    result->z_score = sqrt((elements-3)/1.06)*fisher_transform;
    result->t_score = result->Spearman_Rho * std::sqrt((elements-2)/(1-std::pow(result->Spearman_Rho,2)));
    
    return result;
}

// return the covariance of the linear relation
// of the source vector versus a second vector
template<typename T>
double Vector<T>::covariance(const Vector<T>& other) const {
    if (this->_elements != other._elements) {
        std::cout << "WARNING: Invalid use of method Vector<T>::covariance(); both vectors should have the same number of elements" << std::endl;
    }
    int elements=std::min(this->_elements, other._elements);
    double mean_this = this->mean();
    double mean_other = other.mean();
    double cov = 0;
    for (int i = 0; i < elements; i++) {
        cov += (this->_data[i] - mean_this) * (other._data[i] - mean_other);
    }
    cov /= elements;
    return cov;
}


// returns a histogram of the source vector data
// with the specified number of bars and returns the 
// result as type struct Histogram<T>'
template <typename T>
std::unique_ptr<Histogram<T>> Vector<T>::histogram(uint bars) const {
    std::unique_ptr<Histogram<T>> histogram = std::make_unique<Histogram<T>>(bars);
    // get min and max value from sample
    histogram->min=this->_data[0];
    histogram->max=this->_data[0];
    for (int i=0;i<this->_elements;i++){
        histogram->min=std::fmin(histogram->min,this->_data[i]);
        histogram->max=std::fmax(histogram->max,this->_data[i]);
    }

    // get histogram x-axis scaling
    histogram->width = histogram->max - histogram->min;
    histogram->bar_width = histogram->width / bars;
    
    // set histogram x values, initialize count to zero
    for (int i=0;i<bars;i++){
        histogram->bar[i].lower_boundary = histogram->min + histogram->bar_width * i;
        histogram->bar[i].upper_boundary = histogram->min + histogram->bar_width * (i+1);
        histogram->bar[i].count=0;
    }

    // count absolute occurences per histogram bar
    for (int i=0;i<this->_elements;i++){
        histogram->bar[int((this->_data[i]-histogram->min)/histogram->bar_width)].abs_count++;
    }

    // convert to relative values
    for (int i=0;i<bars;i++){
        histogram->bar[i].rel_count=histogram->bar[i].abs_count/this->_elements;
    }
    return histogram;
}

// Vector binning, i.e. quantisizing into n bins,
// each represented by the mean of the values inside that bin;
// returning the result as pointer to a new Vector of size n
template<typename T>
std::unique_ptr<Vector<T>> Vector<T>::binning(const int bins){
    if (this->_elements == 0) {
        throw std::runtime_error("Cannot bin an empty vector.");
    }
    if (bins <= 0) {
        throw std::invalid_argument("Number of bins must be positive.");
    }
    if (bins >= this->_elements) {
        throw std::invalid_argument("Number of bins must be less than the number of elements in the vector.");
    }
    // prepare the data structure to put the results
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(bins);    
    result->fill->zeros();
    // get a sorted copy of the original data (ascending order)
    auto sorted = this->sort();
    // calculate bin size
    T min = this->min();
    T max = this->max();
    if (min == max) {
        // There's only one unique value in the vector, so we can't bin it
        throw std::runtime_error("Cannot bin a vector with only one unique value.");
    }
    T bin_size = (max - min) / bins;
    int bin = 0;
    int bin_items = 0;
    int i = 0;
    while (bin < bins && i < this->_elements) {
        // until bin is full
        while (sorted->_data[i] <= min + bin_size * (bin + 1)) {
            result->_data[bin] += this->_data[i];
            bin_items++;
            i++;
            if (i == this->_elements) {
                break;
            }
        }
        // calculate mean and move to next bin
        result->_data[bin] /= bin_items;
        bin++;
        bin_items = 0;
    }
    return result;
}


// Matrix transpose
template<typename T>
std::unique_ptr<Matrix<T>> Matrix<T>::transpose() const {
    // create a new matrix with swapped dimensions
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(this->_size[1], this->_size[0]);

    for(int i = 0; i < this->_size[0]; i++){
        for(int j = 0; j < this->_size[1]; j++){
            // swap indices and copy element to result
            result->set(j, i, this->get(i, j));
        }
    }
    return result;
}

// +=================================+   
// | Output                          |
// +=================================+

// prints the Vector/Matrix/Array to the console
template<typename T>
void Array<T>::print(std::string comment, std::string delimiter, std::string line_break, bool with_indices) const {
    if (comment!=""){
        std::cout << comment << std::endl;
    }

    if (this->_dimensions==1){
        // iterate over elements
        for (int i=0;i<this->_elements;i++){
            // add indices
            if (with_indices){
                std::cout << "[" << i << "]=";
            }
            // add value
            std::cout << this->_data[i];
            // add delimiter between elements (except after last value in row)
            if (i != this->_elements-1) {
                std::cout << delimiter;
            }         
        }
        // add line break character(s) to end of the row
        std::cout << line_break;        
        return;       
    }

    // create a vector for temporary storage of the current index (needed for indexing dimensions >=2);
    std::vector<int> index(this->_dimensions,0);
    std::fill(index.begin(),index.end(),0);   

    if (this->_dimensions==2){
        // iterate over rows
        for (int row=0; row < (this->_dimensions==1 ? 1 : this->_size[0]); row++) {
            // iterate over columns
            for (int col=0; col < (this->_dimensions==1 ? this->_size[0] : this->_size[1]); col++) {                
                // add indices
                if (with_indices) {
                    std::cout << "[" << row << "]" << "[" << col << "]=";
                }
                // add value
                index[0]=row; index[1]=col;
                std::cout << this->get(index);
                // add delimiter between columns (except after last value in row)
                if (col != this->_size[1]-1) {
                    std::cout << delimiter;
                }
            }
            // add line break character(s) to end of current row
            std::cout << line_break;            
        }
    }

    else { //=dimensions >=2
        // iterate over rows
        for (int row = 0; row < this->_size[0]; row++) {
            index[0] = row;
            // iterate over columns
            for (int col = 0; col < this->_size[1]; col++) {
                index[1] = col;
                // add opening brace for column
                std::cout << "{";
                // iterate over higher dimensions
                for (int d = 2; d < this->_dimensions; d++) {
                    // add opening brace for dimension
                    std::cout << "{";
                    // iterate over entries in the current dimension
                    for (int i = 0; i < this->_size[d]; i++) {
                        // update index
                        index[d] = i;
                        // add indices
                        if (with_indices) {
                            for (int dd = 0; dd < this->_dimensions; dd++) {
                                std::cout << "[" << index[dd] << "]";
                            }
                            std::cout << "=";
                        }
                        // add value
                        std::cout << this->get(index);
                        // add delimiter between values
                        if (i != this->_size[d] - 1) {
                            std::cout << delimiter;
                        }
                    }
                    // add closing brace for the current dimension
                    std::cout << "}";
                    // add delimiter between dimensions
                    if (d != this->_dimensions - 1) {
                        std::cout << delimiter;
                    }
                }
                // add closing brace for column
                std::cout << "}";
                // add delimiter between columns
                if (col != this->_size[1] - 1) {
                    std::cout << delimiter;
                }
            }
            // add line break character(s) to end of current row
            std::cout << line_break;
        }
    }
}

// helper function to convert an array to
// a std::initializer_list<int>
template<typename T>
std::initializer_list<int> Array<T>::array_to_initlist(int* arr, int size) const {
    return {arr, arr + size};
}  

// helper function to convert a std::initializer_list<int>
// to a one-dimensional integer array
template<typename T>
std::unique_ptr<int[]> Array<T>::initlist_to_array(const std::initializer_list<int>& lst) const {
    std::unique_ptr<int[]> arr(new int[lst.size()]);
    int i = 0;
    for (auto it = lst.begin(); it != lst.end(); ++it) {
        arr[i++] = *it;
    }
    return arr;
}

// +=================================+   
// | Activation Functions            |
// +=================================+

template<typename T>
void Array<T>::Activation::Function::ReLU(){
    for (int i=0;i<this->get_elements();i++){
        this->_data[i] *= this->_data[i]>0;
    }                        
}

template<typename T>
void Array<T>::Activation::Function::lReLU(){
    for (int i=0;i<this->get_elements();i++){
        this->_data[i] = this->_data[i]>0 ? this->_data[i] : this->_data[i]*alpha;
    }                        
}

template<typename T>
void Array<T>::Activation::Function::ELU(){
    for (int i=0;i<this->get_elements();i++){
        this->_data[i] = this->_data[i]>0 ? this->_data[i] : alpha*(std::exp(this->_data[i])-1); 
    }                        
}

template<typename T>
void Array<T>::Activation::Function::sigmoid(){
    for (int i=0;i<this->get_elements();i++){
        this->_data[i] = 1/(1+std::exp(-this->_data[i])); 
    } 
}

template<typename T>
void Array<T>::Activation::Function::tanh(){
    for (int i=0;i<this->get_elements();i++){
        this->_data[i] = std::tanh(this->_data[i]);
    }                        
};

template<typename T>
void Array<T>::Activation::Function::softmax(){
    // TODO !!
};           

template<typename T>
void Array<T>::Activation::Function::ident(){
    // do nothing
    return;
}

template<typename T>
void Array<T>::Activation::Derivative::ReLU(){
    for (int i=0;i<this->get_elements();i++){
        this->_data[i] = this->_data[i]>0 ? 1 : 0;
    }                        
}

template<typename T>
void Array<T>::Activation::Derivative::lReLU(){
    for (int i=0;i<this->get_elements();i++){
        this->_data[i] = this->_data[i]>0 ? 1 : alpha;
    }                        
}

template<typename T>
void Array<T>::Activation::Derivative::ELU(){
    for (int i=0;i<this->get_elements();i++){
        this->_data[i] = this->_data[i]>0 ? 1 : alpha*std::exp(this->_data[i]);
    }                        
}

template<typename T>
void Array<T>::Activation::Derivative::sigmoid(){
    for (int i=0;i<this->get_elements();i++){
        this->_data[i] = std::exp(this->_data[i])/std::pow(std::exp(this->_data[i])+1,2); 
    } 
}

template<typename T>
void Array<T>::Activation::Derivative::tanh(){
    for (int i=0;i<this->get_elements();i++){
        this->_data[i] = 1-std::pow(std::tanh(this->_data[i]),2);
    }                        
}
template<typename T>
void Array<T>::Activation::Derivative::softmax(){
    // TODO !!
};

template<typename T>
void Array<T>::Activation::Derivative::ident(){
    for (int i=0;i<this->get_elements();i++){
        this->_data[i] = 1;
    }                         
}

template<typename T>
void Array<T>::Activation::function(Method method){
    switch (method){
        case ReLU: Function::ReLU(); break;
        case lReLU: Function::lReLU(); break;
        case ELU: Function::ELU(); break;
        case sigmoid: Function::sigmoid(); break;   
        case tanh: Function::tanh(); break;
        case softmax: Function::softmax(); break;
        case ident: Function::ident(); break;
        default: /* do nothing */ break;
    }
}

template<typename T>
void Array<T>::Activation::derivative(Method method){
    switch (method){
        case ReLU: Derivative::ReLU(); break;
        case lReLU: Derivative::lReLU(); break;
        case ELU: Derivative::ELU(); break;
        case sigmoid: Derivative::sigmoid(); break;   
        case tanh: Derivative::tanh(); break;
        case softmax: Derivative::softmax(); break;
        case ident: Derivative::ident(); break;
        default: /* do nothing */ break;
    }
}


// +=================================+   
// | Feature Scaling                 |
// +=================================+

template<typename T>
void Array<T>::Scaling::minmax(T min,T max){
    T data_min = arr._data.min();
    T data_max = arr._data.max();
    double factor = (max-min) / (data_max-data_min);
    for (int i=0; i<arr.get_elements(); i++){
        arr._data[i] = (arr._data[i] - data_min) * factor + min;
    }
}

template<typename T>
void Array<T>::Scaling::mean(){
    T data_min = arr._data.min();
    T data_max = arr._data.max();
    T range = data_max - data_min;
    double mean = arr.mean();
    for (int i=0; i<arr.get_elements(); i++){
        arr._data[i] = (arr._data[i] - mean) / range;
    }
}

template<typename T>
void Array<T>::Scaling::standardized(){
    double mean = arr.mean();
    double stddev = arr.stddev();
    for (int i=0; i<arr.get_elements(); i++){
        arr._data[i] = (arr._data[i] - mean) / stddev;
    }                    
}

template<typename T>
void Array<T>::Scaling::unit_length(){
    // calculate the Euclidean norm of the data array
    T norm = 0;
    int elements = arr.get_elements();
    for (int i = 0; i < elements; i++) {
        norm += std::pow(arr._data[i], 2);
    }
    if (norm==0){return;}
    norm = std::sqrt(norm);
    // scale the data array to unit length
    for (int i = 0; i < elements; i++) {
        arr._data[i] /= norm;
    }                    
}