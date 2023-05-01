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
// as const Vector<int>&
template<typename T>
void Array<T>::set(const Vector<int>& index, const T value){
    // copy Vector to std::vector
    std::vector<int> temp(index.size());
    for (int i=0;i<index.size();i++){
        temp[i] = index[i];
    }
    this->data[get_element(temp)] = value;
}

// assigns a value via its flattened index
template<typename T>
void Array<T>::set(const int index, const T value){
    this->data[index] = value;
};

// assigns a value to an element of a two-dimensional
// matrix via its index (index parameters as const int)
template<typename T>
void Matrix<T>::set(const int row, const int col, const T value){
    this->data[std::fmin(this->data_elements-1,col + row*this->dim_size[1])] = value;
}

// returns the value of an array element via its index
template<typename T>
T Array<T>::get(const std::initializer_list<int>& index) const {
    int element=get_element(index);
    return this->data[element];
}

// returns the value of an array element via
// its index (as type const std::vector<int>&)
template<typename T>
T Array<T>::get(const std::vector<int>& index) const {
    int element=get_element(index);
    return this->data[element];
}

// returns the value of an array element via
// its index (as type const Vector<int>&)
template<typename T>
T Array<T>::get(const Vector<int>& index) const {
    // copy Vector to std::vector
    std::vector<int> temp(index.size());
    for (int i=0;i<index.size();i++){
        temp[i] = index[i];
    }    
    int element=get_element(temp);
    return this->data[element];
}

// returns the value of a 2d matrix element via its index
template<typename T>
T Matrix<T>::get(const int row, const int col) const {
    return this->data[std::fmin(this->data_elements-1,col + row*this->dim_size[1])];
}

// returns the number of dimensions of the array
template<typename T>
int Array<T>::get_dimensions() const {
    return this->dimensions;
}

// returns the number of elements of the specified array dimension
template<typename T>
int Array<T>::get_size(int dimension) const {
    return this->dim_size[dimension];
}

// returns the total number of elements across the array
// for all dimensions
template<typename T>
int Array<T>::get_elements() const {
    return this->data_elements;
}

// converts a multidimensional index (represented by the values of
// a std::initializer_list<int>) to a one-dimensional index (=as a scalar)
template<typename T>
int Array<T>::get_element(const std::initializer_list<int>& index) const {
    // check for invalid index dimensions
    if (index.size()>this->dimensions){
        std::cout << "WARNING: The method 'get_element() has been used with an invalid index:";
        // print the invalid index to the console
        auto iterator = index.begin();
        for (;iterator!=index.end();iterator++){
            std::cout << " " << *iterator;
        }
        std::cout << "\nThe corresponding array has the following dimensions:";
        for (int i=0;i<this->dimensions;i++){
            std::cout << " " << i;
        }
        std::cout << std::endl;
        return -1;
    }
    // deal with the special case of single dimension arrays ("Vector<T>")
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
    result=std::fmin(this->data_elements-1,result);
    return result;
}

// converts a multidimensional index (represented by C-style
// integer array) to a one-dimensional index (=as a scalar)
template<typename T>
int Array<T>::get_element(const std::vector<int>& index)  const {
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
    result=std::fmin(this->data_elements-1,result);
    return result;
}

// converts a one-dimensional ('flattened') index back into
// its multi-dimensional equivalent
template<typename T>
std::vector<int> Array<T>::get_index(int flattened_index) const {
    std::vector<int> result(this->dimensions);
    // deal with the special case of single dimension arrays ("Vector<T>")
    if (this->dimensions == 1){
        result[0] = flattened_index;
        return result;
    }
    // initialize iterator to counter of second last dimension
    auto iterator = result.end()-1;
    // initialize dimension index to last dimension
    int i = this->dimensions-1;
    // decrement iterator down to first dimension
    for (; iterator >= result.begin(); i--, iterator--){
        // calculate index for this dimension
        result[i] = flattened_index % this->dim_size[i];
        // divide flattened_index by size of this dimension
        flattened_index /= this->dim_size[i];
    }
    return result;
}

// Return the subspace size
template<typename T>
int Array<T>::get_subspace(int dimension) const {
    return this->subspace_size[dimension];
}

// +=================================+   
// | Distribution Properties         |
// +=================================+

// returns the lowest value of all values of a vector, matrix or array
template<typename T>
T Array<T>::min() const {
    if (this->data_elements==0){
        std::cout << "WARNING: improper use of method Array<T>::min(): not defined for empty array!\n";
        return T(NAN);
    }
    T result = this->data[0];
    for (int i=0;i<this->data_elements;i++){
        result = std::fmin(result, this->data[i]);
    }
    return result;
}

// returns the highest value of all values of a vector, matrix or array
template<typename T>
T Array<T>::max() const {
    if (this->data_elements==0){
        std::cout << "WARNING: improper use of method Array<T>::min(): not defined for empty array!\n";
        return T(NAN);
    }
    T result = this->data[0];
    for (int i=0;i<this->data_elements;i++){
        result = std::fmax(result, this->data[i]);
    }
    return result;
}

// returns the arrithmetic mean of all values a vector, matrix or array
template<typename T>
double Array<T>::mean() const {
    double sum=0;
    for (int n=0;n<this->data_elements;n++){
        sum+=this->data[n];
    }
    return sum/this->data_elements;
}

// returns the median of all values of a vector, martrix or array
template<typename T>
double Array<T>::median() const {
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

// find the 'mode', i.e. the item that occurs the most number of times
template<typename T>
T Array<T>::mode() const {
    // Sort the array in ascending order
    auto sorted = this->sort();
    // Create an unordered map to store the frequency of each element
    std::unordered_map<T, size_t> freq_map;
    for (size_t i = 0; i < this->data_elements; i++) {
        freq_map[sorted->data[i]]++;
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
    for (int i = 0; i < this->data_elements; i++) {
        double deviation = static_cast<double>(this->data[i]) - mean;
        sum_of_squares += deviation * deviation;
    }
    return sum_of_squares / static_cast<double>(this->data_elements);
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
    for (size_t i = 0; i < this->data_elements; i++) {
        skewness += std::pow((this->data[i] - this->mean()) / this->stddev(), 3);
    }
    skewness /= this->data_elements;
    return skewness;
}

// returns the kurtosis of all data of the vector/matrix/array
template<typename T>
double Array<T>::kurtosis() const {
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
    T result=0;
    for (int i=0; i<this->data_elements; i++){
        result+=this->data[i];
    }
    return result;
}

// elementwise addition of the specified value to all values of the array
template<typename T>
Array<T> Array<T>::operator+(const T value) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i <this->data_elements; i++){
        result->data[i]=this->data[i]+value;
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
        result->fill.values(T(NAN));
    }
    else {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]+other.data[i];
        }
    }
    return std::move(*result);
}

// prefix increment operator;
// increments the values of the array by +1,
// returns a reference to the source array itself
template<typename T>
Array<T>& Array<T>::operator++(){
    for (int i=0; i<this->data_elements; i++){
        this->data[i]+=1;
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
    for (int i=0; i<this->data_elements; i++){
        temp->data[i] = this->data[i];
        this->data[i]++;
    }
    return std::move(*temp);
}

// elementwise addition of the specified
// value to the elements of the array
template<typename T>
void Array<T>::operator+=(const T value) {
    for (int i=0; i<this->data_elements; i++){
        this->data[i]+=value;
    }
}

// elementwise addition of the values of the second
// array to the corresponding values of the current array;
// the dimensions of the arrays must match!
// the function will otherwise turn the source array into
// a NAN array!
template<typename T>
void Array<T>::operator+=(const Array<T>& other){
    if (!equalsize(other)){
        this->fill.values(T(NAN));
        return;
    }
    for (int i=0; i<this->data_elements; i++){
        this->data[i]+=other.data[i];
    }
}

// +=================================+   
// | Substraction                    |
// +=================================+

// elementwise substraction of the specified value from all values of the array
template<typename T>
Array<T> Array<T>::operator-(const T value)  const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i <this->data_elements; i++){
        result->data[i]=this->data[i]-value;
    }
    return std::move(*result);
}

// returns the resulting array of the elementwise substraction of
// two array of equal dimensions (this minus other);
// will return a NAN array if the dimensions don't match!
template<typename T>
Array<T> Array<T>::operator-(const Array<T>& other) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    if (!equalsize(other)){
        result->fill.values(T(NAN));
    }
    else {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]-other.data[i];
        }
    }
    return std::move(*result);
}

// prefix decrement operator;
// first decrements the values of the array by -1,
// then returns the modified array
template<typename T>
Array<T>& Array<T>::operator--(){
    for (int i=0; i<this->data_elements; i++){
        this->data[i]-=1;
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
    Array<T> temp = this->copy();
    for (int i=0; i<this->data_elements; i++){
        this->data[i]-=1;
    }
    return std::move(temp);
}

// elementwise substraction of the specified
// value from the elements of the array
template<typename T>
void Array<T>::operator-=(const T value) {
    for (int i=0; i<this->data_elements; i++){
        this->data[i]-=value;
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
    if (!equalsize(other)){
        this->fill.values(T(NAN));
        return;
    }
    for (int i=0; i<this->data_elements; i++){
        this->data[i]-=other.data[i];
    }
}

// +=================================+   
// | Multiplication                  |
// +=================================+

// returns the product reduction, i.e. the result
// of all individual elements of the array
template<typename T>
T Array<T>::product() const {
    if (this->data_elements==0){
        return T(NAN);
    }
    T result = this->data[0];
    for (int i=1; i<this->data_elements; i++){
        result*=this->data[i];
    }
    return result;
}

// elementwise multiplication with a scalar
template<typename T>
Array<T> Array<T>::operator*(const T factor) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0; i<this->data_elements; i++){
        result->data[i] = this->data[i] * factor;
    }
    return std::move(*result);
}

// elementwise multiplication (*=) with a scalar
template<typename T>
void Array<T>::operator*=(const T factor){
    for (int i=0; i<this->data_elements; i++){
        this->data[i]*=factor;
    }
}

// elementwise multiplication of the values of the current
// array with the corresponding values of a second array,
// resulting in the 'Hadamard product';
// the dimensions of the two arrays must match!
// the function will otherwise return a NAN array!
template<typename T>
Array<T> Array<T>::Hadamard_product(const Array<T>& other)  const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    if(!equalsize(other)){
        result->fill.values(T(NAN));
    }
    else {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]*other.data[i];
        }
    }
    return std::move(*result);
}

// Array tensor reduction ("tensor dotproduct")
template<typename T>
Array<T> Array<T>::tensordot(const Array<T>& other, const std::vector<int>& axes) const {
    // check that the number of axes to contract is the same for both tensors
    if (axes.size() != other.dimensions) {
        throw std::invalid_argument("Number of contraction axes must be the same for both tensors.");
    } 
    
    // check that the sizes of the axes to contract are the same for both tensors
    for (int i = 0; i < axes.size(); i++) {
        if (axes.size() != other.dimensions) {
            throw std::invalid_argument("Invalid use of function Array<T>::tensordot(). Sizes of contraction axes must be the same for both tensors.");
        }         
    }
    
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
    
    // create a new tensor to hold the result
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(out_dims);
    
    // perform the tensor contraction
    std::vector<int> index1(this->dimensions, 0);
    std::vector<int> index2(other.dimensions, 0);
    std::vector<int> index_out(out_dims.size(), 0);
    int contractionsize = 1;
    for (int i = 0; i < axes.size(); i++) {
        contractionsize *= this->dim_size[axes[i]];
    }
    for (int i = 0; i < result->dim_size[0]; i++) {
        index_out[0] = i;
        for (int j = 0; j < contractionsize; j++) {
            for (int k = 0; k < axes.size(); k++) {
                index1[axes[k]] = j % this->dim_size[axes[k]];
                index2[axes[k]] = j % other.dim_size[axes[k]];
            }
            for (int k = 1; k < out_dims.size(); k++) {
                int size = k <= axes[0] ? this->dim_size[k - 1] : other.dim_size[k - axes.size() - 1];
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
    
    return std::move(*result);
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
    if (!this->equalsize(other)){
        std::cout << "WARNING: Invalid usage. The method 'Array<T>::dotproduct() has been used with unequal array shapes:" << std::endl;
        std::cout << "- source array shape:";
        for (int i=0;i<this->dimensions;i++){
            std::cout << " " << this->dim_size[i];
        }
        std::cout << std::endl;
        std::cout << "- second array shape:";
        for (int i=0;i<other.dimensions;i++){
            std::cout << " " << other.dim_size[i];
        }
        std::cout << std::endl;        
        return -1;
    }
    T result = 0;
    for (int i = 0; i < this->data_elements; i++) {
        result += this->data[i] * other.data[i];
    }
    return result;
}


// vector dotproduct ("scalar product")
template<typename T>
T Vector<T>::dotproduct(const Vector<T>& other) const {
    if (this->data_elements != other.get_elements()){
        return T(NAN);
    }
    T result = 0;
    for (int i = 0; i < this->data_elements; i++){
        result += this->data[i] * other.data[i];
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
Matrix<T> Matrix<T>::tensordot(const Matrix<T>& other) const {
    // Create the resulting matrix
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(this->size[0], other.size[1]);
    // Check if the matrices can be multiplied
    if (this->size[1] != other.size[0]){
        result->fill.values(T(NAN));
        return result;
    }
    // Compute the dot product
    for (int i = 0; i < this->size[0]; i++) {
        for (int j = 0; j < other.size[1]; j++) {
            T sum = 0;
            for (int k = 0; k < this->size[1]; k++) {
                sum += this->get(i, k) * other.get(k, j);
            }
            result->set(i, j, sum);
        }
    }
    return std::move(*result);
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
Array<T> Array<T>::operator/(const T quotient) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0; i<this->data_elements; i++){
        result->data[i]=this->data[i]/quotient;
    }
    return std::move(*result);
}

// elementwise division (/=) by a scalar
template<typename T>
void Array<T>::operator/=(const T quotient){
    for (int i=0; i<this->data_elements; i++){
        this->data[i]/=quotient;
    }
}

// elementwise division of the values of the current
// array by the corresponding values of a second array,
// resulting in the 'Hadamard division';
// the dimensions of the two arrays must match!
// the function will otherwise return a NAN array!
template<typename T>
Array<T> Array<T>::Hadamard_division(const Array<T>& other)  const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    if(!equalsize(other)){
        result->fill.values(T(NAN));
    }
    else {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]*other.data[i];
        }
    }
    return std::move(*result);
}

// +=================================+   
// | Modulo                          |
// +=================================+

// elementwise modulo operation, converting the array values
// to the remainders of their division by the specified number
template<typename T>
void Array<T>::operator%=(const double num){
    for (int i=0; i<this->data_elements; i++){
        this->data[i]%=num;
    }
}

// elementwise modulo operation, resulting in an array that
// contains the remainders of the division of the values of
// the original array by the specified number
template<typename T>
Array<double> Array<T>::operator%(const double num) const {
    std::unique_ptr<Array<double>> result = std::make_unique<Array<double>>(this->dim_size);
    for (int i=0; i<this->data_elements; i++){
        result->data[i]=this->data[i]%num;
    }
    return std::move(*result);
}

// +=================================+   
// | Exponentiation                  |
// +=================================+

// elementwise exponentiation to the power of
// the specified exponent
template<typename T>
Array<T> Array<T>::pow(const T exponent){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0; i<this->data_elements; i++){
        result->data[i]=std::pow(this->data[i], exponent);
    }
    return std::move(*result);
}

// elementwise exponentiation to the power of
// the corresponding values of the second array;
// the dimensions of the two array must match!
// the function will otherwise return a NAN array!
template<typename T>
Array<T> Array<T>::pow(const Array<T>& other){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    if (!equalsize(other)){
        result->fill.values(T(NAN));
        return std::move(*result);
    }
    else {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=std::pow(this->data[i], other.data[i]);
        }
    }
    return std::move(*result);
}

// converts the individual values of the array
// elementwise to their square root
template<typename T>
Array<T> Array<T>::sqrt(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0; i<this->data_elements; i++){
        result->data[i]=std::sqrt(this->data[i]);
    }
    return std::move(*result);
}

// converts the individual values of the array
// elementwise to their natrual logarithm
template<typename T>
Array<T> Array<T>::log(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0; i<this->data_elements; i++){
        result->data[i]=std::log(this->data[i]);
    }
    return std::move(*result);
}

// converts the individual values of the array
// elementwise to their base-10 logarithm
template<typename T>
Array<T> Array<T>::log10(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0; i<this->data_elements; i++){
        result->data[i]=std::log10(this->data[i]);
    }
    return std::move(*result);
}

// +=================================+   
// | Rounding                        |
// +=================================+

// rounds the values of the array elementwise
// to their nearest integers
template<typename T>
Array<T> Array<T>::round(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::round(this->data[i]);
    }
    return std::move(*result);
}

// rounds the values of the array elementwise
// to their next lower integers
template<typename T>
Array<T> Array<T>::floor(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::floor(this->data[i]);
    }
    return std::move(*result);
}

// returns a copy of the array that stores the values as rounded
// to their next higher integers
template<typename T>
Array<T> Array<T>::ceil(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::ceil(this->data[i]);
    }
    return std::move(*result);
}

// returns a copy of the array that stores the
// absolute values of the source array
template<typename T>
Array<T> Array<T>::abs(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::fabs(this->data[i]);
    }
    return std::move(*result);
}

// +=================================+   
// | Min, Max                        |
// +=================================+
template<typename T>
Array<T> Array<T>::min(const T value){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::fmin(this->data[i], value);
    }
    return std::move(*result);
}

template<typename T>
Array<T> Array<T>::max(const T value){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::fmax(this->data[i], value);
    }
    return std::move(*result);
}

template<typename T>
Array<T> Array<T>::min(const Array<T>& other){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    if (!equalsize(other)){
        result->fill.values(T(NAN));
        return std::move(*result);
    }    
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::fmin(this->data[i], other.data[i]);
    }
    return std::move(*result);
}

template<typename T>
Array<T> Array<T>::max(const Array<T>& other){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    if (!equalsize(other)){
        result->fill.values(T(NAN));
        return std::move(*result);
    }       
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::fmax(this->data[i], other.data[i]);
    }
    return std::move(*result);
}

// +=================================+   
// | Trigonometric Functions         |
// +=================================+

// elementwise application of the cos() function
template<typename T>
Array<T> Array<T>::cos(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::cos(this->data[i]);
    }
    return std::move(*result);
}

// elementwise application of the sin() function
template<typename T>
Array<T> Array<T>::sin(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::sin(this->data[i]);
    }
    return std::move(*result);
}

// elementwise application of the tan() function
template<typename T>
Array<T> Array<T>::tan(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::tan(this->data[i]);
    }
    return std::move(*result);
}

// elementwise application of the acos() function
template<typename T>
Array<T> Array<T>::acos(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::acos(this->data[i]);
    }
    return std::move(*result);
}

// elementwise application of the asin() function
template<typename T>
Array<T> Array<T>::asin(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::asin(this->data[i]);
    }
    return std::move(*result);
}

// elementwise application of the atan() function
template<typename T>
Array<T> Array<T>::atan(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::atan(this->data[i]);
    }
    return std::move(*result);
}

// +=================================+   
// | Hyperbolic Functions            |
// +=================================+

// elementwise application of the cosh() function
template<typename T>
Array<T> Array<T>::cosh(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::cosh(this->data[i]);
    }
    return std::move(*result);
}

// elementwise application of the sinh() function
template<typename T>
Array<T> Array<T>::sinh(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::sinh(this->data[i]);
    }
    return std::move(*result);
}

// elementwise application of the tanh() function
template<typename T>
Array<T> Array<T>::tanh(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::tanh(this->data[i]);
    }
    return std::move(*result);
}

// elementwise application of the acosh() function
template<typename T>
Array<T> Array<T>::acosh(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::acosh(this->data[i]);
    }
    return std::move(*result);
}

// elementwise application of the asinh() function
template<typename T>
Array<T> Array<T>::asinh(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::asinh(this->data[i]);
    }
    return std::move(*result);
}

// elementwise application of the atanh() function
template<typename T>
Array<T> Array<T>::atanh(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=std::atanh(this->data[i]);
    }
    return std::move(*result);
}

// +=================================+   
// | Find, Replace                   |
// +=================================+

// returns the number of occurrences of the specified value
template<typename T>
int Array<T>::find(const T value) const {
    int counter=0;
    for (int i=0; i<this->data_elements; i++){
        counter+=(this->data[i]==value);
    }
    return counter;
}

// replace all findings of given value by specified new value
template<typename T>
Array<T> Array<T>::replace(const T old_value, const T new_value){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0; i<this->data_elements; i++){
        result->data[i] = this->data[i]==old_value ? new_value : this->data[i];
    }
    return std::move(*result);
}

// returns 1 for all positive elements and -1 for all negative elements
template<typename T>
Array<char> Array<T>::sign(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0; i<this->data_elements; i++){
        result->data[i] = this->data[i]>=0 ? 1 : -1;
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
Array<T> Array<T>::function(const T (*pointer_to_function)(T)){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->dim_size);
    for (int i=0; i<this->data_elements; i++){
        result->data[i] = pointer_to_function(this->data[i]);
    }
    return std::move(*result);
}

// +=================================+   
// | Assignment                      |
// +=================================+

// copy assignment operator with second Array as argument:
// copies the values from a second array into the values of
// the current array;
template<typename T>
Array<T>& Array<T>::operator=(const Array<T>& other) {
    // Check for self-assignment
    if (this != &other) {
        if (!equalsize(other)){
            // Allocate new memory for the array
            std::unique_ptr<T[]> newdata = std::make_unique<T[]>(other.get_elements());
            // Copy the elements from the other array to the new array
            std::copy(other.data.get(), other.data.get() + other.get_elements(), newdata.get());
            // Assign the new data to this object
            this->data = std::move(newdata);
            this->data_elements = other.get_elements();
            this->dim_size = other.dim_size;
            this->subspace_size = other.subspace_size;
        }
        else {
            std::copy(other.data.get(), other.data.get() + other.get_elements(), this->data.get());
        }
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
        if (!equalsize(other)){
            // Allocate new memory for the array
            std::unique_ptr<T[]> newdata = std::make_unique<T[]>(other.et_elements());
            // Copy the elements from the other array to the new array
            std::copy(other.data.get(), other.data.get() + other.get_elements(), newdata.get());
            // Assign the new data to this object
            this->data = std::move(newdata);
            this->data_elements = other.get_elements();
            this->size = other.size;
        }
        else {
            std::copy(other.data.get(), other.data.get() + other.get_elements(), this->get());
        }
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
        if (this->data_elements != other.get_elements()){
            // Allocate new memory for the array
            std::unique_ptr<T[]> newdata = std::make_unique<T[]>(other.get_elements());
            // Copy the elements from the other array to the new array
            std::copy(other.data.get(), other.data.get() + other.get_elements(), newdata.get());
            // Assign the new data to this object
            this->data = std::move(newdata);
            this->data_elements = other.get_elements();
            this->size = other.size;
        }
        else {
            std::copy(other.data.get(), other.data.get() + other.get_elements(), this->get());
        }
    }
    return *this;
}

// indexing operator [] for reading
template<typename T>
T& Vector<T>::operator[](const int index) const {
    if (index>=this->data_elements || index<0){
        throw std::out_of_range("Index out of range");
    }
    return this->data[index];
}

// indexing operator [] for writing
template<typename T>
T& Vector<T>::operator[](const int index) {
    if (index>=this->data_elements || index<0){
        throw std::out_of_range("Index out of range");
    }
    return this->data[index];
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
    for (int i=0; i<this->data_elements; i++){
        result->data[i]=this->data[i]>value;
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
    for (int i=0; i<this->data_elements; i++){
        result->data[i]=this->data[i]>=value;
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
    for (int i=0; i<this->data_elements; i++){
        result->data[i]=this->data[i]==value;
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
    for (int i=0; i<this->data_elements; i++){
        result->data[i]=this->data[i]!=value;
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
    for (int i=0; i<this->data_elements; i++){
        result->data[i]=this->data[i]<value;
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
    for (int i=0; i<this->data_elements; i++){
        result->data[i]=this->data[i]<=value;
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
        result->fill.values(T(NAN));
    }
    else {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]>other.data[i];
        }
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
        result->fill.values(T(NAN));
    }
    else {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]>=other.data[i];
        }
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
        result->fill.values(T(NAN));
    }
    else {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]==other.data[i];
        }
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
        result->fill.values(T(NAN));
    }
    else {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]!=other.data[i];
        }
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
        result->fill.values(T(NAN));
    }
    else {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]<other.data[i];
        }
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
        result->fill.values(T(NAN));
    }
    else {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]<=other.data[i];
        }
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
    for (int i=0; i<this->data_elements; i++){
        result->data[i]=this->data[i]&&value;
    }
    return std::move(*result);
}

// returns a boolean array as the result of the
// logical OR of the source array and the specified
// boolean argument value
template<typename T>
Array<bool> Array<T>::operator||(const bool value) const {
    std::unique_ptr<Array<bool>> result(new Array(this->size));
    for (int i=0; i<this->data_elements; i++){
        result->data[i]=this->data[i]||value;
    }
    return std::move(*result);
}

// returns a boolean array as the result of the
// logical NOT of the source array
template<typename T>
Array<bool> Array<T>::operator!() const {
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->size);
    for (int i=0; i<this->data_elements; i++){
        result->data[i]=!this->data[i];
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
        result->fill.values(T(NAN));
    }
    else {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]&&other.data[i];
        }
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
        result->fill.values(T(NAN));
    }
    else {
        for (int i=0; i<this->data_elements; i++){
            result->data[i]=this->data[i]||other.data[i];
        }
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
    Array<C> result = std::make_unique<Array<C>>(this->dim_size);
    for (int i=0; i<this->data_elements; i++){
        result->data[i]=C(this->data[i]);
    }
    return result;
}

// +=================================+   
// | pointer                         |
// +=================================+

// dereference operator
template<typename T>
Array<T&> Array<T>::operator*() {
    Array<T&> result(this->dim_size);
    for (int i = 0; i < this->data_elements; i++) {
        result.data[i] = *(this->data[i]);
    }
    return result;
}

// 'address-of' operator
template<typename T>
Array<T*> Array<T>::operator&() {
    Array<T*> result(this->dim_size);
    for (int i = 0; i < this->data_elements; i++) {
        result.data[i] = &(this->data[i]);
    }
    return result;
}
// +=================================+   
// | Class Conversion                |
// +=================================+

// flattens an array or matrix into a one-dimensional vector
template<typename T>
Vector<T> Array<T>::flatten() const {
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->data_elements);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=this->data[i];
    }
    return std::move(*result);
}

// converts an array into a 2d matrix of the specified size;
// if the new matrix has less elements in any of the dimensions,
// the surplus elements of the source array will be ignored;
// if the new matrix has more elements than the source array, these
// additional elements will be initialized with the specified value
template<typename T>
Matrix<T> Array<T>::asMatrix(const int rows, const int cols, T init_value) const {
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(rows,cols);
    result->fill.values(init_value);
    std::vector<int> index(this->dimensions);
    // reset the indices of higher dimensions to all zeros
    for (int d=0;d<this->dimensions;d++){
        index[d]=0;
    }
    // iterate over first and second dimension, i.e. keeping
    // the higher dimensions at index zero
    for (int row=0;row<this->dim_size[0];row++){
        index[0]=row;
        for (int col=0;col<this->dim_size[1];col++){
            index[1]=col;
            result->set(row,col,this->get(index));
        }
    }
    return std::move(*result);
}

// converts a matrix into another 2d matrix of different size;
// if the new matrix has less elements in any of the two dimensions,
// the surplus elements of the source will be ignored; if the new matrix
// has more elements, these additional elements will be initialized with
// the specified value
template<typename T>
Matrix<T> Matrix<T>::asMatrix(const int rows, const int cols, T init_value) const {
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(rows,cols);
    result->fill.values(init_value);
    for (int r=0;r<std::fmin(rows,this->dim_size[0]);r++){
        for (int c=0;c<std::fmin(cols,this->dim_size[1]);c++){
            result->set(r,c,this->get(r,c));
        }
    }
    return std::move(*result);
}

template<typename T>
void Matrix<T>::reshape(const int rows, const int cols){
    // TO DO !!
}

template<typename T>
Vector<T> Vector<T>::asVector(const std::vector<T>& other){
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(other.size());
    for (int i=0;i<other.size();i++){
        result.data[i] = other[i];
    }
    return std::move(*result);
}

// converts a vector into a 2d matrix of the specified size;
// the extra rows will be initialized with zeros
// if the new matrix has less columns than the source vector
// has elements, the surplus elements (for row 0) of the source
// will be ignored; if the new matrix has more columns than the
// source vector has elements, these additional elements will
// be initialized with the specified init value;
template<typename T>
Matrix<T> Vector<T>::asMatrix(const int rows, const int cols, T init_value) const {
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(rows,cols);
    result->fill.values(init_value);
    for (int i=0;i<std::fmin(this->data_elements,cols);i++){
        result->set(0,i, this->data[i]);
    }
    return std::move(*result);
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
Matrix<T> Array<T>::asMatrix() const {
    std::unique_ptr<Matrix<T>> result;;
    if (this->dimensions==2){
        result=std::make_unique<Matrix<T>>(this->dim_size[0],this->dim_size[1]);
        int index;
        for (int row=0;row<this->dim_size[0];row++){
            for (int col=0;col<this->dim_size[1];col++){
                index = col +row*this->dim_size[1];
                result->data[index] = this->data[index];
            }
        }
    }
    else {
        result=std::make_unique<Matrix<T>>(this->dim_size[0],this->dim_size[1]);
        std::vector<int> index(this->dimensions);
        // reset dimension indices to all zeros
        std::fill(index.begin(),index.end(),0);
        // iterate over first and second dimension, i.e. keeping
        // the higher dimensions at index zero
        for (int row=0;row<this->dim_size[0];row++){
            index[0]=row;
            for (int col=0;col<this->dim_size[1];col++){
                index[1]=col;
                // assign matching elements with values from source
                result->set(row,col,this->get(index));
            }
        }
    }
    return std::move(*result);
}

// converts a vector into a single row matrix with
// unchanged number of elements
template<typename T>
Matrix<T> Vector<T>::asMatrix() const {
    std::unique_ptr<Matrix<T>> result;;
    result=std::make_unique<Matrix<T>>(1,this->data_elements);
    for (int i=0;i<this->data_elements;i++){
        result->data[i]=this->data[i];
    }
    return std::move(*result);
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
Array<T> Array<T>::asArray(const std::initializer_list<int>& init_list, T init_value) const {
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(init_list);
    result->fill.values(init_value);
    // reset result index to all zeros
    std::vector<int> result_index(result->get_dimensions());
    std::fill(result_index.begin(),result_index.end(),0);
    // reset source index to all zeros
    std::vector<int> source_index(this->dimensions);
    std::fill(source_index.begin(),source_index.end(),0);
    // iterate over source dimensions
    for (int d=0;d<std::fmin(this->dimensions,result->get_dimensions());d++){
        // iterate over elements of given dimension
        for (int i=0;i<std::fmin(this->size[d],result->get_size(d));i++){
            result_index[d]=i;
            source_index[d]=i;
            result->set(result_index,this->data[get_element(source_index)]);
        }
    }
    return std::move(*result);
}

template<typename T>
void Array<T>::reshape(std::vector<int> shape){
    // TO DO !!
}

template<typename T>
void Array<T>::reshape(std::initializer_list<int> shape){
    // TO DO !!
}

template<typename T>
Array<T> Array<T>::concatenate(const Array<T>& other, const int axis){
    // check if both arrays have the same number of dimensions
    if (this->dim_size != other.get_dimensions()){
        throw std::invalid_argument("can't concatenate arrays with unequal number of dimensions");
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
    std::move(*result);
}

template<typename T>
Matrix<T> Matrix<T>::concatenate(const Matrix<T>& other, const int axis){
    // check if both arrays have the same number of dimensions
    if (this->dim_size != other.get_dimensions()){
        throw std::invalid_argument("can't concatenate arrays with unequal number of dimensions");
    }
    // check if all dimensions except for the concatenation axis match
    for (int d=0; d<this->dimensions; d++){
        if (d!=axis && this->dim_size[d] != other.get_size(d)){
            throw std::invalid_argument("can't concatenate arrays with unequal dimension sizes along any axis other than the concatenation axis");
        }
    }
    // check for valid concatenation axis
    if (axis<0 || axis>1){
        throw std::invalid_argument("invalid concatenation axis: must be either 0 (=rows) or 1 (=cols)");
    }
    std::vector<int> shape = this->dim_size;
    shape[axis]+=other.get_size(axis);
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(shape);
    // copy the elements of the current matrix to the equivalent index of the result matrix
    std::vector<int> index;
    for (int i=0;i<this->data_elements;i++){
        index = this->get_index(i);
        result->data[result->get_element(index)] = this->data[i];
    }
    // stitch in the elements of the second matrix
    for (int i=0;i<other.data_elements;i++){
        index = other.get_index(i);
        index[axis]+=this->dim_size[axis];
        result->data[result->get_element(index)] = other.data[i];
    }
    // return result
    std::move(*result);
}

template<typename T>
Vector<T> Vector<T>::concatenate(const Vector<T>& other){
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->data_elements + other.get_elements());
    // copy the elements of the current array to the equivalent index of the result array
    for (int i=0;i<this->data_elements;i++){
        result->data[i] = this->data[i];
    }
    // stitch in the elements of the second array
    for (int i=0;i<other.data_elements;i++){
        result->data[this->data_elements+i] = other.data[i];
    }
    // return result
    std::move(*result);
}

// +=================================+   
// | Constructors & Destructors      |
// +=================================+

// constructor for multi-dimensional array:
// pass dimension size (elements per dimension)
// as an initializer_list, e.g. {3,4,4}
template<typename T>
Array<T>::Array(const std::initializer_list<int>& shape) :
        // member initialization list
        scale(this),
        fill(this),
        activation(this),
        outliers(this),
        pool(this) {
    // set dimensions + check if init_list empty
    this->dimensions = (int)shape.size();
    if (this->dimensions==0){
        return;
    }
    // store size of individual dimensions in std::vector<int> size member variable
    auto iterator=shape.begin();
    int n=0;
    this->dim_size.resize(this->dimensions);
    for (; iterator!=shape.end();n++, iterator++){
        this->dim_size[n]=*iterator;
    }
    // calculate the size of each subspace for each dimension
    int totalsize = 1;
    this->subspace_size.resize(this->dimensions);
    for (int i = 0; i < this->dimensions; i++) {
        this->subspace_size[i] = totalsize;
        totalsize *= this->dim_size[i];
    }    
    // count total number of elements
    this->data_elements=1;
    for (int d=0;d<this->dimensions;d++){
        this->data_elements*=this->dim_size[d];
    }
    // initialize data buffer
    data = std::make_unique<T[]>(this->data_elements);
};

// constructor for multidimensional array:
// pass dimension size (elements per dimension)
// as type std::vector<int>
template<typename T>
Array<T>::Array(const std::vector<int>& shape) {
    // set dimensions + check if init_list empty
    this->dimensions = shape.size();
    if (this->dimensions==0){
        return;
    }
    // store size of individual dimensions in std::vector<int> size member variable
    // and count total number of elements
    this->data_elements=1;
    this->dim_size.resize(this->dimensions);
    for (int i=0;i<this->dimensions;i++){
        this->dim_size[i]=shape[i];
        this->data_elements*=shape[i];
    }
    // calculate the size of each subspace for each dimension
    int totalsize = 1;
    this->subspace_size.resize(this->dimensions);
    for (int i = 0; i < this->dimensions; i++) {
        this->subspace_size[i] = totalsize;
        totalsize *= this->dim_size[i];
    }    
    // initialize data buffer
    this->data = std::make_unique<T[]>(this->data_elements);
    // initialize instances of outsourced classes    
    this->scale = Scaling<T>(this);
    this->fill = Fill<T>(this);
    this->activation = Activation<T>(this);
    this->outliers = Outliers<T>(this);
    this->pool = Pooling<T>(this);
}

// Array move constructor
template<typename T>
Array<T>::Array(Array&& other) noexcept {
    this->data_elements = other.get_elements();
    this->data = std::move(other.data);
    this->dim_size = std::move(other.dim_size);
    other.data.reset();
}

// Array copy constructor
template<typename T>
Array<T>::Array(const Array& other) {
    dimensions = other.dimensions;
    dim_size = other.dim_size;
    subspace_size = other.subspace_size;
    data_elements = other.data_elements;
    data = std::make_unique<T>(data_elements);
    std::copy(other.data, other.data + data_elements, data);
}  

// virtual destructor for parent class Array<T>
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
    this->dim_size.resize(1);
    this->subspace_size.push_back(elements);
    this->dim_size[0]=elements;
    this->data_elements = elements;
    this->capacity = (1.0f+this->_reserve)*elements;
    this->dimensions = 1;
    // initialize data buffer
    this->data = std::make_unique<T[]>(this->data_elements);
    // initialize instances of outsourced classes    
    this->scale = Scaling<T>(this);
    this->fill = Fill<T>(this);
    this->activation = Activation<T>(this);
    this->outliers = Outliers<T>(this);
    this->pool = Pooling<T>(this);         
}

// Vector move constructor
template<typename T>
Vector<T>::Vector(Vector&& other) noexcept {
    this->data_elements = other.get_elements();
    this->data = std::move(other.data);
    this->dim_size = std::move(other.dim_size);
    other.data.reset();
}

// Matrix constructor
template<typename T>
Matrix<T>::Matrix(const int rows, const int cols) {    
    this->data_elements = rows * cols;
    this->dimensions = 2;
    this->dim_size.resize(this->dimensions);
    this->dim_size[0]=rows;
    this->dim_size[1]=cols;
    // calculate the size of each subspace for each dimension
    int totalsize = 1;
    this->subspace_size.resize(2);
    this->subspace_size[0]=rows;
    this->subspace_size[1]=rows*cols; 
    // initialize data buffer
    this->data = std::make_unique<T[]>(this->data_elements); 
    // initialize instances of outsourced classes    
    this->scale = Scaling<T>(this);
    this->fill = Fill<T>(this);
    this->activation = Activation<T>(this);
    this->outliers = Outliers<T>(this);
    this->pool = Pooling<T>(this);     
}

// Matrix move constructor
template<typename T>
Matrix<T>::Matrix(Matrix&& other) noexcept {
    this->data_elements = other.get_elements();
    this->data = std::move(other.data);
    this->dim_size = std::move(other.dim_size);
    other.data.reset();
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

// change the size of a simple C-style array via its std::unique_ptr<T[]>
// by allocating new memory and copying the previous data to the new location
template<typename T>
void Array<T>::resizeArray(std::unique_ptr<T[]>& arr, const int newSize) {
    // Create a new array with the desired size
    std::unique_ptr<T[]> newArr = std::make_unique<T[]>(newSize);
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
    this->data_elements = newSize;
}

// +=================================+   
// | Dynamic Vector Handling         |
// +=================================+

// push back 1 element into the Vector
template<typename T>
int Vector<T>::push_back(const T value){
    this->data_elements++;
    if (this->data_elements>this->_capacity){
        this->_capacity=int(this->data_elements*(1.0+this->_reserve));
        resizeArray(this->data, this->_capacity);
    }
    this->data[this->data_elements-1]=value;
    return this->data_elements;
}

// resize the vector to a new number of elements
template<typename T>
void Vector<T>::resize(const int newsize){
    this->data_elements=newsize;
    if (this->data_elements>this->_capacity){
        this->_capacity=int(this->data_elements*(1.0+this->_reserve));
        resizeArray(this->data, this->_capacity);
    }    
}
// grows the vector size by the specified number of
// additional elements and initializes these new elements
// to the specified value (default=0);
// will only re-allocate memory if the new size exceeds
// the capacity; returns the new total number of elements
template<typename T>
int Vector<T>::grow(const int additional_elements){
    if (additional_elements<1){return 0;}
    int newsize=this->data_elements+additional_elements;
    // re-allocate memory if the new size exceeds the capacity
    if (newsize>this->_capacity){
        this->capacity=int(this->data_elements*(1.0+this->_reserve));
        resizeArray(&this->data, this->_capacity);
    }
    this->data_elements=newsize;
    return newsize;
}

// shrinks the vector size by the specified number of
// elements and returns the resulting new number of
// remaining total elements
template<typename T>
int Vector<T>::shrink(const int remove_amount){
    int newsize=std::fmax(0,this->data_elements-remove_amount);
    this->data_elements=newsize;
    return newsize;
}

// pop 1 element from the end of the Vector
template<typename T>
T Vector<T>::pop_last(){
    this->data_elements--;
    return this->data[this->data_elements];
}

// pop 1 element from the beginning of the Vector
template<typename T>
T Vector<T>::pop_first(){
    T temp = this->data[0];
    // reassign pointer to position of the raw pointer to the element at index 1
    this->data = std::unique_ptr<T[]>(this->data.release() + 1, std::default_delete<T[]>());
    this->data_elements--;
    return temp;
}

// removes the element of the given index and returns its value
template<typename T>
T Vector<T>::erase(const int index){
    T result = this->data[index];
    for (int i=index;i<this->data_elements-1;i++){
        this->data[i] = this->data[i+1];
    }
    this->data_elements--;
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
    return this->data_elements;
}

// +=================================+   
// | Vector Conversion               |
// +=================================+

// returns the vector as a single column matrix, 
// i.e. as transposition with data in rows (single column)
template<typename T>
Matrix<T> Vector<T>::transpose() const {
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(this->data_elements, 1);
    for (int i=0; i<this->data_elements; i++){
        result->set(i, 0, this->data[i]);
    }
    return std::move(*result);
}

// returns a pointer to a reverse order copy of the
// original Vector<T>
template<typename T>
Vector<T> Vector<T>::reverse() const {
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->data_elements);
    for (int i=0;i<this->data_elements;i++){
        result->data[i] = this->data[this->data_elements-1-i];
    }
    return std::move(*result);
}


// +=================================+   
// | Vector Sample Analysis          |
// +=================================+

// returns a vector of integers that represent
// a ranking of the source vector via bubble sorting
// the ranks
template<typename T>
Vector<int> Vector<T>::ranking() const {
    // initialize ranks
    std::unique_ptr<Vector<int>> rank = std::make_unique<Vector<int>>(this->data_elements);
    rank->fill.range(0,1);
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
Vector<T> Vector<T>::exponential_smoothing(bool as_series) const {
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->data_elements);
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

// returns the weighted average of a sample Vector,
// e.g. for time series data
template <typename T>
double Vector<T>::weighted_average(bool as_series) const {
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
double Vector<T>::Dickey_Fuller(DIFFERENCING method, double degree, double fract_exponent) const {
    // make two copies
    Vector<T> data_copy(this->data_elements);
    Vector<T> stat_copy(this->data_elements);    
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
// Make sure that both Vector<T> have the same number of elements! Otherwise the surplus
// elements of the larger vector will be clipped and the result isn't meaningful;
template<typename T>
double Vector<T>::Engle_Granger(const Vector<T>& other) const {
    // make copies of the x+y source data
    int elements = std::fmin(this->data_elements, other.get_elements());
    Vector<T> xdata(elements);
    Vector<T> ydata(elements);
    for (int i=0;i<elements;i++){
        xdata.data[i] = this->data[i];
        ydata.data[i] = other.data[i];
    }
    // make the data stationary
    std::unique_ptr<Vector<T>> x_stat = xdata.stationary();
    std::unique_ptr<Vector<T>> y_stat = ydata.stationary();
    // perform linear regression on x versus y
    std::unique_ptr<typename Array<T>::LinReg> regr_result = x_stat->linear_regression(y_stat);
    // perform a Dickey_Fuller test on the residuals
    Vector<double> residuals(elements);
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
Vector<T> Vector<T>::stationary(DIFFERENCING method, double degree, double fract_exponent)  const {
    // make a copy
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->data_elements);
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
        for (int t = result->size() - 1; t > 0; t--) {
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
Vector<T> Vector<T>::sort(bool ascending) const {
    // make a copy
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->data_elements);
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
Vector<T> Vector<T>::shuffle() const {
    // make a copy
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->data_elements);
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
double Vector<T>::covariance(const Vector<T>& other) const {
    if (this->data_elements != other.get_elements()) {
        std::cout << "WARNING: Invalid use of method Vector<T>::covariance(); both vectors should have the same number of elements" << std::endl;
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



// Vector binning, i.e. quantisizing into n bins,
// each represented by the mean of the values inside that bin;
// returning the result as pointer to a new Vector of size n
template<typename T>
Vector<T> Vector<T>::binning(const int bins){
    if (this->data_elements == 0) {
        throw std::runtime_error("Cannot bin an empty vector.");
    }
    if (bins <= 0) {
        throw std::invalid_argument("Number of bins must be positive.");
    }
    if (bins >= this->data_elements) {
        throw std::invalid_argument("Number of bins must be less than the number of elements in the vector.");
    }
    // prepare the data structure to put the results
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(bins);    
    result->fill.zeros();
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


// Matrix transpose
template<typename T>
Matrix<T> Matrix<T>::transpose() const {
    // create a new matrix with swapped dimensions
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(this->dim_size[1], this->dim_size[0]);

    for(int i = 0; i < this->dim_size[0]; i++){
        for(int j = 0; j < this->dim_size[1]; j++){
            // swap indices and copy element to result
            result->set(j, i, this->get(i, j));
        }
    }
    return std::move(*result);
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
    std::vector<int> index(this->dimensions,0);
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
                if (col != this->dim_size[1]-1) {
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