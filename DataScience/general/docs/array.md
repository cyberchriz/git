[[return to main page]](../../../README.md)
# Vectors, 2d Matrices and n-dimensional Arrays,<br>with sample statistics
usage: `#include <array.h>` or include as part of `<datascience.h>`
___
# `class Array`
An instance of the class can be created by passing its dimensions as type `std::initializer_list<int>` or `std::vector<int>`to the constructor.

For example, for a 3x3x4 array of type `<double>` we can write:
```
Array<double> myArr{3,3,4};
```
Public Methods:


### Getters & Setters

|method|description|
|------|-----------|
| `void set(std::initializer_list<int> index, T value)` | assigns the value at the given index (with the index in curly braces) |
| `void set(const std::vector<int>& index, const T value)`| assigns the value at the given index as type `std::vector<int>`|
| `T get(std::intializer_list<int> index)` | returns the value at the given index (with the index in curly braces) |
| `T get(std::vector<int> index)` | returns the value at the given index (with the index as type `std::vector<int>`) |
| `int get_dimensions() const` | returns an the number of dimensions |
| `int get_size(int dimension) const` | returns the size of the specified dimension as an integer |
| `int get_subspace(int dimension)`|returns the size of the subspace of the given dimension|
| `int get_elements() const` | returns the total number of elements of the entire array (all dimensions) |


### Fill & Initialize

|method|description|
|------|-----------|
| `void fill_values(T value)` | fills the entire array with the specified value (of datatype T) |
| `void fill_zeros()` | fills the entire array with zeros |
| `void fill_identity()` | applies the identity matrix to the array |
| `void fill_random_gaussian(T mu=0, T sigma=1)` | fills the array with random values from a gaussian normal distribution with the given mean and sigma |
| `void fill_random_uniform(T min=0,T max=1.0)` | fills the array with values from a random uniform distribution within the given range |
| `void fill_range(const T start=0,const T step=1)`| fills the array with a continuous range of values in all dimensions, starting from the zero point index; use negative values for descending range |


### Basic Distribution Properties

|method|description|
|------|-----------|
| `mean()` | returns the arrithmetic mean of all values of the array (all dimensions) |
| `median()` | returns the Median of all values of the array (all dimensions) |
| `mode()`| returns the item that occurs the most number of times (for unimodal data)|
| `variance()` | returns the variance of all values of the array (all dimensions) |
| `stddev()` | returns the standard deviation of all values of the array (all dimensions) |
| `skewness()`|returns the skewness of all the entire value population of the array |
| `kurtosis()`|returns the kurtosis of the entire value population of the array|


### Addition

|method|description|
|------|-----------|
| `void sum()` | returns the sum of all values of the array (all dimensions) |
| `std::unique_ptr<Array<T>> operator+(const T value)` | elementwise (scalar) addition of the specified value to all values of the array |
| `std::unique_ptr<Array<T>> operator+(const Array& other)` | elementwise (scalar) addition of a second array of equal dimensions to the current array |
| `std::unique_ptr<Array<T>> operator++(int)` | postfix increment++ all values (elementwise) by +1 |
| `Array<T>& operator++()` | prefix ++increment all values (elementwise) by +1 |
| `void operator+=(const T value)`| elementwise (scalar) addition of the specified value to all values of the array|
| `void operator+=(const Array& other)` | elementwise (scalar) addition of a second array of equal dimensions to the current array|


### Substraction

|method|description|
|------|-----------|
|`std::unique_ptr<Array<T>> operator-(const T value)` | elementwise (scalar) substraction of the specified value from all values of the array |
| `std::unique_ptr<Array<T>> operator-(const Array& other)` | elementwise substraction of a second array of equal dimensions from the current array |
|`std::unique_ptr<Array<T>> operator--(int)`| postfix decrement-- all values (elementwise) by -1 |
|`Array<T>& operator--()`| prefix decrement-- all values (elementwise) by -1 |
|`void operator-=(const T value)` | elementwise (scalar) substraction of the specified value from all values of the array|
|`void operator-=(const Array& other)`| elementwise (scalar) substraction of a second array of equal dimensions from the current array|


### Multiplication

|method|description|
|------|-----------|
|`T product()`| returns the product reduction, i.e. by multiplying all array elements with each other|
| `std::unique_ptr<Array<T>> operator*(const T factor)` | returns the result of elementwise multiplication all values of the array with the given factor|
|`std::unique_ptr<Array<T>> tensordot(const Array<T>& other, const std::vector<int>& axes) const | return the tensor reduction|
|`T dotproduct(const Array<T>& other) const`|returns the dotproduct (aka scalar product) by elementwise multiplication of elements with corresponding indices in both arrays, then adding them up to a single scalar|
|`T operator*(const Array<T>& other) const`|alias for the dotproduct|
| `void operator*=(const T factor)` | elementwise (scalar) multiplication of the individual values of the array by the specified factor|
| `Hadamard(const Array& other)` | elementwise (scalar) multiplication with a second array of equal dimensions, i.e. the Hadamard product |


### Division

|method|description|
|------|-----------|
| `std::unique_ptr<Array<T>> operator/(T quotient)` | divides all values of the array elementwise by the given quotient |
| `void operator/=(const T quotient)` | elementwise (scalar) division of the individual values of the array by the specified quotient|


### Modulo

|method|description|
|------|-----------|
|`void operator%=(const double num)`| sets all values of the array to the remainder of their division by the given number |
| `std::unique_ptr<Array<double>> operator%(const double num)`| returns an Array of type `<double>`that holds the remainders of the division of the original array by the given number |


### Exponentiation & Logarithm

|method|description|
|------|-----------|
| `void pow(const T exponent)` | exponentiates all values of the array to the specified power |
| `void Array<T> pow(const Array& other)` | elementwise exponentiation of the values of the original array to the power of the corresponding values of the second array |
| `void sqrt()` | applies the square roots elementwise to all values of the array |
| `void log()`| takes the natural logarithm (base e) elementwise of all values of the array |
| `void log10()`| takes the base-10 logarithm elementwise of all values of the array |


### Rounding

|method|description|
|------|-----------|
| `void round()`| rounds the values of the array elementwise to their nearest integer|
| `void floor()`| rounds the values of the array elementwise to their next lower integers|
| `void ceil()`| rounds the values of the array elementwise to their next higher integers|

### Find, Replace

|method|description|
|------|-----------|
| `void replace(const T old_value, const T new_value)` | replaces all findings of the specified old value by the specified new value |
| `int find(const T value)` | returns the number of findings of the specified value across the entire array (all dimensions) |


### Custom Functions

|method|description|
|------|-----------|
| `void function(const T (*pointer_to_function)(T))` | applies the specified function to all values of the array (elementwise) |


### Assignment

|method|description|
|------|-----------|
| `Array<T>& operator=(const Array& other)` | copies the values of the specified second array (of equal dimensions) into the current array |


### Elementwise Comparison By Single Value

|method|description|
|------|-----------|
| `std::unique_ptr<Array<bool>> operator>(const T value)`|returns a boolean array with each value representing whether the corresponding value in the original array is greater than the given value |
| `std::unique_ptr<Array<bool>> operator>=(const T value)`|returns a boolean array with each value representing whether the corresponding value in the original array is greater than or equal to the given value |
| `std::unique_ptr<Array<bool>> operator==(const T value)`|returns a boolean array with each value representing whether the corresponding value in the original array is equal to the given value |
| `std::unique_ptr<Array<bool>> operator!=(const T value)`|returns a boolean array with each value representing whether the corresponding value in the original array is unequal to the given value |
| `std::unique_ptr<Array<bool>> operator<(const T value)`|returns a boolean array with weach value representing whether the corresponding value in the original array is less than the given value |
| `std::unique_ptr<Array<bool>> operator<=(const T value)`|returns a boolean array with each value representing whether the corresponding value in the original array is less than or equal to the given value |


### Elementwise Comparison With Second Array

|method|description|
|------|-----------|
// elementwise comparison with second array
| `std::unique_ptr<Array<bool>> operator>(const Array& other)`|returns a boolean array with each value representing whether the corresponding value in the original array is greater than the corresponding value in the second array |
| `std::unique_ptr<Array<bool>> operator>=(const Array& other)`|returns a boolean array with each value representing whether the corresponding value in the original array is greater than or equal to the corresponding value in the second array |
| `std::unique_ptr<Array<bool>> operator==(const Array& other)`|returns a boolean array with each value representing whether the corresponding value in the original array is equal to the corresponding value in the second array |
| `std::unique_ptr<Array<bool>> operator!=(const Array& other)`|returns a boolean array with each value representing whether the corresponding value in the original array is unequal to the corresponding value in the second array |
| `std::unique_ptr<Array<bool>> operator<(const Array& other)`|returns a boolean array with each value representing whether the corresponding value in the original array is less than the corresponding value in the second array |
| `std::unique_ptr<Array<bool>> operator<=(const Array& other)`|returns a boolean array with each value representing whether the corresponding value in the original array is less than or equal to the corresponding value in the second array |


### Elementwise Logical Operations

|method|description|
|------|-----------|
| `std::unique_ptr<Array<bool>> operator&&(const bool value)`|returns a boolean array with each value representing the logical `AND` between the corresponding values of the original array and the given value |
| `std::unique_ptr<Array<bool>> operator\|\|(const bool value)`|returns a boolean array with each value representing the logical `OR` between the corresponding values of the original array and the given value |
| `std::unique_ptr<Array<bool>> operator!()`|returns a boolean array with each value representing the logical `NOT` of the corresponding values of the original array |
| `std::unique_ptr<Array<bool>> operator&&(const Array& other)`|returns a boolean array with each value representing the logical `AND` between the corresponding values of the original array and a second array |
| `std::unique_ptr<Array<bool>> operator\|\|(const Array& other)`|returns a boolean array with each value representing the logical `OR` between the corresponding values of the original array and a second array |


### Type Casting

|method|description|
|------|-----------|
| `template<typename C> operator Array<C>()`| returns an explicit type cast of the original array |

### Class Conversion

|method|description|
|------|-----------|
|`std::unique_ptr<Vector<T>> flatten()`| flattens the array (or matrix) by concatenating its data into a 1d Vector; not implemented for the derived class `Vector<T>` because it wouldn't do anything in this case |
|`virtual std::unique_ptr<Matrix<T>> asMatrix(const int rows, const int cols)`|converts an array, matrix or vector into a 2d matrix of the specified size; if the new matrix has less elements in any of the dimensions, the surplus elements of the source will be ignored; if the new matrix has more elements, these additional elements will be initialized with zeros; this method can also be used to get a resized copy from a 2d source matrix|
|`std::unique_ptr<Matrix<T>> asMatrix()`|converts an array, matrix or vector into a 2d matrix; the exact behavior will depend on the source dimensions: (1.) if the source is one-dimensional (=Vector), the result will be a matrix with a single row; (2.) if the source already is 2-dimensional, the total size and the size per dimension will remain unchanged, only the datatype of the returned object is now 'Matrix<T>'; (3.) if the source has more than 2 dimensions, only values from index 0 of the higher dimensions will be copied into the returned result |
|`std::unique_ptr<Array<T>> asArray(std::initializer_list<int> dim_size)`|converts a vector or matrix into an array or converts a preexisting array into an array of the specified new size; surplus elements of the source that go beyond the limits of the target will be cut off; if the target is bigger, the surplus target elements that have no corresponding index at the source will be initialized with zeros|


### Output
|method|description|
|------|-----------|
| `void print(std::string comment="", std::string delimiter=", ", std::string line_break="\n", bool with_indices=false)`|prints the Array to the console|


### Public member variables
|method|description|
|------|-----------|
| `std::unique_ptr<T[]> _data`| holds a pointer to a copy the source data concatenated into a 1-dimensional ("single row") array |
___

# `class Matrix`
> ### This is a **derived class of `class Array<T>`**. It **inherits all its methods**, but deals with the specific case of 2d arrays.<br>Because the dimensions are already known to be 2d, they don't need to be passed in explicitly as a `std::initializer_list`, so there's an additional constrcutor that allows to pass them by simply passing the size of the x and y dimensions instead (`Matrix(const int rows, const int cols)`).<br>Accordingly, there are additional overloads for the `.get()` and `.set()` methods that equally only need two integers for the x and y indices they refer to (plus in case of the `.set()` method a third parameter for the value that is being assigned).

For example a 2d matrix of size 4x4 and data type `<float>` can be instantiated like this:
```
Matrix<float> myMatrix(4,4);
```
## On top of all the methods given above, inherited from `class Array<T>`, the Matrix class **ADDITIONALLY IMPLEMENTS**:


### Matrix Getters & Setters

|method|description|
|------|-----------|
| `void set(const int row, const int col, const T value)` | assigns the given value to the element at the specified index |
| `T get(const int row, const int col)`| returns the value of the element at the specified index |


### Matrix Transpose

|method|description|
|------|-----------|
|`Matrix<T>::transpose()`|returns the transpose of the current matrix as the resulting new `Matrix<T>`|

___

# `class Vector`
> ### This is also a **derived class of `class Array<T>`** and it also **inherits all its methods**, but deals with the specific case of 1d arrays.Because the dimensions are already known to be 1d, there is no need to pass the dimensions to the constructor as `std::vector<int>` or `std::initializer_list<int>`, hence there is an additional constructor that takes the number of elements is its only argument.<br> Accordingly, there's an additional overload for the `.get()` method that also only needs a single integer for the index it refers to. An additional `.set()` needs needs an integer for the index it refers to, followed by the assigned value as a second argument.

For example a 1d `Vector` with 100 elements of type `<double>` can be instantiated like this:
```
Vector<double> myVec(100);
```

## On top of all the methods inherited from `class Array<T>` this class **ADDITIONALLY IMPLEMENTS**:


### Vector Getters & Setters

|method|description|
|------|-----------|
| `void set(const int index, const T value)` | assigns the given value to the element at the specified index |
| `T get(const int index)`| returns the value of the element at the specified index |


### Dynamic Vector Handling

|method|description|
|------|-----------|
| `int push_back(T value)` | adds 1 element and assigns its value; returns the resulting number of elements |
| `T pop()` | removes the last element and returns its value |
| `T erase(const int index)` | removes the element that has the given index and returns its value |
| `int grow(const int additional_elements,T value=0)`|grows the vector size by the specified number of additional elements and initializes these new elements to the specified value (default=0); will only re-allocate memory if the new size exceeds the current capacity (which is up to +50% by default); returns the resulting new total number of elements|
| `int shrink(const int remove_amount)`|shrinks the vector size by the specified number of elements and returns the resulting new number of remaining total elements|
| `resize(const int new_size)`| changes the vector size to the new number of elements; re-allocation only takes place if the new size exceeds the capacity|
| `int get_capacity()`| returns the current capacity as total elements that are available without memory re-allocations |
| `int size()`| returns the number of elements; equivalent to `int get_elements()`|


### Multiplication

|method|description|
|------|-----------|
| `T dotproduct(const Vector& other)`| returns the dotproduct |
| `T operator*(const Vector& other)`| alias method for the dotproduct |


### Conversion

|method|description|
|------|-----------|
| `Matrix<T> transpose()`| converts the vector to a single column 'vertical' matrix (all data as rows)|
| `std::unique_ptr<Vector<T>> reverse()`| returns a pointer to a reverse order copy of the original `Vector<T>`|


### Vector Sample Analysis

|method|description|
|------|-----------|
| `std::unique_ptr<Vector<int>> ranking()`| returns a pointer to an integer Vector that holds a ranking of the corresponding values contained in the original Vector |
| `std::unique_ptr<Vector<T>> exponential_smoothing(bool as_series=false)`| return a pointer to an exponentially smoothed copy of the original Vector |
| `double weighted_average(bool as_series=true)`|returns the weighted average (e.g. for time series; `as_series`= indexing starts from 0)
| `double Dickey_Fuller()`| returns the result of an augmented Dickey-Fuller unit root test for stationarity; a value of <0.05 usually implies that the Null hypothesis can be rejected, i.e. the sample is stationary; The method for differencing is set to first order integer by default, by can be optionally be changed to other methods via the arguments|
| `double Engle_Granger(const Vector<T>& other)`| takes the source vector and another vector (passed as parameter) and performs an Engle-Granger test in order to test the given numeric sample for cointegration, i.e. checking series data for a long-term relationship. The test was proposed by Clive Granger and Robert Engle in 1987. If the returned p-value is less than a chosen significance level (typically 0.05), it suggests that the two time series are cointegrated and have a long-term relationship.|
| `Vector<T> stationary(DIFFERENCING method=integer,double degree=1,double fract_exponent=2)`| returns a stationary copy (e.g. for time series data) of the original Vector; has options for integer differencing, fractional differencing, mean deviation differencing, logarithmic differencing, higher order differencing|
| `std::unique_ptr<Vector<T>> sort(bool ascending=true)`| returns a pointer to a sorted copy of the original Vector |
| `std::unique_ptr<Vector<T>> shuffle()`| returns a pointer to a randomly shuffled copy of the original Vector|
| `Vector<T> log_transform()`| returns a pointer to a logarithmically transformed copy of the original Vector|
| `std::unique_ptr<LinReg<T>> linear_regression(const Vector<T>& other)`| performs linear regression on the source vector as x_data and a second vector as y_data and returns a pointer to a struct that stores all the results and allows predictions of new values:<br>- `double x_mean=0`<br>- `double y_mean=0`<br>- `double y_intercept`<br>- `double slope`<br>- `double r_squared`<br>- `std::unique_ptr<double[]> y_regression`<br>- `std::unique_ptr<double[]> residuals`<br>- `double SST`<br>- `double SSR`<br>- `T predict(const T x)`<br>- `bool is_good_fit(double threshold=0.95)`|
| `std::unique_ptr<PolyReg<T>> polynomial_regression(const int power)`|performs polynomial regression with the source vector as x_data and a second vector as y_data (to the specified power) and returns a pointer to a struct that stores all the results and allows predictions from new x values:<br>- `double SS_res`<br>- `double SS_tot`<br>- `double RSS`<br>- `double MSE`<br>- `double RSE`<br>- `double y_mean`<br>- `double x_mean`<br>- `double r_squared`<br>- `std::unique_ptr<double[]> coefficient`<br>- `bool is_good_fit(double threshold=0.95)`<br>- `T predict(const T x)`|
|`std::unique_ptr<Correlation<T>> correlation(const Vector<T>& other)`|returns a pointer to a struct that stores the results of the correlation of the source vector as x_data versus a second vector as y_data:<br>- `double x_mean`<br>- `double y_mean`<br>- `double x_stddev`<br> - `double y_intercept`<br> - `double slope`<br> - `double covariance`<br> - `double Pearson_R`<br> - `double Spearman_Rho`<br> - `double r_squared`<br> - `double RSS`<br> - `double SST`<br> - `double SSE`<br> - `double SSR`<br> - `double z_score`<br> - `double t_score`<br> - `std::vector<T> y_predict`<br> - `void print()`|
| `std::unique_ptr<Histogram<T>> histogram(unit bars)`|returns a pointer to a struct that stores the boundaries and counts of histogram bars from the data of the source vector:<br>- `T min`<br>- `T max`<br>- `T bar_width`<br>- `T width`<br>- `std::unique_ptr<Histogram_bar<T>[]> bar`<br>   - `T lower_boundary`<br>   - `T upper_boundary`<br>   - `int abs_count`<br>   - `double rel_count`|



[[return to main page]](../../../README.md)