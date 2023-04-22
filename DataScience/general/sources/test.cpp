#include "../headers/array.h"
#include <iostream>
#include <vector>

int main(){
    std::cout << "Test+Demonstration of the the functions of class Array<T> with the inherited classes Matrix<T> and Vector<T>" << std::endl;

    Array<int> Array1{3,3,3};
    Array1.print("\n\nCreation of a 3x3x3 Array<int> via constructor with std::initializer_list<int>; not initialized:");
    
    std::vector<int> index1 = {3,3,3};
    Array<int> Array2(index1);
    Array2.print("\nCreation of a 3x3x3 Array<int> via constructor with std::vector<int>; not initialized:");
    
    std::initializer_list<int> index2 = {0,0,0};
    Array2.set(index2,10);
    Array2.print("\nTesting the method void set(const std::initializer_list<int>& index, const T value): setting value at index 0,0,0 to 10:");

    index1 = {0,1,0};
    Array2.set(index1,-5);
    Array2.print("\nTesting the method void set(const std::vector<int>& index, const T value): setting value at index 0,1,0 to -5:");

    std::initializer_list<int> index3 = {0,0,0};
    std::cout << "\nTesting value retrieval for index 0,0,0 via method T get(const std::initializer_list<int>& index)" << std::endl;
    std::cout << " Result = " << Array2.get(index3) << std::endl;

    index1 = {0,1,0};
    std::cout << "\nTesting value retrieval for index 0,1,0 via method T get(const std::vector<int>& index);" << std::endl;
    std::cout << "Result = " << Array2.get(index1) << std::endl;
    
    std::cout <<"\nTesting to return the dimensions via method int get_dimensions();" << std::endl;
    std::cout << "Result =" << Array2.get_dimensions() << std::endl;

    std::cout << "\nTesting the method int get_size(int dimension) for all dimensions:" << std::endl;
    for (int i=0;i<Array2.get_dimensions();i++){
        std::cout << "- dimension " << i << ": " << Array2.get_size(i) << std::endl;
    }

    Matrix<int> Matrix1{5,5};
    Matrix1.print("\nTest for creating a 5x5 Matrix<int>, printing with indices:",", ", "\n", true);

    Matrix1.print("\nPrinting the same matrix without indices:");

    Matrix1.fill_values(5);
    Matrix1.print("\nTesting the method void fill_values(T value): filling all elements with '5':");

    Matrix1.fill_zeros();
    Matrix1.print("\nTesting the method void fill_zeros():");

    Matrix1.fill_identity();
    Matrix1.print("\nTesting the method void fill_identity()");

    Vector<double> Vector1(9);
    Vector1.print("\nCreating a Vector<double> with 9 elements:");

    Vector1.fill_random_gaussian();
    Vector1.print("\nTesting the method fill_random_gaussian():");

    Vector1.fill_random_uniform();
    Vector1.print("\nTesting the method fill_random_uniform():");    

    Vector1.fill_range(20,1);
    Vector1.print("\nTesting the method fill_range(), start value 20, step 1.0):");

    std::cout << "\nMean = " << Vector1.mean() << std::endl;
    std::cout << "Median = " << Vector1.median() << std::endl;
    std::cout << "Variance = " << Vector1.variance() << std::endl;
    std::cout << "Standard Deviation = " << Vector1.stddev() << std::endl;
    std::cout << "Sum = " << Vector1.sum() << std::endl;

    auto Vector2 = Vector1+10;
    Vector2->print("\nTesting the operator+ overload with a scalar parameter: adding 10:");

    ++(*Vector2);
    Vector2->print("\nTesting the prefix incement++ operator:");

    (*Vector2)+=5;
    Vector2->print("\nTesting the operator+=, (*this)+=5;");

    Vector<int> Vector3(15);
    Vector3.fill_random_uniform(0,9);
    Vector3.print("\nExample Vector 1: ",", ");
    Vector<int> Vector4(15);
    Vector4.fill_random_uniform(0,9);
    Vector4.print("Example Vector 2: ",", ");

    auto Vector5 = Vector3 + Vector4;
    Vector5->print("\nElementwise addition of Vectors 1+2:");

    Vector5 = Vector3 - Vector4;
    Vector5->print("\nElementwise substraction of Vectors 1-2:");

    Vector5 = Vector3.Hadamard(Vector4);
    Vector5->print("\nElementwise multiplication of Vectors 1*2 ('Hadamard product'):");
}