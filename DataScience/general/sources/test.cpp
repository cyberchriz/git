#include "../headers/array.h"
#include <iostream>

int main(){
    std::cout << "\nLet's create a 3-by-4 matrix:" << std::endl;
    Matrix<int> myMatrix{3,4};
    myMatrix.print();
    
    std::cout << "\nNow let's fill this Matrix with random numbers from 0 to 9:" << std::endl;
    myMatrix.fill_random_uniform(0,9);
    myMatrix.print();

    std::cout << "\nLet's check the indices, too:" << std::endl;
    myMatrix.print(", ", "\n", true);

    std::cout << "\nNow let's transpose the Matrix for fun:" << std::endl;
    auto newMatrix = myMatrix.transpose();
    newMatrix->print(", ", "\n", true);

    std::cout << "\nHow about making the Matrix a little bigger?" << std::endl;
    auto largeMatrix = newMatrix->asMatrix(5, 4);
    largeMatrix->print();
}