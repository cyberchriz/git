#include "../headers/array.h"
#include <iostream>

int main(){
    Matrix<int> myMatrix{3,4};
    myMatrix.fill_zeros();
    myMatrix.print("\nHere's a 3x4 Matrix<int> as an example.");
    
    myMatrix.fill_random_uniform(0,99);
    myMatrix.print("\nFilling the Matrix with random numbers from a uniform distribution 0-99):");

    myMatrix.print("\nLet's check the indices, too:", ", ", "\n", true);

    auto newMatrix = myMatrix.transpose();
    newMatrix->print("\nExample for Matrix transpose:", ", ", "\n", true);

    (*newMatrix)++;
    newMatrix->print("\nTesting the postfix increment operator++:");

    auto largeMatrix = newMatrix->asMatrix(5, 4);
    largeMatrix->print("\nMaking the Matrix a little bigger, padding with 0:");

    Vector<int> myVec(10);
    myVec.fill_range(100,-1);
    myVec.print("\nNow let's try with a vector, filled with a range of numbers, starting from 100, step -1:");

    myVec.shuffle();
    myVec.print("\nRandom shuffle:");
    
    std::cout << "\nMean=" << myVec.mean() << std::endl;
    std::cout << "Median=" << myVec.median() << std::endl;
    std::cout << "Standard Deviation=" << myVec.stddev() << std::endl;
    std::cout << "Ranking: "; (myVec.ranking())->print();
    std::cout << "Stationary (method=integer, degree=1): "; (myVec.stationary(integer))->print();
    
    Vector<int> x_axis(myVec.get_elements());
    x_axis.fill_range(0,1);
    auto regr_result = x_axis.linear_regression(myVec);
    std::cout << "Linear Regression Slope=" << regr_result->slope << std::endl;
    std::cout << "Lin. Reg. r_square=" << regr_result->r_squared << std::endl;

    auto Vec2=myVec.transpose();
    Vec2->print("\nTransposed version of same vector (this will actually create a 'single column matrix'):");

    Array<int> myArr{4,4,2,2};
    myArr.fill_random_uniform(0,9);
    myArr.print("\nTesting the print() function for a 4d array (4x4x2x2), without indices: ", " ", "\n",false);
    myArr.print("\n... and the same WITH indices: ", " ", "\n",true);
}