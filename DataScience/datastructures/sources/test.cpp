#include "../headers/datastructures.h"

int main(){
    Array<int> arr{10,10};
    arr.fill_random_uniform(-10,10);
    arr.print("source array:", " ");
    arr.pool(PoolMethod::MAXABS, {2,2}, {2,2}).print("maxabs pooling:", " ");
}