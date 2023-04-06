// author: cyberchriz (Christian Suer)
// modified code based on a youtube video by "The Cherno" (many thanks!)

#pragma once
#include <chrono>
#include <iostream>

// in order to use this code, simply define the DEBUG flag as a preprocessor directive,
// then write the macro keyword 'TIMER' in any function;
// alternatively (including outside debugging) just create a Timer object without the macro;
// once the local scope of the Timer call ends, the Timer destructor will print out the duration of the Timer's lifetime

//#define DEBUG

#ifdef DEBUG
#define TIMER Timer timer;
#else
#define TIMER
#endif

struct Timer {
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> duration;

    double elapsed_sec(){
        end = std::chrono::high_resolution_clock::now();
        return (end-start).count();          
    }

    double elapsed_ms(){
        end = std::chrono::high_resolution_clock::now();
        return (end-start).count() * 1000;          
    }

    double elapsed_µs(){
        end = std::chrono::high_resolution_clock::now();
        return (end-start).count() * 1000000;          
    }           

    // constructor
    Timer() {
        start = std::chrono::high_resolution_clock::now();
    }
    // destructor
    ~Timer() {
        end = std::chrono::high_resolution_clock::now();
        std::cout << "timer lifetime: " << (end-start).count()*1000 << "ms\n";
    }
};