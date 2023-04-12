// author: cyberchriz (Christian Suer)
// modified code based on a youtube video by "The Cherno" (many thanks!)

#pragma once
#include <chrono>
#include <iostream>

// this code can be used in 2 ways:
// 1. create an instance of Timer, then use the methods elapsed_sec(), elapsed_millisec() or elapsed_microsec on this instance
// define the flag 'TIMELOG' as a preprocessor directive, then use the macro 'TIMER' anywhere in the code start a timer
// --> it will print its lifetime on the console once it goes out of scope, i.e. when the function it lives in ends

//#define TIMELOG

#ifdef TIMELOG
#define TIMER   Timer timer;\
                std::cout << "timer started in scope " << __PRETTY_FUNCTION__ << "; ";
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

    double elapsed_millisec(){
        end = std::chrono::high_resolution_clock::now();
        return (end-start).count() * 1000;          
    }

    double elapsed_microsec(){
        end = std::chrono::high_resolution_clock::now();
        return (end-start).count() * 1000000;          
    }           

    // constructor
    Timer() {
        start = std::chrono::high_resolution_clock::now();
    }
    // destructor
    ~Timer() {
        std::cout << "end of timer lifetime: " << elapsed_millisec() << "ms\n";
    }
};



#include "../sources/timer.cpp"