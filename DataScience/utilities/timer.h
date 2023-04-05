// author: cyberchriz (Christian Suer)
// modified code based on a youtube video by "The Cherno" (many thanks!)

#pragma once
#include <chrono>
#include <iostream>

// in order to use this code, simply define the DEBUG flag as a preprocessor directive,
// then write the macro keyword 'TIMER' in any function;
// once the local scope ends, this will cause the timer struct to print out the duration since the 'TIMER' call

//#define DEBUG

#ifdef DEBUG
#define TIMER ns_timer::Timer timer_object;
#else
#define TIMER
#endif

namespace ns_timer {
    struct Timer {
        std::chrono::high_resolution_clock::time_point start, end;
        std::chrono::duration<float> duration;
        // constructor
        Timer() {
            start = std::chrono::high_resolution_clock::now();
        }
        // destructor
        ~Timer() {
            end = std::chrono::high_resolution_clock::now();
            duration = end - start;
            float ms = duration.count() * 1000.0f;
            std::cout << "timer lifetime: " << ms << "ms\n";
        }
    };
}