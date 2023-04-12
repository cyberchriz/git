// author: cyberchriz (Christian Suer)

// this code logs any heap memory allocations to the console
// by overriding the `new` operator;
// in order to use this, simply #include this file and define the DEBUG flag as a preprocessor directive

#define DEBUG

#ifdef DEBUG
#pragma once
    #include <iostream>
    #include <boost/stacktrace.hpp>

    // returns the name of the function with the specified level on the call stack;
    // index 0 refers to the top level (=current function)
    // index 1 refers to the calling function
    std::string get_function_name(uint32_t stacklevel_index){
        boost::stacktrace::stacktrace trace;
        return trace[stacklevel_index].name();
    }

    void* operator new(size_t size){
        auto stackTrace = boost::stacktrace::stacktrace();   
        std::cout << "memory heap allocation " << size << " bytes in function " << get_function_name(1) << "\n";
        std::cout.flush();
        return malloc(size);
    }
#endif
