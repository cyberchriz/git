#pragma once
#include <cuda_runtime.h> // requires the CUDA toolkit to be installed on the system


/*
usage of this singleton class:

GPU& gpu = GPU::get_instance();
if (gpu.available()) {
    // Do something with the GPU
}

*/

class GPU {
    private:
        // private member variables
        static GPU* instance;
        static bool is_available;  
        
        // private methods
        static GPU& get_instance() {
            if (instance == nullptr) {
                instance = new GPU();
            }
            return *instance;
        }          

    public:
        // public member functions
        static bool available(){return is_available;};

        // constructor
        GPU(){
            // check if gpu is available
            int devices = cudaGetDeviceCount();
            for (int i = 0; i < devices; i++) {
                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, i);
                if (deviceProp.major >= 3 && deviceProp.minor >= 0) {
                is_available = true;
                break;
                }
            }
        }

        // destructor
        ~GPU(){};
};

GPU* GPU::instance = nullptr;