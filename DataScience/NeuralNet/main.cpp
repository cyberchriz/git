#include <neuralnet.h>
#include <cmath>
#include <videostream.h>
#include <vector_dimension.h>
using namespace std;

bool IS_DEBUG = true;

int main(int argc, char *argv[]){

    // set up an autoencoder network for scanning video frames with regard to "familiar" patterns
    NeuralNet scanner("");
    
    ACTIVATION_FUNC act_func=f_LReLU;
    OPTIMIZATION_METHOD opt=Nesterov;
    LAYER_TYPE type=standard;

    scanner.set_training_mode(true);
    scanner.set_learning_rate(0.005);
    scanner.set_learning_rate_decay(0.001);
    scanner.set_learning_momentum(0.9);
    scanner.set_gradient_dependent_attention(false);
    scanner.set_gradient_clipping(true,1);
    scanner.set_scaling_method(normalized);

    int video_width=160;
    int scanner_size=8;
    int scanner_step=(int)(0.5*scanner_size);

    double error_threshold = 1.0;

    vector<vector<vector<double>>> scanner_input;
    for (int x=0;x<scanner_size;x++){
        vector<vector<double>> column;
        for (int y=0;y<scanner_size;y++){
            vector<double> stack;
            for (int z=0;z<3;z++){ // 3 rgb channels
                stack.push_back(0);
            }
            column.push_back(stack);
        }
        scanner_input.push_back(column);
    }

    scanner.add_layer({scanner_size,scanner_size,3},type,opt,act_func);
    scanner.add_layer({100},type,opt,act_func);
    scanner.add_layer({50},type,opt,act_func);
    scanner.add_layer({100},type,opt,act_func);
    scanner.add_layer({scanner_size,scanner_size,3},type,opt,act_func);

    // set up GUI
    GUI gui(video_width +10, video_width*0.75+10,"video_gui");

    // get video stream
    VideoStream stream;
    // use bash command 'youtube-dl -g <link>' to convert a stream link into downloadable link (works with ard mediathek e.g.)
    string url="https://adaptive.ndr.de/i/ndr/2017/0131/TV-20170131-1129-3600.,hq,hd,ln,hi,mn,ao,.mp4.csmil/index-f2-v1-a1.m3u8";
    stream.set_url(url);
    stream.start_stream();
    vector<vector<vector<uchar>>> source_frame;
    vector<vector<vector<uchar>>> target_frame;
    int f_height=0;
    int frames_counter=0;

    // online training
    while (!stream.stopStream && !stream.isPaused){
        source_frame = stream.frame(video_width);
        // set up target frame
        if (target_frame.size()==0){
            auto dim = vector_dimension(source_frame);
            f_height=dim.second[1];
            for (int x=0;x<video_width;x++){
                vector<vector<uchar>> column;
                for (int y=0;y<f_height;y++){
                    vector<uchar> stack;
                    for (int z=0;z<3;z++){
                        stack.push_back(0);
                    }
                    column.push_back(stack);
                }
                target_frame.push_back(column);
            }
        }
        /*
        // reset target frame
        for (int x=0;x<video_width;x++){
            for (int y=0;y<f_height;y++){
                for (int z=0;z<3;z++){
                    target_frame[x][y][z]=0;
                }
            }
        }
        */
       target_frame=source_frame;

        // shift scanner across frame
        for (int roi_x=0;roi_x<video_width-scanner_size;roi_x+=scanner_step){
            for (int roi_y=0;roi_y<f_height-scanner_size;roi_y+=scanner_step){
                // read local frame data into scanner
                for (int x=0;x<scanner_size;x++){
                    for (int y=0;y<scanner_size;y++){
                        for (int z=0;z<3;z++){
                            scanner.set_input({x,y,z},source_frame[roi_x+x][roi_y+y][z]);
                        }
                    }
                }
                // run scanner network
                scanner.forward_pass();
                scanner.autoencode();
                scanner.backpropagate();
                double relative_error = scanner.get_relative_error();
                if (IS_DEBUG){
                    cout << "frames counter:"<<frames_counter<<", roi_x:"<<roi_x<<", roi_y:"<<roi_y<<", loss avg:"<<scanner.get_loss_avg()<<", relative error (rSSG):"<<relative_error<<", avg. h:"<<scanner.get_avg_h()<<", avg. out:"<<scanner.get_avg_output()<<"\n";
                }
                //if (relative_error>error_threshold){
                    // write scanner output to target frame in case relative output error is above error threshold
                    for (int x=0;x<scanner_size;x++){
                        for (int y=0;y<scanner_size;y++){
                            for (int z=0;z<3;z++){
                                // clip pixel value to 0-255
                                double pixel=scanner.get_output({x,y,z});
                                pixel=fmin(pixel,255);
                                pixel=fmax(pixel,0);
                                // write pixel
                                target_frame[roi_x+x][roi_y+y][z]= (uchar)pixel;
                            }
                        }
                    }
                //}
            }
        }
        if(target_frame.empty()){
            cout << "Frame is empty, skipping display" << endl;
            continue;
        }
    
        gui.update_frame(target_frame);
        frames_counter++;
        if (frames_counter%100==0){scanner.balance_weights();}
        if (frames_counter==1000){scanner.reset_weights();}
    }

    return 0;
} 