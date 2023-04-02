#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <thread>
using namespace std;

class VideoStream {
private:
    unique_ptr<cv::VideoCapture> stream;
    string url;
    thread streamThread;
    double currentPosition = 0;

public:
    bool stopStream = false;
    bool isPaused = false;
    void set_url(string url) {
        this->url = url;
    }

    void start_stream() {
        stream = make_unique<cv::VideoCapture>(url);
        if (!stream->isOpened()) {
            throw runtime_error("Cannot open video stream");
        }
        streamThread = thread(&VideoStream::streamLoop, this);
    }

    void stop_stream() {
        stopStream = true;
        streamThread.join();
    }

    void pause_stream() {
        if (!isPaused) {
            isPaused = true;
            currentPosition = stream->get(cv::CAP_PROP_POS_MSEC);
        }
    }

    void resume_stream() {
        if (isPaused) {
            isPaused = false;
            stream->set(cv::CAP_PROP_POS_MSEC, currentPosition);
        }
    }

    vector<vector<vector<unsigned char>>> frame(int max_width) {
        cv::Mat frame;
        stream->read(frame);
        if (frame.cols > max_width) {
            double scale = (double)max_width / frame.cols;
            cv::resize(frame, frame, cv::Size(max_width, frame.rows * scale), 0, 0, cv::INTER_LINEAR);

        }
        vector<vector<vector<unsigned char>>> frame_data;
        for (int x = 0; x < frame.cols; x++) {
            vector<vector<unsigned char>> column;
            for (int y = 0; y < frame.rows; y++) {
                vector<unsigned char> pixel;
                pixel.push_back(frame.at<cv::Vec3b>(y, x)[0]);
                pixel.push_back(frame.at<cv::Vec3b>(y, x)[1]);
                pixel.push_back(frame.at<cv::Vec3b>(y, x)[2]);
                column.push_back(pixel);
            }
            frame_data.push_back(column);
        }
        return frame_data;
    }

    void streamLoop() {
        while (!stopStream) {
            if (!isPaused) {
                // process the frames here
            }
        }
    }
};

class GUI {
private:
    int width, height;
    cv::Mat frame;
    cv::String windowName;
    cv::Scalar color;
    vector<vector<vector<unsigned char>>> frameData;

public:
    GUI(int width, int height, string windowName) : width(width), height(height), windowName(windowName) {
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(windowName, width, height);
        color = cv::Scalar(255, 255, 255);
    }

    void update_frame(vector<vector<vector<unsigned char>>> frameData) {
        this->frameData = frameData;
        frame = cv::Mat(frameData[0].size(), frameData.size(), CV_8UC3);


        for (int x = 0; x < frameData.size(); x++) {
            for (int y = 0; y < frameData[0].size(); y++) {
                frame.at<cv::Vec3b>(y, x)[0] = frameData[x][y][0];
                frame.at<cv::Vec3b>(y, x)[1] = frameData[x][y][1];
                frame.at<cv::Vec3b>(y, x)[2] = frameData[x][y][2];
            }
        }

        if(frame.rows > 0 && frame.cols > 0){
            cv::imshow(windowName, frame);
            cv::waitKey(1);
        }
        else
        {
            cout<<"Invalid frame!"<<endl;
        }
    }
};