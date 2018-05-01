#include <memory>
#include <iostream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <vector>
#include <exception>

#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "DepthSenseGrabberOpenCV.hxx"
#include "../DepthSenseGrabberCore/DepthSenseGrabberCore.hxx"
#include "../shared/ConversionTools.hxx"
#include "../shared/AcquisitionParameters.hxx"



using namespace cv;
using namespace std;

int bgr = 150;

void removeDepthWithoutColor(cv::Mat &depth, const cv::Mat &rgb){
    for (int i = 0; i < rgb.rows; ++i){
        for (int j = 0; j < rgb.cols; ++j){
            unsigned char *pixel = rgb.ptr(i, j);
            unsigned char *pixelDepth = depth.ptr(i, j);
            if(pixel[0] == 0){
                pixelDepth[0] = 0;
                pixelDepth[1] = 0;
            }
        }
    }
}

void removeLowConfidencePixels(cv::Mat &depth, const cv::Mat &confidence){
    u_int32_t count;
    for (int i = 0; i < confidence.rows; ++i){
        for (int j = 0; j < confidence.cols; ++j){
            unsigned short *pixel = (u_short*)confidence.ptr(i, j);
            unsigned short *pixelDepth = (u_short*)depth.ptr(i, j);
//            cerr << *pixelDepth << endl;
            if(*pixelDepth > bgr){
                *pixelDepth = 0;
                count++;
            }
        }
    }
//    cerr << count << endl;
}


void DepthImage_convert_32FC1_to_16UC1(cv::Mat &dest, const cv::Mat &src, float scale) {
    //    assert(src.type() != CV_32FC1 && "DepthImage_convert_32FC1_to_16UC1: source image of different type from 32FC1");
    float *sptr = (float*)src.data;
    int size = src.rows * src.cols;
    float *send = sptr + size;
    dest.create(src.rows, src.cols, CV_16UC1);
    dest.setTo(cv::Scalar(0));
    unsigned short *dptr = (unsigned short*)dest.data;
    while(sptr<send) {
        if(*sptr < std::numeric_limits<float>::max())
            *dptr = scale * (*sptr);
        dptr ++;
        sptr ++;
    }
}

void DepthImage_convert_16UC1_to_32FC1(cv::Mat &dest, const cv::Mat &src, float scale) {
    //    assert(src.type() != CV_16UC1 && "DepthImage_convert_16UC1_to_32FC1: source image of different type from 16UC1");
    const unsigned short *sptr = (const unsigned short*)src.data;
    int size = src.rows * src.cols;
    const unsigned short *send = sptr + size;
    dest.create(src.rows, src.cols, CV_32FC1);
    dest.setTo(cv::Scalar(0.0f));
    float *dptr = (float*)dest.data;
    while(sptr < send) {
        if(*sptr)
            *dptr = scale * (*sptr);
        dptr ++;
        sptr ++;
    }
}

int main(int argc, char *argv[]) {

    system("rm -rf data/depth");
    system("rm -rf data/rgb");
    system("mkdir data/depth");
    system("mkdir data/rgb");

    start_capture(FORMAT_VGA_ID, false, true, true, true);

    uint8_t* pixelsColorAcq = getPixelsColorsAcq();
    uint16_t* pixelsDepthSync = getPixelsDepthSync();
    uint16_t* pixelsConfidenceQVGA = getPixelsConfidenceQVGA();
    uint16_t* pixelsDepthAcq = getPixelsDepthAcqQVGA();
    uint8_t* pixelsColorSync = getPixelsColorSyncQVGA();

    Mat depth;
    Mat color;
    Mat confidence;

    Mat depthDisplay;

    int frameCount = 0;
    bool recording = false;

    while (true){
        Mat(240, 320, CV_16UC1, pixelsDepthAcq).copyTo(depth);
        Mat(240, 320, CV_8UC3, pixelsColorSync).copyTo(color);
        cvtColor(color, color, CV_RGB2BGR);
        Mat(240, 320, CV_16UC1, pixelsConfidenceQVGA).copyTo(confidence);

        removeDepthWithoutColor(depth, color);
        removeLowConfidencePixels(depth, confidence);

        depth.convertTo(depthDisplay, CV_8U, 0.5);


        imshow("depth", depthDisplay);
        imshow("color", color);
        imshow("confidence", confidence);
        moveWindow("depth", 0,0);
        moveWindow("color", 320,0);
        moveWindow("confidence", 640, 0);
        char key = waitKey(10);
        if(key == 27){
            break;
        }
        if(key == 'r'){
            recording = !recording;
        }
        if(recording){
            std::stringstream ss;
            ss << "./data/depth/";
            ss << setfill('0') << setw(7) << frameCount;
            ss << ".png";
            cv::imwrite(ss.str(), depth);

            ss.str(std::string());
            ss.clear();
            ss << "./data/rgb/";
            ss << setfill('0') << setw(7) << frameCount;
            ss << ".png";

            cv::imwrite(ss.str(), color);
            frameCount++;
        }
        if(key == '='){
            bgr += 20;
        }
        if(key == '-'){
            bgr -= 20;
        }

    }

    return 0;
}

