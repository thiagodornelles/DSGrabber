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

void removeDepthWithoutColor(IplImage *depth, IplImage *rgb){
    for (int row = 0; row < rgb->height; row++){
        for (int col = 0; col < rgb->width; col++){
            if(rgb->imageData[rgb->widthStep*row + col * 3] == 0 &&
               rgb->imageData[rgb->widthStep*row + col * 3 + 1] == 0 &&
               rgb->imageData[rgb->widthStep*row + col * 3 + 2] == 0){
               depth->imageData[depth->widthStep*row + col * 2 + 0] = 0;
               depth->imageData[depth->widthStep*row + col * 2 + 1] = 0;
            }
        }
    }
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

    int flagExportType = FILETYPE_JPG; // FILETYPE_NONE, FILETYPE_JPG or FILETYPE_PNM

    int divideDepthBrightnessCV = 6;
    int divideConfidenceBrightnessCV = 3;

    bool interpolateDepthFlag = 0;

    bool dispColorAcqFlag   = 0;
    bool dispDepthAcqFlag   = 1;
    bool dispColorSyncFlag  = 1;
    bool dispDepthSyncFlag  = 0;
    bool dispConfidenceFlag = 0;

    bool saveColorAcqFlag   = 0;
    bool saveDepthAcqFlag   = 0;
    bool saveColorSyncFlag  = 0;
    bool saveDepthSyncFlag  = 0;
    bool saveConfidenceFlag = 0;

    bool buildColorSyncFlag = dispColorSyncFlag || saveColorSyncFlag;
    bool buildDepthSyncFlag = dispDepthSyncFlag || saveDepthSyncFlag;
    bool buildConfidenceFlag = dispConfidenceFlag || saveConfidenceFlag;

    int flagColorFormat = FORMAT_VGA_ID; // VGA, WXGA or NHD

    int widthColor, heightColor;
    switch (flagColorFormat) {
    case FORMAT_VGA_ID:
        widthColor = FORMAT_VGA_WIDTH;
        heightColor = FORMAT_VGA_HEIGHT;
        break;
    case FORMAT_WXGA_ID:
        widthColor = FORMAT_WXGA_WIDTH;
        heightColor = FORMAT_WXGA_HEIGHT;
        break;
    case FORMAT_NHD_ID:
        widthColor = FORMAT_NHD_WIDTH;
        heightColor = FORMAT_NHD_HEIGHT;
        break;
    default:
        printf("Unknown flagColorFormat");
        exit(EXIT_FAILURE);
    }

    int widthDepthAcq, heightDepthAcq;
    if (interpolateDepthFlag) {
        widthDepthAcq = FORMAT_VGA_WIDTH;
        heightDepthAcq = FORMAT_VGA_HEIGHT;
    } else {
        widthDepthAcq = FORMAT_QVGA_WIDTH;
        heightDepthAcq = FORMAT_QVGA_HEIGHT;
    }

    char fileNameColorAcq[50];
    char fileNameDepthAcq[50];
    char fileNameColorSync[50];
    char fileNameDepthSync[50];
    char fileNameConfidence[50];

    char baseNameColorAcq[20] = "colorFrame_0_";
    char baseNameDepthAcq[50] = "data/depth/depthAcqFrame_0_";
    char baseNameColorSync[50] = "data/rgb/colorSyncFrame_0_";
    char baseNameDepthSync[20] = "depthFrame_0_";
    char baseNameConfidence[30] = "depthConfidenceFrame_0_";

    start_capture(flagColorFormat,
                  interpolateDepthFlag,
                  buildColorSyncFlag, buildDepthSyncFlag, buildConfidenceFlag);

    uint16_t* pixelsDepthAcq;
    uint8_t* pixelsColorSync;
    uint8_t* pixelsColorAcq = getPixelsColorsAcq();
    uint16_t* pixelsDepthSync = getPixelsDepthSync();
    uint16_t* pixelsConfidenceQVGA = getPixelsConfidenceQVGA();
    if (interpolateDepthFlag) {
        pixelsDepthAcq = getPixelsDepthAcqVGA();
        pixelsColorSync = getPixelsColorSyncVGA();
    } else {
        pixelsDepthAcq = getPixelsDepthAcqQVGA();
        pixelsColorSync = getPixelsColorSyncQVGA();
    }



    IplImage *cv_depthAcqImage=NULL,
            *cv_depthAcqImageDisplay=NULL,
            *cv_colorAcqImage=NULL, // initialized in main, used in CBs
            *cv_depthSyncImage=NULL, // initialized in main, used in CBs
            *cv_colorSyncImage=NULL, // initialized in main, used in CBs
            *cv_confidenceImage=NULL, // initialized in main, used in CBs
            *cv_emptyImage=NULL; // initialized in main, used in CBs
    CvSize cv_szDepthAcq=cvSize(widthDepthAcq,heightDepthAcq),
            cv_szColorAcq=cvSize(widthColor,heightColor),
            cv_szConfidence=cvSize(FORMAT_QVGA_WIDTH,FORMAT_QVGA_HEIGHT);
    CvSize cv_szDepthSync = cv_szColorAcq, cv_szColorSync = cv_szDepthAcq;

    // VGA format color image
    cv_colorAcqImage=cvCreateImage(cv_szColorAcq,IPL_DEPTH_8U,3);
    if (cv_colorAcqImage==NULL)
    {
        printf("Unable to create color image buffer\n");
        exit(0);
    }
    // QVGA format depth image
    cv_depthAcqImage=cvCreateImage(cv_szDepthAcq,IPL_DEPTH_16U,1);
    cv_depthAcqImageDisplay=cvCreateImage(cv_szDepthAcq,IPL_DEPTH_8U,1);
    if (cv_depthAcqImage==NULL)
    {
        printf("Unable to create depth image buffer\n");
        exit(0);
    }
    // QVGA format depth color image
    cv_depthSyncImage=cvCreateImage(cv_szDepthSync,IPL_DEPTH_8U,1);
    if (cv_depthSyncImage==NULL)
    {
        printf("Unable to create depth color image buffer\n");
        exit(0);
    }
    // QVGA format depth color image
    cv_colorSyncImage=cvCreateImage(cv_szColorSync,IPL_DEPTH_8U,3);
    if (cv_colorSyncImage==NULL)
    {
        printf("Unable to create color depth image buffer\n");
        exit(0);
    }
    // QVGA format confidence image
    cv_confidenceImage=cvCreateImage(cv_szConfidence,IPL_DEPTH_8U,1);
    if (cv_confidenceImage==NULL)
    {
        printf("Unable to create confidence image buffer\n");
        exit(0);
    }
    // Empty image
    cv_emptyImage=cvCreateImage(cv_szColorSync,IPL_DEPTH_8U,1);
    if (cv_emptyImage==NULL)
    {
        printf("Unable to create empty image buffer\n");
        exit(0);
    }


    int frameCountPrevious = -1;
    while (true) {
        int frameCount = getFrameCount();
        int timeStamp = getTimeStamp();
        if (frameCount > frameCountPrevious) {
            frameCountPrevious = frameCount;
            printf("%d\n", frameCount);

            int countDepth = 0;
            for (int i=0; i<heightDepthAcq; i++) {
                for (int j=0; j<widthDepthAcq; j++) {
                    if (dispDepthAcqFlag || (saveDepthAcqFlag && (flagExportType == FILETYPE_JPG))) {
                        cvSet2D(cv_depthAcqImage,i,j,cvScalar(pixelsDepthAcq[countDepth]));
                        cvSet2D(cv_depthAcqImageDisplay,i,j,cvScalar(pixelsDepthAcq[countDepth]/divideDepthBrightnessCV));
                    }
                    if (dispColorSyncFlag || (saveColorSyncFlag && (flagExportType == FILETYPE_JPG))) {
                        cvSet2D(cv_colorSyncImage,i,j,cvScalar(pixelsColorSync[3*countDepth+2],pixelsColorSync[3*countDepth+1],pixelsColorSync[3*countDepth])); //BGR format
                    }
                    countDepth++;
                }
            }
            int countColor = 0;
            for (int i=0; i<heightColor; i++) {
                for (int j=0; j<widthColor; j++) {
                    if (dispColorAcqFlag || (saveColorAcqFlag && flagExportType)) {
                        cvSet2D(cv_colorAcqImage,i,j,cvScalar(pixelsColorAcq[3*countColor+2],pixelsColorAcq[3*countColor+1],pixelsColorAcq[3*countColor])); //BGR format
                    }
                    if (dispDepthSyncFlag || (saveDepthSyncFlag && flagExportType)) {
                        cvSet2D(cv_depthSyncImage,i,j,cvScalar(pixelsDepthSync[countColor]/divideDepthBrightnessCV));
                    }
                    countColor++;
                }
            }
            int countConfidence = 0;
            for (int i=0; i<FORMAT_QVGA_HEIGHT; i++) {
                for (int j=0; j<FORMAT_QVGA_WIDTH; j++) {
                    if (dispConfidenceFlag || (saveConfidenceFlag && flagExportType)) {
                        cvSet2D(cv_confidenceImage,i,j,cvScalar(pixelsConfidenceQVGA[countConfidence]/divideConfidenceBrightnessCV));
                    }
                    countConfidence++;
                }
            }

            removeDepthWithoutColor(cv_depthAcqImage, cv_colorSyncImage);

            if (dispColorAcqFlag) cvShowImage("Acq Color",cv_colorAcqImage);
            if (dispDepthAcqFlag) cvShowImage("Acq Depth",cv_depthAcqImage);
            if (dispDepthSyncFlag) cvShowImage("Synchronized Depth",cv_depthSyncImage);
            if (dispColorSyncFlag) cvShowImage("Synchronized Color",cv_colorSyncImage);
            if (dispConfidenceFlag) cvShowImage("Confidence",cv_confidenceImage);
            if (dispColorAcqFlag+dispColorSyncFlag+dispDepthAcqFlag+dispDepthSyncFlag+dispConfidenceFlag == 0)
                cvShowImage("Empty",cv_emptyImage);


            if (flagExportType == FILETYPE_JPG)
            {
                if (saveDepthAcqFlag) {
                    sprintf(fileNameDepthAcq,"%s%05u.png",baseNameDepthAcq,frameCount);
                    cvSaveImage(fileNameDepthAcq,cv_depthAcqImage);
                }
                if (saveColorAcqFlag) {
                    sprintf(fileNameColorAcq,"%s%05u.png",baseNameColorAcq,frameCount);
                    cvSaveImage(fileNameColorAcq,cv_colorAcqImage);
                }
                if (saveDepthSyncFlag) {
                    sprintf(fileNameDepthSync,"%s%05u.png",baseNameDepthSync,frameCount);
                    cvSaveImage(fileNameDepthSync,cv_depthSyncImage);
                }
                if (saveColorSyncFlag) {
                    sprintf(fileNameColorSync,"%s%05u.png",baseNameColorSync,frameCount);
                    cvSaveImage(fileNameColorSync,cv_colorSyncImage);
                }
                if (saveConfidenceFlag) {
                    sprintf(fileNameConfidence,"%s%05u.png",baseNameConfidence,frameCount);
                    cvSaveImage(fileNameConfidence,cv_confidenceImage);
                }
            } else if (flagExportType == FILETYPE_PNM) {
                if (saveDepthAcqFlag) {
                    sprintf(fileNameDepthAcq,"%s%05u.pnm",baseNameDepthAcq,frameCount);
                    saveDepthFramePNM(fileNameDepthAcq, pixelsDepthAcq, widthDepthAcq, heightDepthAcq, timeStamp);
                }
                if (saveColorAcqFlag) {
                    sprintf(fileNameColorAcq,"%s%05u.pnm",baseNameColorAcq,frameCount);
                    saveColorFramePNM(fileNameColorAcq, pixelsColorAcq, widthColor, heightColor, timeStamp);
                }
                if (saveDepthSyncFlag) {
                    sprintf(fileNameDepthSync,"%s%05u.pnm",baseNameDepthSync,frameCount);
                    saveDepthFramePNM(fileNameDepthSync, pixelsDepthSync, widthColor, heightColor, timeStamp);
                }
                if (saveColorSyncFlag) {
                    sprintf(fileNameColorSync,"%s%05u.pnm",baseNameColorSync,frameCount);
                    saveColorFramePNM(fileNameColorSync, pixelsColorSync, widthDepthAcq, heightDepthAcq, timeStamp);
                }
                if (saveConfidenceFlag) {
                    sprintf(fileNameConfidence,"%s%05u.pnm",baseNameConfidence,frameCount);
                    saveDepthFramePNM(fileNameConfidence, pixelsConfidenceQVGA, FORMAT_QVGA_WIDTH, FORMAT_QVGA_HEIGHT, timeStamp);
                }
            }

            char key = cvWaitKey(10);
            if (key==27)
            {
                printf("Quitting main loop from OpenCV\n");
                stop_capture();
                break;
            }
            if (key == 'r' || key == 'R'){
                saveDepthAcqFlag = !saveDepthAcqFlag;
                saveColorSyncFlag = !saveColorSyncFlag;
            }

        }
    }


    return 0;
}
