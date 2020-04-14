#pragma once
#include "NvInfer.h"
#include "NvUffParser.h"
#include "logger.h"
#include "NvInferRuntimeCommon.h"
#include <io.h>

#include <vector>
#include <stdlib.h>
#include <numeric>
#include <cuda_runtime_api.h>
#include <chrono>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace nvinfer1;
using namespace nvuffparser;
using namespace cv;

extern int IMAGE_HEIGHT;
extern int IMAGE_WIDTH;
extern int IMAGE_CHANNEL;


inline int64_t volume(const nvinfer1::Dims& d)
{
	return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}


inline unsigned int getElementSize(nvinfer1::DataType t)
{
	switch (t)
	{
	case nvinfer1::DataType::kINT32: return 4;
	case nvinfer1::DataType::kFLOAT: return 4;
	case nvinfer1::DataType::kHALF: return 2;
	case nvinfer1::DataType::kINT8: return 1;
	}
	throw std::runtime_error("Invalid DataType.");
	return 0;
}


void GetFiles(string path, vector<string>& files);


vector<float> prepareImage(cv::Mat& img);


void doInferenceUff(IHostMemory* trtModelStream, vector<string> files);
