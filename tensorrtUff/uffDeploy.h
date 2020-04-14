#pragma once
#include "NvInfer.h"
#include "NvUffParser.h"
#include "logger.h"

#include "NvInferRuntimeCommon.h"

#include <stdlib.h>
#include <numeric>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
//#include "resizeNearestPlugin.h"


using namespace std;
using namespace nvinfer1;
using namespace nvuffparser;
using namespace cv;

extern int IMAGE_HEIGHT;
extern int IMAGE_WIDTH;
extern int IMAGE_CHANNEL;
extern string inputTensor;
extern string outputTensor;


struct InferDeleter
{
	template <typename T>
	void operator()(T* obj) const
	{
		if (obj)
		{
			obj->destroy();
		}
	}
};

template <typename T>
using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;


bool uffToTRTModel(const std::string& modelFile,
	const std::string& engineFile,
	IHostMemory*& trtModelStream);


bool readTrtFile(const std::string& engineFile, //name of the engine file
	IHostMemory*& trtModelStream);  //output buffer for the TensorRT model

