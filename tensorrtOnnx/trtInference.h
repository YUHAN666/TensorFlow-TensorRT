#pragma once
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "NvInferRuntimeCommon.h"
#include <io.h>

#include <stdlib.h>
#include <numeric>
#include <cuda_runtime_api.h>
#include <chrono>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <windows.h>
#include <gdiplus.h>

#pragma comment(lib, "gdiplus.lib")



using namespace Gdiplus;
using namespace std;
using namespace nvinfer1;

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


wstring s2ws(const std::string &s);


void GetFiles(string path, vector<wstring>& files, vector<string>& fileNames);


void doInferenceOnnx(IHostMemory* trtModelStream, vector<wstring> files);
