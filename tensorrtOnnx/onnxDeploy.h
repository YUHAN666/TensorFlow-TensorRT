#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"
#include "logger.h"
#include "NvInferRuntimeCommon.h"

#include <stdlib.h>
#include <numeric>
#include <cuda_runtime_api.h>
#include <memory>


#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
//#include "resizeNearestPlugin.h"


using namespace std;
using namespace nvinfer1;


bool onnxToTRTModel(const std::string& modelFile, // onnx文件的名字
	const std::string& filename,  // TensorRT引擎的名字 
	IHostMemory*& trtModelStream); // output buffer for the TensorRT model


bool readTrtFile(const std::string& engineFile, //name of the engine file
	IHostMemory*& trtModelStream);  //output buffer for the TensorRT model
