#include "stdafx.h"
#include "targetver.h"
#include <stdio.h>
#include <tchar.h>
#include "onnxDeploy.h"
#include "opencvInference.h"

//onnx文件地址，如trt引擎不存在则会读取onnx模型并建立trt引擎，如存在trt引擎则直接读取trt引擎
string onnxFile = "F:/CODES/tf_PZT_2020/pbMode/tensorrt_fastscnn.onnx";
//trt引擎文件地址，如引擎不存在则会从onnx模型创建并写入该地址
string onnxEngineFile = "F:/CODES/tf_PZT_2020/pbMode/tensorrtonnx.trt";
//图片文件夹
string path = "F:/TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.0.cudnn7.6/TensorRT-7.0.0.11/data/mnist/images/valid_image/";

extern int IMAGE_HEIGHT = 928;
extern int IMAGE_WIDTH = 320;
extern int IMAGE_CHANNEL = 1;


int main()
{

	vector<wstring> files;
	vector<string> filenames;

	// 读取文件目录
	GetFilesO(path, files, filenames);
	//wstring n = L"n";
	//for (int i = 0; i < files.size(); i++)
	//{
	//	wcout << (files[i].substr(106,1)==n) << endl;
	//}
	//system("pause");


	 //create a TensorRT model from the onnx model and serialize it to a stream
	IHostMemory* trtModelStream{ nullptr };

	// create and load engine
	fstream existEngine;
	existEngine.open(onnxEngineFile, ios::in);
	if (existEngine)
	{
		readTrtFile(onnxEngineFile, trtModelStream);    //从trt文件中读取序列化引擎并反序列化，最后序列化到trtModelStream中
		assert(trtModelStream != nullptr);
	}
	else
	{
		onnxToTRTModel(onnxFile, onnxEngineFile, trtModelStream);       // 将onnx模型转化为trt模型并保存
		assert(trtModelStream != nullptr);
	}


	//do inference
	doInferenceOnnx(trtModelStream, filenames);      //执行推理并打印输出
	system("pause");

	return 0;
}
