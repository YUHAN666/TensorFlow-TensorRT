#include <stdio.h>
#include <tchar.h>
#include "uffDeploy.h"
#include "trtInference.h"

//onnx文件地址，如trt引擎不存在则会读取onnx模型并建立trt引擎，如存在trt引擎则直接读取trt引擎
string uffFile = "F:/CODES/tf_PZT_2020/pbMode/tensorrt_fastscnn.uff";
//trt引擎文件地址，如引擎不存在则会从onnx模型创建并写入该地址
string uffEngineFile = "F:/CODES/tf_PZT_2020/pbMode/tensorrtuff.trt";
//图片文件夹
string path = "F:/TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.0.cudnn7.6/TensorRT-7.0.0.11/data/mnist/images/valid_image/";

extern int IMAGE_HEIGHT = 928;
extern int IMAGE_WIDTH = 320;
extern int IMAGE_CHANNEL = 1;

extern string inputTensor = "Image";
extern string outputTensor = "decision_out";


int main()
{

	vector<string> files;


	// 读取文件目录
	GetFiles(path, files);
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
	existEngine.open(uffEngineFile, ios::in);
	if (existEngine)
	{
		readTrtFile(uffEngineFile, trtModelStream);    //从trt文件中读取序列化引擎并反序列化，最后序列化到trtModelStream中
		assert(trtModelStream != nullptr);
	}
	else
	{
		uffToTRTModel(uffFile, uffEngineFile, trtModelStream);       // 将onnx模型转化为trt模型并保存
		assert(trtModelStream != nullptr);
	}


	//do inference
	doInferenceUff(trtModelStream, files);      //执行推理并打印输出
	system("pause");

	return 0;
}
