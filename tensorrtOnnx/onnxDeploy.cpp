#include "onnxDeploy.h"


bool onnxToTRTModel(const std::string& modelFile, // onnx文件的名字
	const std::string& filename,  // TensorRT引擎的名字 
	IHostMemory*& trtModelStream) // output buffer for the TensorRT model
{
	// 创建builder
	IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	INetworkDefinition* network = builder->createNetworkV2(explicitBatch);


	// 解析ONNX模型
	auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());


	// 判断是否成功解析ONNX模型
	if (!parser->parseFromFile(modelFile.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
	{
		gLogError << "Failure while parsing ONNX file" << std::endl;
		return false;
	}

	// 建立推理引擎
	builder->setMaxBatchSize(1);
	builder->setMaxWorkspaceSize(4600_MiB);
	builder->setFp16Mode(false);
	builder->setInt8Mode(false);

	cout << "start building engine" << endl;
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	cout << "build engine done" << endl;
	assert(engine);

	// 销毁模型解释器
	parser->destroy();

	// 序列化引擎
	trtModelStream = engine->serialize();

	// 保存引擎
	nvinfer1::IHostMemory* data = engine->serialize();
	std::ofstream file;
	file.open(filename, std::ios::binary | std::ios::out);
	cout << "writing engine file..." << endl;
	file.write((const char*)data->data(), data->size());
	cout << "save engine file done" << endl;
	file.close();

	// 销毁所有相关的东西
	engine->destroy();
	network->destroy();
	builder->destroy();

	return true;
}


bool readTrtFile(const std::string& engineFile, //name of the engine file
	IHostMemory*& trtModelStream)  //output buffer for the TensorRT model
{
	using namespace std;
	fstream file;
	cout << "loading filename from:" << engineFile << endl;
	nvinfer1::IRuntime* trtRuntime;
	file.open(engineFile, ios::binary | ios::in);
	file.seekg(0, ios::end);
	int length = file.tellg();
	//cout << "length:" << length << endl;
	file.seekg(0, ios::beg);
	std::unique_ptr<char[]> data(new char[length]);
	file.read(data.get(), length);
	file.close();
	cout << "load engine done" << endl;
	std::cout << "deserializing" << endl;
	trtRuntime = createInferRuntime(gLogger.getTRTLogger());
	ICudaEngine* engine = trtRuntime->deserializeCudaEngine(data.get(), length);
	assert(engine != nullptr);
	cout << "deserialize done" << endl;
	trtModelStream = engine->serialize();

	return true;
}
