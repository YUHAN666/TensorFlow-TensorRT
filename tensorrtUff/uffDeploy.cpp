#include "uffDeploy.h"


bool uffToTRTModel(const std::string& modelFile,
	const std::string& engineFile,
	IHostMemory*& trtModelStream)
{
	SampleUniquePtr<IUffParser> parser{ createUffParser() };
	parser->registerInput("Image", Dims3(IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH), UffInputOrder::kNCHW);
	parser->registerOutput("decision_out");

	SampleUniquePtr<IBuilder> builder{ createInferBuilder(gLogger.getTRTLogger()) };
	if (!builder.get())
	{
		gLogError << "Failed to create infer builder. " << std::endl;
		return false;
	}
	SampleUniquePtr<INetworkDefinition> network{ builder->createNetwork() };

	if (!network.get())
	{
		gLogError << "Failed to create network. " << std::endl;
		return false;
	}

	std::cout << "Parsing Uff model..." << std::endl;
	if (!parser->parse(modelFile.data(), *network, nvinfer1::DataType::kFLOAT))
	{
		gLogError << "Failure while parsing UFF file" << std::endl;
		return false;
	}
	std::cout << "Successfully parsed Uff model" << std::endl;

	SampleUniquePtr<IBuilderConfig> networkConfig{ builder->createBuilderConfig() };
	networkConfig->setMaxWorkspaceSize(4600_MiB);

	const int maxBatchSize = 1;
	builder->setMaxBatchSize(maxBatchSize);
	builder->setFp16Mode(false);
	builder->setInt8Mode(false);
	std::cout << "Building Engine..." << std::endl;
	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *networkConfig);
	std::cout << "Successfully built Engine" << std::endl;

	std::cout << "serializing Engine..." << std::endl;
	trtModelStream = engine->serialize();
	assert(trtModelStream != nullptr);
	std::cout << "Successfully serialized Engine to Stream" << std::endl;


	// 保存引擎
	nvinfer1::IHostMemory* data = engine->serialize();
	std::ofstream file;
	file.open(engineFile, std::ios::binary | std::ios::out);
	cout << "writing engine file..." << endl;
	file.write((const char*)data->data(), data->size());

	cout << "Engine was serialized and saved to" << engineFile << endl;
	file.close();



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