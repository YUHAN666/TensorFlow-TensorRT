#include "opencvInference.h"


wstring s2wsO(const std::string &s)
{
	std::string curLocale = setlocale(LC_ALL, "");
	const char * _Source = s.c_str();
	size_t _Dsize = mbstowcs(NULL, _Source, 0) + 1;
	wchar_t *_Dest = new wchar_t[_Dsize];
	wmemset(_Dest, 0, _Dsize);
	mbstowcs(_Dest, _Source, _Dsize);
	std::wstring result = _Dest;
	delete[]_Dest;
	setlocale(LC_ALL, curLocale.c_str());
	return result;
}


void GetFilesO(string path, vector<wstring>& files, vector<string>& fileNames)
{
	//文件句柄
	intptr_t hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	string p;

	if ((hFile = _findfirst(p.assign(path).append("/*.png").c_str(), &fileinfo)) != -1)
	{
		do
		{
			string q;
			wstring o = s2wsO(q.append(path).append("/").append(fileinfo.name));
			files.push_back(o);

			fileNames.push_back(q);
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}
}

vector<float> prepareImage(cv::Mat& img)
{
	cv::Mat img_float;
	img.convertTo(img_float, CV_32FC1, 1 / 255.0, 0.0);        // 图像归一化处理
															   //cv::Mat img_trans;
															   //cv::transpose(img_float, img_trans);
															   //将图像矩阵以CHW顺序读入vector中

	vector<Mat> input_channels(IMAGE_CHANNEL);
	cv::split(img_float, input_channels);


	vector<float> result(IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL);
	auto data = result.data();
	int channelLength = IMAGE_HEIGHT * IMAGE_WIDTH;
	memcpy(data, input_channels[0].data, channelLength * sizeof(float));

	return result;
}



void doInferenceOnnx(IHostMemory* trtModelStream, vector<string> files)
{
	// get engine
	assert(trtModelStream != nullptr);
	IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
	assert(runtime != nullptr);

	// 创建推理引擎
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size());
	assert(engine != nullptr);
	std::cout << "Successfully deserialized Engine" << std::endl;

	trtModelStream->destroy();
	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);

	// 读取输入数据到缓冲区管理对象中
	assert(engine->getNbBindings() == 2);   // 本模型只有一个输入和一个输出
	void* gpuBuffers[2];       // GPU缓冲区内存指针 *buffer[0]: input; *buffer[1]: output
	std::vector<int64_t> bufferSize;        // 缓冲区内存大小 分为NbBindings块，对应每一个输入和输出
	int nbBindings = engine->getNbBindings();
	bufferSize.resize(nbBindings);

	for (int i = 0; i < nbBindings; ++i)
	{
		nvinfer1::Dims dims = engine->getBindingDimensions(i);      // Binding的维度
		nvinfer1::DataType dtype = engine->getBindingDataType(i);   // Binding的数据类型
		int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);       // 根据维度和数据类型计算出Binding所需的内存大小
		bufferSize[i] = totalSize;
		CHECK(cudaMalloc(&gpuBuffers[i], totalSize));      // 在GPU端为Binding分配相应大小的内存
	}

	// 创建CUDA流以执行此推断
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));       // 创建CUDA流
	int falseAccount = 0;


	for (int i = 0; i < files.size(); i++)
	{
		// 读取图片

		auto t_start_pre = std::chrono::high_resolution_clock::now();
		vector<cv::Mat> inputImgs;
		std::cout << files[i] << std::endl;
		cv::Mat img = cv::imread(files[i]);
		inputImgs.push_back(img);
		// 图片预处理，并以CHW序列化
		vector<float> curInput = prepareImage(img);
		auto t_end_pre = std::chrono::high_resolution_clock::now();
		float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
		std::cout << "prepare image take: " << total_pre << " ms." << endl;

		// 将输入图片从curInput异步复制到GPU内存中
		CHECK(cudaMemcpyAsync(gpuBuffers[0], curInput.data(), bufferSize[0], cudaMemcpyHostToDevice, stream));

		// 执行推理
		auto t_start = std::chrono::high_resolution_clock::now();
		//context->execute(1, gpuBuffers);
		context->executeV2(gpuBuffers);
		auto t_end = std::chrono::high_resolution_clock::now();
		float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
		std::cout << "Inference take: " << total << " ms." << endl;

		// 主机端输出容器out
		int outSize = bufferSize[1] / sizeof(float);
		float* out = new float[outSize];
		//将模型推理输出从gpu buffer中拷贝到主机端out中，本模型只有一个output node
		CHECK(cudaMemcpyAsync(out, gpuBuffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream));
		cudaStreamSynchronize(stream);      //等待流执行完毕

		std::cout << "Output: " << out[0] << endl;		//本模型只有一个output node，即NG概率

														//验证结果(需根据图片命名做相应修改)
		string p = "p";
		string n = "n";
		if ((files[i].substr(106, 1) == p) && (out[0] < 0.5)) {
			falseAccount += 1;
		}
		else if ((files[i].substr(106, 1) == n) && (out[0] > 0.5)) {
			falseAccount += 1;
		}


		cout << "\n" << endl;

		inputImgs.clear();
	}


	cout << "False Account: " << falseAccount << " out of " << files.size() << " images" << endl;
	cout << "Accuracy: " << 1 - float(falseAccount) / float(files.size()) << endl;


	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(gpuBuffers[0]));
	CHECK(cudaFree(gpuBuffers[1]));

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	////cv::imshow("result", img);
	//waitKey(0);

}