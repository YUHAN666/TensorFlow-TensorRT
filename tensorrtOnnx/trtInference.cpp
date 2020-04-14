#include "trtInference.h"



void doInferenceOnnx(IHostMemory* trtModelStream, vector<wstring> files)
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
	GdiplusStartupInput gdiplusstartupinput;
	ULONG_PTR gdiplustoken;
	GdiplusStartup(&gdiplustoken, &gdiplusstartupinput, NULL);

	for (int i = 0; i < files.size(); i++)
	{
		// 读取图片
	
		std::vector<float> img(IMAGE_HEIGHT*IMAGE_WIDTH*IMAGE_CHANNEL);
		// 图片预处理，并以CHW序列化
		std::wcout << files[i] << std::endl;
		auto t_start_pre = std::chrono::high_resolution_clock::now();
		//readPNGFile(filename, img.data(), 320, 928, 3);
		//-------------------------------------------------------
		Bitmap* bmp = new Bitmap(files[i].c_str());
		UINT height = bmp->GetHeight();	//928
		UINT width = bmp->GetWidth();	//320
		UINT channel = height*width;

		Color color;
		//Color color00;
		//bmp->GetPixel(0, 0, &color00);
		//std::cout << ((float)color00.GetRed()) / 255.0 << std::endl;
		for (UINT y = 0; y < height; y++) {
			for (UINT x = 0; x < width; x++) {
				bmp->GetPixel(x, y, &color);
				img[x + y*width] = ((float)color.GetRed()) / 255.0;
				//img[x + y*width + channel] = ((float)color.GetRed())/255.0;
				//img[x + y*width + channel*2] = ((float)color.GetRed())/255.0;
				//img[(x + y*width) * 3] = ((float)color.GetRed()) / 255.0;
				//img[(x + y*width) * 3 + 1] = ((float)cqdfolor.GetRed()) / 255.0;
				//img[(x + y*width) * 3 + 2] = ((float)color.GetRed()) / 255.0;
				//img[(x*height + y) * 3] = ((float)color.GetRed()) / 255.0;
				//img[(x*height + y) * 3 + 1] = ((float)color.GetRed()) / 255.0;
				//img[(x*height + y) * 3 + 2] = ((float)color.GetRed()) / 255.0;
				//img[x*height + y] = ((float)color.GetRed())/255.0;
				//img[x*height + y + channel] = ((float)color.GetRed())/255.0;
				//img[x*height + y + channel*2] = ((float)color.GetRed())/255.0;
			}
		}
		//std::cout << img[890879] << std::endl;
		//---------------------------------------------------------

		auto t_end_pre = std::chrono::high_resolution_clock::now();
		float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
		std::cout << "prepare image take: " << total_pre << " ms." << endl;

		// 将输入图片从curInput异步复制到GPU内存中
		CHECK(cudaMemcpyAsync(gpuBuffers[0], img.data(), bufferSize[0], cudaMemcpyHostToDevice, stream));

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
		wstring p = L"p";
		wstring n = L"n";
		if ((files[i].substr(106, 1)==p) && (out[0] < 0.5)) {
			falseAccount += 1;
		}
		else if ((files[i].substr(106, 1) == n) && (out[0] > 0.5)) {
			falseAccount += 1;
		}


		cout << "\n" << endl;

		img.clear();
	}


	cout << "False Account: " << falseAccount << " out of "<< files.size() << " images" <<endl;
	cout << "Accuracy: " << 1-float(falseAccount)/float(files.size()) << endl;


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


wstring s2ws(const std::string &s)
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


void GetFiles(string path, vector<wstring>& files, vector<string>& fileNames)
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
			wstring o = s2ws(q.append(path).append("/").append(fileinfo.name));
			files.push_back(o);

			fileNames.push_back(fileinfo.name);
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}
}