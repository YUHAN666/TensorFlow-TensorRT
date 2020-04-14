#include "resizeNearestPlugin.h"
#include "plugin.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include "kernel.h"
#include <algorithm>

#define DEBUG 0

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::ResizeNearest;
using nvinfer1::plugin::ResizeNearestPluginCreator;

namespace
{
	const char* RESIZE_PLUGIN_VERSION{ "1" };
	const char* RESIZE_PLUGIN_NAME{ "ResizeNearest_TRT2" };
} // namespace

PluginFieldCollection ResizeNearestPluginCreator::mFC{};
std::vector<PluginField> ResizeNearestPluginCreator::mPluginAttributes;

ResizeNearestPluginCreator::ResizeNearestPluginCreator()
{
	mPluginAttributes.emplace_back(PluginField("scale_x", nullptr, PluginFieldType::kFLOAT32, 1));
	mPluginAttributes.emplace_back(PluginField("scale_y", nullptr, PluginFieldType::kFLOAT32, 1));

	mFC.nbFields = mPluginAttributes.size();
	mFC.fields = mPluginAttributes.data();
}

const char* ResizeNearestPluginCreator::getPluginName() const
{
	return RESIZE_PLUGIN_NAME;
};

const char* ResizeNearestPluginCreator::getPluginVersion() const
{
	return RESIZE_PLUGIN_VERSION;
};

const PluginFieldCollection* ResizeNearestPluginCreator::getFieldNames()
{
	return &mFC;
};

IPluginV2IOExt* ResizeNearestPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
	const PluginField* fields = fc->fields;
	for (int i = 0; i < fc->nbFields; ++i)
	{
		const char* attrName = fields[i].name;
		if (!strcmp(attrName, "scale_x"))
		{
			assert(fields[i].type == PluginFieldType::kFLOAT32);
			mScaleX = *(static_cast<const float*>(fields[i].data));
		}
		if (!strcmp(attrName, "scale_y"))
		{
			assert(fields[i].type == PluginFieldType::kFLOAT32);
			mScaleY = *(static_cast<const float*>(fields[i].data));
		}
	}
	return new ResizeNearest(mScaleX, mScaleY);
};

IPluginV2IOExt* ResizeNearestPluginCreator::deserializePlugin(const char* name, const void* data, size_t length)
{
	return new ResizeNearest(data, length);
};

ResizeNearest::ResizeNearest(float scaleX, float scaleY)
	: mScaleX(scaleX), mScaleY(scaleY)
{
	assert(mScaleX > 0);
	assert(mScaleY > 0);
};

int ResizeNearest::getNbOutputs() const
{
	return 1;
};

Dims ResizeNearest::getOutputDimensions(int index, const Dims* inputDims, int nbInputs)
{
	//assert(nbInputs == 1);
	nvinfer1::Dims const& input = inputDims[0];
	assert(index == 0);
	nvinfer1::Dims output;
	output.nbDims = input.nbDims;
	for (int d = 0; d < input.nbDims; ++d)
	{
		if (d == input.nbDims - 2)
		{
			output.d[d] = int(input.d[d] * mScaleY);
		}
		else if (d == input.nbDims - 1)
		{
			output.d[d] = int(input.d[d] * mScaleX);
		}
		else
		{
			output.d[d] = input.d[d];
		}

	}
	std::cout << "mScaleX:" << mScaleX << std::endl;
	std::cout << "mScaleY:" << mScaleY << std::endl;
	std::cout << "mOutputDims.d[0]:" << output.d[0] << std::endl;
	std::cout << "mOutputDims.d[1]:" << output.d[1] << std::endl;
	std::cout << "mOutputDims.d[2]:" << output.d[2] << std::endl;

	std::cout << "mInputDims.d[0]:" << input.d[0] << std::endl;
	std::cout << "mInputDims.d[1]:" << input.d[1] << std::endl;
	std::cout << "mInputDims.d[2]:" << input.d[2] << std::endl;

	return output;
};

int ResizeNearest::initialize()
{
	return 0;
};

void ResizeNearest::terminate() {

};

void ResizeNearest::destroy() {

};

size_t ResizeNearest::getWorkspaceSize(int) const
{
	return 0;
}

size_t ResizeNearest::getSerializationSize() const
{
	// scale, dimensions: 3 * 2
	return sizeof(float) * 2 + sizeof(int) * 3 * 2;
};

void ResizeNearest::serialize(void* buffer) const
{
	char *d = reinterpret_cast<char*>(buffer), *a = d;
	write(d, mScaleX);
	write(d, mScaleY);
	write(d, mInputDims.d[0]);
	write(d, mInputDims.d[1]);
	write(d, mInputDims.d[2]);
	write(d, mOutputDims.d[0]);
	write(d, mOutputDims.d[1]);
	write(d, mOutputDims.d[2]);
	ASSERT(d == a + getSerializationSize());
};

ResizeNearest::ResizeNearest(const void* data, size_t length)
{
	const char *d = reinterpret_cast<const char*>(data), *a = d;
	mScaleX = read<float>(d);
	mScaleY = read<float>(d);
	mInputDims = Dims3();
	mInputDims.d[0] = read<int>(d);
	mInputDims.d[1] = read<int>(d);
	mInputDims.d[2] = read<int>(d);
	mOutputDims = Dims3();
	mOutputDims.d[0] = read<int>(d);
	mOutputDims.d[1] = read<int>(d);
	mOutputDims.d[2] = read<int>(d);
	//std::cout << "mInputDims.d[0]:" << mInputDims.d[0] << "		mInputDims.d[1]:" << mInputDims.d[1] << "	mInputDims.d[2]:" << mInputDims.d[2] << std::endl;
	//std::cout << "mOutputDims.d[0]:" << mOutputDims.d[0] << "	mOutputDims.d[1]:" << mOutputDims.d[1] << "	mOutputDims.d[2]:" << mOutputDims.d[2] << std::endl;
	ASSERT(d == a + length);
};

const char* ResizeNearest::getPluginType() const
{
	return "ResizeNearest_TRT2";
};

const char* ResizeNearest::getPluginVersion() const
{
	return "1";
};

IPluginV2IOExt* ResizeNearest::clone() const
{
	return new ResizeNearest(*this);
};

void ResizeNearest::setPluginNamespace(const char* libNamespace)
{
	mNameSpace = libNamespace;
};

const char* ResizeNearest::getPluginNamespace() const
{
	return mNameSpace.c_str();
}

// bool ResizeNearest::supportsFormat(DataType type, PluginFormat format) const
// {
//     return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
// };
bool ResizeNearest::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)const
{
	// return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
	return inOut[pos].format == PluginFormat::kNCHW && (inOut[pos].type == DataType::kFLOAT);

}

int ResizeNearest::enqueue(
	int batch_size, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	//std::cout << batch_size << std::endl;
	int nchan = mOutputDims.d[0];
	float scaleX = mScaleX;
	float scaleY = mScaleY;
	int2 osize = { mOutputDims.d[2], mOutputDims.d[1] };
	int istride = mInputDims.d[2];		//input 每行像素数
	int ostride = mOutputDims.d[2];		//output 每行像素数
	int ibatchstride = mInputDims.d[1] * istride;		//input每个channel像素数
	int obatchstride = mOutputDims.d[1] * ostride;		//output每个channel像素数
	dim3 block(32, 16);		//grid线程分布
	dim3 grid((osize.x - 1) / block.x + 1, (osize.y - 1) / block.y + 1, std::min(batch_size * nchan, 65535));		//每个grid内线程
																													//dim3 grid((osize.x - 1) / block.x, (osize.y - 1) / block.y, std::min(batch_size * nchan, 65535));
																													//dim3 grid(osize.x / block.x, osize.y / block.y , std::min(batch_size * nchan, 65535));
																													//std::cout << grid.x << "	" << grid.y << "	" << grid.z << std::endl;
	resizeNearest(grid, block, stream, batch_size * nchan, scaleX, scaleY, osize, static_cast<float const*>(inputs[0]), istride,
		ibatchstride, static_cast<float*>(outputs[0]), ostride, obatchstride);


	return cudaGetLastError() != cudaSuccess;
};

// Return the DataType of the plugin output at the requested index
DataType ResizeNearest::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
	// Only 1 input and 1 output from the plugin layer
	ASSERT(index == 0);

	// Only DataType::kFLOAT is acceptable by the plugin layer
	return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool ResizeNearest::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
	return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool ResizeNearest::canBroadcastInputAcrossBatch(int inputIndex) const
{
	return false;
}

// // Configure the layer with input and output data types.
// void ResizeNearest::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
//     const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
//     const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
// {
//     assert(nbInputs == 1);
//     mInputDims = inputDims[0];

//     assert(nbOutputs == 1);
//     mOutputDims = outputDims[0];
// }
void ResizeNearest::configurePlugin(const PluginTensorDesc * in, int nbInput, const PluginTensorDesc * out, int nbOutput) {
	//assert(nbInput == 1);
	mInputDims = in->dims;

	assert(nbOutput == 1);
	mOutputDims = out->dims;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void ResizeNearest::attachToContext(
	cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void ResizeNearest::detachFromContext() {}