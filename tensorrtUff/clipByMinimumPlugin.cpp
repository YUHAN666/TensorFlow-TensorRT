#include "ClipByMinimumPlugin.h"
#include "plugin.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include "miniKernel.h"
#include <algorithm>

#define DEBUG 0

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::ClipByMinimum;
using nvinfer1::plugin::ClipByMinimumPluginCreator;


namespace
{
	const char* CLIP_MINI_PLUGIN_VERSION{ "1" };
	const char* CLIP_MINI_PLUGIN_NAME{ "ClipByMinimum_TRT2" };
} // namespace

PluginFieldCollection ClipByMinimumPluginCreator::mFC{};
std::vector<PluginField> ClipByMinimumPluginCreator::mPluginAttributes;

ClipByMinimumPluginCreator::ClipByMinimumPluginCreator()
{
	mPluginAttributes.emplace_back(PluginField("minimum", nullptr, PluginFieldType::kFLOAT32, 1));

	mFC.nbFields = mPluginAttributes.size();
	mFC.fields = mPluginAttributes.data();
}

const char* ClipByMinimumPluginCreator::getPluginName() const
{
	return CLIP_MINI_PLUGIN_NAME;
};

const char* ClipByMinimumPluginCreator::getPluginVersion() const
{
	return CLIP_MINI_PLUGIN_VERSION;
};

const PluginFieldCollection* ClipByMinimumPluginCreator::getFieldNames()
{
	return &mFC;
};

IPluginV2IOExt* ClipByMinimumPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
	const PluginField* fields = fc->fields;
	for (int i = 0; i < fc->nbFields; ++i)
	{
		const char* attrName = fields[i].name;
		if (!strcmp(attrName, "minimum"))
		{
			assert(fields[i].type == PluginFieldType::kFLOAT32);
			mMinimum = *(static_cast<const float*>(fields[i].data));
		}

	}
	return new ClipByMinimum(mMinimum);
};

IPluginV2IOExt* ClipByMinimumPluginCreator::deserializePlugin(const char* name, const void* data, size_t length)
{
	return new ClipByMinimum(data, length);
};

ClipByMinimum::ClipByMinimum(float minimum)
	: mMinimum(minimum)
{
};

int ClipByMinimum::getNbOutputs() const
{
	return 1;
};

Dims ClipByMinimum::getOutputDimensions(int index, const Dims* inputDims, int nbInputs)
{
	//assert(nbInputs == 1);
	nvinfer1::Dims const& input = inputDims[0];
	assert(index == 0);
	nvinfer1::Dims output;
	output.nbDims = input.nbDims;
	for (int d = 0; d < input.nbDims; ++d)
	{
		output.d[d] = input.d[d];
	}
	std::cout << "mMinimum:" << mMinimum << std::endl;
	std::cout << "mOutputDims.d[0]:" << output.d[0] << std::endl;
	std::cout << "mOutputDims.d[1]:" << output.d[1] << std::endl;
	std::cout << "mOutputDims.d[2]:" << output.d[2] << std::endl;

	std::cout << "mInputDims.d[0]:" << input.d[0] << std::endl;
	std::cout << "mInputDims.d[1]:" << input.d[1] << std::endl;
	std::cout << "mInputDims.d[2]:" << input.d[2] << std::endl;

	return output;
};

int ClipByMinimum::initialize()
{
	return 0;
};

void ClipByMinimum::terminate() {

};

void ClipByMinimum::destroy() {

};

size_t ClipByMinimum::getWorkspaceSize(int) const
{
	return 0;
}

size_t ClipByMinimum::getSerializationSize() const
{
	// minimum:float, dimensions:Dims 3 * 2
	return sizeof(float) + sizeof(int) * 3 * 2;
};

void ClipByMinimum::serialize(void* buffer) const
{
	char *d = reinterpret_cast<char*>(buffer), *a = d;
	write(d, mMinimum);
	write(d, mInputDims.d[0]);
	write(d, mInputDims.d[1]);
	write(d, mInputDims.d[2]);
	write(d, mOutputDims.d[0]);
	write(d, mOutputDims.d[1]);
	write(d, mOutputDims.d[2]);
	ASSERT(d == a + getSerializationSize());
};

ClipByMinimum::ClipByMinimum(const void* data, size_t length)
{
	const char *d = reinterpret_cast<const char*>(data), *a = d;
	mMinimum = read<float>(d);
	mInputDims = Dims3();
	mInputDims.d[0] = read<int>(d);
	mInputDims.d[1] = read<int>(d);
	mInputDims.d[2] = read<int>(d);
	mOutputDims = Dims3();
	mOutputDims.d[0] = read<int>(d);
	mOutputDims.d[1] = read<int>(d);
	mOutputDims.d[2] = read<int>(d);
	ASSERT(d == a + length);
};

const char* ClipByMinimum::getPluginType() const
{
	return "ClipByMinimum_TRT2";
};

const char* ClipByMinimum::getPluginVersion() const
{
	return "1";
};

IPluginV2IOExt* ClipByMinimum::clone() const
{
	return new ClipByMinimum(*this);
};

void ClipByMinimum::setPluginNamespace(const char* libNamespace)
{
	mNameSpace = libNamespace;
	std::cout << mNameSpace << std::endl;

};

const char* ClipByMinimum::getPluginNamespace() const
{
	return mNameSpace.c_str();
}


bool ClipByMinimum::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)const
{
	// return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
	return inOut[pos].format == PluginFormat::kNCHW && (inOut[pos].type == DataType::kFLOAT);

}

int ClipByMinimum::enqueue(
	int batch_size, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	//std::cout << batch_size << std::endl;
	int nchan = mOutputDims.d[0];
	float mini = mMinimum;
	int ochannel = mOutputDims.d[0];

	dim3 block(ochannel);		//grid线程分布
															//dim3 grid((osize.x - 1) / block.x + 1, (osize.y - 1) / block.y + 1, std::min(batch_size * nchan, 65535));		//每个grid内线程
															//dim3 grid((osize.x - 1) / block.x, (osize.y - 1) / block.y, std::min(batch_size * nchan, 65535));
	dim3 grid(1);
	//std::cout << grid.x << "	" << grid.y << "	" << grid.z << std::endl;
	clipByMinimum(grid, block, stream, batch_size * nchan, mini, ochannel, static_cast<float const*>(inputs[0]), static_cast<float*>(outputs[0]));


	return cudaGetLastError() != cudaSuccess;
};

// Return the DataType of the plugin output at the requested index
DataType ClipByMinimum::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
	// Only 1 input and 1 output from the plugin layer
	ASSERT(index == 0);

	// Only DataType::kFLOAT is acceptable by the plugin layer
	return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool ClipByMinimum::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
	return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool ClipByMinimum::canBroadcastInputAcrossBatch(int inputIndex) const
{
	return false;
}


void ClipByMinimum::configurePlugin(const PluginTensorDesc * in, int nbInput, const PluginTensorDesc * out, int nbOutput) {
	//assert(nbInput == 1);
	mInputDims = in->dims;

	assert(nbOutput == 1);
	mOutputDims = out->dims;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void ClipByMinimum::attachToContext(
	cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void ClipByMinimum::detachFromContext() {}