﻿#define TRT_CLIPBYMAXIMUM_PLUGIN_H

#include <cassert>
#include <cuda_runtime_api.h>
#include <string.h>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "maxKernel.h"

extern int IMAGE_HEIGHT;
extern int IMAGE_WIDTH;
extern int IMAGE_CHANNEL;


namespace nvinfer1
{
	namespace plugin
	{
		class ClipByMaximum : public IPluginV2IOExt
		{
		public:
			ClipByMaximum(float maximum);

			ClipByMaximum(const void* data, size_t length);			//读取层参数

			~ClipByMaximum() override = default;

			int getNbOutputs() const override;

			Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

			int initialize() override;

			void terminate() override;

			void destroy() override;

			size_t getWorkspaceSize(int) const override;

			int enqueue(
				int batch_size, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

			size_t getSerializationSize() const override;

			void serialize(void* buffer) const override;

			// bool supportsFormat(DataType type, PluginFormat format) const override;
			bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)const override;

			const char* getPluginType() const override;

			const char* getPluginVersion() const override;

			IPluginV2IOExt* clone() const override;

			void setPluginNamespace(const char* libNamespace) override;

			const char* getPluginNamespace() const override;

			DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

			bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

			bool canBroadcastInputAcrossBatch(int inputIndex) const override;

			void attachToContext(
				cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

			// void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
			//     const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
			//     const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;
			void configurePlugin(const PluginTensorDesc * in, int nbInput, const PluginTensorDesc * out, int nbOutput)override;

			void detachFromContext() override;

		private:
			float mMaximum;
			Dims mInputDims;
			Dims mOutputDims;
			std::string mNameSpace;
		};

		class ClipByMaximumPluginCreator : public BaseCreator
		{
		public:
			ClipByMaximumPluginCreator();

			~ClipByMaximumPluginCreator() {};

			const char* getPluginName() const override;

			const char* getPluginVersion() const override;

			const PluginFieldCollection* getFieldNames() override;

			IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

			IPluginV2IOExt* deserializePlugin(const char* name, const void* data, size_t length) override;

		private:
			static PluginFieldCollection mFC;
			float mMaximum;
			static std::vector<PluginField> mPluginAttributes;
		};
		REGISTER_TENSORRT_PLUGIN(ClipByMaximumPluginCreator);

	} // namespace plugin
} // namespace nvinfer1

