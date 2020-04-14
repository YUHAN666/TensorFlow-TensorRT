1.使用tensorflow训练数据格式为NCHW的模型并保存为.pb格式


2.安装tf2onnx:  pip install -U tf2onnx		
		pip install onnxruntime


3.使用tf2onnx将模型转化为onnx格式
python -m tf2onnx.convert 
    [--input SOURCE_GRAPHDEF_PB]
    [--graphdef SOURCE_GRAPHDEF_PB]
    [--checkpoint SOURCE_CHECKPOINT]
    [--saved-model SOURCE_SAVED_MODEL]
    [--output TARGET_ONNX_MODEL]
    [--inputs GRAPH_INPUTS]
    [--outputs GRAPH_OUTPUS]
    [--inputs-as-nchw inputs_provided_as_nchw]
    [--opset OPSET]
    [--target TARGET]
    [--custom-ops list-of-custom-ops]
    [--fold_const]
    [--continue_on_error]
    [--verbose]

修改--input(pb模型路径)	--output(输出模型路径)	--inputs(输入tensor名)	--outputs(输出tensor名)  并添加--opset 9 --fold_const 
如： python -m tf2onnx.convert --input F:\CODES\tf_PZT_2020\pbMode\tensorrt_fastscnn.pb --output F:\CODES\tf_PZT_2020\pbMode\tensorrt_fastscnn.onnx --inputs Image:0 --outputs decision_out:0 --opset 9 --fold_const 


4.配置VS项目属性
  常规	目标平台：8.1	平台工具集：Visual Studio 2015 (v140)

  修改C/C++ 附加包含目录至本机路径	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include	(CUDA 头文件路径)
					F:\opencv\build\include		(Opencv头文件路径)
					F:\TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.0.cudnn7.6\TensorRT-7.0.0.11\include	(TensorRT头文件路径)

  预处理器 预处理器定义  添加 _CRT_SECURE_NO_WARNINGS

  修改链接器 输入 至本机路径		F:\TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.0.cudnn7.6\TensorRT-7.0.0.11\lib\*.lib	(TensorRT库文件)
					C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64\*.lib		(CUDA库文件)
					F:\opencv\build\x64\vc15\lib\opencv_world420d.lib	(Opencv库文件, Release时去掉d)

  将F:\opencv\build\x64\vc15\lib\opencv_world420d.dll拷贝至项目Debug目录下  
    F:\opencv\build\x64\vc15\lib\opencv_world420.dll拷贝至项目Release目录下  


5.修改main.cpp中的
onnxFile					onnx模型地址
onnxEngineFile					trt引擎地址
path						图片路径
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL	图片尺寸

以及trtInference.cpp中拷贝输出和验证结果部分



6..第一遍运行程序时会将onnx模型转化为trt引擎并序列化储存，以后如需改变模型需手动删除trt模型

