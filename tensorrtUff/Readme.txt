1.使用tensorflow训练模型并保存为.pb格式


2.下载TenorRT Linux版，并安装uff以及graphsurgeon文件夹下的.whl文件
	pip install uff-0.6.5-py2.py3-none-any.whl
	pip install graphsurgeon-0.4.1-py2.py3-none-any.whl


3.编写config.py将模型中TensorRT不支持的层转化为自定义层

	import graphsurgeon as gs
	import tensorflow as tf

	#自定义层upsample_2D_1，名字为"Upsample2D_1"，操作由自定义层"ResizeNearest_TRT2"实现
	#自定义层需要传入两个参数scale_x和scale_y，分别是长和宽的缩放比例
	upsample_2D_1 = gs.create_plugin_node(name="Upsample2D_1", op="ResizeNearest_TRT2", dtype=tf.float32, scale_x=8.0, scale_y=8.0)
	upsample_2D_2 = gs.create_plugin_node(name="Upsample2D_2", op="ResizeNearest_TRT2", dtype=tf.float32, scale_x=4.0, scale_y=4.0)
	upsample_2D_3 = gs.create_plugin_node(name="Upsample2D_3", op="ResizeNearest_TRT2", dtype=tf.float32, scale_x=10.0, scale_y=29.0)
	upsample_2D_4 = gs.create_plugin_node(name="Upsample2D_4", op="ResizeNearest_TRT2", dtype=tf.float32, scale_x=5.0, scale_y=14.5)
	upsample_2D_5 = gs.create_plugin_node(name="Upsample2D_5", op="ResizeNearest_TRT2", dtype=tf.float32, scale_x=2.5, scale_y=7.25)
	upsample_2D_6 = gs.create_plugin_node(name="Upsample2D_6", op="ResizeNearest_TRT2", dtype=tf.float32, scale_x=1.4285714626312256, scale_y=4.142857074737549)

	namespace_plugin_map = {
		#将模型中名字为"segment/up_sampling2d_1/ResizeNearestNeighbor"转化为自定义层upsample_2D_1
	        "segment/up_sampling2d_1/ResizeNearestNeighbor":upsample_2D_1,		
        	"segment/ff/up_sampling2d/ResizeNearestNeighbor":upsample_2D_2,
	        "segment/gfe/resize/ResizeNearestNeighbor":upsample_2D_3,
	        "segment/gfe/resize_1/ResizeNearestNeighbor":upsample_2D_4,
        	"segment/gfe/resize_2/ResizeNearestNeighbor":upsample_2D_5,
	        "segment/gfe/resize_3/ResizeNearestNeighbor":upsample_2D_6
	}

	def preprocess(dynamic_graph):
	    dynamic_graph.collapse_namespaces(namespace_plugin_map)	#在uff模型中使用自定义层替换原生层
	

4.使用convert-to-uff将pb模型转化为onnx
  	convert-to-uff F:\CODES\tf_PZT_2020\pbMode\tensorrt.pb -p config.py


5.配置VS项目属性（需要使用VS CUDA Runtime创建项目）
  	常规	目标平台：8.1	平台工具集：Visual Studio 2015 (v140)

	  修改C/C++ 附加包含目录至本机路径	
	  							C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include	(CUDA 头文件路径)
								F:\opencv\build\include		(Opencv头文件路径)
								F:\TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.0.cudnn7.6\TensorRT-7.0.0.11\include	(TensorRT头文件路径)

	  预处理器 预处理器定义  添加 _CRT_SECURE_NO_WARNINGS

	  修改链接器 输入 至本机路径 F:\TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.0.cudnn7.6\TensorRT-7.0.0.11\lib\*.lib	(TensorRT库文件)
								C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64\*.lib		(CUDA库文件)
								F:\opencv\build\x64\vc15\lib\opencv_world420d.lib	(Opencv库文件, Release时去掉d)
	
	  将 F:\opencv\build\x64\vc15\lib\opencv_world420d.dll拷贝至项目Debug目录下  
	     F:\opencv\build\x64\vc15\lib\opencv_world420.dll拷贝至项目Release目录下  


5.修改main.cpp中的
	uffFile						onnx模型地址
	uffEngineFile					trt引擎地址
	path						图片路径
	IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL	图片尺寸
	inputTensor, outputTensor			模型输入输出张量名

	以及trtInference.cpp中拷贝输出和验证结果部分


6.第一遍运行程序时会将uff模型转化为trt引擎并序列化储存，以后如需改变模型需手动删除trt模型
