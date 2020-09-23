import tensorflow as tf
import uff

# 需要改成你自己的output_names
output_names = ['output']
frozen_graph_filename = 'yolov3_coco'

# 将frozen graph转换为uff格式
uff_model = uff.from_tensorflow_frozen_model(frozen_graph_filename, output_names)
