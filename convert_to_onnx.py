# convert_to_onnx.py
import tensorflow as tf
import tf2onnx
import onnx

# 加载原始模型
model = tf.keras.models.load_model('models/251229.h5', compile=False)

# 定义输入规范：两个输入均为 (None, 10, 10)
spec = (
    tf.TensorSpec((None, 10, 10), tf.float32, name="node_input"),
    tf.TensorSpec((None, 10, 10), tf.float32, name="adj_input")
)

# 转换模型
output_path = "models/model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
onnx.save(model_proto, output_path)

print(f"✅ 模型已成功转换为: {output_path}")