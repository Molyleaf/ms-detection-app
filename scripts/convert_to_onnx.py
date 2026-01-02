import tensorflow as tf
import tf2onnx
import onnx
import os

# 1. 统一使用 tensorflow.keras 路径，保持与训练代码一致
# 在 Keras 2.13.1 中，无需手动补丁 MultiHeadAttention
# from tensorflow.keras.layers import MultiHeadAttention

model_path = '../models/251229.h5'
output_path = "../models/model.onnx"

# 如果模型中包含其他自定义层（本例中主要是标准层），
# 可以在 custom_objects 中指定，但 MultiHeadAttention 建议使用原生的。
custom_objects = {}

try:
    print(f"正在加载模型: {model_path} ...")
    # 尝试标准加载。compile=False 避开优化器兼容性问题
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("提示：如果依然报错，请检查 H5 文件是否由更高版本的 Keras (如 Keras 3) 生成。")
    exit(1)

# 2. 定义输入规范 (必须与训练代码一致: max_nodes=10, node_dim=10)
# 根据训练代码，node_input 为 (None, 10, 10)，adj_input 为 (None, 10, 10)
spec = (
    tf.TensorSpec((None, 10, 10), tf.float32, name="node_input"),
    tf.TensorSpec((None, 10, 10), tf.float32, name="adj_input")
)

# 3. 执行转换
print("正在转换为 ONNX...")
# 使用 opset 13 是一个稳定的选择，支持复杂的注意力算子
try:
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13
    )

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save(model_proto, output_path)
    print(f"✨ 转换完成! ONNX 模型已保存至: {output_path}")
except Exception as e:
    print(f"❌ ONNX 转换过程中出错: {e}")