import numpy as np
import tensorflow as tf
import onnxruntime as ort

# 1. 准备相同的测试输入
# 根据你的代码，输入维度是 (None, 10, 10)
batch_size = 1
node_input_data = np.random.rand(batch_size, 10, 10).astype(np.float32)
adj_input_data = np.random.rand(batch_size, 10, 10).astype(np.float32)

# 2. 获取 Keras 模型输出
# 注意：加载时必须与转换脚本一致，关闭 compile 以避免层初始化差异
keras_model = tf.keras.models.load_model('../models/251229.h5', compile=False)
keras_output = keras_model.predict([node_input_data, adj_input_data])

# 3. 获取 ONNX 模型输出
ort_session = ort.InferenceSession("../models/model.onnx")
# 获取输入节点名称
input_name_node = ort_session.get_inputs()[0].name
input_name_adj = ort_session.get_inputs()[1].name

onnx_output = ort_session.run(None, {
    input_name_node: node_input_data,
    input_name_adj: adj_input_data
})[0]

# 4. 计算差异
diff = np.abs(keras_output - onnx_output)
max_diff = np.max(diff)
avg_diff = np.mean(diff)

print(f"最大绝对误差: {max_diff:.8f}")
print(f"平均绝对误差: {avg_diff:.8f}")

# 通常 float32 精度下，误差小于 1e-5 即可视为转换正确
if np.allclose(keras_output, onnx_output, atol=1e-5):
    print("✅ 验证通过：Keras 与 ONNX 输出一致！")
else:
    print("❌ 警告：输出差异较大，请检查转换过程。")