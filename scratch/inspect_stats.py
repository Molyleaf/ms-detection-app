import onnx

model = onnx.load("models/model.onnx")
graph = model.graph
print("Last 5 nodes in ONNX graph:")
for node in graph.node[-5:]:
    print(node.op_type, node.input, node.output)
