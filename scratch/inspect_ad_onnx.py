import onnxruntime as ort

sess = ort.InferenceSession("data_processed/ad_checker_model.onnx")
meta = sess.get_modelmeta().custom_metadata_map
print("AD checker ONNX metadata:")
for k, v in meta.items():
    print(k, ":", v)
