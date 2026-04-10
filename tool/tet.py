import onnx
from onnx import version_converter
import os

print("Loading model...")
model = onnx.load(r"C:\Users\admin\Desktop\VTT2\weights\best2.onnx")

print("Converting...")
converted_model = version_converter.convert_version(model, 17)
print("Saving...")
onnx.save(converted_model, "model_opset12.onnx")

print("Done!")

print("Saved at:", os.path.abspath("model_opset12.onnx"))