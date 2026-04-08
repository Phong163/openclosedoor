import onnx
from onnx import version_converter

model = onnx.load(r"C:\Users\admin\Desktop\hanghoavtp\weights\yolo26_goods.onnx")

converted_model = version_converter.convert_version(model, 11)

onnx.save(converted_model, "yolo26_goods_opset11.onnx")