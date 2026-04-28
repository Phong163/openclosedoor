import torch

# Load model trực tiếp
model = torch.load(r"C:\Users\admin\Desktop\VTT_opendoor\weights\openclosedoor21_4.pt", map_location="cpu")

# Nếu model lưu dạng checkpoint
if isinstance(model, dict):
    model = model['model']  # YOLOv5 thường lưu kiểu này

model.eval()
model.float()  # 🔥 FIX QUAN TRỌNG

# Dummy input
dummy = torch.randn(1, 3, 384, 384)
# Export ONNX
torch.onnx.export(
    model,
    dummy,
    "openclosedoor21_4.onnx",
    opset_version=11,  # 🔥 cho onnxruntime 1.6.0
    input_names=['images'],
    output_names=['output'],
    dynamic_axes={
        'images': {0: 'batch'},
        'output': {0: 'batch'}
    }
)

print("Exported classify ONNX!")