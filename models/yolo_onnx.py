import onnxruntime as ort
import cv2
import numpy as np

class YoloONNX:
    def __init__(self, model_path, providers=["CUDAExecutionProvider"]):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, frame, img_size=480):
        img = cv2.resize(frame, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)

        outputs = self.session.run(
            None,
            {self.input_name: img}
        )[0]

        return outputs