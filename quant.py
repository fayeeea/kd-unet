import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


class DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = self._preprocess(
            calibration_image_folder, height, size_limit=0
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def _preprocess(self, images_folder: str, resize_size: int, size_limit=0):
        """
        Loads images from a folder and applies PyTorch-like transforms,
        returning NumPy arrays suitable for ONNX Runtime input.
        """
        image_names = os.listdir(images_folder)
        if size_limit > 0 and len(image_names) >= size_limit:
            batch_filenames = [image_names[i] for i in range(size_limit)]
        else:
            batch_filenames = image_names

        # PyTorch-style transforms
        preprocess = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),  # [0,255] -> [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        batch_data_list = []

        for image_name in batch_filenames:
            image_path = os.path.join(images_folder, image_name)
            img = Image.open(image_path).convert("RGB")
            tensor_img = preprocess(img)  # [C,H,W], float32
            np_img = tensor_img.unsqueeze(0).unsqueeze(0).numpy()  # [1,C,H,W] for ONNX
            batch_data_list.append(np_img)

        # Concatenate into a single batch: [B,C,H,W]
        batch_data = np.concatenate(batch_data_list, axis=0)
        return batch_data

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

def to_onnx(model, dummy, output_path='model.onnx'):
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=13,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    
    print(f'File {output_path} created')
    return
    
def quantize_onnx(weight_type, input_path='model.onnx', output_path='model_quant.onnx'):
    
    if weight_type.lower() == 'int8':
        weight_type = QuantType.QInt8,
    elif weight_type.lower() == 'float8':
        weight_type = QuantType.QFLOAT8E4M3FN
    else:
        print('Quantization Failed')
        return
    
    dr = DataReader(calibration_image_folder='data/images',model_path='model.onnx')
    quantize_static(
        model_input=input_path,
        model_output=output_path,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
    )
    print(f'File {output_path} created')
    return

    
