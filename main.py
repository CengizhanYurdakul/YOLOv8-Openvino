import argparse
from src.Converter import Converter

parser = argparse.ArgumentParser()
parser.add_argument("--modelPath", default="src/Models/best.pt", help="Model path to convert from torch2onnx2openvino")
parser.add_argument("--imagePath", default="src/Assets/testImage.jpg", help="Test image to check model")
parser.add_argument("--datasetPath", default="src/datasets/val/images", help="Dataset path to calibrate model for quantization")
parser.add_argument("--imageSize", default=480, help="Input shape of model")
args = parser.parse_args()

converter = Converter(args)
converter.convert()