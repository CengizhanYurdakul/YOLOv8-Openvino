# YOLOv8 Openvino Conversion+Quantization

## Installation
```
conda create --name yolorepo python==3.8.5
conda activate yolorepo
python -m pip install --upgrade pip
pip install -r requirements.txt -f https://download.pytorch.org/whl/cpu
```
## Inference (Converting & Quantization)
Add the YOLOv8 model you want to convert to the `src/Models` path. Then add the necessary parameters to the terminal command below and run it. 
```
python main.py --modelPath src/Models/best.pt --imagePath src/Assets/testImage.jpg --datasetPath src/datasets/val/images --imageSize 480
```
In the `Output.png` image you can see the results of Torch, Openvino and Quantized Openvino models respectively.

## Test (Quantized Model)
You can try the quantized model in the `test.py` file so that you can try it on a single image.