from time import time
import matplotlib.pyplot as plt
from openvino.runtime import Core

from src.utils import *

#! ------------ Modify only these informations -----------------
modelPath = "src/Models/best_openvino_model/quantized_best.xml"
imagePath = "src/Assets/testImage.jpg"
imageSize = 480
#! -------------------------------------------------------------

# Initialize model
core = Core()
model = core.read_model(modelPath)
device = "CPU"
compiledModel = core.compile_model(model, device)

# Read and process input image
frame = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
outputLayer = compiledModel.output(0)
frame = letterbox(frame, new_shape=(imageSize, imageSize))[0]
inputTensor = np.expand_dims(frame, 0)
input_hw = frame.shape[:2]

# Inference quantized model
timeList = []
for _ in range(200):
    s = time()
    result = compiledModel(inputTensor)[outputLayer]
    e = time()
    timeList.append(e-s)
    
print("Test FPS: ", 1/np.mean(timeList))

# Process outputs of model
detections = postprocess(result, input_hw, frame)

for i in detections[0]["det"]:
    cv2.rectangle(frame, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255, 0, 0), 3)

# Show image
plt.imshow(frame)
plt.show()