import os
import cv2
import nncf
import numpy as np
from time import time
from ultralytics import YOLO

from openvino.runtime import Core
from openvino.runtime import serialize
from openvino.runtime import Type, Layout
from openvino.preprocess import PrePostProcessor

from src.utils import *

class Converter:
    def __init__(self, args):
        self.args = args
        self.initVaribles()
        
    def initVaribles(self):
        self.torchModel = None
        self.openvinoModel = None
        
        self.openvinoModelPath = os.path.join(self.args.modelPath.replace(self.args.modelPath.split("/")[-1], "%s_openvino_model" % self.args.modelPath.split("/")[-1].replace(".pt", "")), self.args.modelPath.split("/")[-1].replace("pt", "xml"))
        
        self.torchTimeList = []
        self.vinoTimeList = []
        self.vinoQTimeList = []
        
        self.torchMeanTime = None
        self.vinoMeanTime = None
        self.vinoQMeanTime = None

        self.testImage = cv2.imread(self.args.imagePath)
        self.torchOutputImage = self.testImage.copy()
        self.vinoOutputImage = self.testImage.copy()
        self.vinoQOutputImage = self.testImage.copy()
        
    def initTorchModel(self):
        self.torchModel = YOLO(self.args.modelPath)
    
    def inferenceTorch(self):
        # Initialize
        self.initTorchModel()
        
        # Speed test
        for i in range(20):
            if i < 10:
                self.torchModel(self.testImage)
            else:
                startTime = time()
                self.torchModel(self.testImage)
                self.torchTimeList.append(time() - startTime)
        
        # Visualize
        results = self.torchModel(self.testImage)
        for xyxy in results[0].boxes.xyxy:
            cv2.rectangle(self.torchOutputImage, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 3)
        
        self.torchMeanTime = np.mean(self.torchTimeList)
        
    def torch2openvino(self):
        self.torchModel.export(format="openvino", opset=11)
        del self.torchModel
        
    def initOpenvinoModel(self):
        core = Core()
        self.model = core.read_model(self.openvinoModelPath)
        self.openvinoModel = core.compile_model(self.model, "CPU")
    
    def inferenceOpenvino(self):
        # Initialize
        self.initOpenvinoModel()
        
        preprocessedImage = letterbox(self.testImage, new_shape=(self.args.imageSize, self.args.imageSize))[0]
        preprocessedImage = preprocessImage(preprocessedImage)
        inputTensor = image2tensor(preprocessedImage)
        
        # Speed test
        for i in range(20):
            if i < 10:
                self.openvinoModel(inputTensor)
            else:
                startTime = time()
                self.openvinoModel(inputTensor)
                self.vinoTimeList.append(time() - startTime)
        
        # Visualize
        result = self.openvinoModel(inputTensor)
        boxes = result[self.openvinoModel.output(0)]
        input_hw = inputTensor.shape[2:]
        detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=self.vinoOutputImage)
        for i in detections[0]["det"]:
            cv2.rectangle(self.vinoOutputImage, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255, 0, 0), 3)
              
        self.vinoMeanTime = np.mean(self.vinoTimeList)
        
    def quantizeOpenvino(self):
        dataset = CustomDataset(self.args.datasetPath, self.args.imageSize)
        def transformFunction(data_item):
            return data_item["inputImage"]
        
        quantizationDataset = nncf.Dataset(dataset, transformFunction)
    
        ignoredScope = nncf.IgnoredScope(
        types=["Multiply", "Subtract", "Sigmoid"],
        names=[
                "/model.22/dfl/conv/Conv",
                "/model.22/Add",
                "/model.22/Add_1",
                "/model.22/Add_2",
                "/model.22/Add_3",
                "/model.22/Add_4",   
                "/model.22/Add_5",
                "/model.22/Add_6",
                "/model.22/Add_7",
                "/model.22/Add_8",
                "/model.22/Add_9",
                "/model.22/Add_10"
            ]
        )

        quantizedModel = nncf.quantize(
            self.model, 
            quantizationDataset,
            preset=nncf.QuantizationPreset.MIXED,
            ignored_scope=ignoredScope
        )
        
        del self.model
        del self.openvinoModel
        del dataset
        del quantizationDataset
        
        core = Core()
        quantizedCompiledModel = core.compile_model(quantizedModel, "CPU")
        
        frame = cv2.cvtColor(cv2.imread(self.args.imagePath), cv2.COLOR_BGR2RGB)
        
        preprocessedImage = letterbox(self.testImage, new_shape=(self.args.imageSize, self.args.imageSize))[0]
        preprocessedImage = preprocessImage(preprocessedImage)
        inputTensor = image2tensor(preprocessedImage)
        
        self.vinoQTimeList = []
        for _ in range(200):
            s = time()
            result = quantizedCompiledModel(inputTensor)
            e = time()
            self.vinoQTimeList.append(e-s)
            
        self.vinoQMeanTime = np.mean(self.vinoQTimeList)
            
        boxes = result[quantizedCompiledModel.output(0)]
        input_hw = inputTensor.shape[2:]
        detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=self.vinoQOutputImage)
        
        for i in detections[0]["det"]:
            cv2.rectangle(self.vinoQOutputImage, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255, 0, 0), 3)
        
        ppp = PrePostProcessor(quantizedModel)
        ppp.input(0).tensor().set_shape([1, self.args.imageSize, self.args.imageSize, 3]).set_element_type(Type.u8).set_layout(Layout('NHWC'))
        ppp.input(0).preprocess().convert_element_type(Type.f32).convert_layout(Layout('NCHW')).scale([255., 255., 255.])
        quantizedModelWithPreprocess = ppp.build()
        serialize(quantizedModelWithPreprocess, "%s/best_openvino_model/quantized_best.xml" % self.args.modelPath.replace("/" + self.args.modelPath.split("/")[-1], ""))
    
    def visualizeOutputs(self):
        concat = np.concatenate([self.torchOutputImage, self.vinoOutputImage, self.vinoQOutputImage], axis=1)
        cv2.imwrite("Output.png", concat)
    
    def printSpeedTest(self):
        print("Vino Mean FPS: \t", 1/self.vinoMeanTime)
        print("Quantized Vino Mean FPS: \t", 1/self.vinoQMeanTime)
        print("DONE!")
    
    def convert(self):
        self.inferenceTorch()
        self.torch2openvino()
        self.inferenceOpenvino()
        self.quantizeOpenvino()
        self.visualizeOutputs()
        self.printSpeedTest()