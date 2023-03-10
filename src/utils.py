import os
import cv2
import torch
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import torch.utils.data as data
from ultralytics.yolo.utils import ops
import torchvision.transforms as transforms


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scale_fill=False, scaleup=False, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  
    if auto:  
        dw, dh = np.mod(dw, stride), np.mod(dh, stride) 
    elif scale_fill: 
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  

    dw /= 2  
    dh /= 2

    if shape[::-1] != new_unpad:  
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)
 
def preprocessImage(img):
    
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img

def image2tensor(image):
    input_tensor = image.astype(np.float32)
    input_tensor /= 255.0  
    
    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor

def postprocess(pred_boxes, input_hw, orig_img, min_conf_threshold=0.25, nms_iou_threshold=0.7, agnosting_nms=False, max_detections=300, pred_masks=None, retina_mask=False):
    nms_kwargs = {"agnostic": agnosting_nms, "max_det":max_detections}
    if pred_masks is not None:
        nms_kwargs["nm"] = 32
    preds = ops.non_max_suppression(
        torch.from_numpy(pred_boxes),
        min_conf_threshold,
        nms_iou_threshold,
        **nms_kwargs
    )
    results = []
    proto = torch.from_numpy(pred_masks) if pred_masks is not None else None

    for i, pred in enumerate(preds):
        shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
        if not len(pred):
            results.append({"det": [], "segment": []})
            continue
        if proto is None:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            results.append({"det": pred})
            continue
        if retina_mask:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], shape[:2])  # HWC
            segments = [ops.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
        else:
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], input_hw, upsample=True)  # HWC
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            segments = [ops.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
        results.append({"det": pred[:, :6].numpy(), "segment": segments})
    return results

class CustomDataset(data.Dataset):
    def __init__(self, imagePath:str, imageSize:int):

        self.imagePath = imagePath
        self.imageSize = imageSize
                
        self.imageNames = None
        self.transform = None

        self.initInformations()
        self.initTransform()
            
    def __getitem__(self, index):        
        inputImage = None
        
        while inputImage is None:
            # Read input image and label
            imageName = self.imageNames[index]
            inputImage = Image.open(imageName)
            inputImage = np.array(inputImage)     
               
            preprocessedImage = letterbox(inputImage, new_shape=(self.imageSize, self.imageSize))[0]

            inputImage = Image.fromarray(preprocessedImage)    
                    
        transformedImage = self.transform(inputImage)
        
        batch = {
            "inputImage": transformedImage
        }
        
        return batch
            
    def __len__(self):
        return len(self.imageNames)
    
    def initTransform(self):
        transformList = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        self.transform = transforms.Compose(transformList)
    
    def initInformations(self):
        self.imageNames = glob(os.path.join(self.imagePath, "*.jpg"))