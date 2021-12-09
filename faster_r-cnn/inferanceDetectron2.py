import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode#, Visualizer
import torch
from os import listdir

from torchvision.transforms import Compose

from detectron2.modeling import build_model

import json
from detectron2.data.datasets import register_coco_panoptic_separated, register_coco_instances

from vis import Visualizer
import time
device = torch.device("cuda")

register_coco_instances(name="vis_test", metadata={}, json_file="cocoData/test/annotations/instances_Test.json", image_root="cocoData/test/images")

MetadataCatalog.get("vis_test").set(thing_classes=['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'])
MetadataCatalog.get("vis_test").set(thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7:6, 8:7})

data = DatasetCatalog.get('vis_test')

print(MetadataCatalog.get("vis_test"))
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.merge_from_list(['MODEL.DEVICE', 'cuda'])
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8 
cfg.MODEL.WEIGHTS = "output50/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("vis_test", )
cfg.freeze()

metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
print(metadata)
print(len(metadata.thing_classes))
print(metadata.thing_classes)
print(metadata.thing_dataset_id_to_contiguous_id)


WINDOW_NAME = "WebCamTest"

video_visualizer = VideoVisualizer(metadata, ColorMode.IMAGE)
predictor = DefaultPredictor(cfg)

typeOfFrame = "Image"
index = 0

startPath = "cocoData/test/images"

files = listdir(startPath)
files.sort()

print(files)

typeOfAnalytics = "both"

saving = True


if (typeOfFrame == "Video"):
    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
elif (typeOfFrame == "VideoOptak"):
    cam = cv2.VideoCapture("/home/potetsos/skule/2021Host/prosjekt/dataFFI/ffiBilder/filmer/rockQuarryIntoWoodsDrive.mp4")
    num_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    codec, file_ext = ("mp4v", ".mp4")
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fname = "/home/potetsos/skule/2021Host/prosjekt/dataFFI/ffiBilder/rockQuarryIntoWoodsDrive_Analyzed" + file_ext
    frames_per_second = cam.get(cv2.CAP_PROP_FPS)
    print("FPS", frames_per_second)
    output_file = cv2.VideoWriter(
                    filename=output_fname,
                    # some installation of opencv may not support x264 (due to its license),
                    # you can try other format (e.g. MPEG)
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(frames_per_second),
                    #frameSize=(3840, 1080),
                    frameSize=(4096, 1536),
                    #frameSize=(2048, 540),
                    isColor=True,
            )
    print("\n")
    print(output_file)
    print("\n")

def getFrame(index):
    if (typeOfFrame == "Video" or typeOfFrame == "VideoOptak"):
        success, frame = cam.read()
    if (typeOfFrame == "Image"):
        print(str(index) + "/" +str(len(files)-1))
        frame = read_image(startPath + "/" +files[index], format="BGR")
        index+=1



    return(frame)

def visualise_predicted_frame(frame, predictions):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    instances = predictions["instances"].to('cpu')
    confident_detections = instances[instances.scores > 0.3] 
    vis_frame = video_visualizer.draw_instance_predictions(frame, confident_detections)
    vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
    return(vis_frame)

totalTime = 0
avgTime = 0

counter = 0
for index in range(len(files)):
    counter += 1

    if (typeOfFrame == "Image" and index==len(files)):
        print("sett alle bilder")
        break

    if (typeOfAnalytics == "both"):
        
        frame = getFrame(index)
        timeStart = time.time()
        predicted = predictor(frame)
        print(time.time() - timeStart)
        totalTime += time.time() - timeStart
        print("avgTimge", totalTime/counter)

        if (typeOfFrame != "VideoOptak"):
            instances = predicted["instances"].to('cpu')
            visulizer = Visualizer(frame, metadata)
            vis_frame = visulizer.draw_instance_predictions(instances).get_image()

            visulizer = Visualizer(frame, metadata)
            out = visulizer.draw_dataset_dict(data[counter-1]).get_image()
            combinedFrame = np.vstack((vis_frame, out))

            if (saving == True):
                cv2.imwrite("outputData/testSett5/" + files[index], combinedFrame)
                
            


