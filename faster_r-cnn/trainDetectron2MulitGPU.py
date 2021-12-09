from torch._C import device
import detectron2
import numpy as np
import os
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog

from detectron2.data.datasets import register_coco_panoptic
from detectron2.data.datasets import register_coco_panoptic_separated, register_coco_instances
from detectron2.engine import DefaultTrainer, launch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor

from detectron2.evaluation.panoptic_evaluation import COCOPanopticEvaluator
from detectron2.evaluation import inference_on_dataset
from detectron2.modeling import build_model
from detectron2.evaluation import inference_on_dataset

from detectron2.engine import DefaultPredictor
from detectron2.data.detection_utils import read_image
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
import cv2
from detectron2.data import build_detection_test_loader
import sys

import torch
from detectron2.evaluation import COCOEvaluator

antall = torch.cuda.device_count()

print(antall)

for i in range(antall):
    print(torch.cuda.get_device_name(i))


register_coco_instances(name="vis_train", metadata={}, json_file="cocoData/train/annotations/instances_Train.json", image_root="cocoData/train/images")
register_coco_instances(name="vis_test", metadata={}, json_file="cocoData/test/annotations/instances_Test.json", image_root="cocoData/test/images")


MetadataCatalog.get("vis_train").set(thing_classes=['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'])
MetadataCatalog.get("vis_train").set(thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7:6, 8:7})

MetadataCatalog.get("vis_test").set(thing_classes=['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'])
#MetadataCatalog.get("vis_test").set(thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7:6, 8:7})
MetadataCatalog.get("vis_test").set(thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7:6, 8:7})

register_coco_instances(name="vis_val", metadata={}, json_file="cocoData/validate/annotations/instances_Validation.json", image_root="cocoData/validate/images")

MetadataCatalog.get("vis_val").set(thing_classes=['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'])
#MetadataCatalog.get("vis_test").set(thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7:6, 8:7})
MetadataCatalog.get("vis_val").set(thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7:6, 8:7})


#print(torch.cuda.memory_summary(device=None, abbreviated=False))
def train():
    cfg = get_cfg()
    
    cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    #cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.merge_from_list(['MODEL.DEVICE', 'cuda'])
    cfg.MODEL.WEIGHTS = "models/model_final_280758.pkl"
    #cfg.MODEL.WEIGHTS = "models/model_final_f6e8b1.pkl"

    cfg.DATASETS.TRAIN = ("vis_val", )
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.MAX_ITER = 1803*2
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8

    #print(cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    cfg.OUTPUT_DIR = "./outputAllLabels"
    print(cfg)

    cfg.freeze()

    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    #os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    #DatasetCatalog.get('coco_2017_train_panoptic')
    #print(cfg.MODEL.ROI_HEADS.NUM_CLASSES)

    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    return (trainer.train())
#metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
#print(m_train2017etadata)
['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider']


#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
#predictor = DefaultPredictor(cfg)


#res = evaluator.evaluate()
#print(res)




def validate():
    cfg2 = get_cfg()

    cfg2.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    #cfg.merge_from_list(['MODEL.DEVICE', 'cuda', 'MODEL.WEIGHTS', 'detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl'])

    cfg2.merge_from_list(['MODEL.DEVICE', 'cuda'])
    cfg2.MODEL.WEIGHTS = "output50_00028/model_final.pth"
    #cfg2.MODEL.WEIGHTS = "detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl"
    #cfg2.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0
    cfg2.DATASETS.TEST = ("vis_test", )

    cfg2.MODEL.ROI_HEADS.NUM_CLASSES = 8

    cfg2.freeze()

    metadata = MetadataCatalog.get(cfg2.DATASETS.TEST[0])
    print(metadata)


    video_visualizer = VideoVisualizer(metadata, ColorMode.IMAGE)
    print(cfg2)
    #model = build_model(cfg2)
    #print(model)
    predictor = DefaultPredictor(cfg2)

    #frame = read_image("datasettVariert/coco/images/3Kara.jpg", format="BGR")
    frame = read_image("cocoData/test/images/frame_000000.PNG", format="BGR")
    outputs = predictor(frame)
    instances = outputs["instances"].to('cpu')
    visulizer = Visualizer(frame, metadata)
    vis_frame = visulizer.draw_instance_predictions(instances).get_image()
    vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite("stille.png", vis_frame)

    evaluator_test = COCOEvaluator(dataset_name="vis_test")
    val_loader = build_detection_test_loader(cfg2, "vis_test")
    print(inference_on_dataset(predictor.model, val_loader, evaluator_test))

if __name__ == "__main__":
    print("for lunsj")
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    print("port", port)
    #launch(
    #    train,
    #    1,
    #    num_machines=1,
    #    machine_rank=0,
    #    dist_url="tcp://127.0.0.1:{}".format(port),
    #)
    print("etter lunsj")

    validate()

#evaluator_train = COCOPanopticEvaluator(dataset_name="ffi_train_separated")
#val_loader = build_detection_test_loader(cfg2, "ffi_train_separated")
#print(inference_on_dataset(predictor.model, val_loader, evaluator_train))




#evaluator_test = COCOPanopticEvaluator(dataset_name="ffi_test_separated")
#val_loader = build_detection_test_loader(cfg2, "ffi_test_separated")
#print(inference_on_dataset(predictor.model, val_loader, evaluator_test))


#res = trainer.test(cfg=cfg2, model=predictor.model, evaluators=[evaluator])

#inference_on_dataset()

#print(res)