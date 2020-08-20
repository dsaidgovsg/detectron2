import os
import cv2

from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.structures.boxes import BoxMode

from detectron2.utils.visualizer import Visualizer

def set_gpu(gpu_id):
    # According to Detectron2, set gpu by using CUDA_VISIBLE_DEVICES environ var
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

class Faster_RCNN(object):
    def __init__(self, config_path=None, weight_path=None, labels_path=None, gpu_id=0):
        if not weight_path or not os.path.exists(weight_path):
            raise FileNotFoundError
        if not labels_path or not os.path.exists(labels_path):
            raise FileNotFoundError

        set_gpu(gpu_id)

        # Set up cfg
        cfg = get_cfg()
        if not os.path.exists(config_path):
            cfg.merge_from_file(model_zoo.get_config_file(config_path))
        else:
            # Note that the Base-RCNN-FPN.yaml is expected to be in the parent dir of config_path
            cfg.merge_from_file(config_path)
        
        cfg.MODEL.WEIGHTS = weight_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.45

        self.predictor = DefaultPredictor(cfg)
        self.cfg = cfg

        with open(labels_path) as namesFH:
            names_list = namesFH.read().strip().split("\n")
            self.labels = [x.strip() for x in names_list]
        self.num_classes = len(self.labels)

    def get_class_name(self, i):
        return self.labels[i]

    def detect(self, frame, thresh=0.5, nms=0.45):
        # Detectron2 does not seem to have hier_thresh


        # DefaultPredictor takes in BGR image
        outputs = self.predictor(frame)
        
        # Output is XYXY_ABS - Mobius expects XYWH_ABS where XY is the bbox center (but BoxMode XYWH_ABS is top left corner)
        # Take note that Detectron2 model only outputs score for the best class
        pred_boxes = BoxMode.convert(outputs["instances"].get("pred_boxes").tensor, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        pred_scores = outputs["instances"].get("scores")
        pred_classes = outputs["instances"].get("pred_classes")
        
        # Mobius expects a list of [pred_class, pred_score, [Xc, Yc, W, H]]
        outputs = []
        for i in range(len(pred_boxes)):
                pred_box = pred_boxes[i]
                # Convert to XY of bbox center
                pred_box[0] += pred_box[2]/2.0
                pred_box[1] += pred_box[3]/2.0
                pred_score = pred_scores[i]
                pred_class = pred_classes[i]
                outputs.append([pred_class.tolist(),
                                pred_score.tolist(),
                                pred_box.tolist()])

        return outputs

    def get_num_classes(self):
        return self.num_classes
