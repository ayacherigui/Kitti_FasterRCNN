'''
USAGE
This code performs predictions on test images and outputs a folder containing prediction annotation to use for evaluation softwares like the companion evaluation tool

the format is <object-name> <score> <left> <top> <right> <bottom>

Generate predictions by:
python inference.py -d <path to the images folder> -dest <path to the results folder>
'''

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import torch
import torchvision
from torchvision import transforms

import argparse
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d', '--images-dir', default=None, type=str, dest='images-path',
        help='path to the images folder to perform prediction upon'
    )

    parser.add_argument(
        '-dest', '--dest-dir',
        default=None, type=str, dest='result-dir',
        help='path to the destination folder'
    )

    args = vars(parser.parse_args())
    return args


def main(args):

    images_path = args['images-path']
    path = args['result-dir']

    NUM_CLASSES = 15
    KITTI_INSTANCE_CATEGORY_NAMES = {0: u'Cyclist', 1: u'DontCare', 2: u'Misc', 3: u'Person_sitting', 4: u'Tram', 5: u'Truck', 6: 'Van', 7: u'car', 8: u'person', 9: u'Tram',
                                     10: u'People', 11: u'Bus', 12: u'Vehicle-with-trailer', 13: u'Special-vehicle', 15: u'Pickup'}

    DEVICE = torch.device('cpu')

    # model=torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    checkpoint = torch.load(
        '/home/aya/Desktop/Kitti_FasterRCNN/outputs/training/res/best_model.pth', map_location=DEVICE)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # Get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    print("pretrained model imported....")
    os.makedirs(path, exist_ok=True)  # make the destination dir

    print("prediction...")
    for i in os.listdir(images_path):

        img = Image.open(images_path+i)
        # convert the PIL image to a tensor
        ConvertTensor = transforms.ToTensor()
        img = ConvertTensor(img)

        with torch.no_grad():  # not to use the gradient function of pytorch (grad will not be calculated)
            pred = model([img],)

        # seperate the 3 keys (boxes, labels, scores)
        bboxes, labels, scores = pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]
        OBJECT = ""

        for j in range(len(labels)):
            OBJECT += KITTI_INSTANCE_CATEGORY_NAMES[int(labels[j])]
            OBJECT += " "
            OBJECT += str(float(scores[j]))
            OBJECT += " "
            OBJECT += str(int(bboxes[j][0]))
            OBJECT += " "
            OBJECT += str(int(bboxes[j][1]))
            OBJECT += " "
            OBJECT += str(int(bboxes[j][2]))
            OBJECT += " "
            OBJECT += str(int(bboxes[j][3]))
            OBJECT += "\n"

        fileName = i.split(".")
        fileName = fileName[0]+".txt"
        with open(path+fileName, 'w') as f:
            f.write(OBJECT)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
