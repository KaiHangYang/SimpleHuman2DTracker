from tracker import *
import display_utils
import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import json

def VisualRelation(leg_relations, color_table):
  if leg_relations[0] == 1:
    color_table[J13_KNEE_R] = [255, 255, 255]
  elif leg_relations[0] == 2:
    color_table[J13_KNEE_R] = [0, 0, 0]
  
  if leg_relations[1] == 1:
    color_table[J13_ANKLE_R] = [255, 255, 255]
  elif leg_relations[1] == 2:
    color_table[J13_ANKLE_R] = [0, 0, 0]
  
  if leg_relations[2] == 1:
    color_table[J13_KNEE_L] = [255, 255, 255]
  elif leg_relations[2] == 2:
    color_table[J13_KNEE_L] = [0, 0, 0]
  
  if leg_relations[3] == 1:
    color_table[J13_ANKLE_L] = [255, 255, 255]
  elif leg_relations[3] == 2:
    color_table[J13_ANKLE_L] = [0, 0, 0]

  return color_table

if __name__ == "__main__":
  label_path = "./dataset/annotations/train_fu_labeled.json"
  img_dir = "./dataset/images"

  datas = json.load(open(label_path))
  img_dict = {}
  for cur_img in datas["images"]:
    img_dict[cur_img["id"]] = cur_img["file_name"]

  for cur_data in datas["annotations"]:

    img = cv2.imread(os.path.join(img_dir, img_dict[cur_data["image_id"]]))
    j2ds = np.array(cur_data["keypoints"])
    rels = np.array(cur_data["rel"])
    bbox = np.array(cur_data["bbox"])
    ###### Visualization ######
    color_table = (np.ones([13, 3]) * 128).astype(np.uint8)
    VisualRelation(rels, color_table)

    display_img = display_utils.drawPoints(img.copy(), j2ds.copy(), point_ratio=10, color_table=color_table)

    bbox_joints = np.array([[bbox[0], bbox[1]],
                            [bbox[0] + bbox[2], bbox[1]],
                            [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                            [bbox[0], bbox[1] + bbox[3]]]).astype(np.int)
    display_img = display_utils.drawLines(display_img, bbox_joints, np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))
    cv2.imshow("display_img", display_img)
    cv2.waitKey(3)