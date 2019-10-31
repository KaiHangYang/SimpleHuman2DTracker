from tracker import *
import display_utils
import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import json

class Interpreter:
  def __init__(self):
    self.pre_coord = np.array([[[90.0, 128.0], [78.0, 176.0], [80.0, 215.0], [116.0, 128.0], [118.0, 178.0], [109.0, 223.0], [88.0, 20.0], [77.0, 61.0], [68.0, 84.0], [83.0, 113.0], [115.0, 56.0], [101.0, 61.0], [89.0, 57.0]]])
    self.interpreter = tf.lite.Interpreter("relhm-16-temporal.tflite")

    self.input_details = self.interpreter.get_input_details()
    self.output_details = self.interpreter.get_output_details()
    self.interpreter.allocate_tensors()
    
    self.tracker = Tracker(input_width=1080,
                      input_height=1920,
                      output_width=192,
                      output_height=256,
                      blf_threshold=0.0)

    self._last_j2ds = None
    self._last_believes = None

  def _Inference(self, img):
    img = np.copy(img[np.newaxis])
    self.interpreter.set_tensor(self.input_details[0]["index"], img.astype(self.input_details[0]["dtype"]))
    self.interpreter.set_tensor(self.input_details[1]["index"], self.pre_coord.astype(self.input_details[1]["dtype"]))
    self.interpreter.invoke()
    output_hms = self.interpreter.get_tensor(self.output_details[0]["index"])
    output_hms = np.squeeze(output_hms)
    joints_2d = np.array(np.unravel_index(np.argmax(output_hms.reshape([-1, 13]), axis=0), shape=[64, 48])).T * 4
    joints_2d = joints_2d[:, ::-1]
    believes = np.max(output_hms.reshape([-1, 13]), axis=0)

    return joints_2d, believes

  def Detect(self, img, first_frame=True):
    raw_img = img.copy()
    leasting_frames = 1

    if first_frame:
      self.tracker.Reset();
      self._last_j2ds = None
      self._last_believes = None
      leasting_frames = 10

    for _ in range(leasting_frames):
      cropped_img = self.tracker.Track(raw_img, self._last_j2ds, self._last_believes)

      output_j2ds, output_believes = self._Inference(cropped_img)
      global_believes = output_believes / 255.0
      global_j2ds = self.tracker.PutIntoGlobal(output_j2ds)

      self._last_j2ds, self._last_believes = global_j2ds, global_believes

    return self._last_j2ds, self.tracker.GetBoundingBox(is_tight=True), self._last_believes

def GetImageLists(img_dir):
  img_list, label_list = [], []
  for cur_file in os.listdir(img_dir):
    if cur_file.split(".")[-1] == "json":
      label_list.append(cur_file)
    else:
      img_list.append(cur_file)

  img_list.sort()
  label_list.sort()
  data_list = []
  
  for cur_img in img_list:
    cur_name = cur_img.split(".")[0]
    
    if cur_name + ".json" in label_list:
      data_list.append((cur_img, cur_name + ".json"))
    else:
      data_list.append((cur_img, data_list[-1][1]))
  return data_list

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
  interpreter = Interpreter()
  processed_data = dict(annotations=[], images=[])

  cur_img_id = 0
  cur_lbl_id = 0
  lbl_img_path = "train/fu_labeled"
  save_img_path = os.path.join("./dataset/images", lbl_img_path)
  save_lbl_path = os.path.join("./dataset/annotations", "train_fu_labeled.json")
  if not os.path.isdir(save_img_path):
    os.makedirs(save_img_path)

  if not os.path.isdir(os.path.dirname(save_lbl_path)):
    os.makedirs(os.path.dirname(save_lbl_path))

  for v_idx in range(0, 20):
    data_dir = "E:/DataSets/relation/image/{:02d}".format(v_idx + 1)
    data_list = GetImageLists(data_dir)

    for idx, cur_data in enumerate(data_list):
      sys.stderr.write("\rCurrently processing: {:05d}".format(cur_lbl_id))
      sys.stderr.flush()

      img = cv2.imread(os.path.join(data_dir, cur_data[0]))

      # Read labeled relations.
      lbl = json.load(open(os.path.join(data_dir, cur_data[1])))
      leg_relations = np.array([int(lbl["shapes"][0]["label"]),
                                int(lbl["shapes"][1]["label"]),
                                int(lbl["shapes"][2]["label"]),
                                int(lbl["shapes"][3]["label"])]).astype(np.int)
      leg_relations = np.array([-1, 1, 2, 0])[leg_relations]
      label_relations = np.zeros([13])
      label_relations[0:4] = leg_relations

      j2ds, bbox, believes = interpreter.Detect(img, idx == 0)
      color_table = (np.ones([13, 3]) * 128).astype(np.uint8)
      color_table = VisualRelation(label_relations, color_table)

      extra_pad = max(bbox[2], bbox[3]) * 0.3
      bbox_pad = max(bbox[2], bbox[3]) * 0.125

      affine_matrix = np.array([[1.0, 0.0, -(bbox[0] - extra_pad)], [0.0, 1.0, -(bbox[1] - extra_pad)]])
      label_img = cv2.warpAffine(img, affine_matrix, dsize=(int(bbox[2] + 2*extra_pad), int(bbox[3] + 2*extra_pad)))
      label_img_name = os.path.join(save_img_path, cur_data[0])

      label_j2ds = j2ds - np.array([bbox[0], bbox[1]]) + np.array([extra_pad, extra_pad])
      label_bbox = np.array([0, 0, bbox[2] + 2*extra_pad, bbox[3] + 2*extra_pad])
      label_bbox = np.array([bbox_pad, bbox_pad,
                             label_bbox[2] - 2*bbox_pad, label_bbox[3] - 2*bbox_pad])

      processed_data["annotations"].append({
        "id": cur_lbl_id,
        "image_id": cur_img_id,
        "keypoints": np.concatenate([label_j2ds, np.ones([13, 1])], axis=-1).tolist(),
        "bbox": label_bbox.tolist(),
        "rel": label_relations.tolist()
      })
      cur_lbl_id += 1

      processed_data["images"].append({
        "id": cur_img_id,
        "file_name": label_img_name,
        "src": "fu_labeled"
      })
      cur_img_id += 1
      cv2.imwrite(os.path.join(save_img_path, cur_data[0]), label_img)
      ###### Visualization ######
      # display_img = display_utils.drawPoints(label_img.copy(), label_j2ds.copy(), point_ratio=10, color_table=color_table)

      # # cv2.imwrite(os.path.join(save_image_path, cur_img_name), cur_img)
      # bbox_joints = np.array([[label_bbox[0], label_bbox[1]],
      #                         [label_bbox[0] + label_bbox[2], label_bbox[1]],
      #                         [label_bbox[0] + label_bbox[2], label_bbox[1] + label_bbox[3]],
      #                         [label_bbox[0], label_bbox[1] + label_bbox[3]]]).astype(np.int)
      # display_img = display_utils.drawLines(display_img, bbox_joints, np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))
      # cv2.imshow("display_img", display_img)
      # cv2.waitKey(30)

  with open(save_lbl_path, "w") as f:
    json.dump(processed_data, f)