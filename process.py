from tracker import Tracker
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

    return self._last_j2ds, self.tracker.GetBoundingBox(), self._last_believes

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

if __name__ == "__main__":
  interpreter = Interpreter()

  for v_idx in range(0, 20):
    data_dir = "E:/DataSets/relation/image/{:02d}".format(v_idx + 1)
    data_list = GetImageLists(data_dir)

    for idx, cur_data in enumerate(data_list):
      img = cv2.imread(os.path.join(data_dir, cur_data[0]))
      lbl = json.load(open(os.path.join(data_dir, cur_data[1])))
      leg_relations = np.array([int(lbl["shapes"][0]["label"]),
                                int(lbl["shapes"][1]["label"]),
                                int(lbl["shapes"][2]["label"]),
                                int(lbl["shapes"][3]["label"])]) - 1
      print(leg_relations)
      j2ds, bbox, believes = interpreter.Detect(img, idx == 0)

      display_img = display_utils.drawPoints(img, j2ds, point_ratio=2)
      bbox_joints = np.array([[bbox[0], bbox[1]],
                              [bbox[0] + bbox[2], bbox[1]],
                              [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                              [bbox[0], bbox[1] + bbox[3]]])
      display_img = display_utils.drawLines(display_img, bbox_joints, np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))
      display_img = np.transpose(display_img, [1, 0, 2])
      cv2.imshow("display_img", display_img)
      cv2.waitKey(30)