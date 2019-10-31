from tracker import Tracker
import display_utils
import tensorflow as tf
import numpy as np
import cv2
import os
import sys

sys.path.append("E:/Projects/lite_pose")

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

  def Detect(self, img):
    raw_img = img.copy()
    self.tracker.Reset();
    last_j2ds = None
    last_believes = None
    for _ in range(10):
      cropped_img = self.tracker.Track(raw_img, last_j2ds, last_believes)

      output_j2ds, output_believes = self._Inference(cropped_img)
      global_believes = output_believes / 255.0
      global_j2ds = self.tracker.PutIntoGlobal(output_j2ds)

      last_j2ds, last_believes = global_j2ds, global_believes

    return last_j2ds, self.tracker.GetBoundingBox(), last_believes

if __name__ == "__main__":
  interpreter = Interpreter()
  img = cv2.imread("E:/DataSets/relation/image/07/070001.jpg")

  for v_idx in range(0, 20):
    img = cv2.imread("E:/DataSets/relation/image/{:02d}/{:02d}{:04d}.jpg".format(v_idx + 1, v_idx + 1, 1))
    j2ds, bbox, believes = interpreter.Detect(img)

    display_img = display_utils.drawPoints(img, j2ds, point_ratio=2)
    cv2.imshow("display_img", display_img)
    cv2.waitKey()
