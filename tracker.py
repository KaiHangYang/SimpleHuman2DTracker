import numpy as np
import sys
import os
import cv2
import copy

J13_HIP_R = 0
J13_KNEE_R = 1
J13_ANKLE_R = 2
J13_HIP_L = 3
J13_KNEE_L = 4
J13_ANKLE_L = 5
J13_HEAD_M = 6
J13_SHOULDER_R = 7
J13_ELBOW_R = 8
J13_WRIST_R = 9
J13_SHOULDER_L = 10
J13_ELBOW_L = 11
J13_WRIST_L = 12

TRACKER_EMPTY = 0
TRACKER_DETECTING = 1
TRACKER_TRACKING = 2

class Tracker(object):
    def __init__(self,
                 input_width, input_height,
                 output_width, output_height,
                 blf_threshold=0.1):
        self.blf_threshold = blf_threshold
        self.input_width = input_width
        self.input_height = input_height
        self.output_width = output_width
        self.output_height = output_height

        self.global_scale = 1.0
        self.global_offset = np.array([0, 0])

        initial_width = self.input_width * 0.3
        initial_height = initial_width / self.output_width * self.output_height
        initial_cen_x = self.input_width * 0.5
        initial_cen_y = self.input_height * 0.4
        self.default_bounding_box = {
            "x": initial_cen_x - initial_width * 0.5,
            "y": initial_cen_y - initial_height * 0.5,
            "w": initial_width,
            "h": initial_height
        }
        self.bounding_box = copy.deepcopy(self.default_bounding_box)

        self.detecting_frame_cnt = 0
        self.tracking_faliure_frame_cnt = 0

        self.state = TRACKER_EMPTY

    def Reset(self):
        self.SetState(TRACKER_EMPTY)
        self.ResetBoundingBox()

    def SetState(self, state):
        self.state = state
        self.detecting_frame_cnt = 0
        self.tracking_faliure_frame_cnt = 0

    def CheckValid(self, valid_flags):
        return (valid_flags[J13_HIP_R] and valid_flags[J13_HIP_L] and
                valid_flags[J13_HEAD_M] and valid_flags[J13_SHOULDER_R] and
                valid_flags[J13_SHOULDER_L])

    def GetValidFlags(self, believes):
        valid_flags = np.zeros_like(believes).astype(np.bool)
        for idx, blf in enumerate(believes):
            valid_flags[idx] = blf > self.blf_threshold
        return valid_flags

    def PutIntoGlobal(self, joints):
        joints = np.copy(joints)
        joints = joints * self.global_scale + self.global_offset
        return joints

    def UpdateBoundingBox(self, joints, valid_flags):
        min_x, max_x = self.input_width - 1, 0
        min_y, max_y = self.input_height - 1, 0

        for idx, valid in enumerate(valid_flags):
            if valid:
                min_x = min(min_x, joints[idx, 0])
                max_x = max(max_x, joints[idx, 0])
                min_y = min(min_y, joints[idx, 1])
                max_y = max(max_y, joints[idx, 1])

        if min_x >= max_x or min_y >= max_y:
            self.ResetBoundingBox()
            return

        if (max_x - min_x + 1) / self.input_width < 0.05 or (max_y - min_y + 1) / self.input_height < 0.05:
            # bounding box is to small
            cen_x = (min_x + max_x) * 0.5
            cen_y = (min_y + max_y) * 0.5
            initial_half_size = (
                min(self.input_width, self.input_height) * 0.8) * 0.5
            min_x = cen_x - initial_half_size
            max_x = cen_x + initial_half_size

            min_y = cen_y - initial_half_size
            max_y = cen_y + initial_half_size

        self.bounding_box["x"] = min_x
        self.bounding_box["y"] = min_y
        self.bounding_box["w"] = max_x - min_x + 1
        self.bounding_box["h"] = max_y - min_y + 1

    def ResetBoundingBox(self):
        self.global_scale = 1.0
        self.global_offset = np.array([0, 0])
        self.bounding_box = copy.deepcopy(self.default_bounding_box)

    def GetBoundingBox(self, is_tight=False):
        if is_tight:
            extra_scale = 1.0
        else:
            extra_scale = 1.32

        if self.bounding_box["w"] / self.bounding_box["h"] > self.output_width / self.output_height:
            box_w = self.bounding_box["w"] * extra_scale
            box_h = box_w / self.output_width * self.output_height
        else:
            box_h = self.bounding_box["h"] * extra_scale
            box_w = box_h / self.output_height * self.output_width
        cen_x = self.bounding_box["x"] + self.bounding_box["w"] * 0.5
        cen_y = self.bounding_box["y"] + self.bounding_box["h"] * 0.5

        return np.array([cen_x - box_w * 0.5, cen_y - box_h * 0.5, box_w, box_h])

    def Track(self, frame, joints=None, belief=None):
        
        if joints is None or belief is None:
            self.SetState(TRACKER_EMPTY)
            self.ResetBoundingBox()
        else:
            valid_flags = self.GetValidFlags(belief)
            if self.state == TRACKER_EMPTY:
                self.state = TRACKER_DETECTING

            if self.state == TRACKER_DETECTING:
                if self.CheckValid(valid_flags):
                    # if valid
                    self.state = TRACKER_TRACKING
                else:
                    if self.detecting_frame_cnt >= 5:
                        self.SetState(TRACKER_EMPTY)
                    else:
                        self.UpdateBoundingBox(joints, valid_flags)
                        self.detecting_frame_cnt += 1

            if self.state == TRACKER_TRACKING:
                self.UpdateBoundingBox(joints, valid_flags)

                if self.CheckValid(valid_flags):
                    self.SetState(TRACKER_TRACKING)
                else:
                    if self.tracking_faliure_frame_cnt >= 4:
                        self.SetState(TRACKER_EMPTY)
                        self.ResetBoundingBox()
                    else:
                        self.tracking_faliure_frame_cnt += 1
        cur_bounding_box = self.GetBoundingBox()

        self.global_offset = np.array(
            [cur_bounding_box[0], cur_bounding_box[1]])
        self.global_scale = cur_bounding_box[2] / self.output_width

        affine_matrix = np.array(
            [[1.0, 0, -self.global_offset[0]], [0.0, 1.0, -self.global_offset[1]]]) / self.global_scale
        affined_frame = cv2.warpAffine(frame, affine_matrix, dsize=(
            int(self.output_width), int(self.output_height)))

        return affined_frame
