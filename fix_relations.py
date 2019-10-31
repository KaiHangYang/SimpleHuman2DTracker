import os
import json
import sys
import numpy as np

if __name__ == "__main__":
    annotations_path = "./dataset/annotations/train_fu_labeled.json"
    saved_ = "./dataset/annotations/train_fu_labeled_changed.json"
    datas = json.load(open(annotations_path))

    anns = datas["annotations"]

    for ann_idx, cur_ann in enumerate(anns):
      anns[ann_idx]["rel"] = np.array(cur_ann["rel"])[0:11].astype(np.int).tolist()

    with open(saved_, "w") as f:
        json.dump(datas, f)
