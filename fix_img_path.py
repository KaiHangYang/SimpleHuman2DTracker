import os
import json
import sys

if __name__ == "__main__":
    annotations_path = "./dataset/annotations/train_fu_labeled.json"
    datas = json.load(open(annotations_path))

    imgs = datas["images"]

    for img_idx, cur_img in enumerate(imgs):
        img_name = os.path.join("train/fu_labeled", cur_img["file_name"].split("\\")[-1])
        datas["images"][img_idx]["file_name"] = img_name

    with open(annotations_path, "w") as f:
        json.dump(datas, f)
