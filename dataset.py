# -*- coding: utf-8 -*-

import os
import json

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class KeypointDataset(Dataset):
    def __init__(self, args, split='train'):
        self.kpt_num = args.kpt_num  # 这里有4个关键点，分别为：左上、右上、左下、右下
        self.image_size = args.image_size  # 输入到网络的尺寸
        self.heatmap_size = args.heatmap_size  # 关键点热图尺寸
        self.sigma = args.sigma
        self.data_dir = args.data_dir

        self.image_dir = os.path.join(self.data_dir, "%s/images" % split)
        self.label_dir = os.path.join(self.data_dir, "%s/labels" % split)

        self.im_ids = []
        self.images = []
        self.labels = []
        for file in os.listdir(self.image_dir):
            if os.path.isfile(os.path.join(self.image_dir, file)) and not file.startswith("."):
                no_ext_file_name = os.path.splitext(file)[0]  # 获取文件名，不包括拓展名
                self.im_ids.append(no_ext_file_name)
                image = os.path.join(self.image_dir, no_ext_file_name + ".jpg")
                label = os.path.join(self.label_dir, no_ext_file_name + ".json")
                assert os.path.isfile(image)
                assert os.path.isfile(label)
                self.images.append(image)
                self.labels.append(label)

        assert (len(self.images) == len(self.labels))
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

        if split == "train":
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # print(self.images[index])
        image = Image.open(self.images[index]).convert('RGB')
        w, h = image.size
        w_scale = self.image_size / w
        h_scale = self.image_size / h
        image = image.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        image = self.transform(image)

        points = self.json_to_numpy(self.labels[index], w_scale, h_scale)  # [kpt_num, 2], 相对image size的坐标
        target = self.generate_target(points)

        target = torch.from_numpy(target)

        return image, target

    def json_to_numpy(self, label_path, w_scale, h_scale):
        with open(label_path, "r") as fp:
            json_data = json.load(fp)
            points = json_data['shapes']

        kpts = np.zeros((self.kpt_num, 2), dtype=np.float32)  # 按照 左上、右上、左下、右下 顺序存放关键点坐标
        for point in points:
            if point["label"] == "left_top":
                kpts[0] = point["points"][0]
            elif point["label"] == "right_top":
                kpts[1] = point["points"][0]
            elif point["label"] == "left_bottom":
                kpts[2] = point["points"][0]
            elif point["label"] == "right_bottom":
                kpts[3] = point["points"][0]
            else:
                raise KeyError("Unknown keypoint : %s for file : %s" % (point["label"], label_path))

        kpts[:, 0] *= w_scale
        kpts[:, 1] *= h_scale

        return kpts

    def generate_target(self, points):
        """
        :param points: [kpt_num, 2]
        """
        target = np.zeros((self.kpt_num, self.heatmap_size, self.heatmap_size), dtype=np.float32)

        tmp_size = self.sigma * 3

        for point_id in range(self.kpt_num):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(points[point_id][0] / feat_stride + 0.5)
            mu_y = int(points[point_id][1] / feat_stride + 0.5)
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            # Generate Gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size)
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size)

            target[point_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target

    @staticmethod
    def show_heatmap(heatmaps):
        for heatmap in heatmaps:
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Keypoint Dataset")
    parser.add_argument('--kpt-num', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=448)
    parser.add_argument('--heatmap-size', type=int, default=112)
    parser.add_argument('--sigma', type=float, default=3.0)
    parser.add_argument('--data-dir', type=str, default="./data/car_plate")

    flags = parser.parse_args()

    dataset = KeypointDataset(flags, split="val")
    for i, (img, tgt) in enumerate(dataset):
        img = img.numpy()
        tgt = tgt.numpy()
        print(img.shape, tgt.shape)
        KeypointDataset.show_heatmap(tgt)
        if i == 4:
            break
