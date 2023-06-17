# -*- coding:utf-8 -*-

import os

import cv2
import numpy as np
import torch

from kpt_resnet50 import KptResNet50

model_file = "./model/best.pth"
kpt_num = 4
input_height = 448
input_width = 448
heatmap_height = 112
heatmap_width = 112
x_feat_stride = input_width / heatmap_width
y_feat_stride = input_height / heatmap_height

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = KptResNet50(kpt_num=kpt_num, pretrained=False)
model.load_state_dict(torch.load(model_file, map_location='cpu'))
model.to(device)

point_color_dict = {
    0: (0, 0, 255),  # 左上
    1: (0, 255, 0),  # 右上
    2: (255, 0, 0),  # 左下
    3: (255, 0, 255),  # 右下
}


def image_preprocess(np_img):
    img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)  # bgr to rgb
    # resize
    img = cv2.resize(img, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    # normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data = img.astype(np.float32)
    data = (data / 255. - np.array(mean)) / np.array(std)
    # transpose
    data = data.transpose((2, 0, 1)).astype(np.float32)  # HWC to CHW

    return data


def image_postprocess(heatmaps):
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    width = heatmaps.shape[3]
    heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def inference_one(data_input):
    image_tensor = torch.from_numpy(data_input)
    image_tensor = image_tensor.unsqueeze(0).float()  # add batch dimension
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        model.eval()
        out = model(image_tensor)
    heatmaps = out.cpu().detach().numpy()

    return heatmaps


def visual(origin_img, coords, maxvals, save_name):
    dst_img = origin_img.copy()
    for point_id in range(4):
        x, y = coords[point_id]
        cv2.circle(dst_img, (x, y), 5, point_color_dict[point_id], -1)

        # text = str(round(maxvals[point_id], 2))
        # cv2.putText(origin_img, text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imwrite(save_name, dst_img)


def adjust(origin_img, coords, save_name):
    dst_cols, dst_rows = 1000, 400
    dst_coords = np.array([[50, 50],
                           [dst_cols - 50, 50],
                           [50, dst_rows - 50],
                           [dst_cols - 50, dst_rows - 50]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(coords.astype(np.float32), dst_coords)
    dst_img = cv2.warpPerspective(origin_img, M, (dst_cols, dst_rows))

    cv2.imwrite(save_name, dst_img)


if __name__ == '__main__':
    image_dir = "./data/car_plate/val/images"
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        if image_path.endswith("jpg"):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # read image
            origin_height, origin_width = image.shape[:2]  # get origin resolution
            w_scale = origin_width / input_width
            h_scale = origin_height / input_height

            input_data = image_preprocess(image)
            output = inference_one(input_data)
            coords, maxvals = image_postprocess(output)  # 相对 heatmap 的坐标

            coords = coords.reshape((4, 2))
            maxvals = maxvals.reshape(-1)

            # 转换为相对输入图像数据的坐标
            coords[:, 0] *= x_feat_stride
            coords[:, 1] *= y_feat_stride

            # 转换为相对原始图像的坐标
            coords[:, 0] *= w_scale
            coords[:, 1] *= h_scale
            coords = (coords + 0.5).astype(np.int32)

            # 在原图中绘制检测出的关键点
            save_path = os.path.join(output_dir, image_name)
            visual(image, coords, maxvals, save_path)

            # 透视变换矫正图像
            adjusted_path = os.path.join(output_dir, "_adjust.".join(image_name.split(".")))
            adjust(image, coords, adjusted_path)
