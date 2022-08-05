"""
author: hova88
date: 2021/03/16
"""
import numpy as np
from visual_tools import draw_clouds_with_boxes
import open3d as o3d
import sys


if __name__ == "__main__":
    
    cloud_path, boxes_path, save_file =  sys.argv[1], sys.argv[2], sys.argv[3]
    
    print(f'cloud path {cloud_path}')

    cloud = np.fromfile(cloud_path, dtype=np.float32).reshape(-1,4)
    boxes = np.loadtxt(boxes_path).reshape(-1,9)
    boxes = boxes[boxes[:, -1] > 0.01][:, :7] # score thr = 0.01

    draw_clouds_with_boxes(cloud, boxes, save_file)
