# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import cv2
import glob
import re
import argparse

numbers = re.compile(r'(\d+)')
def sequence_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

parser = argparse.ArgumentParser("python3 bbox_video.py")
parser.add_argument("images_path", help="Absolute path to input images folder", type=str)
parser.add_argument("output_path", help="Absolute path to save output video", type=str)
args = parser.parse_args()

img_array = []
images_path = args.images_path + '*.png'
for filename in sorted(glob.glob(images_path), key=sequence_sort):
    print(f"Current File Being Processed is: {filename}")
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

# Can specify the desired FPS (frames per second) in place of '1' below
out = cv2.VideoWriter(args.output_path,cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
