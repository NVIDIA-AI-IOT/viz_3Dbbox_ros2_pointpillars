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


import os
import argparse
from subprocess import call

parser = argparse.ArgumentParser("python3 viz_all_pcs.py")
parser.add_argument("input_bin_path", help="Absolute path to input bin files", type=str)
parser.add_argument("input_txt_path", help="Absolute path to input text files", type=str)
parser.add_argument("output_images_path", help="Absolute path to save resulting images", type=str)
args = parser.parse_args()

files = os.listdir(args.input_bin_path)
files.sort()
for file in files:
    cloud_path = args.input_bin_path + file
    boxes_path = args.input_txt_path + os.path.splitext(file)[0]+'.txt'
    save_file = args.output_images_path + os.path.splitext(file)[0]+'.png'
    call(f'python3 visualize/viewer.py {cloud_path} {boxes_path} {save_file}', shell=True)
