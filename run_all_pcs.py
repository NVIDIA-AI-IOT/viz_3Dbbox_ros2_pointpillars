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

parser = argparse.ArgumentParser("python3 run_all_pcs.py")
parser.add_argument("class_names", help="List of object classes detected by the model", type=str)
parser.add_argument("nms_iou_thresh", help="NMS IOU threshold", type=float)
parser.add_argument("pre_nms_top_n", help="Number of top n boxes to use for NMS", type=float)
parser.add_argument("engine_path", help="Absolute path to TensorRT engine file", type=str)
parser.add_argument("input_bin_path", help="Absolute path to input bin files", type=str)
parser.add_argument("data_type", help="Data type (fp32 or fp16)", type=str)
parser.add_argument("output_path", help="Absolute path to output folder", type=str)
parser.add_argument('--do_profile', action='store_true')
parser.add_argument('--no_profile', dest='do_profile', action='store_false')
parser.set_defaults(do_profile=False)
args = parser.parse_args()


files = os.listdir(args.input_bin_path)
for file in files:
    input_file = args.input_bin_path + file
    
    if args.do_profile:
        call(f'./pointpillars -e {args.engine_path} -l {input_file} -o {args.output_path} -t {args.nms_iou_thresh} -c {args.class_names} -n {args.pre_nms_top_n} -p -d {args.data_type}', shell=True)
    else:
        call(f'./pointpillars -e {args.engine_path} -l {input_file} -o {args.output_path} -t {args.nms_iou_thresh} -c {args.class_names} -n {args.pre_nms_top_n}  -d {args.data_type}', shell=True)
