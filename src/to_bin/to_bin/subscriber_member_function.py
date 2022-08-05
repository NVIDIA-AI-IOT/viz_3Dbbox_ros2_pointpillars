################################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2

import numpy as np
from pypcd import pypcd
from sensor_msgs.msg import PointField
import sensor_msgs_py.point_cloud2 as pc2

pc_num = 0

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        
        self.declare_parameter('output_path')
        self.declare_parameter('intensity_range')

        global param_output_path
        param_output_path = self.get_parameter('output_path').value
        global param_intensity_range
        param_intensity_range = self.get_parameter('intensity_range').value

        self.subscription = self.create_subscription(
            PointCloud2,
            '/point_cloud',
            self.listener_callback,
            700)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        cloud_points = list(pc2.read_points(msg, skip_nans=True, field_names = ("x", "y", "z", "intensity")))
        
        pc_arr = np.array(cloud_points).flatten()
        # dividing by param_intensity_range to get intensity in desired range
        pc_arr[3::4] /= param_intensity_range
        global pc_num
        pc_num += 1
        print(pc_num)
        output_file = param_output_path + str(pc_num) + '.bin'
        pc_arr.astype('float32').tofile(output_file)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
