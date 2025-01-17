# Copyright (c) 2022，Horizon Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
import os
from launch.substitutions import LaunchConfiguration

def declare_configurable_parameters(parameters):
    return [DeclareLaunchArgument(param['name'], default_value=param['default_value'], description=param['description']) for param in parameters]

def set_configurable_parameters(parameters):
    return dict([(param['name'], LaunchConfiguration(param['name'])) for param in parameters])

def generate_launch_description():

    stereo_calib_file_path =  os.path.join(
        get_package_share_directory('stereonet_model'),
        'config',
        'stereo2.yaml'
    )

    stereonet_model_file_path =  os.path.join(
        get_package_share_directory('stereonet_model'),
        'config',
        'model.hbm'
    )
    local_image_path =  os.path.join(
        get_package_share_directory('stereonet_model'),
        'config'
    )

    node_params = [
        {'name':'stereo_image_topic', 'default_value':'/image_combine_raw', 'description': 'stereo_image_topic'},
        {'name':'camera_cx', 'default_value':'659.710', 'description': 'rectified_camera_cx'},
        {'name':'camera_cy', 'default_value':'360.584', 'description': 'rectified_camera_cy'},
        {'name':'camera_fx', 'default_value':'527.1931', 'description': 'rectified_camera_fx'},
        {'name':'camera_fy', 'default_value':'527.1931', 'description': 'rectified_camera_fy'},
        {'name':'need_rectify', 'default_value':'True', 'description': 'whether need_rectify or not'},
        {'name':'need_pcl_filter', 'default_value':'False', 'description': 'whether need_pcl_filter or not'},
        {'name':'base_line', 'default_value':'0.119893', 'description': 'base_line of stereo'},
        {'name':'height_min', 'default_value':'0.03', 'description': 'height_min'},
        {'name':'height_max', 'default_value':'5.0', 'description': 'height_max'},
        {'name':'save_image', 'default_value':'False', 'description': 'save_image'},
        {'name':'use_local_image', 'default_value':'False', 'description': 'use_local_image'},
        {'name':'use_usb_camera', 'default_value':'True', 'description': 'use_usb_camera'},
        {'name':'stereo_combine_mode', 'default_value':'1', 'description': 'stereo_combine_mode'},
        {'name':'stereo_calib_file_path', 'default_value': stereo_calib_file_path, 'description': 'stereo_calib_file_path'},
        {'name':'stereonet_model_file_path', 'default_value': stereonet_model_file_path, 'description': 'stereonet_model_file_path'},
        {'name':'local_image_path', 'default_value': local_image_path, 'description': 'local_image_path'},
        {'name':'log_level', 'default_value':'info', 'description': 'log_level'},
        {'name':'leaf_size', 'default_value':'0.05', 'description': 'leaf_size'},
        {'name':'stdv', 'default_value':'0.01', 'description': 'stdv'},
        {'name':'KMean', 'default_value':'10', 'description': 'KMean'},
        {'name':'alpha', 'default_value':'2', 'description': 'visual_alpha'},
        {'name':'beta', 'default_value':'0', 'description': 'visual_beta'},
    ]

    launch = declare_configurable_parameters(node_params)
    launch.append(Node(
        package='stereonet_model',
        executable='stereonet_model_node',
        output='screen',
        parameters=[set_configurable_parameters(node_params)],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    ))

    return LaunchDescription(launch)
