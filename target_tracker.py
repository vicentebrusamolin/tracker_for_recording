#!/usr/bin/env python3

import sys
sys.path.insert(1, '/home/vicente_brusamolin/ASV-CollisionAvoidance/src/radar/import')

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D
from vision_msgs.msg import BoundingBox2DArray
from protobuf_client_interfaces.msg import Gateway

from cv_bridge import CvBridge

import cv2
import numpy as np

import os

from Cluster import Cluster
from TargetTracker import TargetTracker

class TargetFinderNode(Node):
    def __init__(self):
        super().__init__('target_tracker')

        self.get_logger().info('TargetFinder has been started')

        ### AFTER RECORDING, ERASE ###
        self.image_dir = "/home/vicente_brusamolin/ASV-CollisionAvoidance/src/images/"
        if os.path.exists(self.image_dir):
            for filename in os.listdir(self.image_dir):
                file_path = os.path.join(self.image_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        ###

        # parameters
        self.declare_parameter("dist_tolerance", 100)
        self.declare_parameter("max_speed", 20)
        self.declare_parameter("frame_side", 976)
        self.declare_parameter("sweep_radius", 926)

        # distance tolerance (in pixels) for obstacle matching between frames 
        # -> eventually update to meters, km or nautical miles
        self.dist_tolerance = self.get_parameter("dist_tolerance").value

        # maximum speed for obstacle validation (unused)
        self.max_speed = self.get_parameter("max_speed").value

        # sweep frame side length in pixels 
        # -> set in launch and validate with images provided on topic "radar_sweep"
        self.frame_side = self.get_parameter("frame_side").value

        # sweep radius in meters: used to estimate spatial resolution, and real world distances
        self.sweep_radius = self.get_parameter("sweep_radius").value
        
        # subscriber - receives updated cluster detections
        self.bb_subscription = self.create_subscription(
            BoundingBox2DArray,
            'r_centroid_arr',
            self.bb_callback,
            10
        )
        self.bb_subscription  # Prevent unused variable warning

        # MOOS variables
        self.gl_x    = 0 # global x coordinates in meters
        self.gl_y    = 0 # global y coordinates in meters
        self.gl_head = 0 # global heading in degrees
        
        ### AFTER RECORDING, ERASE ###
        self.heave = 0 # global heave in meters
        self.roll =  0 # euler roll in degrees
        self.pitch = 0 # euler pitch in degrees
        self.dbtime = 0
        self.i = 0
        self.last_radar = np.zeros((1080, 1920, 3))
        self.last_camera = np.zeros((2160, 3840, 3)) # MUDAR PARA A RESOLUCAO DA TELA

        self.gl_x_arr = np.array([])
        self.gl_y_arr = np.array([])
        self.gl_head_arr = np.array([])
        self.heave_arr = np.array([])
        self.roll_arr = np.array([])
        self.pitch_arr = np.array([])
        self.dbtime_arr = np.array([])

        ####

        # subscriber - receives data from MOOSDB
        self.moos_subscription = self.create_subscription(
            Gateway,
            '/gateway_msg', 
            self.moos_callback, 
            10
        )
        self.moos_subscription  # Prevent unused variable warning

        # subscriber - receives updated radar sweep
        ### AFTER RECORDING, ERASE ###
        self.radar_subscription = self.create_subscription(
            Image,
            'radar_sweep',
            self.sweep_callback,
            10
        )
        self.radar_subscription  # Prevent unused variable warning
        ###

        # subscriber - receives images from camera view
        ### AFTER RECORDING, ERASE ###
        self.camera_subscription = self.create_subscription(
            Image,
            'camera_view',
            self.camera_view_callback,
            10
        )
        self.camera_subscription  # Prevent unused variable warning
        ###

        # conversion between OpenCV and ROS2 image types
        ### AFTER RECORDING, ERASE ###
        self.bridge = CvBridge()
        ###

        # publisher - publishes obstacle information
        self.publisher = self.create_publisher(Gateway, f'/send_to_gateway', 10)

        # TargetTracker instance: adds the core tracking functionality
        # Requires cluster information to be instanced
        self.tt = None

        # pixel spatial resolution
        self.spatial_resolution = self.sweep_radius/(self.frame_side/2)

        self._timer = self.create_timer(1, self.timer_callback)


    ### AFTER RECORDING, ERASE ###
    def sweep_callback(self, msg):
        self.get_logger().info('Received radar')

        # convert to OpenCV image type
        self.last_radar = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    
    def camera_view_callback(self, msg):
        self.get_logger().info('Received camera')
        self.last_camera = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def timer_callback(self):
        self.gl_x_arr       = np.append(self.gl_x_arr, self.gl_x)
        self.gl_y_arr       = np.append(self.gl_y_arr, self.gl_y)
        self.gl_head_arr    = np.append(self.gl_head_arr, self.gl_head)
        self.heave_arr      = np.append(self.heave_arr, self.heave)
        self.roll_arr       = np.append(self.roll_arr, self.roll)
        self.pitch_arr      = np.append(self.pitch_arr, self.pitch)
        self.pitch_arr      = np.append(self.pitch_arr, self.pitch)
        self.dbtime_arr     = np.append(self.dbtime_arr, self.dbtime)

        radar_img_name = os.path.join(self.image_dir, f"r_{self.i}_{self.dbtime}.jpg")
        camera_img_name = os.path.join(self.image_dir, f"c_{self.i}_{self.dbtime}.jpg")
        cv2.imwrite(radar_img_name, self.last_radar)
        cv2.imwrite(camera_img_name, self.last_camera)

        self.i += 1
        self.get_logger().info('Saving state')

    ####

    def bb_callback(self, msg: BoundingBox2DArray):
        self.get_logger().info(f'Received')

        self.get_logger().error(f'REPLACE BY MESSAGE TIME')
        time = self.get_clock().now().nanoseconds

        clusters = []

        header = msg.header
        bb_arr = msg.boxes

        for bb in bb_arr:
            centroid_x = bb.center.position.x
            centroid_y = bb.center.position.y
            width = bb.size_x
            height = bb.size_y

            cluster = Cluster([0, 0])
            cluster.x_max = centroid_x + width/2
            cluster.x_min = centroid_x - width/2
            cluster.y_max = centroid_y + height/2
            cluster.y_min = centroid_y - height/2

            clusters.append(cluster)

        if self.tt == None:
            self.tt = TargetTracker(
                self.dist_tolerance, 
                self.max_speed,
                self.frame_side,
                self.spatial_resolution, 
                clusters,
                time*1e-9
            )
            return
        
        self.tt.update(clusters, time*1e-9)
        frame = np.zeros((976,976,3))
        frame = self.tt.draw_obstacles(frame, time*1e-9)

        rotate_matrix = cv2.getRotationMatrix2D(center=(frame.shape[0]/2, frame.shape[0]/2), angle=-self.gl_head, scale=1) 
  
        # rotate the image using cv2.warpAffine  
        # 90 degree anticlockwise 
        height, width = frame.shape[:2] 
        rotated_image = cv2.warpAffine( 
            src=frame, M=rotate_matrix, dsize=(width, height)) 

        cv2.imshow("img", frame)
        # cv2.imshow("rot", rotated_image)
        cv2.waitKey(round(1/60*1000))

        # publish message
        self.publish_obstacles(self.tt.obstacles, time, frame)
        # self.publish_obstacles(time)

    #def moos_receiver_callback(self, msg):
    #    print('Entrou no  callback do moos_receiver')
    #    if msg.gateway_key == 'NAV_X':
    #        self.NAV_X = msg.gateway_double
    #        print("NAV_X = {}".format(self.NAV_X))
    #    elif msg.gateway_key == 'NAV_Y':
    #        self.NAV_Y = msg.gateway_double
    #        print("NAV_Y = {}".format(self.NAV_X))
    #    elif msg.gateway_key == 'NAV_HEADING':
    #        self.NAV_HEADING = msg.gateway_double
    #        print("NAV_HEADING = {}".format(self.NAV_X))
    #    #print('Received from own ship: heading {}, x-y {},{}'.format(self.NAV_X, self.NAV_Y, self.NAV_HEADING))

    def moos_callback(self, msg: Gateway):
        key = msg.gateway_key
        if key == "NAV_X":
            self.gl_x = msg.gateway_double
            print(f'NAV_X = {self.gl_x}')
        elif key == "NAV_Y":
            self.gl_y = msg.gateway_double
        elif key == "NAV_HEADING":
            self.gl_head = msg.gateway_double
        elif key == "NAV_Z":
            self.heave = msg.gateway_double
        elif key == "IMU_PITCH":
            self.pitch = msg.gateway_double
        elif key == "IMU_ROLL":
            self.roll = msg.gateway_double
        elif key == "DB_TIME":
            self.dbtime = msg.gateway_double
        else:
            pass


    # def relative_to_global(self, obstacle):
    #     '''
    #     Fixes TargetTraker coordinates/heading (relative) to be consistent with MOOS coordinates 
    #     output: [id, abs_speed, global_x, global_y, heading]
    #     '''
    #     x, y = obstacle.state[0, 0:2]
    #     vx, vy = obstacle.state[0, 2:4]

    #     gl_x = (x-488) * self.spatial_resolution + self.gl_x
    #     gl_y = -(y-488) * self.spatial_resolution + self.gl_y

    #     abs_spd = np.linalg.norm([vx, vy]) * self.spatial_resolution

    #     if (vy == 0):
    #         if (vx > 0):
    #             head = 90
    #         else:
    #             head = 270
    #     else:
    #         head = np.rad2deg(np.arctan(vx/vy)) # clock-wise heading in degrees (north up)

    #     gl_obstacle = []
    #     gl_obstacle.append(obstacle.id)
    #     gl_obstacle.append(abs_spd)
    #     gl_obstacle.append(gl_x)
    #     gl_obstacle.append(gl_y)
    #     gl_obstacle.append(head)

    #     return gl_obstacle

    ### MAIS ATUALIZADO ABAIXO


    def relative_to_global(self, obstacle, rot=None, mode="head_up"):
        '''
        Fixes TargetTraker coordinates/heading (relative) to be consistent with MOOS coordinates 
        output: [id, abs_speed, global_x, global_y, heading]
        '''
        vy, vx = obstacle.state[0, 2:4]

        if mode == "north_up":
            # rotation matrix is ignored in this mode
            y, x = obstacle.state[0, 0:2]
            gl_x = self.gl_x + (x - self.frame_side/2) * self.spatial_resolution
            gl_y = self.gl_y - (y - self.frame_side/2) * self.spatial_resolution

            abs_spd = np.linalg.norm([vx, vy]) * self.spatial_resolution

            if (vy == 0):
                if (vx > 0):
                    head = 90
                else:
                    head = 270
            else:
                head = np.rad2deg(np.arctan2(vx/vy)) # clock-wise heading in degrees (north up)

        elif mode == "head_up":
            # assert(rot != None) # checks for missing rotation matrix 
            # print(f'Rot dentro do relative_to_global {rot}')
            rel_pos = rot @ (obstacle.state[0, 0:2] - np.array([self.frame_side/2, self.frame_side/2])).reshape(2, 1)
            rel_pos = (rot @ (obstacle.state[0, 0:2] - np.array([self.frame_side/2, self.frame_side/2])).reshape(2, 1)).ravel() 
            rel_pos_true_dist = -(rel_pos * self.spatial_resolution)

            gl_x = self.gl_x - rel_pos_true_dist[0]
            gl_y = self.gl_y + rel_pos_true_dist[1]

            print(f'embarcação {obstacle.id} - global x: {gl_x}')
            
            abs_spd = np.linalg.norm([vx, vy]) * self.spatial_resolution

            gl_vx, gl_vy = (rot @ np.array([[vx], [vy]])).ravel()

            if (gl_vy == 0):
                if (gl_vx > 0):
                    head = 90
                else:
                    head = 270
            else:
                head = np.rad2deg(np.arctan(gl_vx/gl_vy)) # clock-wise heading in degrees (north up)
        else:
            print('mode not authorized')
            return

        head = head % 360 if head > 0 else 360 - (head % 360)

        gl_obstacle = []
        gl_obstacle.append(obstacle.id)
        gl_obstacle.append(abs_spd)
        gl_obstacle.append(gl_x)
        gl_obstacle.append(gl_y)
        gl_obstacle.append(head)
        # print(gl_obstacle)

        return gl_obstacle

    # def publish_obstacles(self, time):
    #     i=0
    #     for obstacle in self.tt.obstacles:
    #         # gl_obstacle = self.relative_to_global(obstacle)

    #         x, y = [self.gl_x, self.gl_y]+(obstacle.state[0, 0:2]-[488,488])*self.tt.spatial_res
    #         # print(x)
    #         vx, vy = obstacle.state[0, 2:4]
    #         dist = np.round(np.linalg.norm(obstacle.centroid() - [488, 488])*self.tt.spatial_res)

    #         gtw = Gateway()
    #         gtw.gateway_key = "NODE_REPORT"
    #         # gtw.gateway_string = f"NAME={gl_obstacle[0]},TYPE=KAYAK,UTC_TIME={time},X={gl_obstacle[2]},Y={gl_obstacle[3]},SPD={gl_obstacle[1]},HDG={gl_obstacle[4]},DEPTH=0"

    #         send_string = f"NAME={obstacle.id},TYPE=KAYAK,UTC_TIME={time},X={x},Y={y},SPD={np.linalg.norm([vx, vy])},HDG={0},DEPTH=0"
    #         gtw.gateway_string = send_string
    #         print(gtw.gateway_string)
    #         #print("Contato {} possui coordenads x-y: {:.2f}, {:.2f}. Distância de {}".format(obstacle.id, x, y, dist))
    #         self.publisher.publish(gtw)
    #         i += 1
            

    #     self.get_logger().info(f'Published: {i} obstacles, from {len(self.tt.obstacles)}')

    ### MAIS ATUALIZADO ABAIXO

    def publish_obstacles(self, obstacles, time, frame):

        rot1_orig = cv2.getRotationMatrix2D([0, 0], -self.gl_head, 1.0)
        rot = rot1_orig[:2, :2]
        # print(f'Rot dentro do publish_obstacle {rot}')

        canvas = np.zeros_like(frame)
        
        rot2_orig = cv2.getRotationMatrix2D([frame.shape[0]/2, frame.shape[0]/2], -self.gl_head, 1.0)
        rot2 = rot2_orig[:2, :2]
        
        i = 0
        for obstacle in obstacles:


            y, x = obstacle.state[0, 0:2]
            rel_pos = ((rot2 @ (obstacle.state[0, 0:2] - np.array([self.frame_side/2, self.frame_side/2])).reshape(2, 1)).ravel() + np.array([self.frame_side/2, self.frame_side/2])).astype(np.int16) 
            cv2.circle(canvas, rel_pos, 4, (255, 255, 255), -1)
            cv2.putText(canvas, f'{obstacle.id}', rel_pos + np.array([20, 20]), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1, cv2.LINE_AA)


            gl_obstacle = self.relative_to_global(obstacle, rot=rot, mode="head_up")

            dist = np.round(np.linalg.norm(obstacle.centroid() - [488, 488])*self.tt.spatial_res)

            gtw = Gateway()
            gtw.gateway_key = "NODE_REPORT"
            send_string = f"NAME={gl_obstacle[0]},TYPE=KAYAK,UTC_TIME={time},X={gl_obstacle[2]},Y={gl_obstacle[3]},SPD={gl_obstacle[1]},HDG={gl_obstacle[4]},DEPTH=0"
            gtw.gateway_string = send_string
            # print(gtw.gateway_string)
            # print("Contato {} possui coordenads x-y: {:.2f}, {:.2f}. Distância de {}".format(gl_obstacle[0], gl_obstacle[2], gl_obstacle[3], dist))
            
            self.publisher.publish(gtw)
            i += 1
  
        # rotate the image using cv2.warpAffine  
        # 90 degree anticlockwise 
        height, width = frame.shape[:2] 
        rotated_image = cv2.warpAffine(src=frame, M=rot2_orig, dsize=(width, height)) 

        # cv2.imshow("canvas", canvas)
        # cv2.imshow("publish_obstacles frame", frame)
        # cv2.imshow("publish_obstacles rot", rotated_image)
        # cv2.waitKey(round(1/60*1000))

        self.get_logger().info(f'Published: {i} obstacles, from {len(obstacles)}')

def main(args=None):
    rclpy.init(args=args)
    node = TargetFinderNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()