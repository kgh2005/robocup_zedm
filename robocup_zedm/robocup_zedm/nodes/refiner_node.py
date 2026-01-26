#!/usr/bin/env python3
import threading
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from vision_interfaces.msg import BoundingBox
from humanoid_interfaces.msg import Robocupvision25, Robocupvision25feature

# utils로 분리한 함수들
from robocup_zedm.utils.utils import pixel_to_cam_coords


class RefinerNode(Node):
    def __init__(self):
        super().__init__('refiner_node')

        # ---------- Publishers ----------
        self.vision_pub = self.create_publisher(Robocupvision25, 'vision', 10)
        self.vision_feature_pub = self.create_publisher(Robocupvision25feature, 'vision_feature', 10)

        # ---------- Parameters ----------
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('camera_info_topic', '/zedm/zed_node/rgb/color/rect/image/camera_info')
        self.declare_parameter('bbox_topic', '/Bounding_box')

        self.declare_parameter('half_win', 1)
        self.declare_parameter('remove_space_dis', 3000)  # mm

        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.bbox_topic = self.get_parameter('bbox_topic').value

        self.half_win = int(self.get_parameter('half_win').value)
        self.remove_space_dis = int(self.get_parameter('remove_space_dis').value)

        # clamp (C++와 동일)
        self.half_win = int(np.clip(self.half_win, 0, 6))

        self.get_logger().info("=========================")
        self.get_logger().info("Parameters loaded:")
        self.get_logger().info(f"half_win: {self.half_win}")
        self.get_logger().info(f"remove_space_dis: {self.remove_space_dis}")

        # ---------- Camera intrinsics ----------
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.width = 0
        self.height = 0
        self.have_caminfo = False

        # ---------- Depth storage ----------
        self.bridge = CvBridge()
        self.depth_lock = threading.Lock()
        self.latest_depth_m = None  # float32 meters

        # ---------- Detection caches ----------
        self.det_ball = []   # list of (score, x1,y1,x2,y2)
        self.det_line = []
        self.det_robot = []

        # ---------- Messages ----------
        self.vision_msg = Robocupvision25()
        self.vision_feature_msg = Robocupvision25feature()

        # ---------- Subscriptions ----------
        self.create_subscription(BoundingBox, self.bbox_topic, self.bbox_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        self.get_logger().info("RefinerNode initialized.")

    # ---------------- Callbacks ----------------
    def camera_info_callback(self, msg: CameraInfo):
        # K: [fx 0 cx; 0 fy cy; 0 0 1]
        k = msg.k
        self.fx = float(k[0])
        self.fy = float(k[4])
        self.cx = float(k[2])
        self.cy = float(k[5])
        self.width = int(msg.width)
        self.height = int(msg.height)
        self.have_caminfo = True

    def depth_callback(self, msg: Image):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)

            # 여기서 바로 체크
            if depth_image.dtype != np.float32:
                self.get_logger().warn(f"depth dtype is {depth_image.dtype}, expected float32")
                return

            with self.depth_lock:
                self.latest_depth_m = depth_image.copy()

        except Exception as e:
            self.get_logger().error(f"cv_bridge exception: {e}")



    def bbox_callback(self, msg: BoundingBox):
        self.det_ball.clear()
        self.det_line.clear()
        self.det_robot.clear()

        n = len(msg.class_ids)

        # size mismatch guard
        if not (len(msg.score) == n and len(msg.x1) == n and len(msg.y1) == n and len(msg.x2) == n and len(msg.y2) == n):
            self.get_logger().warn("BoundingBox arrays size mismatch.")
            return

        for i in range(n):
            cls = int(msg.class_ids[i])
            score = float(msg.score[i])
            x1 = int(msg.x1[i]); y1 = int(msg.y1[i])
            x2 = int(msg.x2[i]); y2 = int(msg.y2[i])

            if cls == 0:
                self.det_ball.append((score, x1, y1, x2, y2))
            elif cls == 1:
                self.det_line.append((score, x1, y1, x2, y2))
            elif cls == 2:
                self.det_robot.append((score, x1, y1, x2, y2))

        self.bbox_processing()

    # ---------------- Processing ----------------
    def bbox_processing(self):
        if not self.have_caminfo:
            return

        with self.depth_lock:
            if self.latest_depth_m is None:
                return
            depth_copy = self.latest_depth_m.copy()

        h, w = depth_copy.shape[:2]

        # ===== BALL =====
        ball_cam_u = 0
        ball_cam_v = 0
        ball_dist_mm = 0.0
        ball_2d_x_mm = 0.0
        ball_2d_y_mm = 0.0

        if len(self.det_ball) > 0:
            score, x1, y1, x2, y2 = self.det_ball[0]
            u = x1 + (x2 - x1) // 2
            v = y1 + (y2 - y1) // 2

            u = int(np.clip(u, 0, w - 1))
            v = int(np.clip(v, 0, h - 1))

            X, Y, Z = pixel_to_cam_coords(depth_copy, u, v, self.fx, self.fy, self.cx, self.cy, self.half_win)

            if Z > 0.0:
                ball_cam_u = u
                ball_cam_v = v

                ball_dist_mm = Z * 1000.0
                ball_2d_x_mm = X * 1000.0 * (-1.0)
                ball_2d_y_mm = Z * 1000.0

                self.get_logger().info("========== ball ==========")
                self.get_logger().info(f"pixel (u,v) = ({u},{v})")
                self.get_logger().info(f"cam_pt = [X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}] meters")
                self.get_logger().info(f"cam_pt = [X={X*100.0:.1f}, Y={Y*100.0:.1f}, Z={Z*100.0:.1f}] cm")
            else:
                self.get_logger().info(f"ball depth invalid at (u={u},v={v})")

        # ===== LINE FEATURES =====
        for score, x1, y1, x2, y2 in self.det_line:
            u = x1 + (x2 - x1) // 2
            v = y1 + (y2 - y1) // 2
            u = int(np.clip(u, 0, w - 1))
            v = int(np.clip(v, 0, h - 1))

            X, Y, Z = pixel_to_cam_coords(depth_copy, u, v, self.fx, self.fy, self.cx, self.cy, self.half_win)
            if Z <= 0.0:
                continue

            dist_mm = Z * 1000.0
            if dist_mm < float(self.remove_space_dis):
                self.vision_feature_msg.confidence.append(float(score))
                self.vision_feature_msg.distance.append(float(dist_mm))
                self.vision_feature_msg.point_vec_x.append(float(X * 1000.0 * (-1.0)))
                self.vision_feature_msg.point_vec_y.append(float(Z * 1000.0))

        # ===== ROBOTS =====
        robot_vec_x_mm = []
        robot_vec_y_mm = []
        for score, x1, y1, x2, y2 in self.det_robot:
            u = x1 + (x2 - x1) // 2
            v = y1 + (y2 - y1) // 2
            u = int(np.clip(u, 0, w - 1))
            v = int(np.clip(v, 0, h - 1))

            X, Y, Z = pixel_to_cam_coords(depth_copy, u, v, self.fx, self.fy, self.cx, self.cy, self.half_win)
            if Z <= 0.0:
                continue

            robot_vec_x_mm.append(X * 1000.0 * (-1.0))
            robot_vec_y_mm.append(Z * 1000.0)

        # publish (C++와 동일한 필드 의미로 채움)
        self.vision_msg.ball_cam_x = int(ball_cam_u)
        self.vision_msg.ball_cam_y = int(ball_cam_v)


        if ball_dist_mm == 0.0:
            self.vision_msg.ball_2d_x = 0.0
            self.vision_msg.ball_2d_y = 0.0
        else:
            self.vision_msg.ball_2d_x = float(ball_2d_x_mm)
            self.vision_msg.ball_2d_y = float(ball_2d_y_mm)

        self.vision_msg.ball_d = float(ball_dist_mm)

        self.vision_msg.robot_vec_x = [float(x) for x in robot_vec_x_mm]
        self.vision_msg.robot_vec_y = [float(y) for y in robot_vec_y_mm]

        self.vision_pub.publish(self.vision_msg)
        
        self.vision_msg.robot_vec_x = []
        self.vision_msg.robot_vec_y = []


        if len(self.vision_feature_msg.confidence) > 0:
            self.vision_feature_pub.publish(self.vision_feature_msg)
            self.vision_feature_msg.confidence = []
            self.vision_feature_msg.distance = []
            self.vision_feature_msg.point_vec_x = []
            self.vision_feature_msg.point_vec_y = []


def main(args=None):
    rclpy.init(args=args)
    node = RefinerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
