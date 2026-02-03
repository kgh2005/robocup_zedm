#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from transitions import Machine

from vision_interfaces.msg import Robocupvision
from dynamixel_rdk_msgs.msg import DynamixelPanTiltMsgs


class PanTiltNode(Node):
    states = ["lost", "found"]

    def __init__(self):
        super().__init__("pantilt_node")
        
        self.vision_sub = self.create_subscription(Robocupvision, "vision", self.vision_callback, 1,)
        self.pantilt_pub = self.create_publisher(DynamixelPanTiltMsgs, "pantilt_dxl", 1,)

        # params
        self.declare_parameter("rate_hz", 50.0)
        
        self.declare_parameter("pan_id", 22)
        self.declare_parameter("pan_max_deg", 70.0)
        self.declare_parameter("pan_min_deg", -70.0)
        
        self.declare_parameter("tilt_id", 23)
        self.declare_parameter("tilt_max_deg", 0.0)
        self.declare_parameter("tilt_min_deg", -30.0)
        
        self.declare_parameter("scan_pan_speed_deg_s", 35.0)
        self.declare_parameter("scan_tilt_speed_deg_s", 35.0)
        
        self.declare_parameter("img_w", 960)
        self.declare_parameter("img_h", 540)
        
        self.declare_parameter("pan_dir", 1.0)
        self.declare_parameter("tilt_dir", 1.0)
        
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        
        self.pan_id = int(self.get_parameter("pan_id").value)
        self.pan_max_deg = float(self.get_parameter("pan_max_deg").value)
        self.pan_min_deg = float(self.get_parameter("pan_min_deg").value)
        
        self.tilt_id = int(self.get_parameter("tilt_id").value)
        self.tilt_max_deg = float(self.get_parameter("tilt_max_deg").value)
        self.tilt_min_deg = float(self.get_parameter("tilt_min_deg").value)
        
        self.scan_pan_speed = float(self.get_parameter("scan_pan_speed_deg_s").value)
        self.scan_tilt_speed = float(self.get_parameter("scan_tilt_speed_deg_s").value)
        
        self.img_w = int(self.get_parameter("img_w").value)
        self.img_h = int(self.get_parameter("img_h").value)
        
        self.roi_hw = self.img_w // 6
        self.roi_hh = self.img_h // 6
        
        self.pan_dir = float(self.get_parameter("pan_dir").value)
        self.tilt_dir = float(self.get_parameter("tilt_dir").value)
        
        
        self.get_logger().info("PantiltNode initialized.")
        self.get_logger().info(f"rate_hz: {self.rate_hz}")
        self.get_logger().info(f"pan_id: {self.pan_id}, pan_max_deg: {self.pan_max_deg}, pan_min_deg: {self.pan_min_deg}")
        self.get_logger().info(f"tilt_id: {self.tilt_id}, tilt_max_deg: {self.tilt_max_deg}, tilt_min_deg: {self.tilt_min_deg}")
        self.get_logger().info(f"scan_pan_speed: {self.scan_pan_speed} deg/s, scan_tilt_speed: {self.scan_tilt_speed} deg/s")
        self.get_logger().info(f"img_w: {self.img_w}, img_h: {self.img_h}")
        self.get_logger().info(f"roi_hw: {self.roi_hw}, roi_hh: {self.roi_hh}")
        self.get_logger().info(f"pan_dir: {self.pan_dir}, tilt_dir: {self.tilt_dir}")
        
        # 스캔 꼭짓점 시퀀스
        self.scan_points = [
            (self.pan_max_deg, self.tilt_min_deg), # 왼쪽 아래
            (0.0, self.tilt_min_deg), # 가운데 아래
            (self.pan_min_deg, self.tilt_min_deg), # 오른쪽 아래
            (self.pan_min_deg, self.tilt_max_deg), # 오른쪽 위
            (0.0, self.tilt_max_deg), # 가운데 위
            (self.pan_max_deg, self.tilt_max_deg) # 왼쪽 위
            # (self.pan_max_deg, self.tilt_min_deg), # 오른쪽 아래
        ]

        self.scan_i = 0
        self.scan_target_pan, self.scan_target_tilt = self.scan_points[self.scan_i]


        # internal state (vision side will update these)
        self.ball_seen = False 
        
        self.ball_x, self.ball_y = 0.0, 0.0
        
        self.ball_cam_x, self.ball_cam_y = 0.0, 0.0
        
        self.pan_deg, self.tilt_deg = 0.0, 0.0
        
        self.angle_deg = 0.0
        
        self.just_lost = False
        self.jump_mode = False
        self.jump_target_i = 0


        
        self.pantilt = DynamixelPanTiltMsgs()
        

        # FSM
        self.machine = Machine(
            model=self,
            states=PanTiltNode.states,
            initial="lost",
        )
        self.machine.add_transition("see_ball", "lost", "found")
        self.machine.add_transition("lose_ball", "found", "lost")

        # control loop
        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)
    
    def pantilt_publish(self):
        self.pantilt.pan_id = self.pan_id
        self.pantilt.pan_goal_position = self.pan_deg
        self.pantilt.pan_profile_acceleration = 0.0
        self.pantilt.pan_profile_velocity = 0.0
        
        self.pantilt.tilt_id = self.tilt_id
        self.pantilt.tilt_goal_position = self.tilt_deg
        self.pantilt.tilt_profile_acceleration = 0.0
        self.pantilt.tilt_profile_velocity = 0.0
        
        self.pantilt_pub.publish(self.pantilt)
        
    def _move_toward(self, cur, target, step):
        # step은 항상 양수 (deg/tick)
        if cur < target:
            return min(cur + step, target)
        else:
            return max(cur - step, target)

    def _scan_update_constant_speed(self, dt):
        pan_step = abs(self.scan_pan_speed) * dt
        tilt_step = abs(self.scan_tilt_speed) * dt

        self.pan_deg = self._move_toward(self.pan_deg, self.scan_target_pan, pan_step)
        self.tilt_deg = self._move_toward(self.tilt_deg, self.scan_target_tilt, tilt_step)

        # 도달 확인 (부동소수점 오차 허용)
        pan_arrived = abs(self.pan_deg - self.scan_target_pan) < 0.1
        tilt_arrived = abs(self.tilt_deg - self.scan_target_tilt) < 0.1
        
        if pan_arrived and tilt_arrived:
            self.scan_i = (self.scan_i + 1) % len(self.scan_points)
            self.scan_target_pan, self.scan_target_tilt = self.scan_points[self.scan_i]
    
    def _track_roi_px_simple(self, dt):
        cx, cy = self.img_w * 0.5, self.img_h * 0.5
        left, right = cx - self.roi_hw, cx + self.roi_hw
        top, bottom = cy - self.roi_hh, cy + self.roi_hh

        x, y = self.ball_x, self.ball_y

        # ROI 안이면 정지
        if left <= x <= right and top <= y <= bottom:
            return

        # x가 왼쪽이면 pan 한쪽, 오른쪽이면 반대쪽
        if x < left:
            self.pan_deg -= self.pan_dir * self.scan_pan_speed * dt
        elif x > right:
            self.pan_deg += self.pan_dir * self.scan_pan_speed * dt

        # y가 위/아래면 tilt 조절 (필요하면 부호만 뒤집기)
        if y < top:
            self.tilt_deg += self.tilt_dir * self.scan_tilt_speed * dt
        elif y > bottom:
            self.tilt_deg -= self.tilt_dir * self.scan_tilt_speed * dt
            
        # self.get_logger().info(f"Tracking ball at px ({self.ball_x}, {self.ball_y}), pan_deg: {self.pan_deg}, tilt_deg: {self.tilt_deg}")

        # 리밋
        self.pan_deg = max(self.pan_min_deg, min(self.pan_max_deg, self.pan_deg))
        self.tilt_deg = max(self.tilt_min_deg, min(self.tilt_max_deg, self.tilt_deg))
        
        
    def angle_to_scan_index(self, angle_deg: float):
        a = angle_deg % 360.0

        # 0~60: 오른쪽 위(RT=5)
        if 0.0 <= a < 60.0:
            return 'RT'
        # 60~120: 위(MT=4)
        elif 60.0 <= a < 120.0:
            return 'MT'
        # 120~180: 왼쪽 위(LT=3)
        elif 120.0 <= a < 180.0:
            return 'LT'
        # 180~240: 왼쪽 아래(LB=2)
        elif 180.0 <= a < 240.0:
            return 'LB'
        # 240~300: 아래(MB=1)
        elif 240.0 <= a < 300.0:
            return 'MB'
        # 300~360: 오른쪽 아래(RB=0)
        else:
            return 'RB'
        
    def _move_to_target(self, dt):
        pan_step = abs(self.scan_pan_speed) * dt
        tilt_step = abs(self.scan_tilt_speed) * dt
        self.pan_deg = self._move_toward(self.pan_deg, self.scan_target_pan, pan_step)
        self.tilt_deg = self._move_toward(self.tilt_deg, self.scan_target_tilt, tilt_step)
        
        # 부동소수점 오차 허용 (0.1도 이내)
        pan_arrived = abs(self.pan_deg - self.scan_target_pan) < 0.1
        tilt_arrived = abs(self.tilt_deg - self.scan_target_tilt) < 0.1
        
        return pan_arrived and tilt_arrived


    def angle_deg_360(self, X, Y):
        return (math.degrees(math.atan2(Y, X)) + 360.0) % 360.0


    def _tick(self):
        # 1) 상태 전이(비전 결과 기반)
        if self.ball_seen and self.state == "lost":
            self.see_ball()
        elif (not self.ball_seen) and self.state == "found":
            # lost로 전환될 때 각도 저장
            self.angle_deg = self.angle_deg_360(self.ball_cam_x, self.ball_cam_y)
            angle_ = self.angle_to_scan_index(self.angle_deg)

            match angle_:
                case 'LB': self.jump_target_i = 0
                case 'MB': self.jump_target_i = 1
                case 'RB': self.jump_target_i = 2
                case 'RT': self.jump_target_i = 3
                case 'MT': self.jump_target_i = 4
                case 'LT': self.jump_target_i = 5

            # 점프 목표 설정
            self.scan_i = self.jump_target_i
            self.scan_target_pan, self.scan_target_tilt = self.scan_points[self.scan_i]
            self.jump_mode = True

            self.get_logger().info(
                f"lost: jump start -> {angle_} idx={self.jump_target_i} "
                f"target=({self.scan_target_pan},{self.scan_target_tilt})"
            )
            
            self.lose_ball()
            
        dt = 1.0 / self.rate_hz
            
        match self.state:
            case "lost":
                if self.jump_mode:
                    arrived = self._move_to_target(dt)
                    if arrived:
                        self.jump_mode = False
                        self.get_logger().info(f"lost: jump arrived, start scan at scan_i={self.scan_i}")
                else:
                    self._scan_update_constant_speed(dt)
                
            case "found":
                self._track_roi_px_simple(dt)
        
        self.pantilt_publish()

    
    def vision_callback(self, msg: Robocupvision):
        if msg.ball_d == 0 and msg.ball_x == 0 and msg.ball_y == 0:
            
            self.ball_seen = False
        else:
            self.ball_x = msg.ball_x
            self.ball_y = msg.ball_y
            
            self.ball_cam_x = msg.ball_cam_x * 0.1 # cm
            self.ball_cam_y = -msg.ball_cam_y * 0.1 # cm
            
            self.ball_seen = True
            
            # self.get_logger().info(f"vision_callback: ball_seen={self.ball_seen}, ball_x={msg.ball_x}, ball_y={msg.ball_y}")
            # self.get_logger().info(f"vision_callback: ball_cam_x={msg.ball_cam_x}, ball_cam_y={msg.ball_cam_y}")
        
        # self.get_logger().info(f"vision_callback: ball_seen={self.ball_seen}, ball_2d_x={msg.ball_2d_x}, ball_2d_y={msg.ball_2d_y}")



def main(args=None):
    rclpy.init(args=args)
    node = PanTiltNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()