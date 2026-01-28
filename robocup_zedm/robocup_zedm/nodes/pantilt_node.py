#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from transitions import Machine

from vision_interfaces.msg import Robocupvision


class PanTiltNode(Node):
    states = ["lost", "found"]

    def __init__(self):
        super().__init__("pantilt_node")
        
        self.vision_sub = self.create_subscription(Robocupvision, "vision", self.vision_callback, 10,)

        # params
        self.declare_parameter("rate_hz", 50.0)
        rate_hz = float(self.get_parameter("rate_hz").value)
        
        self.get_logger().info("PantiltNode initialized.")
        self.get_logger().info(f"rate_hz: {rate_hz}")

        # internal state (vision side will update these)
        self.ball_seen = False 
        self.pan = 0.0
        self.tilt = 0.0
        
        self.ball_x, self.ball_y = 0.0, 0.0

        # FSM
        self.machine = Machine(
            model=self,
            states=PanTiltNode.states,
            initial="lost",
        )
        self.machine.add_transition("see_ball", "lost", "found")
        self.machine.add_transition("lose_ball", "found", "lost")

        # control loop
        self.timer = self.create_timer(1.0 / rate_hz, self._tick)

    def _tick(self):
        # 1) 상태 전이(비전 결과 기반)
        if self.ball_seen and self.state == "lost":
            self.see_ball()
        elif (not self.ball_seen) and self.state == "found":
            self.lose_ball()

        # 2) 상태별 행동(여기만 커지면 됨)
        if self.state == "lost":
            pass
            # self.get_logger().info("lost state: searching for ball...")
        else:  # found
            pass
            # self.get_logger().info(f"found state: tracking ball at ({self.ball_x}, {self.ball_y})")
    
    def vision_callback(self, msg: Robocupvision):
        if msg.ball_d == 0 and msg.ball_2d_x == 0 and msg.ball_2d_y == 0:
            self.ball_seen = False
        else:
            self.ball_x = msg.ball_cam_x
            self.ball_y = msg.ball_cam_y
            self.ball_seen = True
        
        # self.get_logger().info(f"vision_callback: ball_seen={self.ball_seen}, ball_2d_x={msg.ball_2d_x}, ball_2d_y={msg.ball_2d_y}")



def main(args=None):
    rclpy.init(args=args)
    node = PanTiltNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
