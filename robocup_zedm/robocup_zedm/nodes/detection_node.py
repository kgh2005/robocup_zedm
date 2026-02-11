#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np

from ultralytics import YOLO

from vision_interfaces.msg import BoundingBox


CONFIDENCE_THRESHOLDS = {
    0: 0.5, # 공
    1: 0.5, # 로봇
    2: 0.5, # ㄱ
    3: 0.5, # t
    4: 0.5, # x
}

COLORS = {
    0: (0, 165, 255),   # 주황색 (Orange)
    1: (0, 255, 255),   # 노란색 (Yellow)
    2: (255, 0, 255),   # 보라색 (Purple)
    3: (0, 0, 255),     # 빨간색 (Red)
    4: (0, 255, 0),     # 초록색 (Green)
}

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        # ---- parameters ----
        self.declare_parameter('model_path', '')
        self.declare_parameter('rgb_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('device', 'cuda:0')      # 'cuda:0' / 'cpu'
        self.declare_parameter('imgsz', 640)            # YOLO 입력 크기
        self.declare_parameter('show_debug', True)

        self.model_path = self.get_parameter('model_path').value
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.device = self.get_parameter('device').value
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.show_debug = bool(self.get_parameter('show_debug').value)

        if not self.model_path:
            self.get_logger().fatal("Parameter 'model_path' is empty.")
            raise RuntimeError("model_path is required")

        # ---- YOLO (Ultralytics) ----
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info(f"Loaded YOLO model: {self.model_path} (device={self.device})")
        except Exception as e:
            self.get_logger().fatal(f"YOLO load failed: {e}")
            raise

        self.bridge = CvBridge()
        self.busy = False  # 프레임 드롭(원하면 제거 가능)

        self.pub = self.create_publisher(BoundingBox, "/Bounding_box", 1)
        self.sub = self.create_subscription(Image, self.rgb_topic, self.image_callback, 1)

        # ball best
        self.ball_best_conf = 0.0
        self.ball_best = None  # (cls, conf, x1,y1,x2,y2)

    def image_callback(self, msg: Image):
        if self.busy:
            return
        self.busy = True

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gry = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            self.process_image(bgr, gry)
        except Exception as e:
            self.get_logger().error(f"image_callback error: {e}")
        finally:
            self.busy = False

    def process_image(self, bgr: np.ndarray, gry: np.ndarray = None):
        if bgr is None or bgr.size == 0:
            return
        
        if gry is None or gry.size == 0:
            return

        results = self.model.predict(
            source=gry,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        if not results:
            return

        r = results[0]
        boxes = getattr(r, 'boxes', None)

        bbox_msg = BoundingBox()
        self.ball_best_conf = 0.0
        self.ball_best = None

        if boxes is not None and len(boxes) > 0:
            # xyxy: (N,4), conf: (N,), cls: (N,)
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            clss = boxes.cls.detach().cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
                thr = CONFIDENCE_THRESHOLDS.get(int(cls), 0.5)
                if float(conf) < thr:
                    continue

                bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)
                if bx2 <= bx1 or by2 <= by1:
                    continue

                if int(cls) == 0:
                    if float(conf) > self.ball_best_conf:
                        self.ball_best_conf = float(conf)
                        self.ball_best = (int(cls), float(conf), bx1, by1, bx2, by2)
                else:
                    bbox_msg.class_ids.append(int(cls))
                    bbox_msg.score.append(float(conf))
                    bbox_msg.x1.append(bx1)
                    bbox_msg.y1.append(by1)
                    bbox_msg.x2.append(bx2)
                    bbox_msg.y2.append(by2)

                    if self.show_debug:
                        cv2.rectangle(bgr, (bx1, by1), (bx2, by2), COLORS.get(int(cls), (255, 255, 255)), 2)

        # ball best 추가
        if self.ball_best is not None:
            cls, conf, bx1, by1, bx2, by2 = self.ball_best
            bbox_msg.class_ids.append(cls)
            bbox_msg.score.append(conf)
            bbox_msg.x1.append(bx1)
            bbox_msg.y1.append(by1)
            bbox_msg.x2.append(bx2)
            bbox_msg.y2.append(by2)

            if self.show_debug:
                cv2.rectangle(bgr, (bx1, by1), (bx2, by2), COLORS.get(cls, (255, 255, 255)), 2)

        self.pub.publish(bbox_msg)

        if self.show_debug:
            cv2.imshow("Detection", bgr)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
