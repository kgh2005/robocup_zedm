#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import time
import cv2
import numpy as np

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from ultralytics import YOLO
from vision_interfaces.msg import BoundingBox


CONFIDENCE_THRESHOLDS = {
    0: 0.5,  # 공
    1: 0.5,  # 로봇
    2: 0.5,  # ㄱ
    3: 0.5,  # t
    4: 0.5,  # x
}

COLORS = {
    0: (0, 165, 255),   # 주황
    1: (0, 255, 255),   # 노랑
    2: (255, 0, 255),   # 보라
    3: (0, 0, 255),     # 빨강
    4: (0, 255, 0),     # 초록
}


class DetectionNode(Node):
    def __init__(self):
        super().__init__("detection_node")

        # ---- parameters ----
        self.declare_parameter("model_path", "")
        self.declare_parameter("rgb_topic", "/zed/zed_node/rgb/image_rect_color")
        self.declare_parameter("device", "0")             # "0" / "cpu"
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("half", True)
        self.declare_parameter("bbox_topic", "/Bounding_box")

        # 디버그: imshow는 기본 OFF 권장
        self.declare_parameter("show_imshow", False)
        self.declare_parameter("debug_topic", "/debug_image")
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("debug_rate_hz", 10.0)

        self.model_path = self.get_parameter("model_path").value
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.device = str(self.get_parameter("device").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.half = bool(self.get_parameter("half").value)
        self.bbox_topic = self.get_parameter("bbox_topic").value

        self.show_imshow = bool(self.get_parameter("show_imshow").value)
        self.debug_topic = self.get_parameter("debug_topic").value
        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        self.debug_rate_hz = float(self.get_parameter("debug_rate_hz").value)

        if not self.model_path:
            raise RuntimeError("model_path is required")

        # ---- YOLO ----
        self.model = YOLO(self.model_path)
        try:
            self.model.fuse()
        except Exception:
            pass

        self.bridge = CvBridge()

        # ---- pubs ----
        self.pub = self.create_publisher(BoundingBox, self.bbox_topic, 1)
        self.debug_pub = self.create_publisher(Image, self.debug_topic, 1)

        self.sub = self.create_subscription(Image, self.rgb_topic, self.image_callback, 1)

        # ---- state ----
        self.busy = False  # 콜백 중첩 방지(지연 누적 방지)
        self._last_debug_pub = 0.0

    def image_callback(self, msg: Image):
        # 추론이 밀리면 최신 프레임만 처리하고 나머지는 드랍
        if self.busy:
            return
        self.busy = True

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if bgr is None or bgr.size == 0:
                return

            # 1채널 입력
            gry = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gry1 = gry[..., None]  # (H,W,1)

            results = self.model.predict(
                source=gry1,
                imgsz=self.imgsz,
                device=self.device,
                half=self.half,
                verbose=False,
            )
            if not results:
                return

            r = results[0]
            boxes = getattr(r, "boxes", None)

            bbox_msg = BoundingBox()

            ball_best_conf = 0.0
            ball_best = None

            # 디버그 그리기(필요할 때만 copy)
            dbg = None
            need_draw = self.show_imshow or self.publish_debug
            if need_draw:
                dbg = bgr.copy()

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                clss = boxes.cls.detach().cpu().numpy().astype(int)

                for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
                    if float(conf) < CONFIDENCE_THRESHOLDS.get(int(cls), 0.5):
                        continue

                    bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)
                    if bx2 <= bx1 or by2 <= by1:
                        continue

                    if int(cls) == 0:
                        if float(conf) > ball_best_conf:
                            ball_best_conf = float(conf)
                            ball_best = (int(cls), float(conf), bx1, by1, bx2, by2)
                    else:
                        bbox_msg.class_ids.append(int(cls))
                        bbox_msg.score.append(float(conf))
                        bbox_msg.x1.append(bx1)
                        bbox_msg.y1.append(by1)
                        bbox_msg.x2.append(bx2)
                        bbox_msg.y2.append(by2)

                        if dbg is not None:
                            cv2.rectangle(dbg, (bx1, by1), (bx2, by2), COLORS.get(int(cls), (255, 255, 255)), 2)

            # ball best 추가
            if ball_best is not None:
                cls, conf, bx1, by1, bx2, by2 = ball_best
                bbox_msg.class_ids.append(cls)
                bbox_msg.score.append(conf)
                bbox_msg.x1.append(bx1)
                bbox_msg.y1.append(by1)
                bbox_msg.x2.append(bx2)
                bbox_msg.y2.append(by2)

                if dbg is not None:
                    cv2.rectangle(dbg, (bx1, by1), (bx2, by2),  COLORS.get(cls, (255, 255, 255)), 2)

            # publish bbox
            self.pub.publish(bbox_msg)

            # 디버그: imshow (권장: OFF)
            if self.show_imshow and dbg is not None:
                cv2.imshow("Detection", dbg)
                cv2.waitKey(1)

            # 디버그: 토픽 publish (rate limit)
            if self.publish_debug and dbg is not None:
                now = time.time()
                if now - self._last_debug_pub >= (1.0 / max(1.0, self.debug_rate_hz)):
                    dbg_msg = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
                    self.debug_pub.publish(dbg_msg)
                    self._last_debug_pub = now

        except Exception as e:
            self.get_logger().error(f"image_callback error: {e}")
        finally:
            self.busy = False


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


if __name__ == "__main__":
    main()
