import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
import numpy as np

class SceneGraphVisualizer(Node):
    def __init__(self):
        super().__init__('scene_graph_visualizer')
        self.bridge = CvBridge()
        self.latest_graph = None

        # Subscribe to camera image
        self.sub_img = self.create_subscription(
            Image,
            '/camera/rgbd_camera/image_raw',
            self.image_callback,
            10)

        # Subscribe to scene graph
        self.sub_graph = self.create_subscription(
            String,
            '/conceptgraph/scene_graph',
            self.graph_callback,
            10)

        self.get_logger().info('✅ Visualizer started!')

    def graph_callback(self, msg):
        self.latest_graph = json.loads(msg.data)

    def image_callback(self, msg):
        if self.latest_graph is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        for obj in self.latest_graph['objects']:
            x1 = obj['bbox']['x1']
            y1 = obj['bbox']['y1']
            x2 = obj['bbox']['x2']
            y2 = obj['bbox']['y2']
            cx = obj['position_2d']['x']
            cy = obj['position_2d']['y']
            yolo_label = obj['yolo_label']
            clip_label = obj['clip_label']
            conf = obj['yolo_confidence']

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label background
            label = f"{yolo_label}+{clip_label} ({conf})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)

            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Draw center point
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Show frame count
        frame_text = f"Frame: {self.latest_graph['frame']} | Objects: {self.latest_graph['object_count']}"
        cv2.putText(frame, frame_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('ConceptGraph Scene', frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = SceneGraphVisualizer()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()