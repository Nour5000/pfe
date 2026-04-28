import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')
        
        # Subscribe to robot camera
        self.sub = self.create_subscription(
            Image,
            '/camera/rgbd_camera/image_raw',
            self.callback,
            10)
        
        # Publish annotated image
        self.pub = self.create_publisher(Image, '/yolo/image_annotated', 10)
        
        self.get_logger().info('✅ YOLO Node started — waiting for camera...')

    def callback(self, msg):
        # Convert ROS image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        # Log detections
        for r in results:
            for box in r.boxes:
                cls = self.model.names[int(box.cls)]
                conf = float(box.conf)
                self.get_logger().info(f'Detected: {cls} ({conf:.2f})')
        
        # Publish annotated image
        annotated = results[0].plot()
        ros_img = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        self.pub.publish(ros_img)

def main():
    rclpy.init()
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()