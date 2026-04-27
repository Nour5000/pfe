import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image,
            '/camera/rgbd_camera/image_raw',
            self.callback,
            10)
        self.saved = False
        self.get_logger().info('Waiting for image from robot camera...')

    def callback(self, msg):
        if not self.saved:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv2.imwrite('/home/msi/go2_ws/scripts/robot_frame.jpg', cv_image)
            self.get_logger().info('✅ Image saved to scripts/robot_frame.jpg')
            self.saved = True

def main():
    rclpy.init()
    node = ImageSaver()
    rclpy.spin_once(node, timeout_sec=10.0)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()