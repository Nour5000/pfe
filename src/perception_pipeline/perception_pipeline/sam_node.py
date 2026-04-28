import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
from mobile_sam import sam_model_registry, SamPredictor

class SamNode(Node):
    def __init__(self):
        super().__init__('sam_node')
        self.bridge = CvBridge()

        # Load MobileSAM model
        self.get_logger().info('Loading MobileSAM model...')
        model_type = 'vit_t'
        checkpoint = '/home/msi/models/sam/mobile_sam.pt'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        self.device = device
        self.get_logger().info(f'✅ MobileSAM loaded on {device}')

        # Subscribe to camera
        self.sub = self.create_subscription(
            Image,
            '/camera/rgbd_camera/image_raw',
            self.callback,
            10)

        # Publish segmentation mask
        self.pub = self.create_publisher(Image, '/sam/segmentation', 10)

        self.get_logger().info('✅ SAM Node started — waiting for camera...')

    def callback(self, msg):
        # Convert to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Set image in predictor
        self.predictor.set_image(rgb)

        # Use center point as prompt
        h, w = rgb.shape[:2]
        center = np.array([[w // 2, h // 2]])
        labels = np.array([1])

        # Generate mask
        masks, scores, _ = self.predictor.predict(
            point_coords=center,
            point_labels=labels,
            multimask_output=False
        )

        # Visualize mask
        mask = masks[0].astype(np.uint8) * 255
        colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.7, colored, 0.3, 0)

        # Publish
        ros_img = self.bridge.cv2_to_imgmsg(overlay, 'bgr8')
        self.pub.publish(ros_img)

        self.get_logger().info(f'Mask generated — score: {scores[0]:.2f}')

def main():
    rclpy.init()
    node = SamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()