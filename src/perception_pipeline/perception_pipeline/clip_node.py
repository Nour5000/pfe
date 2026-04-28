import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import clip
from PIL import Image as PILImage

class ClipNode(Node):
    def __init__(self):
        super().__init__('clip_node')
        self.bridge = CvBridge()

        # Load CLIP model
        self.get_logger().info('Loading CLIP model...')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.preprocess = clip.load('ViT-B/32', device=self.device)
        self.get_logger().info(f'✅ CLIP loaded on {self.device}')

        # Hospital-relevant object labels
        self.labels = [
            'person', 'bed', 'chair', 'table', 'door',
            'window', 'wheelchair', 'monitor', 'cabinet', 'floor'
        ]

        # Precompute text embeddings
        text_tokens = clip.tokenize(self.labels).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        # Subscribe to camera
        self.sub = self.create_subscription(
            Image,
            '/camera/rgbd_camera/image_raw',
            self.callback,
            10)

        self.get_logger().info('✅ CLIP Node started — waiting for camera...')

    def callback(self, msg):
        # Convert to PIL image
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        # Preprocess and encode image
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        # Compute similarity with all labels
        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(3)

        # Log top 3 matches
        self.get_logger().info('--- CLIP Top 3 matches ---')
        for value, index in zip(values, indices):
            self.get_logger().info(f'  {self.labels[index]:15s} {value.item():.1f}%')

def main():
    rclpy.init()
    node = ClipNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()