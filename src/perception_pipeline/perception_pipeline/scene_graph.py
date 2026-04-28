import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import clip
import json
from PIL import Image as PILImage
from ultralytics import YOLO

class SceneGraphNode(Node):
    def __init__(self):
        super().__init__('scene_graph_node')
        self.bridge = CvBridge()

        # Load YOLO
        self.get_logger().info('Loading YOLO...')
        self.yolo = YOLO('yolov8n.pt')

        # Load CLIP
        self.get_logger().info('Loading CLIP...')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_model, self.preprocess = clip.load('ViT-B/32', device=self.device)

        # Hospital labels for CLIP
        self.labels = [
            'person', 'bed', 'chair', 'table', 'door',
            'window', 'wheelchair', 'monitor', 'cabinet', 'floor'
        ]
        text_tokens = clip.tokenize(self.labels).to(self.device)
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        # Subscribe to camera
        self.sub = self.create_subscription(
            Image,
            '/camera/rgbd_camera/image_raw',
            self.callback,
            10)

        # Publish scene graph
        self.pub = self.create_publisher(String, '/conceptgraph/scene_graph', 10)

        self.frame_count = 0
        self.get_logger().info('✅ Scene Graph Node started!')

    def callback(self, msg):
        # Only process every 10 frames for performance
        self.frame_count += 1
        if self.frame_count % 10 != 0:
            return

        # Convert image
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        # YOLO detections
        yolo_results = self.yolo(frame, verbose=False)
        objects = []
        for r in yolo_results:
            for i, box in enumerate(r.boxes):
                cls_name = self.yolo.names[int(box.cls)]
                conf = float(box.conf)
                if conf < 0.3:
                    continue

                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Crop region for CLIP
                crop = pil_img.crop((x1, y1, x2, y2))
                if crop.size[0] < 10 or crop.size[1] < 10:
                    continue

                # CLIP embedding on crop
                img_input = self.preprocess(crop).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    img_feat = self.clip_model.encode_image(img_input)
                    img_feat /= img_feat.norm(dim=-1, keepdim=True)
                sim = (100.0 * img_feat @ self.text_features.T).softmax(dim=-1)
                top_val, top_idx = sim[0].topk(1)
                clip_label = self.labels[top_idx[0]]
                clip_score = top_val[0].item()

                objects.append({
                    'id': i,
                    'yolo_label': cls_name,
                    'yolo_confidence': round(conf, 2),
                    'clip_label': clip_label,
                    'clip_score': round(clip_score, 2),
                    'position_2d': {'x': cx, 'y': cy},
                    'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                })

        # Build scene graph
        scene_graph = {
            'frame': self.frame_count,
            'object_count': len(objects),
            'objects': objects
        }

        # Publish as JSON
        msg_out = String()
        msg_out.data = json.dumps(scene_graph, indent=2)
        self.pub.publish(msg_out)

        # Log summary
        self.get_logger().info(f'--- Scene Graph (frame {self.frame_count}) ---')
        self.get_logger().info(f'Objects detected: {len(objects)}')
        for obj in objects:
            self.get_logger().info(
                f'  [{obj["id"]}] YOLO:{obj["yolo_label"]}({obj["yolo_confidence"]}) '
                f'CLIP:{obj["clip_label"]}({obj["clip_score"]}%) '
                f'pos:({obj["position_2d"]["x"]},{obj["position_2d"]["y"]})'
            )

def main():
    rclpy.init()
    node = SceneGraphNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()