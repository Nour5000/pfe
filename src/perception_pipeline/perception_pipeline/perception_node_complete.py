#!/usr/bin/env python3
"""
Nœud ROS2 - Pipeline COMPLET
YOLOv8 + MobileSAM + CLIP + Projection 3D
Output: 518-dim features pour LTN (Mayssa)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import (
    Image,
    CameraInfo,
    RegionOfInterest,
    PointCloud2,
    PointField,
)
from geometry_msgs.msg import Point, Vector3
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor
import clip

from perception_msgs.msg import ObjectDetection3D, DetectionArray


class PerceptionNodeComplete(Node):

    def __init__(self):
        super().__init__("perception_node_complete")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        self.get_logger().info("Loading YOLOv8...")
        self.yolo = YOLO("yolov8n.pt")
        self.get_logger().info("✅ YOLOv8 ready")

        self.get_logger().info("Loading MobileSAM...")
        sam = sam_model_registry["vit_t"](
            checkpoint="/home/msi/models/sam/mobile_sam.pt"
        )
        sam.to(device=self.device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)
        self.get_logger().info("✅ MobileSAM ready")

        self.get_logger().info("Loading CLIP...")
        self.clip_model, self.clip_preprocess = clip.load(
            "ViT-B/32", device=self.device
        )
        self.get_logger().info("✅ CLIP ready")

        self.camera_matrix = None
        self.create_subscription(
            CameraInfo,
            "/camera/rgbd_camera/camera_info",
            self.camera_info_callback,
            10,
        )

        self.bridge = CvBridge()
        self.rgb_sub = Subscriber(self, Image, "/camera/rgbd_camera/image_raw")
        self.depth_sub = Subscriber(
            self, Image, "/camera/rgbd_camera/depth/image_raw"
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1,
        )
        self.sync.registerCallback(self.perception_callback)

        self.detections_pub = self.create_publisher(
            DetectionArray, "/perception/detections", 10
        )
        self.viz_pub = self.create_publisher(
            Image, "/perception/visualization", 10
        )

        self.frame_count = 0
        self.get_logger().info("✅ Pipeline COMPLET ready!")
        self.get_logger().info(
            "Publishing 518-dim features for LTN on /perception/detections"
        )

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
            self.get_logger().info(
                f"Camera matrix received: fx={fx:.1f}, fy={fy:.1f}, "
                f"cx={cx:.1f}, cy={cy:.1f}"
            )

    def perception_callback(self, rgb_msg: Image, depth_msg: Image):
        self.frame_count += 1
        if self.frame_count % 3 != 0:
            return

        try:
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            depth_np = np.array(depth_img, dtype=np.float32)

            results = self.yolo(rgb_img, conf=0.4, verbose=False)[0]

            if results.boxes is None or len(results.boxes) == 0:
                return

            boxes = results.boxes[:5]

            rgb_for_sam = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(rgb_for_sam)

            detection_array = DetectionArray()
            detection_array.header = rgb_msg.header

            viz = rgb_img.copy()

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_name = self.yolo.names[int(box.cls[0])]

                mask = self.segment_with_sam([x1, y1, x2, y2])
                if mask is None:
                    continue

                position, dimensions = self.project_mask_to_3d(mask, depth_np)
                if position is None:
                    continue

                clip_embedding = self.extract_clip_features(rgb_img, mask)

                detection = ObjectDetection3D()
                detection.header = rgb_msg.header
                detection.object_id = idx
                detection.class_name = class_name
                detection.confidence = confidence

                detection.bbox_2d = RegionOfInterest(
                    x_offset=x1,
                    y_offset=y1,
                    height=(y2 - y1),
                    width=(x2 - x1),
                    do_rectify=False,
                )

                mask_uint8 = (mask.astype(np.uint8) * 255)
                mask_msg = self.bridge.cv2_to_imgmsg(mask_uint8, encoding="mono8")
                mask_msg.header = rgb_msg.header
                detection.segmentation_mask = mask_msg

                detection.position = Point(
                    x=float(position[0]),
                    y=float(position[1]),
                    z=float(position[2]),
                )

                detection.dimensions = Vector3(
                    x=float(dimensions[0]),
                    y=float(dimensions[1]),
                    z=float(dimensions[2]),
                )

                detection.object_cloud = self.mask_to_pointcloud(
                    mask, depth_np, rgb_msg.header
                )

                detection.clip_embedding = clip_embedding.tolist()

                detection_array.detections.append(detection)

                color = (
                    int((idx * 80) % 255),
                    int((idx * 120 + 80) % 255),
                    int((idx * 160 + 160) % 255),
                )

                viz[mask] = (
                    viz[mask] * 0.5 + np.array(color) * 0.5
                ).astype(np.uint8)

                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(viz, contours, -1, color, 2)

                label = f"{class_name} {confidence:.0%}"
                cv2.putText(
                    viz,
                    label,
                    (x1, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                pos_text = (
                    f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})m"
                )
                cv2.putText(
                    viz,
                    pos_text,
                    (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )

                cv2.putText(
                    viz,
                    "CLIP✓ 512-dim",
                    (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )

            self.detections_pub.publish(detection_array)

            viz_msg = self.bridge.cv2_to_imgmsg(viz, encoding="bgr8")
            viz_msg.header = rgb_msg.header
            self.viz_pub.publish(viz_msg)

            self.get_logger().info(
                f"Frame {self.frame_count} — "
                f"{len(detection_array.detections)} detections "
                f"[geo(6) + CLIP(512) = 518-dim] ✅"
            )

        except Exception as e:
            self.get_logger().error(f"Error: {e}")
            import traceback

            traceback.print_exc()

    def segment_with_sam(self, bbox):
        try:
            masks, _, _ = self.sam_predictor.predict(
                box=np.array(bbox),
                multimask_output=False,
            )
            return masks[0]
        except Exception as e:
            self.get_logger().warning(f"SAM failed: {e}")
            return None

    def extract_clip_features(self, image, mask):
        try:
            masked_img = image.copy()
            masked_img[~mask] = 0

            coords = np.column_stack(np.where(mask))
            if len(coords) == 0:
                return np.zeros(512, dtype=np.float32)

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            pad = 5
            y_min = max(0, y_min - pad)
            x_min = max(0, x_min - pad)
            y_max = min(image.shape[0], y_max + pad)
            x_max = min(image.shape[1], x_max + pad)

            cropped = masked_img[y_min:y_max, x_min:x_max]
            if cropped.size == 0:
                return np.zeros(512, dtype=np.float32)

            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(cropped_rgb)

            processed = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.clip_model.encode_image(processed)

            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().flatten().astype(np.float32)

        except Exception as e:
            self.get_logger().warning(f"CLIP failed: {e}")
            return np.zeros(512, dtype=np.float32)

    def mask_to_pointcloud(self, mask, depth_img, header):
        if self.camera_matrix is None:
            fx, fy, cx, cy = 554.25, 554.25, 320.0, 240.0
        else:
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]

        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return PointCloud2()

        depths = depth_img[y_coords, x_coords]
        valid = np.isfinite(depths) & (depths > 0.1) & (depths < 10.0)

        if valid.sum() == 0:
            return PointCloud2()

        x_coords = x_coords[valid]
        y_coords = y_coords[valid]
        depths = depths[valid]

        x_points = (x_coords - cx) * depths / fx
        y_points = (y_coords - cy) * depths / fy
        z_points = depths

        points = np.column_stack([x_points, y_points, z_points]).astype(np.float32)

        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = points.shape[0]
        cloud_msg.is_dense = False
        cloud_msg.is_bigendian = False
        cloud_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg.point_step = 12
        cloud_msg.row_step = cloud_msg.point_step * points.shape[0]
        cloud_msg.data = points.tobytes()

        return cloud_msg

    def project_mask_to_3d(self, mask, depth_img):
        if self.camera_matrix is None:
            fx, fy, cx, cy = 554.25, 554.25, 320.0, 240.0
        else:
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]

        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return None, None

        depths = depth_img[y_coords, x_coords]
        valid = np.isfinite(depths) & (depths > 0.1) & (depths < 10.0)

        if valid.sum() < 10:
            return None, None

        x_coords = x_coords[valid]
        y_coords = y_coords[valid]
        depths = depths[valid]

        x_points = (x_coords - cx) * depths / fx
        y_points = (y_coords - cy) * depths / fy
        z_points = depths

        points_3d = np.column_stack([x_points, y_points, z_points])

        position = points_3d.mean(axis=0)
        dimensions = points_3d.max(axis=0) - points_3d.min(axis=0)

        return position, dimensions


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNodeComplete()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()