#!/usr/bin/env python3
"""
Test script - verifies perception pipeline output for LTN
"""
import rclpy
from rclpy.node import Node
from perception_msgs.msg import DetectionArray


class TestPerception(Node):
    def __init__(self):
        super().__init__('test_perception')
        self.sub = self.create_subscription(
            DetectionArray,
            '/perception/detections',
            self.callback,
            10
        )
        self.get_logger().info('Listening on /perception/detections...')

    def callback(self, msg):
        self.get_logger().info(f'\n{"="*60}')
        self.get_logger().info(f'Frame received: {len(msg.detections)} objects')

        for det in msg.detections:
            self.get_logger().info(f'\n  Object #{det.object_id}: {det.class_name}')
            self.get_logger().info(f'    Confidence : {det.confidence:.2%}')
            self.get_logger().info(
                f'    Position 3D: ({det.position.x:.2f}, '
                f'{det.position.y:.2f}, {det.position.z:.2f}) m'
            )
            self.get_logger().info(
                f'    Dimensions : ({det.dimensions.x:.2f}, '
                f'{det.dimensions.y:.2f}, {det.dimensions.z:.2f}) m'
            )

            if len(det.clip_embedding) == 512:
                norm = sum(x**2 for x in det.clip_embedding)**0.5
                self.get_logger().info(
                    f'    CLIP embedding: ✅ 512-dim (norm={norm:.3f})'
                )
                # Total features for LTN
                geo = [
                    det.position.x, det.position.y, det.position.z,
                    det.dimensions.x, det.dimensions.y, det.dimensions.z
                ]
                total_dims = len(geo) + len(det.clip_embedding)
                self.get_logger().info(
                    f'    LTN features: ✅ {total_dims}-dim (6 geo + 512 CLIP)'
                )
            else:
                self.get_logger().warning(
                    f'    CLIP embedding: ❌ {len(det.clip_embedding)}-dim'
                )


def main():
    rclpy.init()
    node = TestPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()