#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from perception_msgs.msg import DetectionArray
from sensor_msgs.msg import PointCloud2


class ObjectCloudExtractor(Node):

    def __init__(self):
        super().__init__('object_cloud_extractor')

        self.sub = self.create_subscription(
            DetectionArray,
            '/perception/detections',
            self.callback,
            10
        )

        self.pub = self.create_publisher(
            PointCloud2,
            '/perception/object_cloud',
            10
        )

        self.get_logger().info('Publishing object cloud to /perception/object_cloud')

    def callback(self, msg):
        if len(msg.detections) == 0:
            return

        cloud = msg.detections[0].object_cloud
        cloud.header = msg.header
        self.pub.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectCloudExtractor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()