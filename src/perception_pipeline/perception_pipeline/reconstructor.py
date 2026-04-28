import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
import cv2
import struct

class ReconstructorNode(Node):
    def __init__(self):
        super().__init__('reconstructor_node')
        self.bridge = CvBridge()

        # Camera intrinsics (Gazebo default)
        self.fx = 554.25
        self.fy = 554.25
        self.cx = 320.0
        self.cy = 240.0

        # Subscribe to depth camera
        self.sub_depth = self.create_subscription(
            Image,
            '/camera/rgbd_camera/depth/image_raw',
            self.callback,
            10)

        # Publish point cloud
        self.pub = self.create_publisher(
            PointCloud2,
            '/reconstruction/point_cloud',
            10)

        self.get_logger().info('✅ Reconstructor Node started — waiting for depth image...')

    def callback(self, msg):
        # Convert depth image
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        h, w = depth.shape

        # Generate 3D points
        points = []
        step = 8  # Sample every 8 pixels for performance
        for v in range(0, h, step):
            for u in range(0, w, step):
                z = float(depth[v, u])
                if z > 0.1 and z < 10.0 and not np.isnan(z):
                    x = (u - self.cx) * z / self.fx
                    y = (v - self.cy) * z / self.fy
                    points.append((x, y, z))

        if len(points) == 0:
            return

        # Build PointCloud2 message
        cloud_msg = PointCloud2()
        cloud_msg.header = msg.header
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.is_dense = True
        cloud_msg.is_bigendian = False
        cloud_msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg.point_step = 12
        cloud_msg.row_step = 12 * len(points)

        # Pack points
        data = []
        for x, y, z in points:
            data.append(struct.pack('fff', x, y, z))
        cloud_msg.data = b''.join(data)

        self.pub.publish(cloud_msg)
        self.get_logger().info(f'Published point cloud: {len(points)} points')

def main():
    rclpy.init()
    node = ReconstructorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()