std_msgs/Header header

uint32 object_id
string class_name
float32 confidence

# 2D detection
sensor_msgs/RegionOfInterest bbox_2d
sensor_msgs/Image segmentation_mask

# 3D information
geometry_msgs/Point position
geometry_msgs/Vector3 dimensions
sensor_msgs/PointCloud2 object_cloud

# Semantic features
float32[] clip_embedding