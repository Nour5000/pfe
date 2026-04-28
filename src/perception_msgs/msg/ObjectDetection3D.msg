# ObjectDetection3D.msg
std_msgs/Header header

# Identification
uint32 object_id
string class_name
float32 confidence

# 2D Detection
sensor_msgs/RegionOfInterest bbox_2d

# 3D Position and dimensions
geometry_msgs/Point position
geometry_msgs/Vector3 dimensions

# Semantic features (512-dim CLIP embedding)
float32[] clip_embedding