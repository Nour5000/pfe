from setuptools import setup

package_name = 'perception_pipeline'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='msi',
    maintainer_email='msi@todo.todo',
    description='ConceptGraphs perception pipeline for Go2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'yolo_node = perception_pipeline.yolo_node:main',
            'sam_node = perception_pipeline.sam_node:main',
            'clip_node = perception_pipeline.clip_node:main',
            'reconstructor = perception_pipeline.reconstructor:main',
            'scene_graph = perception_pipeline.scene_graph:main',
            'perception_node_complete = perception_pipeline.perception_node_complete:main',
        ],
    },
)
