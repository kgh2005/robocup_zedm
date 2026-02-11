from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'robocup_zedm'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament index
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        # package.xml
        ('share/' + package_name, ['package.xml']),

        # install config/launch/model into share
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.rviz')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'model'), glob('model/*')),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='robit',
    maintainer_email='leokim0503@kw.ac.kr',
    description='ZED Mini + YOLO detection/refiner nodes for RoboCup',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            # robocup_zedm/robocup_zedm/nodes/detection_node.py 의 main() 기준
            'detection_node = robocup_zedm.nodes.detection_node:main',
            'refiner_node   = robocup_zedm.nodes.refiner_node:main',
            'pantilt_node   = robocup_zedm.nodes.pantilt_node:main',
        ],
    },
)
