from setuptools import setup
import os
from glob import glob

package_name = 'demonstration3'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gowri',
    maintainer_email='gowrishankar2001@hotmail.com',
    description='Line follower package',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'line_follower = demonstration3.line_follower:main',
        ],
    },
)
