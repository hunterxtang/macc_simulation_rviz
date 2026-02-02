import os 
from glob import glob
from setuptools import find_packages, setup

package_name = 'macc_rviz'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/macc_rviz']),
    	('share/macc_rviz', ['package.xml']),
    	(os.path.join('share', 'macc_rviz', 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hunter',
    maintainer_email='hunter@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
		'macc_rviz_sim = macc_rviz.macc_rviz_sim:main'
        ],
    },
)
