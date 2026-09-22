"""Checks for distributable packages and source-verified forward kinematics."""
import unittest

import numpy as np

from scripts.robot.urdf_model import parse_urdf
from scripts.robot.kinematics import fk_poses
from scripts.robot.check_packages import check_package, MODELS


class ModelTests(unittest.TestCase):
    def test_yam_source_verified_pose(self):
        robot = parse_urdf(MODELS / 'yam/robot.urdf')
        poses = fk_poses(robot, {'arm_left_joint2': 1.047, 'arm_left_joint3': 1.047})
        # ABC source MJCF home pose, verified against MuJoCo before packaging.
        reference = np.array([
            [0, 0, 1, 0.62945483630618],
            [0, 1, 0, 0.31],
            [-1, 0, 0, 1.1426046253784932],
            [0, 0, 0, 1],
        ])
        np.testing.assert_allclose(poses['arm_left_grasp'], reference, atol=1e-9)

    def test_packaged_yam_finger_coupling(self):
        robot = parse_urdf(MODELS / 'yam/robot.urdf')
        zero = fk_poses(robot, {})
        opened = fk_poses(robot, {'arm_left_finger2_joint': -0.02})
        local_motion = []
        for finger in (1, 2):
            link = f'arm_left_finger{finger}_link'
            local_motion.append((np.linalg.inv(zero[link]) @ opened[link])[:3, 3])
        np.testing.assert_allclose(local_motion[0], [0, 0, 0.02], atol=1e-10)
        np.testing.assert_allclose(local_motion[1], [0, 0, -0.02], atol=1e-10)

    def test_all_distributable_packages(self):
        packages = sorted(MODELS.glob('*/robot.json'))
        self.assertEqual({p.parent.name for p in packages}, {'yam', 'agibot_g2', 'galaxea_r1lite'})
        for package in packages:
            with self.subTest(robot=package.parent.name):
                self.assertEqual(check_package(package.parent)['status'], 'passed')


if __name__ == '__main__':
    unittest.main()
