import math
import json
from pathlib import Path
import unittest

import numpy as np

from scripts.convert.agibotworld2026 import (
    G2FK, ROOT, STATE_FIELDS, pose_matrix,
    camera_id, instructions, openness, seconds_to_ns, source_fields,
    verify_stationary_base,
)


class AgibotWorld2026MappingTests(unittest.TestCase):
    def test_source_time_is_used_without_fps_reconstruction(self):
        self.assertEqual(seconds_to_ns(0.03333333507180214), 33_333_335)
        with self.assertRaises(ValueError):
            seconds_to_ns(math.nan)

    def test_gripper_zero_is_visually_open_and_negative_closes(self):
        self.assertEqual(openness(0), 1)
        self.assertEqual(openness(-0.455), 0.5)
        self.assertEqual(openness(-0.91), 0)
        self.assertEqual(openness(-1), 0)
        with self.assertRaises(ValueError):
            openness(math.inf)

    def test_depth_is_excluded_and_missing_odometry_is_preserved(self):
        names = (
            'state/left_effector/position', 'state/right_effector/position',
            'state/end/arm_position', 'state/end/arm_orientation', 'state/waist/position',
            'state/robot/position', 'state/robot/orientation',
        )
        lengths = (1, 1, 6, 8, 5, 0, 0)
        fields = {}
        offset = 0
        for name, length in zip(names, lengths):
            fields[name] = {'indices': list(range(offset, offset + length))}
            offset += length
        info = {'robot_type': 'g2a', 'features': {
            'observation.state': {'field_descriptions': fields},
            'observation.images.head_depth': {'video_info': {'video.is_depth_map': True}},
            'observation.images.hand_left': {'video_info': {'video.is_depth_map': False}},
        }}
        indices, cameras = source_fields(info)
        self.assertEqual(indices['state/robot/position'], [])
        self.assertEqual(cameras, {'observation.images.hand_left': 'hand_left'})
        self.assertEqual(camera_id('observation.images.hand_left'), 'hand_left')

    def test_instruction_segments_use_sample_times_and_fill_gaps(self):
        info = {'instruction_segments': {'3': [
            {'track': 'default', 'instruction': 'Pick', 'start_frame_index': 1,
             'end_frame_index': 3},
            {'track': 'Left arm', 'instruction': 'Overlapping arm label',
             'start_frame_index': 0, 'end_frame_index': 4},
        ]}}
        times = [0, 31, 68, 102]
        self.assertEqual(instructions(info, 3, 'Whole task', times), [
            {'start_ns': 0, 'end_ns': 31, 'text': 'Whole task'},
            {'start_ns': 31, 'end_ns': 102, 'text': 'Pick'},
        ])

    def test_missing_base_pose_requires_zero_base_commands(self):
        info = {'features': {'action': {'field_descriptions': {
            'action/robot/velocity': {'indices': [1, 3]},
        }}}}
        verify_stationary_base(info, [[4, 0, 5, 0], [6, 0, 7, 0]])
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            verify_stationary_base(info, [[4, 0, 5, 0], [6, 0, 7, 0.1]])
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            verify_stationary_base(info, [[4, 0, 5, math.nan]])


@unittest.skipUnless((ROOT / 'assets/robot_models/agibot_g2/robot.json').exists(), 'G2 model assets absent')
class AgibotWorld2026PoseTests(unittest.TestCase):
    def setUp(self):
        self.fk = G2FK(ROOT / 'assets/robot_models/agibot_g2')
        self.samples = json.loads((Path(__file__).parent / 'fixtures/agibotworld2026_pose_samples.json').read_text())['samples']

    @staticmethod
    def pack(sample):
        state, indices = [], {}
        for key in STATE_FIELDS:
            indices[key] = list(range(len(state), len(state) + len(sample[key])))
            state.extend(sample[key])
        return state, indices

    def test_contract_axes_in_source_flange(self):
        np.testing.assert_array_equal(self.fk.axes[:, 0], [0, 0, 1])
        np.testing.assert_array_equal(self.fk.axes[:, 2], [1, 0, 0])
        np.testing.assert_array_equal(self.fk.axes[:, 1], np.cross(self.fk.axes[:, 2], self.fk.axes[:, 0]))
        self.assertAlmostEqual(np.linalg.det(self.fk.axes), 1)

    def test_recorded_wrist_cameras_are_rigidly_attached_to_exported_tools(self):
        # Real source rows with independently recorded wrist extrinsics. Wrong
        # arm geometry, torso omission, frame order or quaternion order break
        # this invariant as the arms and torso move through the episode.
        relative = [[], []]
        for sample in self.samples:
            state, indices = self.pack(sample)
            for i, (side, pose) in enumerate(zip(('left', 'right'), self.fk.poses(state, indices, True))):
                pose = np.asarray(pose)
                rotation = np.column_stack((pose[3:6], pose[6:9], np.cross(pose[3:6], pose[6:9])))
                prefix = f'extrinsic_end_T_hand_{side}_rgbd_aligned/'
                camera_p = np.asarray(sample[prefix + 'translation_vector'])
                camera_r = np.asarray(sample[prefix + 'rotation_matrix']).reshape(3, 3)
                relative[i].append(np.concatenate((rotation.T @ (camera_p - pose[:3]), (rotation.T @ camera_r).ravel())))
        for samples in relative:
            np.testing.assert_allclose(samples, np.broadcast_to(samples[0], np.shape(samples)), atol=2e-6, rtol=0)

    def test_mobile_base_is_composed_on_the_left(self):
        sample = self.samples[2].copy()
        state, indices = self.pack(sample)
        fixed = self.fk.poses(state, indices, True)
        sample['state/robot/position'] = [2, -1, 0.3]
        sample['state/robot/orientation'] = [0, 0, math.sqrt(0.5), math.sqrt(0.5)]
        world = pose_matrix(sample['state/robot/position'], sample['state/robot/orientation'])
        state, indices = self.pack(sample)
        for old, new in zip(fixed, self.fk.poses(state, indices)):
            np.testing.assert_allclose(new[:3], world[:3, :3] @ old[:3] + world[:3, 3])
            np.testing.assert_allclose(new[3:6], world[:3, :3] @ old[3:6])
            np.testing.assert_allclose(new[6:9], world[:3, :3] @ old[6:9])
        sample['state/end/arm_orientation'] = [0] * 8
        with self.assertRaisesRegex(ValueError, 'quaternion'):
            self.fk.poses(*self.pack(sample))


if __name__ == '__main__':
    unittest.main()
