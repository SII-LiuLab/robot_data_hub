import math
import unittest

from scripts.convert.agibotworld2026 import (
    camera_id, instructions, openness, seconds_to_ns, source_fields,
    verify_stationary_base,
)


class AgibotWorld2026MappingTests(unittest.TestCase):
    def test_source_time_is_used_without_fps_reconstruction(self):
        self.assertEqual(seconds_to_ns(0.03333333507180214), 33_333_335)
        with self.assertRaises(ValueError):
            seconds_to_ns(math.nan)

    def test_gripper_negative_opening_and_nonfinite_rejection(self):
        self.assertEqual(openness(0), 0)
        self.assertEqual(openness(-0.455), 0.5)
        self.assertEqual(openness(-0.91), 1)
        self.assertEqual(openness(-1), 1)
        with self.assertRaises(ValueError):
            openness(math.inf)

    def test_depth_is_excluded_and_missing_odometry_is_preserved(self):
        names = (
            'state/left_effector/position', 'state/right_effector/position',
            'state/joint/position', 'state/waist/position',
            'state/robot/position', 'state/robot/orientation',
        )
        lengths = (1, 1, 14, 5, 0, 0)
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


if __name__ == '__main__':
    unittest.main()
