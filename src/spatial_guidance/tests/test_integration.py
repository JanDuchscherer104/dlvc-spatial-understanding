import unittest

from src.audio_guidance.spatial_audio import SpatialAudio

from spatial_guidance.audio_guidance.audio_generator import AudioGenerator
from spatial_guidance.scene_understanding.depth_estimation import DepthEstimation
from spatial_guidance.scene_understanding.object_detection import ObjectDetection


class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.object_detector = ObjectDetection()
        self.depth_estimator = DepthEstimation()
        self.audio_generator = AudioGenerator()
        self.spatial_audio = SpatialAudio()

    def test_integration_object_detection_depth_estimation(self):
        test_image = "path/to/test/image.jpg"
        detected_objects = self.object_detector.detect(test_image)
        depth_data = self.depth_estimator.estimate_depth(test_image)

        self.assertIsNotNone(detected_objects)
        self.assertIsNotNone(depth_data)
        self.assertGreater(len(detected_objects), 0)

    def test_integration_audio_generation(self):
        test_image = "path/to/test/image.jpg"
        detected_objects = self.object_detector.detect(test_image)
        depth_data = self.depth_estimator.estimate_depth(test_image)
        audio_instructions = self.audio_generator.generate(detected_objects, depth_data)

        self.assertIsNotNone(audio_instructions)
        self.assertTrue(isinstance(audio_instructions, str))

    def test_integration_spatial_audio_rendering(self):
        test_image = "path/to/test/image.jpg"
        detected_objects = self.object_detector.detect(test_image)
        depth_data = self.depth_estimator.estimate_depth(test_image)
        audio_instructions = self.audio_generator.generate(detected_objects, depth_data)
        audio_output = self.spatial_audio.render(audio_instructions)

        self.assertIsNotNone(audio_output)
        self.assertTrue(isinstance(audio_output, bytes))


if __name__ == "__main__":
    unittest.main()
