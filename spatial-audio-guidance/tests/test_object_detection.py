import unittest
from src.scene_understanding.object_detection import ObjectDetector

class TestObjectDetection(unittest.TestCase):

    def setUp(self):
        self.detector = ObjectDetector()

    def test_detect_objects(self):
        test_image = "path/to/test/image.jpg"
        detected_objects = self.detector.detect(test_image)
        self.assertIsInstance(detected_objects, list)
        self.assertGreater(len(detected_objects), 0)

    def test_detect_objects_empty_image(self):
        empty_image = "path/to/empty/image.jpg"
        detected_objects = self.detector.detect(empty_image)
        self.assertEqual(detected_objects, [])

    def test_detect_objects_invalid_input(self):
        with self.assertRaises(ValueError):
            self.detector.detect(None)

if __name__ == '__main__':
    unittest.main()