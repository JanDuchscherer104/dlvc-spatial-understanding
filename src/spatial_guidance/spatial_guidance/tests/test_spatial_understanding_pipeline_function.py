import unittest

from zenml.pipelines.pipeline_definition import Pipeline

from spatial_guidance.pipeline.pipeline import (
    PipelineConfig,
    PipelineIn,
    SpatialUnderstandingPipeline,
    spatial_understanding,
)


class DummyStep:
    def __init__(self):
        self.called = False

    def __call__(self, input_data):
        self.called = True
        return input_data


class TestSpatialUnderstandingPipelineFunction(unittest.TestCase):
    def test_pipeline_decorator_returns_pipeline(self):
        # The decorated function should return a Pipeline when called
        pipe = spatial_understanding
        self.assertTrue(isinstance(pipe, Pipeline))

    def test_default_config_fields(self):
        config = PipelineConfig()
        # Config should have stage configs and global settings
        self.assertTrue(hasattr(config, "dataset_config"))
        self.assertTrue(hasattr(config, "detection_config"))
        self.assertTrue(hasattr(config, "visualization_config"))
        self.assertTrue(isinstance(config.enable_cache_global, bool))
        self.assertTrue(isinstance(config.verbose, bool))

    def test_custom_pipeline_execution_flow(self):
        # Patch setup_target to use DummyStep instances and verify calls
        config = PipelineConfig()
        ds = DummyStep()
        det = DummyStep()
        vis = DummyStep()
        config.dataset_config.target = lambda cfg: ds
        config.detection_config.target = lambda cfg: det
        config.visualization_config.target = lambda cfg: vis
        # Create a barebones pipeline and run
        pip = SpatialUnderstandingPipeline(config)
        output = pip.run(idx=0, user_prompt="test")
        # Verify each stage was called
        self.assertTrue(ds.called, "Dataset stage was not called")
        self.assertTrue(det.called, "Detection stage was not called")
        self.assertTrue(vis.called, "Visualization stage was not called")
        # Output should forward output of visualization stage, i.e., the VisualizationIn instance
        from spatial_guidance.pipeline.pipeline import VisualizationIn

        self.assertIsInstance(output, VisualizationIn)
        # Ensure input propagation into VisualizationIn
        self.assertEqual(
            output.detection_output, det(ds(PipelineIn(idx=0, user_prompt="test")))
        )


if __name__ == "__main__":
    unittest.main()
