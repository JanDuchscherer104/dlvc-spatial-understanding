import unittest
from types import SimpleNamespace
from unittest.mock import patch

from spatial_guidance.pipeline.data_contracts import DataModel, DatasetOut
from spatial_guidance.pipeline.pipeline import SpatialUnderstandingPipeline
from spatial_guidance.pipeline.pipeline_stage import PipelineStage


class DummyOutput:
    pass


class DummyFinalStep:
    def __init__(self, output):
        self.output = SimpleNamespace(load=lambda: output)


class DummyRun:
    def __init__(self, steps):
        self.steps = steps


class DummyPipelineModel:
    def __init__(self, last_run):
        self.last_run = last_run


class TestGetLatestOutput(unittest.TestCase):
    def test_no_runs(self):
        dummy_client = SimpleNamespace(
            get_pipeline=lambda name: DummyPipelineModel(None)
        )
        with patch(
            "spatial_guidance.pipeline.pipeline.Client", return_value=dummy_client
        ):
            with self.assertRaises(RuntimeError) as cm:
                SpatialUnderstandingPipeline.get_latest_output()
            self.assertIn("No runs found", str(cm.exception))

    def test_no_steps(self):
        dummy_run = DummyRun(steps={})
        dummy_client = SimpleNamespace(
            get_pipeline=lambda name: DummyPipelineModel(dummy_run)
        )
        with patch(
            "spatial_guidance.pipeline.pipeline.Client", return_value=dummy_client
        ):
            with self.assertRaises(RuntimeError) as cm:
                SpatialUnderstandingPipeline.get_latest_output()
            self.assertIn("No steps found", str(cm.exception))

    def test_success(self):
        expected = DummyOutput()
        final_step = DummyFinalStep(expected)
        steps = {"first_step": SimpleNamespace(), "final_step": final_step}
        dummy_run = DummyRun(steps=steps)
        dummy_client = SimpleNamespace(
            get_pipeline=lambda name: DummyPipelineModel(dummy_run)
        )
        with patch(
            "spatial_guidance.pipeline.pipeline.Client", return_value=dummy_client
        ):
            output = SpatialUnderstandingPipeline.get_latest_output()
            self.assertIs(output, expected)


if __name__ == "__main__":
    unittest.main()
