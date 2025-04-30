import os
import tempfile
import unittest

import numpy as np
from PIL import Image
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

from spatial_guidance.pipeline.data_contracts import (
    DataModel,
    DetectedObject,
    DetectionStageOut,
    VisualizationIn,
    VisualizationOut,
)
from spatial_guidance.pipeline.materializer import PydanticNumpyMaterializer
from spatial_guidance.pipeline.pipeline import (
    PipelineConfig,
    PipelineIn,
    SpatialUnderstandingPipeline,
)
from spatial_guidance.pipeline.pipeline_stage import PipelineStage
from spatial_guidance.visualization.scene_visualizer import (
    SceneVisualizer,
    SceneVisualizerConfig,
)

from ..utils import PathConfig


class TestSceneVisualizerStage(unittest.TestCase):
    def test_visualizer_empty_objects(self):
        # Create a simple RGB and depth image
        rgb = Image.new("RGB", (100, 100), color="gray")
        depth = Image.new("RGB", (100, 100), color="black")
        # No objects; use proper DetectionStageOut
        vis = SceneVisualizerConfig().setup_target()
        detection_output = DetectionStageOut(objects=[])
        out = vis.entrypoint(
            VisualizationIn(
                rgb_image=rgb,
                depth_image=depth,
                detection_output=detection_output,
            )
        )
        self.assertIsInstance(out, VisualizationOut)
        self.assertEqual(out.object_count, 0)
        self.assertIsInstance(out.visualization, Image.Image)

    def test_visualizer_with_object(self):
        # Single object at center
        rgb = Image.new("RGB", (100, 100), color="white")
        depth = Image.new("RGB", (100, 100), color="white")
        # Construct a full DetectedObject
        obj = DetectedObject(
            aabb_2d=[250, 250, 750, 750],
            points_2d=[(500, 500)],
            label="test",
            approx_distance=1.0,
            height="unknown",
            description="",
            is_hazard=False,
            is_dynamic=False,
            hazard_type=None,
        )
        vis = SceneVisualizerConfig(show_debug_info=True, point_radius=5).setup_target()
        detection_output = DetectionStageOut(objects=[obj])
        out = vis.entrypoint(
            VisualizationIn(
                rgb_image=rgb,
                depth_image=depth,
                detection_output=detection_output,
            )
        )
        self.assertEqual(out.object_count, 1)
        self.assertIsInstance(out.visualization, Image.Image)


class TestPydanticNumpyMaterializer(unittest.TestCase):
    def test_save_and_load_datamodel(self):
        # Define a simple DataModel subclass
        class Model(DataModel):
            arr: np.ndarray
            img: Image.Image
            val: int

        arr = np.arange(6).reshape(2, 3)
        img = Image.new("RGB", (10, 10), color="blue")
        model = Model(arr=arr, img=img, val=7)
        # materializer with explicit URI
        with tempfile.TemporaryDirectory() as tmpdir:
            m = PydanticNumpyMaterializer(uri=tmpdir)
            m.save(model)
            # load back
            loaded = m.load(Model)
        self.assertIsInstance(loaded, Model)
        self.assertTrue((loaded.arr == arr).all())
        self.assertIsInstance(loaded.img, Image.Image)
        self.assertEqual(loaded.val, 7)


class TestFullPipelineRun(unittest.TestCase):
    def test_pipeline_run_sequence(self):
        # Ensure the default dataset_dir exists so PathConfig validation passes
        pc = PathConfig()
        dummy_dir = pc.root / pc.data / "SmartAIs Recorded Data/5"
        dummy_dir.mkdir(parents=True, exist_ok=True)

        # Dummy steps that return the input data directly
        class DummyStep(PipelineStage):
            def entrypoint(self, input_data):
                return input_data

        # Setup config and override all stages to DummyStep
        config = PipelineConfig()
        ds = DummyStep(config.dataset_config)
        det = DummyStep(config.detection_config)
        vis = DummyStep(config.visualization_config)
        config.dataset_config.target = lambda cfg: ds
        config.detection_config.target = lambda cfg: det
        config.visualization_config.target = lambda cfg: vis

        # Run the pipeline
        pipe = SpatialUnderstandingPipeline(config)
        output = pipe.run(idx=5, user_prompt="hello")

        # Verify each stage was called
        self.assertTrue(hasattr(ds, "config"))
        self.assertTrue(hasattr(det, "config"))
        self.assertTrue(hasattr(vis, "config"))
        # Since DummyStep echoes the input, output should be the PipelineIn instance
        from spatial_guidance.pipeline.pipeline import PipelineIn

        self.assertIsInstance(output, PipelineIn)
        self.assertEqual(output.idx, 5)
        self.assertEqual(output.user_prompt, "hello")


if __name__ == "__main__":
    unittest.main()
