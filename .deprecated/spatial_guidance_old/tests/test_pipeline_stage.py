import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from zenml import pipeline
from zenml.config import DockerSettings

from spatial_guidance.pipeline.data_contracts import DataModel
from spatial_guidance.pipeline.materializer import PydanticNumpyMaterializer
from spatial_guidance.pipeline.pipeline_stage import PipelineStage, StepConfig


# Define test data models
class TextInput(DataModel):
    """Simple text input model."""

    text: str
    metadata: Optional[Dict[str, str]] = None


class ProcessedText(DataModel):
    """Text with processing results."""

    original_text: str
    word_count: int
    char_count: int
    is_processed: bool = False


class ImageWithText(DataModel):
    """Combined image and text data model."""

    text_data: ProcessedText
    image_data: np.ndarray
    label: str


class ChartOutput(DataModel):
    """Output containing a chart image."""

    chart_image: np.ndarray
    summary_text: str
    metrics: Dict[str, float]


# Define test pipeline stages
class TextProcessingStage(PipelineStage[TextInput, ProcessedText]):
    """Simple text processing stage that counts words and characters."""

    def entrypoint(self, input_data: TextInput) -> ProcessedText:
        original_text = input_data.text
        word_count = len(original_text.split())
        char_count = len(original_text)

        # Store results for test verification
        test_results["text_processing"] = {
            "original_text": original_text,
            "word_count": word_count,
            "char_count": char_count,
        }

        return ProcessedText(
            original_text=original_text,
            word_count=word_count,
            char_count=char_count,
            is_processed=True,
        )


class TextProcessingStageConfig(StepConfig):
    """Configuration for the text processing stage."""

    target: type = TextProcessingStage
    capitalize_output: bool = False


# Register the default config for TextProcessingStage
TextProcessingStage.register_config(TextProcessingStageConfig())


class DockerizedAnalysisStage(PipelineStage[TextInput, ProcessedText]):
    """Text processing stage intended to run in Docker."""

    def entrypoint(self, input_data: TextInput) -> ProcessedText:
        # Record environment details that might differ in the container
        environment_info = {
            "docker": "true",  # Will be set if running in Docker
            "os_info": os.uname().sysname if hasattr(os, "uname") else "unknown",
        }

        text = input_data.text
        word_count = len(text.split())
        char_count = len(text)

        # Store in test_results for verification
        test_results["docker_test"] = {
            "environment": environment_info,
            "text": text,
            "word_count": word_count,
        }

        return ProcessedText(
            original_text=text,
            word_count=word_count,
            char_count=char_count,
            is_processed=True,
        )


class DockerizedAnalysisConfig(StepConfig):
    """Configuration for dockerized analysis stage."""

    target: type = DockerizedAnalysisStage
    docker_settings: DockerSettings = DockerSettings(
        parent_image="zenmldocker/zenml:0.80.1-py3.11"
    )


# Register the default config for DockerizedAnalysisStage
DockerizedAnalysisStage.register_config(DockerizedAnalysisConfig())


# Classes that were previously defined inside test methods - moved to module level
class ConfigAwareStage(PipelineStage[TextInput, ProcessedText]):
    """Custom stage that uses its configuration."""

    def entrypoint(self, input_data: TextInput) -> ProcessedText:
        text = input_data.text
        if hasattr(self.config, "capitalize_output") and self.config.capitalize_output:
            text = text.upper()

        result = ProcessedText(
            original_text=text,
            word_count=len(text.split()),
            char_count=len(text),
            is_processed=True,
        )

        # Store config details for test verification
        test_results["config_test"] = {
            "text": text,
            "capitalize_was_applied": text.isupper(),
        }

        return result


class ConfigAwareStageConfig(StepConfig):
    """Configuration with custom settings."""

    target: type = ConfigAwareStage
    capitalize_output: bool = True


# Register the default config for ConfigAwareStage
ConfigAwareStage.register_config(ConfigAwareStageConfig())


class DefaultStage(PipelineStage[TextInput, ProcessedText]):
    """Stage that runs in default environment."""

    def entrypoint(self, input_data: TextInput) -> ProcessedText:
        test_results["default_env"] = {"text": input_data.text}
        return ProcessedText(
            original_text=input_data.text,
            word_count=len(input_data.text.split()),
            char_count=len(input_data.text),
            is_processed=True,
        )


class DefaultStageConfig(StepConfig):
    """Config for default environment."""

    target: type = DefaultStage


# Register the default config for DefaultStage
DefaultStage.register_config(DefaultStageConfig())


class AnotherDockerConfig(StepConfig):
    """Config for another Docker environment."""

    target: type = DockerizedAnalysisStage
    docker_settings: DockerSettings = DockerSettings(
        parent_image="zenmldocker/zenml:0.70.0-py3.9"  # Different version
    )


class ImageGenerationStage(PipelineStage[ProcessedText, ImageWithText]):
    """Stage that generates an image based on text analysis."""

    def entrypoint(self, input_data: ProcessedText) -> ImageWithText:
        # Create a simple visualization based on word count
        width = min(500, input_data.word_count * 50)
        height = 200

        # Generate simple bar chart as numpy array
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Add a simple bar representing word count (blue bar)
        bar_height = int(input_data.word_count * 15)
        if bar_height > 0:
            image[-bar_height:, 50:150, 0] = 50
            image[-bar_height:, 50:150, 2] = 200

        # Add another bar for character count (green bar)
        char_bar_height = min(height - 10, int(input_data.char_count))
        if char_bar_height > 0:
            image[-char_bar_height:, 200:300, 1] = 200

        output = ImageWithText(
            text_data=input_data,
            image_data=image,
            label=f"Analysis of '{input_data.original_text[:10]}...'",
        )

        # Store data for verification
        test_results["image_generation"] = {
            "image_shape": image.shape,
            "label": output.label,
            "original_text": input_data.original_text,
        }

        return output


class ImageGenerationStageConfig(StepConfig):
    """Configuration for the image generation stage."""

    target: type = ImageGenerationStage
    chart_type: str = "bar"
    color_scheme: str = "default"


# Register the default config for ImageGenerationStage
ImageGenerationStage.register_config(ImageGenerationStageConfig())


class ChartFinalizerStage(PipelineStage[ImageWithText, ChartOutput]):
    """Final stage that produces chart output with metrics."""

    def entrypoint(self, input_data: ImageWithText) -> ChartOutput:
        # Add decorative elements to the chart
        image = input_data.image_data.copy()

        # Add a border
        image[0:3, :, :] = 255
        image[-3:, :, :] = 255
        image[:, 0:3, :] = 255
        image[:, -3:, :] = 255

        # Calculate some metrics
        metrics = {
            "words_per_char": input_data.text_data.word_count
            / max(1, input_data.text_data.char_count),
            "avg_word_length": input_data.text_data.char_count
            / max(1, input_data.text_data.word_count),
        }

        result = ChartOutput(
            chart_image=image,
            summary_text=f"Analysis complete for text with {input_data.text_data.word_count} words",
            metrics=metrics,
        )

        # Store data for verification
        test_results["chart_finalizer"] = {
            "metrics": metrics,
            "summary": result.summary_text,
        }

        return result


class ChartFinalizerStageConfig(StepConfig):
    """Configuration for the chart finalizer stage."""

    target: type = ChartFinalizerStage
    add_title: bool = True
    enable_cache: bool = False  # Override the default cache behavior


# Register the default config for ChartFinalizerStage
ChartFinalizerStage.register_config(ChartFinalizerStageConfig())


# Global storage for test results
test_results: Dict[str, Any] = {}


class TestPipelineStage(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test artifacts
        self.test_dir = tempfile.mkdtemp()
        os.environ["ZENML_ANALYTICS_OPT_IN"] = "false"
        test_results.clear()

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_single_stage_execution(self):
        """Test execution of a single pipeline stage."""

        @pipeline(enable_cache=False)
        def single_stage_pipeline():
            # Create a stage with default configuration
            config = TextProcessingStageConfig()
            stage = config.setup_target()

            # Create input
            input_data = TextInput(text="This is a test sentence for processing.")

            # Process and capture the output for verification
            stage(input_data)

            # No need to return or access the result attribute directly
            # The stage will store results in the test_results dictionary

        # Run the pipeline
        single_stage_pipeline()

        # Verify results through our global test_results dictionary
        self.assertIn("text_processing", test_results)
        self.assertEqual(
            test_results["text_processing"]["word_count"], 7
        )  # 7 words in test sentence
        self.assertEqual(
            test_results["text_processing"]["char_count"], 39
        )  # 39 chars including spaces

    def test_multi_stage_pipeline(self):
        """Test a complete multi-stage pipeline."""

        @pipeline(enable_cache=False)
        def text_visualization_pipeline():
            # Configure and create stages
            text_processor = TextProcessingStageConfig().setup_target()
            image_generator = ImageGenerationStageConfig().setup_target()
            chart_finalizer = ChartFinalizerStageConfig(
                enable_cache=False
            ).setup_target()

            # Input data
            text_input = TextInput(
                text="The quick brown fox jumps over the lazy dog.",
                metadata={"source": "test", "priority": "high"},
            )

            # Execute the pipeline stages
            processed_text = text_processor(text_input)
            image_with_text = image_generator(processed_text)
            chart_finalizer(image_with_text)

            # We don't directly access the outputs - our stages store data
            # in test_results

        # Run the pipeline
        text_visualization_pipeline()

        # Verify results using our global test_results dictionary
        self.assertIn("text_processing", test_results)
        self.assertIn("image_generation", test_results)
        self.assertIn("chart_finalizer", test_results)

        self.assertEqual(test_results["text_processing"]["word_count"], 9)
        self.assertEqual(test_results["text_processing"]["char_count"], 44)
        self.assertIsInstance(test_results["image_generation"]["image_shape"], tuple)
        self.assertGreater(
            test_results["image_generation"]["image_shape"][1], 0
        )  # Width > 0
        self.assertAlmostEqual(
            test_results["chart_finalizer"]["metrics"]["avg_word_length"],
            44 / 9,
            places=2,
        )

    def test_dockerized_stage_configuration(self):
        """Test setting up a stage with Docker configuration."""

        # Create the dockerized stage config
        docker_config = DockerizedAnalysisConfig()

        # Verify docker settings
        self.assertIsNotNone(docker_config.docker_settings)
        self.assertEqual(
            docker_config.docker_settings.parent_image,
            "zenmldocker/zenml:0.80.1-py3.11",
        )

        # Check that docker settings are properly included in step kwargs
        step_kwargs = docker_config.get_step_kwargs()
        self.assertIn("settings", step_kwargs)
        self.assertIn("docker", step_kwargs["settings"])
        self.assertEqual(
            step_kwargs["settings"]["docker"].parent_image,
            "zenmldocker/zenml:0.80.1-py3.11",
        )

        # Create the stage
        docker_stage = docker_config.setup_target()

        # The actual stage should have the config with Docker settings
        self.assertEqual(
            docker_stage.config.docker_settings.parent_image,
            "zenmldocker/zenml:0.80.1-py3.11",
        )

    def test_dockerized_stage_in_pipeline(self):
        """Test a pipeline with a dockerized stage."""

        @pipeline(enable_cache=False)
        def docker_pipeline():
            # Create dockerized stage
            docker_config = DockerizedAnalysisConfig()
            docker_stage = docker_config.setup_target()

            # Create input
            input_data = TextInput(
                text="This text will be processed in a Docker container."
            )

            # Execute dockerized stage
            docker_stage(input_data)

            # Data is stored in test_results, not accessed here directly

        # Run the pipeline
        docker_pipeline()

        # Verify the result was produced (Docker may not be available in test env)
        self.assertIn("docker_test", test_results)
        self.assertEqual(test_results["docker_test"]["word_count"], 9)

    def test_configuration_propagation(self):
        """Test that configuration parameters are properly propagated to stages."""

        @pipeline(enable_cache=False)
        def config_test_pipeline():
            # Create configured stage
            stage_config = ConfigAwareStageConfig(capitalize_output=True)
            stage = stage_config.setup_target()

            # Process input
            input_data = TextInput(text="This should be capitalized")
            stage(input_data)

            # Results are stored in test_results

        # Run pipeline
        config_test_pipeline()

        # Verify configuration was applied
        self.assertIn("config_test", test_results)
        self.assertTrue(test_results["config_test"]["capitalize_was_applied"])
        self.assertEqual(
            test_results["config_test"]["text"], "THIS SHOULD BE CAPITALIZED"
        )

    def test_materializer_integration(self):
        """Test pipeline stages with the PydanticNumpyMaterializer."""

        @pipeline(enable_cache=False)
        def materializer_test_pipeline():
            # Configure and create stages
            text_processor = TextProcessingStageConfig().setup_target()
            image_generator = ImageGenerationStageConfig().setup_target()

            # Input with materializer
            text_input = TextInput(text="Testing materializer integration")

            # Execute with materializer handling - results stored in test_results
            processed_text = text_processor(text_input)
            image_generator(processed_text)

        # Run the pipeline
        materializer_test_pipeline()

        # Verify results
        self.assertIn("image_generation", test_results)
        self.assertIsNotNone(test_results["image_generation"]["image_shape"])
        self.assertGreater(len(test_results["image_generation"]["label"]), 0)
        self.assertEqual(
            test_results["image_generation"]["image_shape"][2], 3
        )  # RGB image

    def test_mixed_docker_environments(self):
        """Test a pipeline with stages running in different environments."""

        @pipeline(enable_cache=False)
        def mixed_docker_pipeline():
            # Create stages with different environments
            default_stage = DefaultStageConfig().setup_target()
            docker1_stage = DockerizedAnalysisConfig().setup_target()
            docker2_stage = AnotherDockerConfig().setup_target()

            # Input data
            input_text = TextInput(text="Testing multiple Docker environments")

            # Store settings for verification
            test_results["docker_settings"] = {
                "default": getattr(default_stage.config, "docker_settings", None),
                "docker1": docker1_stage.config.docker_settings.parent_image,
                "docker2": docker2_stage.config.docker_settings.parent_image,
            }

            # Need to actually use the stages in the pipeline for ZenML to recognize them
            default_stage(input_text)
            docker1_stage(input_text)
            docker2_stage(input_text)

        # Run the pipeline
        mixed_docker_pipeline()

        # Verify that docker settings were properly configured
        self.assertIn("docker_settings", test_results)
        self.assertIsNone(test_results["docker_settings"]["default"])
        self.assertEqual(
            test_results["docker_settings"]["docker1"],
            "zenmldocker/zenml:0.80.1-py3.11",
        )
        self.assertEqual(
            test_results["docker_settings"]["docker2"], "zenmldocker/zenml:0.70.0-py3.9"
        )


if __name__ == "__main__":
    unittest.main()
