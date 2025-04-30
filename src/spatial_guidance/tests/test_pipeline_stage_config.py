import unittest

from zenml.config import DockerSettings, ResourceSettings, StepRetryConfig

from spatial_guidance.pipeline.pipeline_stage import StepConfig


class TestPipelineStageConfig(unittest.TestCase):
    def test_get_step_kwargs_with_docker_and_resources(self):
        docker = DockerSettings(parent_image="python:3.8-slim")
        resources = ResourceSettings(cpu=2, memory="1Gi")
        config = StepConfig(
            docker_settings=docker,
            resources=resources,
            enable_cache=True,
        )
        kwargs = config.get_step_kwargs()
        self.assertTrue(kwargs["enable_cache"])
        self.assertIn("settings", kwargs)
        settings = kwargs["settings"]
        self.assertIn("docker", settings)
        self.assertEqual(settings["docker"], docker)
        self.assertIn("resources", settings)
        self.assertEqual(settings["resources"], resources)

    def test_retry_and_callbacks(self):
        retry = StepRetryConfig(max_retries=3, delay=10)

        def on_success(step):
            pass

        def on_failure(step, exc):
            pass

        config = StepConfig(
            retry=retry,
            on_success=on_success,
            on_failure=on_failure,
        )
        kwargs = config.get_step_kwargs()
        self.assertEqual(kwargs["retry"], retry)
        self.assertEqual(kwargs["on_success"], on_success)
        self.assertEqual(kwargs["on_failure"], on_failure)

    def test_none_settings_filtered(self):
        config = StepConfig()
        kwargs = config.get_step_kwargs()
        self.assertNotIn("settings", kwargs)
        self.assertFalse("enable_cache" in kwargs) or self.assertIn(
            "enable_cache", kwargs
        )


if __name__ == "__main__":
    unittest.main()
