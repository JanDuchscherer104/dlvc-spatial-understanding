from spatial_guidance.data_contracts.aabb_segmentation import RawAABBDetSeg
from spatial_guidance.data_contracts.dataset import PipelineIn
from spatial_guidance.scene_understanding.gemini_aabb_detection import (
    GeminiAABBDetSegConfig,
)
from spatial_guidance.utils import Console
from spatial_guidance.utils.configs import PathConfig

CONSOLE = Console()
paths = PathConfig()

from spatial_guidance.data_handling.stray_scanner.data_parser import (
    StrayScannerDataParserConfig,
)
from spatial_guidance.data_handling.stray_scanner.stray_dataset import (
    StrayDatasetConfig,
)
from spatial_guidance.data_handling.stray_scanner.stray_scanner_paths import (
    StrayScannerPaths,
)

ds_config = StrayDatasetConfig(
    data_parser_config=StrayScannerDataParserConfig(
        paths=StrayScannerPaths(
            dataset_dir=paths.data / "SmartAIs-Recorded-Data/baustelle"
        )
    )
)
ds = ds_config.setup_target()

gemini_config = GeminiAABBDetSegConfig(
    temperature=0.2, model_name="gemini-2.5-flash-preview-05-20"
)
gemini = gemini_config.setup_target()

# Get a sample from the dataset
sample_idx = 300
pipeline_input = PipelineIn(idx=sample_idx)  # Example prompt
dataset_output = ds.entrypoint(pipeline_input)

# Run detection
# The result object now contains visualization_rgb and visualization_depth
result = gemini.entrypoint(dataset_output)


CONSOLE.plog(result)  # Log the full result object for inspection

# Display the generated visualizations
if result.visualization_rgb:
    CONSOLE.log(f"Displaying RGB visualization for sample {sample_idx}...")
    result.visualization_rgb.show()
else:
    CONSOLE.log(f"No RGB visualization generated for sample {sample_idx}.")

if result.visualization_depth:
    CONSOLE.log(f"Displaying Depth visualization for sample {sample_idx}...")
    result.visualization_depth.show()
else:
    CONSOLE.log(
        f"No Depth visualization generated or available for sample {sample_idx}."
    )

# Example of how to access original image if needed for comparison
# original_rgb_img = Image.fromarray(ds.get_rgb(sample_idx))
# original_rgb_img.show()

CONSOLE.log("Test script finished.")
pass
