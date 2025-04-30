from zenml import step

from ..pipeline.data_contracts import DatasetOut, DetectionStageOut, VisualizationIn


@step
def get_visualization_in(
    dataset_out: DatasetOut, detection_output: DetectionStageOut
) -> VisualizationIn:
    """
    Create the visualization input from dataset and detection outputs.

    Args:
        dataset_out: Output from the dataset stage.
        detection_output: Output from the detection stage.

    Returns:
        VisualizationIn: The combined input for the visualization stage.
    """
    return VisualizationIn(
        rgb_image=dataset_out.rgb_image,
        depth_image=dataset_out.depth_image,
        detection_output=detection_output,
    )
