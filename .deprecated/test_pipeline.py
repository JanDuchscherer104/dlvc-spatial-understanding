from spatial_guidance import PipelineConfig
from spatial_guidance.utils import CONSOLE

pipeline = PipelineConfig(
    stack="local_docker_stack",
).setup_target()

ret = pipeline(420, None)

for stage in pipeline.get_names().values():
    stage_out = pipeline.get_output_of_stage(stage)
    CONSOLE.log(f"Stage {stage} output:")
    CONSOLE.plog(stage_out, title=f"Stage {stage} Output")

pipeline.get_latest_output().visualization.show()
