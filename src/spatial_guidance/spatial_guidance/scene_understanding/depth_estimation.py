from typing import Type

import numpy as np
from pydantic import Field

from ..pipeline.data_contracts import DepthEstimationInput, DepthEstimationOutput
from ..pipeline.pipeline_stage import StepConfig
from ..utils import BaseConfig
