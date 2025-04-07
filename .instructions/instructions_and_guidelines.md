# Project Coding Guidelines & Pipeline Overview

## 1. Project Context & Architecture

**Project Focus**: A modular pipeline for 3D scene understanding to help blind users navigate their environment.

**Core Architecture**:

- **Input Stage**: Processes user prompts, video frames, and depth data
- **Detection Stage**: Performs initial object/scene detections
- **Refinement Stage**: Enhances scene understanding iteratively

**Design Philosophy**: Configuration objects set up and validate; Target objects handle runtime functionality.

---

## 2. Core Design Patterns

### Config-as-Factory Pattern

This pattern uses strongly-typed Pydantic objects as factories that validate their parameters before instantiating runtime components:

1. The Factory Method pattern via the `setup_target()` method
2. The Builder pattern (constructing complex objects step by step)
3. Dependency Injection (providing an object with its dependencies via their respective configs)

All configuration classes should inherit from `BaseConfig` (which is derived from `pydantic.BaseModel`) from the `utils` module.

```python
from utils import BaseConfig
from pydantic import Field, field_validator, ValidationInfo

class MyModelConfig(BaseConfig["MyModel"]):
    target: Type["MyModel"] = Field(default_factory=lambda: MyModel)

    child: "ChildConfig" = ChildConfig(field1="value1")
    model_name: str = "default_model"

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str | Path, info: ValidationInfo) -> Path:
        if v not in ["model_a", "model_b", "default_model"]:
            raise ValueError(f"Invalid model name: {v}")
        return v

class MyModel:
    def __init__(self, config: MyModelConfig):
        self.config = config
        self.child = config.child.setup_target()
```

### Singleton Configuration Pattern

For global resources using `SingletonConfig`:

```python
from .utils import SingletonConfig

class PathConfig(SingletonConfig):
    root: Path = Field(default_factory=lambda: Path(__file__).parents[3])
    data: Path = Field(default=".data")

    def get_api_key(self, key_name: Literal["GOOGLE_API_KEY", "OPENAI_API_KEY", ...]) -> Optional[str]:
        # Implementation that ensures consistent access to keys
```

The class `PathConfig` should be used as basis for interacting with all files in this project. You may add fields to it, or simply use it as is.

```py
from utils import PathConfig

stray_dataset = PathConfig().data / "stray_dataset"
api_key = PathConfig().get_api_key("GOOGLE_API_KEY")
```

### Enum-Based Model Selection Pattern

In case of multiple particular runtime classes within a higher-level class, use enums to define available model types with a `setup_target()` method for instantiation:

```python
from enum import Enum, auto

class ModelType(Enum):
    MODEL_A = auto()
    MODEL_B = auto()
    MODEL_C = auto()

    @classmethod
    def setup_target(cls, config: "ModelConfig") -> Union[ModelA, ModelB, ModelC]:
        match config.model:
            case cls.MODEL_A:
                return ModelA(params.child)
            case cls.MODEL_B:
                return ModelB(params.child)
            case cls.MODEL_C:
                return ModelC(params.child)
            case _:
                raise NotImplementedError(f"Model type {params.model} not implemented")


class SystemConfig(BaseConfig["System"]):
    model_type: ModelType = ModelType.MODEL_A
    target: Type["System"] = Field(default_factory=lambda: System)
    # Additional configuration parameters

    def setup_target(self) -> "System":
        """Factory method to instantiate the model based on type"""
        return self.model.setup_target(self)


class System:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.model = config.model_type.setup_target(config)
```

### Validator Patterns

**Field Validators**: Transform and validate individual fields

```python
@field_validator("dataset_dir", mode="before")
@classmethod
def validate_dataset_dir(cls, v: str | Path, info: ValidationInfo) -> Path:
    if isinstance(v, (str, Path)) and not Path(v).is_absolute():
        pth = PathConfig().data / v
    else:
        pth = Path(v)
    assert pth.exists(), f"Dataset root directory {pth} does not exist."
    return pth.resolve()
```

**Model Validators**: Validate entire model after all fields are processed

```python
@model_validator(mode="after")
def validate(self) -> Self:
    required_files = [self.get_camera_matrix_path(), self.get_odometry_path()]
    missing_files = list(filter(lambda f: not f.exists(), required_files))
    if missing_files:
        CONSOLE.warn(f"[red]Required files missing: {', '.join(str(f) for f in missing_files)}")
    return self
```

---

## 3. Implementation Requirements

### Strict Type Enforcement

- All objects, functions, and methods must have explicit type annotations
- Use Pydantic's rich type validation system
- Enable IDE type checking and validation

```python
def process_image(
    image: np.ndarray,
    scale_factor: float = 1.0,
    normalize: bool = True
) -> Tuple[np.ndarray, Dict[str, float]]:
    # Implementation with proper type annotations
```

### Verbose Logging & Debugging

Use centralized `CONSOLE` from `utils` for consistent logging:

```python
from utils import CONSOLE

CONSOLE.log("Starting [blue]detection process[/blue]")
CONSOLE.warn("Missing depth data, falling back to RGB only")
```

### Target Setup Pattern

Consistent instantiation pattern for runtime objects:

```python
from spatial_guidance import StrayScannerPaths, StrayScannerDataParserConfig

# Nested configuration with shared field propagation
dataset = StrayDatasetConfig(
    data_parser_config=StrayScannerDataParserConfig(
        paths=StrayScannerPaths(dataset_dir="SmartAIs Recorded Data/5")
    ),
    is_rotated=True,
).setup_target()
```

**Benefits**:

- Shared fields automatically propagate to nested configs
- All validations occur before runtime instantiation
- Clear separation between setup and execution phases

---

## 4. Pipeline Design Pattern

### Modular Pipeline Architecture

Our spatial understanding pipeline follows a robust, flexible design that enables swappable components while maintaining type safety and standardized data flow. It combines several design patterns:

1. **Chain of Responsibility Pattern**: Process data through a sequence of handlers (stages)
2. **Strategy Pattern**: Swap implementations of each stage using configuration
3. **Data Transfer Object Pattern**: Standardized data exchange between stages
4. **Adapter Pattern**: Convert data formats between incompatible interfaces
5. **Decorator Pattern**: Enhance stage functionality without modifying core components

#### 4.1 Standardized Data Exchange with Dynamic Parameter Matching

All pipeline stages exchange data through strongly typed `PydanticModel` objects. We use an enhanced pattern that automatically unpacks only the required parameters for each stage method:

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Tuple, get_type_hints, Callable
import inspect
import numpy as np

class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    label: str
    confidence: float

class SegmentationMask(BaseModel):
    mask: np.ndarray
    label: str
    confidence: float

class PipelineData(BaseModel):
    """Data object shared between pipeline stages"""
    rgb_frame: np.ndarray
    depth_frame: Optional[np.ndarray] = None
    user_query: Optional[str] = None
    detections: List[Union[BoundingBox, SegmentationMask]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Ensure numpy arrays are handled correctly
    model_config = {"arbitrary_types_allowed": True}

    def extract_kwargs_for(self, func: Callable) -> Dict[str, Any]:
        """Extract only the parameters needed for the given function

        Args:
            func: The function whose parameters should be matched

        Returns:
            Dictionary containing only the fields that match parameter names
        """
        # Get parameter names from the function signature
        sig = inspect.signature(func)
        param_names = set(sig.parameters.keys())

        # Only include fields that match parameter names
        available_fields = self.model_dump()
        kwargs = {k: v for k, v in available_fields.items() if k in param_names}

        return kwargs
```

#### 4.2 Parameter-Aware Stage Interface

Each pipeline stage uses a enhanced interface that processes only needed data fields:

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Type, Dict, Any

DataType = TypeVar('DataType', bound=BaseModel)

class PipelineStage(Generic[DataType], ABC):
    """Abstract base class for pipeline stages with automatic parameter matching"""

    @abstractmethod
    def process(self, data: DataType) -> DataType:
        """Process input data and return processed output"""
        pass

    def _call_with_matching_params(self, method: Callable, data: DataType) -> Any:
        """Call a method with only the parameters it needs from the data object

        Args:
            method: The method to call
            data: The data object containing potential parameters

        Returns:
            The result of calling the method with matched parameters
        """
        kwargs = data.extract_kwargs_for(method)
        return method(**kwargs)


class DetectionStage(PipelineStage[PipelineData]):
    """Detection stage that automatically unpacks only needed parameters"""

    def process(self, data: PipelineData) -> PipelineData:
        # Call actual implementation with only the parameters it needs
        results = self._call_with_matching_params(self._detect_objects, data)

        # Update the data object with results
        data.detections = results
        return data

    def _detect_objects(self, rgb_frame: np.ndarray, depth_frame: Optional[np.ndarray] = None) -> List[BoundingBox]:
        """The actual detection implementation, only receives parameters it needs

        Even if PipelineData contains many other fields, this method only receives
        the rgb_frame and depth_frame parameters that it requests.
        """
        # Object detection implementation
        # ...
        return [BoundingBox(x_min=10, y_min=20, x_max=100, y_max=200, label="chair", confidence=0.95)]
```

#### 4.3 Stage Configuration with Specialized Processing Methods

Each stage can specify exactly which data it needs, reducing coupling:

```python
class GeminiVisionStage(PipelineStage[PipelineData]):
    """Stage that uses Gemini Vision API for detection"""

    def __init__(self, config: "GeminiVisionConfig"):
        self.config = config
        self.processor = GeminiVisionProcessor(config)

    def process(self, data: PipelineData) -> PipelineData:
        """Process the data using Gemini Vision API"""
        # Only the rgb_frame parameter will be extracted and passed
        detections = self._call_with_matching_params(self._analyze_image, data)

        # Update data with new detections
        data.detections.extend(detections)
        return data

    def _analyze_image(self, rgb_frame: np.ndarray) -> List[BoundingBox]:
        """Analyze image with Gemini Vision - only receives the rgb_frame parameter"""
        # Preprocess
        inputs = self.processor.preprocess(rgb_frame)

        # Call API
        api_response = self._call_gemini_api(inputs)

        # Postprocess
        return self.processor.postprocess(api_response, original_shape=rgb_frame.shape[:2])
```

#### 4.4 Pipeline Composition with Parameter-Aware Stages

The pipeline orchestrates stages, each accessing only the data they need:

```python
class Pipeline:
    """Main pipeline orchestrating the flow of data through stages"""

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Initialize stages from configuration
        self.input_stage = config.input_stage_config.setup_target()
        self.detection_stage = config.detection_stage_config.setup_target()
        self.refinement_stage = config.refinement_stage_config.setup_target()
        self.spatial_analysis = config.spatial_analysis_config.setup_target()
        self.response_generation = config.response_generation_config.setup_target()

    def process(self, rgb_frame: np.ndarray, depth_frame: Optional[np.ndarray] = None, user_query: Optional[str] = None) -> Dict[str, Any]:
        """Process input data through the pipeline"""
        # Initialize pipeline data object
        data = PipelineData(
            rgb_frame=rgb_frame,
            depth_frame=depth_frame,
            user_query=user_query
        )

        # Process through each stage - stages only access fields they need
        data = self.input_stage.process(data)
        data = self.detection_stage.process(data)
        data = self.refinement_stage.process(data)
        data = self.spatial_analysis.process(data)
        result = self.response_generation.process(data)

        return result.model_dump()
```

#### 4.5 Parallel Processing Pattern with Dynamic Parameter Extraction

For stages that can run models in parallel while maintaining loose coupling:

```python
class ParallelDetectionStage(PipelineStage[PipelineData]):
    """Detection stage that can run multiple models in parallel"""

    def __init__(self, config: ParallelDetectionConfig):
        self.config = config
        self.models = [model_config.setup_target() for model_config in config.model_configs]

    def process(self, data: PipelineData) -> PipelineData:
        """Run all detection models and merge results"""
        all_detections = []

        # Process with each model (can be parallelized with ThreadPoolExecutor)
        for model in self.models:
            # Each model only receives the parameters it declares in its detect method
            detections = self._call_with_matching_params(model.detect, data)
            all_detections.extend(detections)

        # Update pipeline data
        data.detections.extend(all_detections)
        return data
```

#### 4.6 Extension Methods for Enhanced Functionality

The pipeline data object can be extended with utility methods:

```python
class PipelineData(BaseModel):
    """Data object shared between pipeline stages with enhanced functionality"""
    # ...fields as defined earlier

    def get_closest_object(self) -> Optional[Tuple[BoundingBox, float]]:
        """Find the closest object based on depth data

        Returns:
            Tuple of (bounding box, distance in meters) or None if no objects
        """
        if not self.detections or self.depth_frame is None:
            return None

        closest_obj = None
        min_distance = float('inf')

        for detection in self.detections:
            if isinstance(detection, BoundingBox):
                # Calculate center point of bbox
                center_x = (detection.x_min + detection.x_max) // 2
                center_y = (detection.y_min + detection.y_max) // 2

                # Get depth at center point
                distance = self.depth_frame[center_y, center_x]

                if distance < min_distance:
                    min_distance = distance
                    closest_obj = detection

        return (closest_obj, min_distance) if closest_obj else None

    def filter_detections(self, min_confidence: float = 0.5, labels: Optional[List[str]] = None) -> List[Union[BoundingBox, SegmentationMask]]:
        """Filter detections based on confidence and labels

        Args:
            min_confidence: Minimum confidence threshold
            labels: Optional list of labels to include (if None, include all)

        Returns:
            Filtered list of detections
        """
        filtered = [d for d in self.detections if d.confidence >= min_confidence]

        if labels:
            filtered = [d for d in filtered if d.label in labels]

        return filtered
```

#### 4.7 Usage Example with Automatic Parameter Matching

```python
def spatial_analysis_example():
    # Create detection model that only needs RGB
    rgb_model = RGBDetectionModel(RGBModelConfig())

    # Create detection model that needs both RGB and depth
    rgbd_model = RGBDDetectionModel(RGBDModelConfig())

    # Create stage with both models
    detection_stage = ParallelDetectionStage(
        ParallelDetectionConfig(model_configs=[rgb_model, rgbd_model])
    )

    # Create pipeline data with all available information
    data = PipelineData(
        rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
        depth_frame=np.zeros((480, 640), dtype=np.float32),
        user_query="What objects are in front of me?"
    )

    # Process data - each model automatically receives only the parameters it needs
    result = detection_stage.process(data)

    # The RGB model only received rgb_frame
    # The RGBD model received both rgb_frame and depth_frame
    # Both models worked with the same data object without coupling
```

### 5. Implementation Checklist

**Include**:

- ✓ Explicit type annotations on all methods and functions
- ✓ Config classes inherit from `utils.BaseConfig`
- ✓ Parameter-aware stage methods that only receive data they explicitly request
- ✓ Dynamic parameter extraction from pipeline data objects
- ✓ Enum-based model selection for swappable components
- ✓ Logging through centralized `utils.CONSOLE`
- ✓ Separation of validation, configuration, instantiation and runtime functionality
- ✓ Standardized data exchange using Pydantic models
- ✓ Clear stage interfaces with well-defined input/output types and doc-strings
- ✓ Pre- and post-processing patterns for model-specific requirements
- (✓) Support for parallel model execution within stages
- ✓ Proper error handling and logging at stage boundaries
- ✓ Functional approaches over loops/comprehensions when appropriate

**Avoid**:

- ✗ Directly accessing data fields that aren't explicitly declared as parameters
- ✗ Tight coupling between stages (stages should be independently replaceable)
- ✗ Embedding validation logic in runtime classes (targets)
- ✗ Variables whose types are not inferrable
- ✗ Hard-coded model parameters (use configuration)
- ✗ Mixed responsibility in classes (follow single responsibility principle)
- ✗ Silently failing operations (always log or raise exceptions)

By following these guidelines, we ensure a maintainable, type-safe, and well-documented codebase with clear traceability between configuration and runtime components, while enabling easy modification and extension of the pipeline.
