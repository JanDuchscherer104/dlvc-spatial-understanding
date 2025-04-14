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

### 4. Implementation Checklist

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
