# Installation & Setup

```bash
conda create -n dlvc python=3.11
conda activate dlvc

conda install -c conda-forge open3d

git clone <repository-url>
cd dlvc-04-spatial-understanding
pip install -r src/spatial_guidance/requirements.txt

pip install -e src/utils
pip install -e src/spatial_guidance
```

## API Keys

[Create a Gemini API key](https://aistudio.google.com/app/apikey) and add it to the `.env` file.
<!-- [Create an OpenAI API key](https://platform.openai.com/api-keys) and add it to the `.env` file. -->

```bash
GOOGLE_API_KEY=<your-api-key>
# OPENAI_API_KEY=sk-<...>
```

## Setup Docker and ZenML

Use the provided Makefile to set up Docker and ZenML. The Makefile contains several targets to help you with the setup process.

1. **`registry-init`**: Launches a local Docker registry container on `localhost:<PORT>` (default 8000), mapping container port 5000 to your specified port.
Should print `{"repositories":[]}` when no images are present.
2. **`zenml-init`**: Initializes ZenML, installs the numpy integration, registers the local docker orchestrator, sets up a stack called `local_docker_stack`, registers a container registry at `${REGISTRY_HOST}`, and updates the stack to use it.

- **`build-base`**: Builds the base image from `Dockerfile.base` with tag `latest` (or specified via `TAG`), namespaced under `${NAMESPACE}` and pushed to the local registry.
After building `make registry-check` should yield something like `{"repositories":["spatialunderstanding/base"]}`

- **`zenml-reinit`**: Completely resets ZenML, then repeats the full stack setup as done in `zenml-init`, ensuring a clean environment with a fresh stack and registry configuration.

Optional arguments:

- `PORT=<your-port>`: Specify an unused port for the Docker registry (default is 8000).
- `NAMESPACE=<your-namespace>`: Specify a custom namespace for the Docker image.
- `TAG=<your-tag>`: Specify a custom tag for the Docker image.
- `BASE_DOCKER_FILE=<your-docker-file>`: Specify a custom Dockerfile for the base image.
