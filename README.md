# 3D Spatial Scene Understanding and Interactive Audio Guidance for Blind Users

## Overview

This project aims to develop an application that provides 3D spatial scene understanding and interactive audio guidance specifically designed for blind users. By leveraging advanced techniques in depth estimation, object detection, and spatial audio rendering, the application seeks to enhance navigation and interaction in various environments.

- **3D Spatial Scene Understanding**: Utilizes depth estimation and object detection to create a comprehensive understanding of the user's surroundings.
- **Interactive Audio Guidance**: Generates spatial audio cues and spoken instructions to assist users in navigating their environment effectively.

See [Setup](SETUP.md) for installation instructions.

### Setup ZenML

```bash
zenml init
zenml integration install numpy
```

```bash
zenml orchestrator register local_docker --flavor=local_docker

# Register and activate a stack with the new orchestrator
zenml stack register local_docker_stack -o local_docker -a default --set
```

### Fixing ZenML

```bash
zenml clean --yes
zenml init
```

## Folder Structure

## Usage

- Example usage of the `StrayScanner` dataset is provided in this [notebook](notebooks/test_stray.ipynb).

## Style Guide

### Connecting to the GHCR

```bash
export CR_PAT=<your_personal_access_token>
echo $CR_PAT | docker login ghcr.io -u <your_github_username> --password-stdin

docker push ghcr.io/JanDuchscherer104/spatialunderstanding-base:latest
```

#### Register the GHCR with ZenML

```bash
zenml container-registry register spatial-understanding-registry \
    --flavor=github \
    --uri=ghcr.io/janduchscherer104

zenml stack update -c spatial-understanding-registry
```

Verify the current stack:

```bash
zenml stack describe
```

Output should look like this:

Stack Configuration

|COMPONENT_TYPE     | COMPONENT_NAME                 |
|--------------------|--------------------------------|
|ARTIFACT_STORE     | default                        |
|ORCHESTRATOR       | local_docker                   |
|CONTAINER_REGISTRY | spatial-understanding-registry |

'local_docker_stack' stack (ACTIVE)

Create local docker registry:

```bash
docker run -d \
  --restart=always \
  --name registry \
  -p 5001:5000 \
  registry:2
```

- or simply use `-p 5000:5000` if the default port is not in use

# Build the base image

```bash
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.base \
  -t localhost:5001/spatialunderstanding/base:latest \
  --push \
  .
```

Edit Docker Deamon settings and subsequently restart Docker:

```json
{
    "insecure-registries": [
        "localhost:8000",
        "host.docker.internal:8000"
  ]
}
```

❯ zenml container-registry register local-registry \
  --flavor=default \
  --uri=localhost:5001
Successfully registered container_registry `local-registry`.
You can display various ZenML entities including pipelines, runs, stacks and much more on the ZenML Dashboard. You can try it locally, by running `zenml login --local`, or remotely, by deploying ZenML on the infrastructure of your choice.
❯ zenml stack update -c local-registry

❯ zenml container-registry register local-registry \
  --flavor=default \
  --uri=$REGISTRY_URI

❯ zenml stack update -c local-registry

docker run -d \
  --restart=always \
  --name registry \
  -p 8000:5000 \
  registry:2

# Build image

docker build -f Dockerfile.base -t localhost:8000/spatialunderstanding/base:latest .

# Push manually

docker push localhost:8000/spatialunderstanding/base:latest

Check if the image was stored correctly:

```bash
curl http://localhost:8000/v2/_catalog
```

should return:

```json
{"repositories":["spatialunderstanding/base"]}
```
