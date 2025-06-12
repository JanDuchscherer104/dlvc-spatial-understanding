PORT ?= 8000
NAMESPACE ?= spatialunderstanding
BASE_IMAGE_NAME ?= base
TAG ?= latest
BASE_DOCKERFILE ?= Dockerfile.base

IMAGE_NAME = ${NAMESPACE}/${BASE_IMAGE_NAME}
REGISTRY_HOST = localhost:${PORT}
FULL_IMAGE_NAME = ${REGISTRY_HOST}/${IMAGE_NAME}

registry-init:
	docker run -d --restart=always --name registry -p ${PORT}:5000 registry:2
	curl http://localhost:${PORT}/v2/_catalog

registry-check:
	curl http://localhost:${PORT}/v2/_catalog

build-base:
	echo "Currently building is not done from a custom dockerfile!"
# docker build -f ${BASE_DOCKERFILE} -t $(FULL_IMAGE_NAME):$(TAG) .
# docker push $(FULL_IMAGE_NAME):$(TAG)

zenml-init:
	zenml init
	zenml integration install numpy
	zenml orchestrator register local_docker --flavor=local_docker
	zenml stack register local_docker_stack -o local_docker -a default --set
	zenml container-registry register local-registry --flavor=default --uri=$(REGISTRY_HOST)
	zenml stack update -c local-registry
	zenml stack describe

zenml-reinit:
	zenml clean --yes
	make zenml-init

