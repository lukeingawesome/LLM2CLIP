.PHONY: configure up exec build start down run ls check cache-env

# Convenience `make` recipes for Docker Compose.
# See URL below for documentation on Docker Compose.
# https://docs.docker.com/engine/reference/commandline/compose

# **Change `SERVICE` to specify other services and projects.**
SERVICE = dev
COMMAND = /bin/zsh

# `PROJECT` is equivalent to `COMPOSE_PROJECT_NAME`.
# Project names are made unique for each user to prevent name clashes.
# MS accounts has period (".") within username. It should be removed to meet docker container name rules.
USR_SAFE := $(shell id -un | sed 's/\.//g')
PROJECT := $(shell hostname)-${USR_SAFE}
PROJECT_ROOT := /opt/project

# Environment variables
BASE_IMAGE = ghcr.io/irail/cxr-dev-base:20240320
GID := $(shell id -g)
UID := $(shell id -u)
USR := $(shell id -un)
GRP := $(shell id -gn)
HOSTNAME := $(shell hostname)
HOME := $(shell echo ~${USR})
IMAGE_NAME := irail-dev:${USR_SAFE}
DOTCACHE_DIR = ${HOME}/.cache

COMPOSE_DOCKER_CLI_BUILD = 1
DOCKER_BUILDKIT = 1

export BASE_IMAGE GID UID GRP USR HOSTNAME IMAGE_NAME PROJECT PROJECT_ROOT DOTCACHE_DIR
export COMPOSE_DOCKER_CLI_BUILD DOCKER_BUILDKIT

ENV_FILE := .env
cache-env:  # Not actually necessary, but it's helpful for debugging w/o Makefile
	@rm -rf "${ENV_FILE}"; \
	printf "### Environment variables:\n"; \
	printf "BASE_IMAGE=$$BASE_IMAGE\n" >> "${ENV_FILE}";\
	printf "GID=$$GID\n" >> "${ENV_FILE}"; \
	printf "UID=$$UID\n" >> "${ENV_FILE}"; \
	printf "GRP=$$GRP\n" >> "${ENV_FILE}"; \
	printf "USR=$$USR\n" >> "${ENV_FILE}"; \
	printf "HOSTNAME=$$HOSTNAME\n" >> "${ENV_FILE}"; \
	printf "IMAGE_NAME=$$IMAGE_NAME\n" >> "${ENV_FILE}"; \
	printf "PROJECT=$$PROJECT\n" >> "${ENV_FILE}"; \
	printf "PROJECT_ROOT=$$PROJECT_ROOT\n" >> "${ENV_FILE}"; \
	printf "DOTCACHE_DIR=$$DOTCACHE_DIR\n" >> "${ENV_FILE}"; \
	printf "COMPOSE_DOCKER_CLI_BUILD=$$COMPOSE_DOCKER_CLI_BUILD\n" >> "${ENV_FILE}"; \
	printf "DOCKER_BUILDKIT=$$DOCKER_BUILDKIT\n" >> "${ENV_FILE}"; \
	cat "${ENV_FILE}"; \
	printf "\n";

dotcache:  # Creates `${HOME}/.cache` dir if it does not exist.
	@if [ ! -d "${DOTCACHE_DIR}" ]; then \
		mkdir -p -m 755 ${DOTCACHE_DIR}; \
	fi && \
	if [ "$(shell stat --format '%U%G' ${DOTCACHE_DIR})" != "${USR}${GRP}" ]; then \
		chown ${USR}:${GRP} ${DOTCACHE_DIR}; \
	fi;

configure: dotcache cache-env
PROJECT=mm_hbko
build: configure # Start service. Rebuilds the image from the Dockerfile before creating a new container. # + Monorepo pip editable install included.
	docker compose -p ${PROJECT} up --build -d ${SERVICE} && \
	printf "\n### Installing user-requirements.txt..\n\n" && \
	docker commit ${PROJECT}-${SERVICE}-1 ${IMAGE_NAME} && \
	printf "\n### Restarting the service (down & up) to make container has the updated image id.\n\n" && \
	docker compose -p ${PROJECT} down && \
	docker compose -p ${PROJECT} up --no-build -d ${SERVICE}
up:  # Start service. Creates a new container from the image.
	docker compose -p ${PROJECT} up -d ${SERVICE}
exec:  # Execute service. Enter interactive shell.
	docker compose -p ${PROJECT} exec ${SERVICE} ${COMMAND}
start:  # Start a stopped service without recreating the container. Useful if the previous container must not deleted.
	docker compose -p ${PROJECT} start ${SERVICE}
stop:
	docker compose -p ${PROJECT} stop ${SERVICE}
down:  # Shut down service and delete containers, volumes, networks, etc.
	docker compose -p ${PROJECT} down
run:  # Used for debugging cases where service will not start.
	docker compose -p ${PROJECT} run --rm ${SERVICE} ${COMMAND} --gpus all
ls:  # List all services.
	docker compose ls -a

OVERRIDE_FILE = docker-compose.override.yaml
# Makefiles do not read the initial spaces, hence the inclusion of
# indentation at the end of the line. Not pretty but it works.
OVERRIDE_TEMPLATE = "$\
\# You can customize volume mounts over the given docker-compose.yaml\n$\
\# You better not mount directly on container user home dir (\`/home/${USER}\`),\n$\
\# since it may get affected by Dockerfile build routines.\n$\
\# FYI, ${PROJECT_ROOT} will be your workdir by default.\n$\
services:\n  $\
  ${SERVICE}:\n    $\
    volumes:\n      $\
      - /data:/data\n      $\
	  - /data2:/data2\n      $\
	  - /data3:/data3\n	$\
"
# Create override file for who does need customized volume mount on Docker Compose configurations.
${OVERRIDE_FILE}:  # It does not overwrite the existing file
	printf ${OVERRIDE_TEMPLATE} >> ${OVERRIDE_FILE}

overrides: ${OVERRIDE_FILE}

# Auto run full (frequently-used) routines
irail: build exec
test:
	docker compose -p ${PROJECT} exec ${SERVICE} /bin/zsh test.sh
