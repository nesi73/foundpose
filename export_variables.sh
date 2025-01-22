#!/bin/bash
OBJECT=$1

export REPO_PATH="/mnt/foundpose"

export BOP_PATH="/mnt/foundpose/datasets/$OBJECT"

export PYTHONPATH="$REPO_PATH:$REPO_PATH/external/bop_toolkit:$REPO_PATH/external/dinov2"
