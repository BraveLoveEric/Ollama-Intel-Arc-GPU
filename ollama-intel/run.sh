#!/bin/bash
source ipex-llm-init -g --device Arc
export OLLAMA_NUM_GPU=999
export no_proxy=localhost,127.0.0.1
export ZES_ENABLE_SYSMAN=1
# source /opt/intel/oneapi/setvars.sh
source /opt/intel/1ccl-wks/setvars.sh
export SYCL_CACHE_PERSISTENT=1
export OLLAMA_INTEL_GPU=true
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
# [optional] if you want to run on single GPU, use below command to limit GPU may improve performance
export ONEAPI_DEVICE_SELECTOR=level_zero:0
# init ollama
mkdir -p /var/ollama
cd /var/ollama
init-ollama
# start ollama service
./ollama $1