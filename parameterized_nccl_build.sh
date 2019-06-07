# Script to build nccl, nccl-test and aws-ofi for specific version of nccl
#
# Paths specialized for DLAMI 23, cuda-10.
# Applies ~/aws-ofi-nccl.patch if present
#
# Parameters:
# NCCL_VERSION_TAG: for folder name, all results go to
#      FOLDER_ROOT=~/nccl/nccl-$NCCL_VERSION_TAG
# GIT_CHECKOUT_COMMAND: command called after git clone, right before building
# NCCL_WIPE_PREVIOUS_BUILD: if set, will get rid of previous build artifacts
#
# Result:
#  $FOLDER_ROOT/nccl/build/lib/libnccl.so
#  $FOLDER_ROOT/aws-ofi-nccl/install/{bin, lib}
#  $FOLDER_ROOT/nccl-tests/build/all_gather_perf
      
#
# Examples:
# NCCL_VERSION_TAG=2.3.7
# GIT_CHECKOUT_CMD="git checkout v2.3.7-1"
# source parameterized_nccl_build.sh
#
# NCCL_VERSION_TAG=2.4.7
# GIT_CHECKOUT_CMD="git checkout v2.4.7-1"
# source parameterized_nccl_build.sh
#
# NCCL_VERSION_TAG=2.4.7ms0
# GIT_CHECKOUT_CMD="git checkout dev/kwen/multi-socket"
# source parameterized_nccl_build.sh

export FOLDER_ROOT=~/nccl/nccl-$NCCL_VERSION_TAG

# CUDA_HOME is used for nccl:nccl-tests:aws-ofi, contains bin, lib, include
export CUDA_HOME=/usr/local/cuda-10.0

# MPI_HOME is used for aws-ofi:nccl-test, contains bin, lib, include
export MPI_HOME=$HOME/anaconda3

# NCCL_HOME is used for aws-ofi:nccl-tests, contains include, lib
export NCCL_HOME=$FOLDER_ROOT/nccl/build

export EFA_HOME=/opt/amazon/efa

# PyTorch install asks for NCCL_ROOT_DIR, contains build, README.md, src
# export NCCL_ROOT_DIR=$FOLDER_ROOT/nccl

logtag="parameterized_nccl_build.sh: "

pushd .

if [ -z ${NCCL_VERSION_TAG+x} ]; then
    echo "$logtag Error: Must set NCCL_VERSION_TAG"
    exit
fi

if [ -z ${GIT_CHECKOUT_CMD+x} ]; then
    echo "$logtag Error: Must set GIT_CHECKOUT_CMD"
    exit
fi


if [ -z ${NCCL_WIPE_PREVIOUS_BUILD+x} ]; then
    echo "NCCL_WIPE_PREVIOUS_BUILD not set, reusing previous build"
else
    echo "NCCL_WIPE_PREVIOUS_BUILD is set, building from scratch"
    rm -Rf $FOLDER_ROOT
fi

if [ -z ${GIT_CHECKOUT_CMD+x} ]; then
    GIT_CHECKOUT_CMD="skipping checkout"
    echo "$logtag Using default git checkout cmd $GIT_CHECKOUT_CMD"
else
    echo "$logtag Using existing git checkout cmd $GIT_CHECKOUT_CMD"
fi

echo "$logtag Installing nccl"
mkdir -p $FOLDER_ROOT && cd $FOLDER_ROOT
git clone https://github.com/NVIDIA/nccl.git || echo "exists"

cd nccl
$GIT_CHECKOUT_CMD

unset NVCC_GENCODE
# Only compile for Pascal/cuda 9+
# https://github.com/NVIDIA/nccl/issues/165
# remove 3.0 from list of supported architectures.
export NVCC_GENCODE="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=compute_70"

make -j src.build


echo "$logtag Installing aws-ofi-nccl"
mkdir -p $FOLDER_ROOT && cd $FOLDER_ROOT
git clone https://github.com/aws/aws-ofi-nccl.git || echo exists
cd aws-ofi-nccl

if [ -f "$HOME/aws-ofi-nccl.patch" ]; then
    echo "using ~/aws-ofi-nccl.patch version of aws-ofi-nccl"
    # git apply fails with "patch does not apply", use patch command instead
    #    git apply ~/aws-ofi-nccl.patch
    patch -p1 < ~/aws-ofi-nccl.patch
else
    echo "$logtag using master version of aws-ofi-nccl"
    #    git apply ~/aws-ofi-nccl.patch
fi

make clean
./autogen.sh
mkdir install
./configure --prefix=$FOLDER_ROOT/aws-ofi-nccl/install \
            --with-mpi=$MPI_HOME \
            --with-libfabric=/opt/amazon/efa \
            --with-nccl=$NCCL_HOME \
            --with-cuda=$CUDA_HOME
make && make install


echo "Installing nccl-examples"
mkdir -p $FOLDER_ROOT && cd $FOLDER_ROOT
git clone https://github.com/NVIDIA/nccl-tests.git || echo "exists"
cd nccl-tests

# TODO(y): is this same as as MPI_HOME, or are there extra things in anaconda3?
export LD_LIBRARY_PATH=$HOME/anaconda3/lib/:$LD_LIBRARY_PATH
make MPI=1

popd
