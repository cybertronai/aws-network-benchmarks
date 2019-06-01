# Builds nccl, nccl-tests and aws-ofi-nccl for specific nccl version.
# Paths specialized to DLAMI 23, cuda-10
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


pushd .

if [ -z ${NCCL_VERSION_TAG+x} ]; then
    echo "Error: Must set NCCL_VERSION_TAG"
    exit
fi

if [ -z ${GIT_CHECKOUT_CMD+x} ]; then
    echo "Error: Must set GIT_CHECKOUT_CMD"
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
    echo "Using default git checkout cmd $GIT_CHECKOUT_CMD"
else
    echo "Using existing git checkout cmd $GIT_CHECKOUT_CMD"
fi

echo "Installing nccl"
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


# PyTorch install asks for NCCL_ROOT_DIR, contains build, README.md, src
# export NCCL_ROOT_DIR=$FOLDER_ROOT/nccl

echo "Installing aws-ofi-nccl"
mkdir -p $FOLDER_ROOT && cd $FOLDER_ROOT
git clone https://github.com/aws/aws-ofi-nccl.git || echo exists
cd aws-ofi-nccl
# make clean
# git apply aws_ofi_nccl.patch
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

$MPI_HOME/bin/mpirun \
-x FI_PROVIDER="efa" \
-x FI_OFI_RXR_RX_COPY_UNEXP=1 -x FI_OFI_RXR_RX_COPY_OOO=1 \
-x FI_EFA_MR_CACHE_ENABLE=1 -x FI_OFI_RXR_INLINE_MR_ENABLE=1 \
-x LD_LIBRARY_PATH=$FOLDER_ROOT/aws-ofi-nccl/install/lib/:$NCCL_HOME/lib:$CUDA_HOME/lib64:$EFA_HOME/lib64:$MPI_HOME/lib:$LD_LIBRARY_PATH \
-x NCCL_DEBUG=INFO -x NCCL_TREE_THRESHOLD=0 --host localhost -n 2 -N 2 \
--mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
--oversubscribe \
~/nccl/nccl-$NCCL_VERSION_TAG/nccl-tests/build/all_reduce_perf -b 8 -e 1M -f 2 -g 1 -c 1 -n 2

popd
