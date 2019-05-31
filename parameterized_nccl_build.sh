# Builds nccl and nccl-tests for specific nccl version.
# Paths specialized to DLAMI 22, cuda-10
#
# Parameters:
# MPI_HOME:  (must be set for the nccl-tests to build, ie MPI_HOME=/usr/local/mpi)
# NCCL_VERSION_TAG: for folder name, git clone called from
#     ~/nccl/nccl-$NCCL_VERSION_TAG
# GIT_CHECKOUT_COMMAND: command called after git clone, right before building
# NCCL_WIPE_PREVIOUS_BUILD: if set, will get rid of previous build artifacts
#
# Result:
#  ~/nccl/nccl-$NCCL_VERSION_TAG/nccl/build/lib/libnccl.so
#  ~/nccl/nccl-$NCCL_VERSION_TAG/nccl-tests/build/build/all_reduce_perf
#
# Examples:
# MPI_HOME=/usr/local/mpi
# NCCL_VERSION_TAG=2.3.7
# GIT_CHECKOUT_CMD="git checkout v2.3.7-1"
# source parameterized_nccl_build.sh
#
# MPI_HOME=/usr/local/mpi
# NCCL_VERSION_TAG=2.4.7
# GIT_CHECKOUT_CMD="git checkout v2.4.7-1"
# source parameterized_nccl_build.sh
#
# MPI_HOME=/usr/local/mpi
# NCCL_VERSION_TAG=2.4.7ms0
# GIT_CHECKOUT_CMD="git checkout dev/kwen/multi-socket"
# source parameterized_nccl_build.sh

pushd .

# change default cuda to cuda-10, possibly not needed
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-10.0 /usr/local/cuda

if [ -z ${NCCL_VERSION_TAG+x} ]; then
    NCCL_VERSION_TAG="default"
    echo "Using default tag $NCCL_VERSION_TAG"
else
    echo "Using existing tag $NCCL_VERSION_TAG"
fi

if [ -z ${GIT_CHECKOUT_CMD+x} ]; then
    GIT_CHECKOUT_CMD="skipping checkout"
    echo "Using default git checkout cmd $GIT_CHECKOUT_CMD"
else
    echo "Using existing git checkout cmd $GIT_CHECKOUT_CMD"
fi

mkdir -p ~/nccl/nccl-$NCCL_VERSION_TAG
cd ~/nccl/nccl-$NCCL_VERSION_TAG
git clone https://github.com/NVIDIA/nccl.git || echo "exists"

cd nccl
$GIT_CHECKOUT_CMD
if [ -z ${NCCL_WIPE_PREVIOUS_BUILD+x} ]; then
    echo "NCCL_WIPE_PREVIOUS_BUILD not set, reusing previous build"
else
    echo "NCCL_WIPE_PREVIOUS_BUILD is set, building from scratch"
    rm -Rf build
fi

make -j src.build


# Library goes to ~/nccl/nccl-$NCCL_VERSION_TAG/nccl/build/lib/libnccl.so
# .so is taken from LD_LIBRARY_PATH by default
export LD_LIBRARY_PATH=~/nccl/nccl-$NCCL_VERSION_TAG/nccl/build/lib:$LD_LIBRARY_PATH
# PyTorch install asks for NCCL_ROOT_DIR
export NCCL_ROOT_DIR=~/nccl/nccl-$NCCL_VERSION_TAG/nccl
# NCCL_HOME is used for linking of nccl-tests
export NCCL_HOME=~/nccl/nccl-$NCCL_VERSION_TAG/nccl/build

# Build nccl examples
cd ~/nccl/nccl-$NCCL_VERSION_TAG
git clone https://github.com/NVIDIA/nccl-tests.git || echo "exists"

cd ~/nccl/nccl-$NCCL_VERSION_TAG/nccl-tests
if [ -z ${NCCL_WIPE_PREVIOUS_BUILD+x} ]; then
    echo "NCCL_WIPE_PREVIOUS_BUILD not set, reusing previous examples build"
else
    echo "NCCL_WIPE_PREVIOUS_BUILD is set, building from scratch"
    make clean
fi

if [ -z ${MPI_HOME+x} ]; then
    echo "Warning, MPI_HOME is not set"
else
    echo "Using MPI_HOME=$MPI_HOME"
fi

make MPI=1 CUDA_HOME=/usr/local/cuda-10.0

# install aws-ofi-nccl
mkdir -p ~/nccl/aws-ofi-nccl-$NCCL_VERSION_TAG
cd ~/nccl/aws-ofi-nccl-$NCCL_VERSION_TAG
git clone https://github.com/aws/aws-ofi-nccl.git || echo exists
cd aws-ofi-nccl
make clean
git apply aws_ofi_nccl.patch
./autogen.sh
mkdir install
./configure --prefix=$HOME/aws-ofi-nccl/install --with-mpi=$HOME/anaconda3 \
            --with-libfabric=/opt/amazon/efa \
            --with-nccl=$NCCL_HOME \
            --with-cuda=/usr/local/cuda-10.0

make && make install

popd
