set -y

sudo yum update -y
sudo yum groupinstall "Development Tools" -y

mkdir ~/packages
cd ~/packages
wget https://s3-us-west-2.amazonaws.com/aws-efa-installer/aws-efa-installer-latest.tar.gz
tar -xf aws-efa-installer-latest.tar.gz
cd aws-efa-installer
sudo ./efa_installer.sh -y
# sudo reboot  # doesn't seem needed

cd ~/packages
wget http://us.download.nvidia.com/tesla/418.40.04/NVIDIA-Linux-x86_64-418.40.04.run
sudo bash NVIDIA-Linux-x86_64-418.40.04.run --no-drm --disable-nouveau --dkms --silent --install-libglvnd

cd ~/packages
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
chmod +x cuda_10.0.130_410.48_linux
sudo ./cuda_10.0.130_410.48_linux --silent --override --toolkit --samples --no-opengl-libs

cd ~/packages
git clone https://github.com/NVIDIA/nccl.git
cd nccl
git checkout dev/kwen/multi-socket
make -j src.build
make pkg.txz.build
cd build/pkg/txz
tar xvfJ nccl_2.4.7ms1-1+cuda10.0_x86_64.txz
sudo cp -r nccl_2.4.7ms1-1+cuda10.0_x86_64/* /usr/local/cuda-10.0/

cd ~/packages
git clone https://github.com/aws/aws-ofi-nccl.git || echo exists
cd aws-ofi-nccl
wget https://s3.amazonaws.com/yaroslavvb2/data/aws-ofi-nccl.patch
patch -p1 < aws-ofi-nccl.patch

./autogen.sh
./configure --prefix=/usr --with-mpi=/opt/amazon/efa --with-libfabric=/opt/amazon/efa/ --with-cuda=/usr/local/cuda --with-nccl=/usr/local/cuda
sudo yum install libudev-devel -y
LDFLAGS="-L/opt/amazon/efa/lib64" make MPI=1 MPI_HOME=/opt/amazon/efa/ CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/cuda
sudo make install

cd ~/packages
wget https://s3.amazonaws.com/yaroslavvb2/data/cudnn-10.0-linux-x64-v7.6.0.64.tgz
tar zxvf cudnn-10.0-linux-x64-v7.6.0.64.tgz
sudo cp -r cuda/* /usr/local/cuda-10.0

sudo update-alternatives --set gcc "/usr/bin/gcc48"
sudo update-alternatives --set g++ "/usr/bin/g++48"

cd ~/packages
wget https://github.com/bazelbuild/bazel/releases/download/0.20.0/bazel-0.20.0-installer-linux-x86_64.sh
sudo bash bazel-0.20.0-installer-linux-x86_64.sh

cd ~/packages
wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh -b

sudo sh -c 'echo "/opt/amazon/efa/lib64/" > mpi.conf'
sudo sh -c 'echo "/usr/local/cuda/lib/" > nccl.conf'
sudo sh -c 'echo "/usr/local/cuda/lib64/" > cuda.conf'
sudo ldconfig

cd /usr/local/lib
sudo ln -s /opt/amazon/efa/lib64/libmpi.so ./libmpi.so

cd ~/packages
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 MPI_HOME=/opt/amazon/efa/ CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/cuda

# test all_reduce_perf
export CUDA_HOME=/usr/local/cuda
export EFA_HOME=/opt/amazon/efa
bin=$HOME/packages/nccl-tests/build/all_reduce_perf
LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$EFA_HOME/lib64 $bin -b 8 -e 8

# test MPI EFA
/opt/amazon/efa/bin/mpirun -n 8 -x NCCL_DEBUG=INFO -x FI_PROVIDER=efa -x LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$EFA_HOME/lib64 $bin -b 8 -e 8


# build pytorch, follow https://github.com/pytorch/pytorch#from-source

export PATH=$HOME/anaconda3/bin:$PATH
conda create -n pytorch_p36 python=3.6 -y
source activate pytorch_p36

conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing -y
conda install -c pytorch magma-cuda100 -y

cd ~/packages
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v1.1.0
git submodule sync
git submodule update --init --recursive

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install

# extra useful packages

wget https://download-ib01.fedoraproject.org/pub/epel/7/x86_64/Packages/e/epel-release-7-11.noarch.rpm
sudo rpm -Uvh epel-release*rpm
sudo yum install nload -y

sudo yum install -y mosh
sudo yum install -y htop
sudo yum install -y gdb
sudo yum install -y tmux

set +y
