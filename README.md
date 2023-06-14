# TwoFieldMPM - Run by Bow
![C/C++ CI](https://github.com/penn-graphics-research/Bow/workflows/C/C++%20CI/badge.svg)

# Compile CRAMP (and replicate current 3D issue)
``` bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make cramp -j8
cd bin
./cramp 300
```
or 
``` bash
python build.py
```
Excutables are collected in the `bin` folder

### Compile on OSX
Alternative to gcc, one can also use `brew install llvm` to install llvm and specify the compiler in cmake ` cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/local/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/usr/local/opt/llvm/bin/clang++` to compile.

# Optional Dependencies
### CUDA (used by AMGCL)

Put the following in ~/.bashrc for correct CUDA linking
``` bash
export PATH="/usr/local/cuda-10.1/bin/:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda-10.1"
export CUDA_PATH="/usr/local/cuda-10.1"
```

### SuiteSparse 

1. Install [MKL](https://software.intel.com/content/www/us/en/develop/articles/qualify-for-free-software.html#student) (Intel Math Kernel Library, free tools for students)

2. Build SuiteSparse from source (with MKL linking flags)
``` bash
sudo apt install libomp-dev libmpc-dev

## Add the following lines into ~/.zshrc
export PATH=/snap/clion/current/bin/cmake/linux/bin:$PATH
export LIBRARY_PATH=/opt/intel/oneapi/mkl/2021.3.0/lib/intel64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/2021.3.0/lib/intel64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2021.3.0/linux/compiler/lib/intel64_lin:$LD_LIBRARY_PATH
export LD_PRELOAD=/opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_def.so.1:/opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_avx2.so.1:/opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_core.so:/opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_lp64.so:/opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_thread.so:/opt/intel/oneapi/compiler/2021.3.0/linux/compiler/lib/intel64_lin/libiomp5.so

git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
cd SuiteSparse
vim SuiteSparse_config/SuiteSparse_config.mk

## Modify CUDA_PATH like: CUDA_PATH = /usr/local/cuda-10.1
## Update CUDA architecture (e.g. remove -gencode=arch=compute_30,code=sm_30 \)

make library BLAS='-lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -lmkl_blacs_intelmpi_lp64 -liomp5' LAPACK='-lmkl_scalapack_lp64' -j 12
sudo cp -r lib /usr/local
sudo cp -r include /usr/local
```

3. Set environment variable
Add the following line to your `.bashrc`:
``` bash
export SuiteSparse_ROOT=/home/xuan/code/SuiteSparse
```

4. Set thread number per linear solver
```
export MKL_NUM_THREADS=16
export OMP_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16
```
Note that different OS can have different environment variable to control it.

# Create your own projects
CMake will automatically add subdirectories under `projects` folder which contain `CMakeLists.txt`. To create a new project that depends on `Bow`, clone your cmake project under project folder and compile `Bow` as usual.
