import subprocess
from sys import platform

baseCommand = ''

if platform == "linux" or platform == "linux2":
    # linux
    baseCommand += 'export SuiteSparse_ROOT=~/Documents/SuiteSparse\n'
    baseCommand += 'export LIBRARY_PATH=/opt/intel/oneapi/mkl/2021.2.0/lib/intel64:$LIBRARY_PATH\n'
    baseCommand += 'export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/2021.2.0/lib/intel64:$LD_LIBRARY_PATH\n'
    runCommand = baseCommand + 'mkdir build\ncd build\nrm -rf CMakeCache.txt\ncmake -DCMAKE_BUILD_TYPE=Release ..\nmake -j 8'
elif platform == "darwin":
    # OS X, use g++-10 to support -fopenmp
    baseCommand += 'export SuiteSparse_ROOT=/usr/local/Cellar/suite-sparse/5.10.1\n'
    baseCommand += 'export LIBRARY_PATH=/usr/local/Cellar/metis/5.1.0/lib:$LD_LIBRARY_PATH\n'
    runCommand = baseCommand + 'mkdir build\ncd build\nrm -rf CMakeCache.txt\ncmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-11 ..\nmake -j 8'
elif platform == "win32":
    # Windows
    print('compile not tested on win32!\n')
    runCommand = baseCommand + 'mkdir build\ncd build\nrm -rf CMakeCache.txt\ncmake -DCMAKE_BUILD_TYPE=Release ..\nmake -j 8'

subprocess.call([runCommand], shell=True)