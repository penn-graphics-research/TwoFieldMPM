import subprocess

numThreads = '16'
baseCommand = 'export MKL_NUM_THREADS=' + numThreads + \
    '\nexport OMP_NUM_THREADS=' + numThreads + \
    '\nexport VECLIB_MAXIMUM_THREADS=' + numThreads + \
    '\n'

# for nSeg in ['24']:
#     # for nuC in ['0.75']:
#     for nuC in ['1000']:
#         runCommand = baseCommand + ' bin/ipc_2d 1 0 ' + nSeg + ' ' + nuC
#         subprocess.call([runCommand], shell=True)

# for nSeg in ['1 8', '2 16', '4 32', '8 64', '16 128']:
# for nSeg in ['32 1']:
#     for dHat in ['0.1 2e6']:
#         runCommand = baseCommand + ' bin/ipc_2d 2 0 ' + nSeg + ' ' + dHat
#         subprocess.call([runCommand], shell=True)

# for nSeg in ['1 8', '2 16', '4 32', '8 64']:
# # for nSeg in ['16 1']:
#     for dHat in ['0.1 1e6']:
#         runCommand = baseCommand + ' bin/ipc_3d 1 0 ' + nSeg + ' ' + dHat
#         subprocess.call([runCommand], shell=True)

for nSeg in ['10', '20', '40', '80', '160']:
# for nSeg in ['160']:
    # runCommand = baseCommand + ' bin/fem_2d 5 0 ' + nSeg # static
    # runCommand = baseCommand + ' bin/fem_2d 4 0 ' + nSeg + ' 1'# dynamic
    # runCommand = baseCommand + ' bin/ipc_2d 3 0 ' + nSeg # static
    runCommand = baseCommand + ' bin/ipc_2d 0 0 ' + nSeg + ' 1' # dynamic
    subprocess.call([runCommand], shell=True)