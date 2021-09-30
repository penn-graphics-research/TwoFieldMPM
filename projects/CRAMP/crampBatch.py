import subprocess

#TEST CONTROL CENTER
#Set which tests you want to run in the following lists of demos, then see next section of controls
sectorA = [1]                #[uniaxialTension]

#TEST CONTROL SUBSTATION
#Set what runs you want for each demo (e.g. run 0 degree and 90 degree fibers whenever diskShoot is run)
test1 = [1, 0, 0, 0, 0]                 #uniaxialTension: [control, eta, zeta, p, dMin]

if sectorA[0]:
    constants = [10, 10000, 0.03, 0.4]
    etaArray = [1, 100, 1000]
    zetaArray = [100, 1000, 100000]
    pArray = [0.0275, 0.0325, 0.035]
    dMinArray = [0.25, 0.3, 0.35]
    if test1[0]:
        eta = constants[0]
        zeta = constants[1]
        p = constants[2]
        dMin = constants[3]
        runCommand = './cramp 204 ' + str(eta) + ' ' + str(zeta) + ' ' + str(p) + ' ' + str(dMin)
        print(runCommand)
        subprocess.call([runCommand], shell=True)
    if test1[1]:
        for eta in etaArray:
            zeta = constants[1]
            p = constants[2]
            dMin = constants[3]
            runCommand = './cramp 204 ' + str(eta) + ' ' + str(zeta) + ' ' + str(p) + ' ' + str(dMin)
            print(runCommand)
            subprocess.call([runCommand], shell=True)
    if test1[2]:
        for zeta in zetaArray:
            eta = constants[0]
            p = constants[2]
            dMin = constants[3]
            runCommand = './cramp 204 ' + str(eta) + ' ' + str(zeta) + ' ' + str(p) + ' ' + str(dMin)
            print(runCommand)
            subprocess.call([runCommand], shell=True)
    if test1[3]:
        for p in pArray:
            eta = constants[0]
            zeta = constants[1]
            dMin = constants[3]
            runCommand = './cramp 204 ' + str(eta) + ' ' + str(zeta) + ' ' + str(p) + ' ' + str(dMin)
            print(runCommand)
            subprocess.call([runCommand], shell=True)
    if test1[4]:
        for dMin in dMinArray:
            eta = constants[0]
            zeta = constants[1]
            p = constants[2]
            runCommand = './cramp 204 ' + str(eta) + ' ' + str(zeta) + ' ' + str(p) + ' ' + str(dMin)
            print(runCommand)
            subprocess.call([runCommand], shell=True)

