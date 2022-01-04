import subprocess

#TEST CONTROL CENTER
#Set which tests you want to run in the following lists of demos, then see next section of controls
sectorA = [0, 0, 1]                #[uniaxialTension with AnisoMPM, SENT with Rankine Damage, SENT with Tanh Damage]

#TEST CONTROL SUBSTATION
#Set what runs you want for each demo (e.g. run 0 degree and 90 degree fibers whenever diskShoot is run)
test1 = [1, 0, 0, 0, 0]                 #uniaxialTension: [control, eta, zeta, p, dMin]
test2 = [1, 1, 1, 1, 1, 1]              #SENT with displacement BCs and Rankine Damage: [control, Gf, sigmaC, alpha, dMin, minDp]
test3 = [1, 1, 1, 1, 1, 1]              #SENT with displacement BCs and Tanh Damage: [control, lamC, width, alpha, dMin, mnDp]

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

if sectorA[1]:
    constants = [0.0223, 2600, 1.0, 0.25, 1.0] #Gf, sigmaC, alpha, dMin, minDp
    GfArray = [0.015, 0.02, 0.025, 0.03]
    sigmaCArray = [2400, 2500, 2700, 2800]
    alphaArray = [0.5, 0.75, 1.25, 1.5]
    dMinArray = [0.15, 0.2, 0.3, 0.35]
    minDpArray = [0.8, 0.9, 0.95]
    if test2[0]:
        Gf = constants[0]
        sigmaC = constants[1]
        alpha = constants[2]
        dMin = constants[3]
        minDp = constants[4]
        runCommand = './cramp 209 ' + str(Gf) + ' ' + str(sigmaC) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
        print(runCommand)
        subprocess.call([runCommand], shell=True)
    if test2[1]:
        for Gf in GfArray:
            sigmaC = constants[1]
            alpha = constants[2]
            dMin = constants[3]
            minDp = constants[4]
            runCommand = './cramp 209 ' + str(Gf) + ' ' + str(sigmaC) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
            print(runCommand)
            subprocess.call([runCommand], shell=True)
    if test2[2]:
        for sigmaC in sigmaCArray:
            Gf = constants[0]
            alpha = constants[2]
            dMin = constants[3]
            minDp = constants[4]
            runCommand = './cramp 209 ' + str(Gf) + ' ' + str(sigmaC) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
            print(runCommand)
            subprocess.call([runCommand], shell=True)
    if test2[3]:
        for alpha in alphaArray:
            Gf = constants[0]
            sigmaC = constants[1]
            dMin = constants[3]
            minDp = constants[4]
            runCommand = './cramp 209 ' + str(Gf) + ' ' + str(sigmaC) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
            print(runCommand)
            subprocess.call([runCommand], shell=True)
    if test2[4]:
        for dMin in dMinArray:
            Gf = constants[0]
            sigmaC = constants[1]
            alpha = constants[2]
            minDp = constants[4]
            runCommand = './cramp 209 ' + str(Gf) + ' ' + str(sigmaC) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
            print(runCommand)
            subprocess.call([runCommand], shell=True)
    if test2[5]:
        for minDp in minDpArray:
            Gf = constants[0]
            sigmaC = constants[1]
            alpha = constants[2]
            dMin = constants[3]
            runCommand = './cramp 209 ' + str(Gf) + ' ' + str(sigmaC) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
            print(runCommand)
            subprocess.call([runCommand], shell=True)

if sectorA[2]:
    constants = [1.1, 0.01, 1.0, 0.25, 1.0] #lamC, tanhWidth, alpha, dMin, minDp
    # lamCArray = [0.015, 0.02, 0.025, 0.03]
    # tanhWidthArray = [2400, 2500, 2700, 2800]
    # alphaArray = [0.5, 0.75, 1.25, 1.5]
    # dMinArray = [0.15, 0.2, 0.3, 0.35]
    # minDpArray = [0.8, 0.9, 0.95]
    if test3[0]:
        lamC = constants[0]
        tanhWidth = constants[1]
        alpha = constants[2]
        dMin = constants[3]
        minDp = constants[4]
        runCommand = './cramp 210 ' + str(lamC) + ' ' + str(tanhWidth) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
        print(runCommand)
        subprocess.call([runCommand], shell=True)




