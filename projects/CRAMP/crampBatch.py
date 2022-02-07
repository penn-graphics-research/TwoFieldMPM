import subprocess

#TEST CONTROL CENTER
#Set which tests you want to run in the following lists of demos, then see next section of controls
sectorA = [0, 0, 0, 0]            #[uniaxialTension with AnisoMPM, SENT with Rankine Damage, SENT with Tanh Damage, Shear Fracture with Rankine] 
sectorB = [0, 0]                  #[Damage Suite FCR, Damage Suite NH]
sectorC = [0]                     #[Numerical Fracture Exploration (SENT, 150% Displacement Stretch, No Damage)]
sectorD = [1]                     #[Pipe Flow with Viscous Fluid]

#TEST CONTROL SUBSTATION
#Set what runs you want for each demo (e.g. run 0 degree and 90 degree fibers whenever diskShoot is run)
test1 = [1, 0, 0, 0, 0]                 #uniaxialTension: [control, eta, zeta, p, dMin]
test2 = [0, 1, 1, 1, 0, 0]              #SENT with displacement BCs and Rankine Damage: [control, Gf, sigmaC, alpha, dMin, minDp]
test3 = [0, 1, 1, 0, 0, 0]              #SENT with displacement BCs and Tanh Damage: [control, lamC, width, alpha, dMin, mnDp]
test4 = [1, 1, 1, 1]                    #Damage Test Suite FCR [SENT FCR stress, SENT FCR stretch, shear FCR stress, shear FCR stretch]
test5 = [0, 0, 0, 0, 0, 1]              #Damage Test Suite NH [SENT NH stress, SENT NH stretch, shear NH stress, shear NH stretch, LARGER shear stretch with NH, LARGER SENT stretchDamage with NH]
test6 = [0, 1]                          #Num Frax Exploration [Variable dx, Variable PPC]
pipeFlowTests = [0, 0, 1]               #Pipe Flow Tests [Horizontal with Dirichlet, Vertical with Dirichlet, Horizontal with Elastic Walls]

################################
########### SECTOR A ###########
################################

#Uniaxial Tension Test with Displacement BCs and AnisoMPM Damage
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

#SENT with Displacement BCs and Rankine Damage
if sectorA[1]:
    constants = [0.0223, 2600, 1.0, 0.25, 1.0] #Gf, sigmaC, alpha, dMin, minDp
    GfArray = [0.005, 0.04] #[0.015, 0.02, 0.025, 0.03]
    sigmaCArray = [2000, 3000] #[2400, 2500, 2700, 2800]
    alphaArray = [0.9, 0.95, 0.99, 1.01, 1.05, 1.1] #[0.5, 0.75, 1.25, 1.5]
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

#SENT with Displacement BCs and Tanh Damage
if sectorA[2]:
    constants = [1.5, 0.1, 1.0, 0.25, 1.0] #lamC, tanhWidth, alpha, dMin, minDp
    lamCArray = [1.3, 1.4, 1.6, 1.75, 1.8, 2.0] #[1.001, 1.003, 1.007, 1.009]
    tanhWidthArray = [0.25, 0.2, 0.15, 0.125, 0.05]
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
    if test3[1]:
        for lamC in lamCArray:
            tanhWidth = constants[1]
            alpha = constants[2]
            dMin = constants[3]
            minDp = constants[4]
            runCommand = './cramp 210 ' + str(lamC) + ' ' + str(tanhWidth) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
            print(runCommand)
            subprocess.call([runCommand], shell=True)
    if test3[2]:
        for tanhWidth in tanhWidthArray:
            lamC = constants[0]
            alpha = constants[2]
            dMin = constants[3]
            minDp = constants[4]
            runCommand = './cramp 210 ' + str(lamC) + ' ' + str(tanhWidth) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
            print(runCommand)
            subprocess.call([runCommand], shell=True)

################################
########### SECTOR B ###########
################################

#Damage Suite FCR
if sectorB[0]:
    if test4[0]: #SENT, stress based
        Gf = 0.0223
        sigmaC = 2600
        alpha = 1.0
        dMin = 0.25
        minDp = 1.0
        runCommand = './cramp 209 ' + str(Gf) + ' ' + str(sigmaC) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
        print(runCommand)
        subprocess.call([runCommand], shell=True)
    if test4[1]: #SENT, stretch based
        lamC = 1.5
        tanhWidth = 0.15
        alpha = 1.0
        dMin = 0.25
        minDp = 1.0
        runCommand = './cramp 210 ' + str(lamC) + ' ' + str(tanhWidth) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
        print(runCommand)
        subprocess.call([runCommand], shell=True)
    if test4[2]: #shear fracture, stress based
        Gf = 22300
        sigmaC = 570000000
        alpha = 1.0
        dMin = 0.25
        minDp = 1.0
        runCommand = './cramp 211 ' + str(Gf) + ' ' + str(sigmaC) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
        print(runCommand)
        subprocess.call([runCommand], shell=True)
    if test4[3]: #shear fracture, stretch based
        lamC = 1.006
        tanhWidth = 0.001
        alpha = 1.0
        dMin = 0.25
        minDp = 1.0
        runCommand = './cramp 212 ' + str(lamC) + ' ' + str(tanhWidth) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
        print(runCommand)
        subprocess.call([runCommand], shell=True)

#Damage Suite NH
if sectorB[1]:
    if test5[0]: #SENT, stress based
        Gf = 0.0223
        sigmaC = 2600
        alpha = 1.0
        dMin = 0.25
        minDp = 1.0
        runCommand = './cramp 213 ' + str(Gf) + ' ' + str(sigmaC) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
        print(runCommand)
        subprocess.call([runCommand], shell=True)
    if test5[1]: #SENT, stretch based
        lamC = 1.5
        tanhWidth = 0.15
        alpha = 1.0
        dMin = 0.25
        minDp = 1.0
        runCommand = './cramp 214 ' + str(lamC) + ' ' + str(tanhWidth) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
        print(runCommand)
        subprocess.call([runCommand], shell=True)
    if test5[2]: #shear fracture, stress based
        Gf = 22300
        sigmaC = 570000000
        alpha = 1.0
        dMin = 0.25
        minDp = 1.0
        runCommand = './cramp 215 ' + str(Gf) + ' ' + str(sigmaC) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
        print(runCommand)
        subprocess.call([runCommand], shell=True)
    if test5[3]: #shear fracture, stretch based
        lamC = 1.006
        tanhWidth = 0.001
        alpha = 1.0
        dMin = 0.25
        minDp = 1.0
        runCommand = './cramp 216 ' + str(lamC) + ' ' + str(tanhWidth) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
        print(runCommand)
        subprocess.call([runCommand], shell=True)
    if test5[4]: #LARGER shear fracture, stretch based
        lamC = 1.8
        tanhWidth = 0.2
        alpha = 1.0
        dMin = 0.25
        minDp = 1.0
        runCommand = './cramp 219 ' + str(lamC) + ' ' + str(tanhWidth) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
        print(runCommand)
        subprocess.call([runCommand], shell=True)
    if test5[5]: #LARGER uniaxial tension fracture, stretch based
        lamC = 2.5
        tanhWidth = 0.08
        alpha = 1.0
        dMin = 0.25
        minDp = 1.0
        runCommand = './cramp 221 ' + str(lamC) + ' ' + str(tanhWidth) + ' ' + str(alpha) + ' ' + str(dMin) + ' ' + str(minDp)
        print(runCommand)
        subprocess.call([runCommand], shell=True)

################################
########### SECTOR C ###########
################################

#Numerical Fracture Exploration
if sectorC[0]:
    if test6[0]:
        dxArray = [0.0001, 0.00025, 0.0005] #[0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.004]
        ppc = 4
        for dx in dxArray:
            runCommand = './cramp 217 ' + str(dx) + ' ' + str(ppc)
            print(runCommand)
            subprocess.call([runCommand], shell=True)
    if test6[1]:
        ppcArray = [4, 9, 16]
        dx = 0.00025
        for ppc in ppcArray:
            runCommand = './cramp 217 ' + str(dx) + ' ' + str(ppc)
            print(runCommand)
            subprocess.call([runCommand], shell=True)

################################
########### SECTOR D ###########
################################

#Pipe Flow with Viscous Fluid
if sectorD[0]:
    if pipeFlowTests[0]:
        #horizontal pipe with dirichlet BCs
        bulk = 10000
        gamma = 7
        viscosityArray = [0.0075, 0.01]
        for viscosity in viscosityArray:
            runCommand = './cramp 222 ' + str(bulk) + ' ' + str(gamma) + ' ' + str(viscosity)
            print(runCommand)
            subprocess.call([runCommand], shell=True)
    if pipeFlowTests[1]:
        #vertical pipe with dirichlet BCs
        bulk = 10000
        gamma = 7
        viscosityArray = [0.004]
        for viscosity in viscosityArray:
            runCommand = './cramp 223 ' + str(bulk) + ' ' + str(gamma) + ' ' + str(viscosity)
            print(runCommand)
            subprocess.call([runCommand], shell=True)
    if pipeFlowTests[2]:
        #horizontal with deformable pipe walls
        bulk = 10000
        gamma = 7
        viscosityArray = [0.004]
        for viscosity in viscosityArray:
            runCommand = './cramp 224 ' + str(bulk) + ' ' + str(gamma) + ' ' + str(viscosity)
            print(runCommand)
            subprocess.call([runCommand], shell=True)
