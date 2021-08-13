import numpy as np
from scipy.constants import c as c0

#######
# SIMULATION PARAMETERS
#######


# COMPUTE DEFAULT GRID RESOLUTION
dz1 = np.min(LAMBDA)/nmax/NRES
dz2 = dmin/NDRES
dz = np.min(dz1, dz2)

# SNAP GRID TO CRITICAL DIMENSIONS
N = np.ceil(dc/dz)
dz = dc/N

#######
# BUILD DEVICE ON GRID
#######


# COMPUTE TIME STEP
dt = N*dz/(2*t)

# INITIALIZE MATERIALS TO FREE SPACE
ER = np.ones(1, Nz)
UR = np.ones(1, Nz)

# COMPUTE UPDATE COEFFICIENTS
mEy = (c0*dt)/ER;
mHx = (c0*dt)/UR;

# INITIALIZE FIELDS
Ey = np.zeros(N)
Hx = Ey

# SOURCE PARAMETERS
f2 = 5.0 * gigahertz
NFREQ = 1000;
FREQ = np.linspace(0, f2, NFREQ)

# COMPUTE SOURCE
tau = 0.5*fmax
t1 = 6*tau

def e_source(t):
    return np.exp(-((t-t1)/tau)**2)

A = -np.sqrt(ER/UR)
ddt =(N*dz)/(2*c0)+dt/2

def h_source(t):
    return A*e_source(t+ddt)

# INITIALIZE FOURIER TRANSFORMS
K = np.exp(-li*2*np.pi*dt*FREQ)
REF = zeros(1, NFREQ)
TRN = zeros(1, NFREQ)
SRC = zeros(1, NFREQ)


# INITIALIZE BOUNDARY FIELDS
H1, H2, H3, E1, E2, E3 = 0, 0, 0, 0, 0, 0

#######
# MAIN FDTD LOOP (Ey/Hx MODE)
#######

for T in range(1,STEPS):
    ######
    # CODE FOR UPDATE EQUATIONS
    ######
    # Update H from E (Dirichlet Boundary Conditions)
    for nz in range(1, Nz - 1):
        Hx(nz) = Hx(nz) + mHx(nz)*(Ey(nz+1) - Ey(nz))
    Hx(Nz) = Hx(Nz) + mHx(Nz)*(E3 - Ey(Nz))
    H3 = H2
    H2 = H1
    H1 = Hx(1)
    
    # Update E from H (Dirichlet Boundary Conditions)
    Ey(1) = Ey(1) + mEy(1)*(Hx(1) - H3)
    for nz in range(1, Nz - 1):
        Ey(nz) = Ey(nz) + mEy(nz)*(Hx(nz) - Hx(nz-1))
    E3 = E2
    E2 = E1
    E1 = Ey(1)
    
    ######
    # CODE FOR FOURIER TRANSFORMS
    ######
    for nf in range(1:NFREQ):
        REF(nf) = REF(nf) + (K(nf)**T)*Ey(1)
        REF(nf) = REF(nf) + (K(nf)**T)*Ey(1)
        REF(nf) = REF(nf) + (K(nf)**T)*Ey(1)
    
    ######
    # CODE FOR VISUALIZING
    ######
    
    
# FINISH FOURIER TRANSFORMS   
REF = REF*dt
TRN = TRN*dt
SRC = SRC*dt

# COMPUTE REFLECTANCE, TRANSMITTANCE AND CONSERVATION
REFLEC = np.abs(REF/SRC)**2
TRANS = np.abs(TRN/SRC)**2
CONS = RELEC + TRANS

# SETTING UP TIME AND FREQUENCY AXIS
T = np.arange(0,STEPS-1)*dt
fmax = 0.5/dt
freq = np.linspace(-fmax, fmax, STEPS)      #steps should be odd

######
# CODE FOR VISUALIZING
######
