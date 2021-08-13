import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.constants import c as c0

#######
# CONSTANTS
#######

m = 1
cm = 0.01
mm = 0.001

s = 1
GHz = 10**9

#######
# SIMULATION PARAMETERS
#######
materialER = 1              # Largest material relative permiability
materialUR = 1              # Largest materail relative permissivity
fmax = 1*GHz                # Gigahertz
dmin = 30.24*cm                # Critical length in the model (cm)
materialThickness = 30.24*cm   # Materal thickness

# COMPUTE DEFAULT GRID RESOLUTION
NRES = 20                               # Number of cells allocated to min wavelength
nmax = np.sqrt(materialER*materialUR)   # Maximum refractive index
minLambda = c0/(fmax*nmax)
dz1 = minLambda/NRES

NDRES = 40                               # Number of cells allocated to critical length
dz2 = dmin/NDRES
dz = np.amin([dz1, dz2])                # Minimum of the resolutions

# SNAP GRID TO CRITICAL DIMENSIONS
N = int(np.ceil(materialThickness/dz))                 # Size of device
dz = materialThickness/N                               # Size of a cell

#######
# BUILD DEVICE ON GRID
#######
NSPACE = 50                             # Number of free space around device
NTRN, NREF, NSRC = 1, 1, 1
NRECORD = NTRN + NREF + NSRC            # Number of cells for recording TRN, REF and SRC ( 1 each )
Nz = N + 2*NSPACE + NRECORD             # Total number of grids
Nz1 = NSPACE + NREF + NSRC + 1          # Beginning of material
Nz2 = Nz1 + N - 1

# INITIALIZE MATERIALS TO FREE SPACE
ER = np.ones(Nz)
UR = np.ones(Nz)

# INITIALIZE FIELDS
Ey = np.zeros(Nz)
Hx = Ey

# INITIALIZE DEVICE INTO SPACE
UR[Nz1:Nz2] = materialUR
ER[Nz1:Nz2] = materialER

# COMPUTE TIME STEP
Nt = 1                                  # Number of cells devoted to half of dt
dt = Nt*dz/(2*c0)

# COMPUTE UPDATE COEFFICIENTS
mEy = (c0*dt)/ER;
mHx = (c0*dt)/UR;

# COMPUTE SOURCE PARAMETERS
tau = 1/(2*fmax)
t1 = 6*tau

# COMPUTE NUMBER OF TIME STEPS
tprop = (nmax*Nz*dz)/c0                 # Time taken for a wave to propagate the whole grids
                                        # ones
tTotal = 12*tau + 5*tprop
STEPS = int(np.ceil(tTotal/dt))

# COMPUTE SOURCE
t = np.arange(0,STEPS-1)*dt
ERsrc = 1
URsrc = 1
A = -np.sqrt(ERsrc/URsrc)
delta =(N*dz)/(2*c0)+dt/2

Esrc = np.exp(-((t-t1)/tau)**2)
Hsrc = A*np.exp(-((t+delta-t1)/tau)**2)

# INITIALIZE FOURIER TRANSFORMS
NFREQ = 1000
FREQ = np.linspace(0, fmax, NFREQ)
K = np.exp(-1.0j*2*np.pi*dt*FREQ)
REF = np.zeros(NFREQ)
TRN = np.zeros(NFREQ)
SRC = np.zeros(NFREQ)


# INITIALIZE BOUNDARY FIELDS
H1, H2, H3, E1, E2, E3 = 0, 0, 0, 0, 0, 0

#######
# MAIN FDTD LOOP (Ey/Hx MODE)
#######
z = np.arange(0,Nz)
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
ax.axis('off')
ims = []

for T in range(1,STEPS):
    ######
    # CODE FOR UPDATE EQUATIONS
    ######
    # Update H from E (Dirichlet Boundary Conditions)
    for nz in range(1, Nz - 1):
        Hx[nz] = Hx[nz] + mHx[nz]*(Ey[nz+1] - Ey[nz])
    Hx[Nz-1] = Hx[Nz-1] + mHx[Nz-1]*(E3 - Ey[Nz-1])
    H3 = H2
    H2 = H1
    H1 = Hx[0]
    
    # Update E from H (Dirichlet Boundary Conditions)
    Ey[0] = Ey[0] + mEy[0]*(Hx[0] - H3)
    for nz in range(1, Nz - 1):
        Ey[nz] = Ey[nz] + mEy[nz]*(Hx[nz] - Hx[nz-1])
    E3 = E2
    E2 = E1
    E1 = Ey[0]
    
    ######
    # CODE FOR FOURIER TRANSFORMS
    ######
    # for nf in range(1,NFREQ):
    #     REF[nf] = REF[nf] + (K[nf]**T)*Ey[0]
    #     REF[nf] = REF[nf] + (K[nf]**T)*Ey[Nz-1]
    #     REF[nf] = REF[nf] + (K[nf]**T)*Esrc[T]
    
    Ey[6] = Ey[6] + Esrc[T-1]
    Hx[6] = Hx[6] + Hsrc[T-1]
    
    ######
    # CODE FOR VISUALIZING
    ######
    if T%5==0:
        imHx, = ax.plot(Hx, animated=True)
        imEy, = ax.plot(-Ey, animated=True)
        imb = ax.text(0.05, 0.95, 'frame = {0:d} OF {1:0.0f}'.format(T, STEPS), va='top', ha='left', color=[0, 0, 0], transform=ax.transAxes)
        ims.append([imHx,imEy,imb])
        
ani = anim.ArtistAnimation(fig, artists=ims, interval=10, repeat_delay=0)
fig.show()
    
'''   
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
'''