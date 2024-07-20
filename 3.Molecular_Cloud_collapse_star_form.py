'''THIS IS A CODE SIMULATING A COLLAPSING MOLECULAR CLOUD IN PYTHON USING DEDALUS LIBRARY. 
IT SOLVES THE CONTINUITY EQUATION , MOMENTUM EQUATIONs, ENERGY EQUATION,INTERNAL ENERGY EQUATION,RELATION BETWEEN PRESSURE AND INTERNAL ENERGY 
USING RUNGE KUTTA METHOD . 
IT ALSO SOLVES FOR THE GRAVITATIONAL ACCELERATION WITH BOUNDARY VALUE OF 0. THIS IS DONE USING FIPY. '''


import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid3D, DiffusionTerm
from numba import jit
import dedalus.public as d3
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

# ConstantsS
c = 3e8   # speed of light
G = 6.67430e-11  # universal gravitational constant
Mstar = 1.989e32  # mass of gas cloud
Rstar = 10.7e14 # Dimensions of gas cloud
pi = np.pi  # pi
Nx = Ny = Nz = 30  # number of grid spacings in x, y, and z
stop_sim_time = 5e10  # Simulation stops at this time
timestep = 1e8 # timestep
dtype = np.float64  # data type for arrays
gamma = 1.5  # average ratio of C_p/C_v for the gas (due to presence of H and traces of He)
k_B = 1.380649e-16  # Boltzmann constant in cgs units (erg/K)
T = 2.73  # Temperature in Kelvin
mu = 2.3  # Mean molecular weight for molecular hydrogen
m_H = 1.6735575e-24  # Mass of a hydrogen atom in grams

# Coordinates and distributor
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype)


#opening file handler
f=open("Molecular_cloud.txt",'w')

# Bases
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Rstar, Rstar))
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Rstar, Rstar))
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Rstar, Rstar))

# Fields
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
p = dist.Field(name='p', bases=(xbasis, ybasis, zbasis)) #pressure
rho = dist.Field(name='rho', bases=(xbasis, ybasis, zbasis)) #density
vx = dist.Field(name='vx', bases=(xbasis, ybasis, zbasis)) #velocity in x dir.
vy = dist.Field(name='vy', bases=(xbasis, ybasis, zbasis)) #velocity in y dir.
vz = dist.Field(name='vz', bases=(xbasis, ybasis, zbasis)) #velocity in z dir.

E = dist.Field(name='E', bases=(xbasis, ybasis, zbasis))  # Net Energy
U = dist.Field(name='U', bases=(xbasis, ybasis, zbasis))  # internal energy
potential = dist.Field(name='potential', bases=(xbasis, ybasis, zbasis))  #gravitational potential

np.random.seed(42)
# Initial Conditions
vx['g'] = 0
vy['g'] = 0
vz['g'] = 0
rho['g'] = (1e-13)+2e-14*(np.random.random(rho['g'].shape))   
p['g'] = (rho['g'] * k_B * T) / (mu * m_H)
U['g'] = p['g']/(gamma-1)


# Set up FiPy grid
nx = ny = nz = Nx
dx = dy = dz = 2 * Rstar / (nx - 1)
mesh = Grid3D(dx=dx, dy=dy, dz=dz, nx=nx, ny=ny, nz=nz)

# Initial density for FiPy
values = rho['g'].flatten()

density_fipy = CellVariable(name="density", mesh=mesh, value=values)

# Potential field for FiPy
pot = CellVariable(name="pot", mesh=mesh)

# Boundary conditions for FiPy
pot.constrain(0, mesh.facesLeft)
pot.constrain(0, mesh.facesBottom)
pot.constrain(0, mesh.facesTop)
pot.constrain(0, mesh.facesRight)
pot.constrain(0, mesh.facesFront)
pot.constrain(0, mesh.facesBack)

# Poisson equation in FiPy
poisson_eq = DiffusionTerm(coeff=1.0) == 4*pi*G*density_fipy

poisson_eq.solve(var=pot)
E['g']=U['g']+potential['g']*rho['g']

problem1 = d3.IVP([rho, vx, vy, vz, E, U, p], namespace=locals())
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])
dz = lambda A: d3.Differentiate(A, coords['z'])

# Correctly define equations using Dedalus differential operators
equations1 = [
    "dt(rho) = -dx(rho * vx)-dy(rho * vy)-dz(rho * vz)",
    "dt(vx) = (-rho * dx(potential)  - dx(p) - dx(rho * vx * vx) - dy(rho * vx * vy) - dz(rho * vx * vz) + (dx(rho * vx) + dy(rho * vy) + dz(rho * vz)) * vx) / rho",
    "dt(vy) = (-rho * dy(potential) - dy(p) - dx(rho * vy * vx) - dy(rho * vy * vy) - dz(rho * vy * vz) + (dx(rho * vx) + dy(rho * vy) + dz(rho * vz)) * vy) / rho",
    "dt(vz) = (-rho * dz(potential)- dz(p) - dx(rho * vz * vx) - dy(rho * vz * vy) - dz(rho * vz * vz) - (dx(rho * vx) + dy(rho * vy) + dz(rho * vz)) * vz) / rho",
    "dt(E) = -dx((E + p) * vx) - dy((E + p) * vy) - dz((E + p) * vz) +rho * (dx(potential)* vx + dy(potential)* vy + dz(potential) * vz)",
    "U = E - 0.5 * rho * (vx * vx + vy * vy + vz * vz)",
    "p = (gamma - 1) * U"
]

for i, eq in enumerate(equations1):
    try:
        print(f"Adding equation {i}: {eq}")
        problem1.add_equation(eq)
    except Exception as e:
        logger.error("Error adding equation %d: %s", i, e)
# Solve the problem
current_sim_time = 0
try:
    solver = problem1.build_solver(d3.RK443)
    solver.stop_sim_time = stop_sim_time

    while solver.proceed:
        # Update density for FiPy
        density_fipy.value = rho['g'].flatten()
        
        # Solve Poisson equation for potential
        poisson_eq.solve(var=pot)
    
        # Compute gravitational accelerations from potential
        pot_array = np.reshape(pot.value, (nx, ny, nz))

        potential['g']=pot_array
    
        # Update Dedalus fields with computed accelerations
    

        solver.step(timestep)
        current_sim_time += timestep

        if solver.iteration % 10== 0:
            #printing the maximum and minimum for various parameters
            f.write(f"Time={current_sim_time}\n")
            f.write(f'Completed iteration {solver.iteration}\n')
            f.write(f"Min density: {np.min(rho['g'])}\n")
            f.write(f"Max density: {np.max(rho['g'])}\n")
            f.write(f"Min pressure: {np.min(p['g'])}\n")
            f.write(f"Max pressure: {np.max(p['g'])}\n")
            f.write(f"Min energy: {np.min(E['g'])}\n")
            f.write(f"Max energy: {np.max(E['g'])}\n")
            f.write(f"Max velx: {np.max(vx['g'])}\n")
            f.write(f"Max vely: {np.max(vy['g'])}\n")
            f.write(f"Max velz: {np.max(vz['g'])}\n")
            f.write(f"Max Internal Energy: {np.max(U['g'])}\n")
            f.write(f"Min Internal Energy: {np.min(U['g'])}\n")
            f.write("\n")

            rho_data = (rho(z=0).evaluate()['g']).reshape(Nx, Ny)
            p_data = (p(z=0).evaluate()['g']).reshape(Nx, Ny)
            potential_data = (potential(z=0).evaluate()['g']).reshape(Nx, Ny)
            energy_data = (E(z=0).evaluate()['g']).reshape(Nx, Ny)
            U_data = (U(z=0).evaluate()['g']).reshape(Nx, Ny)
            X_data = np.linspace(-Rstar, Rstar, Nx)
            Y_data = np.linspace(-Rstar, Rstar, Ny)

            fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 2 rows, 2 columns
            fig.suptitle(f"VALUE OF PARAMETERS IN Z=0 PLANE\nCurrent Simulation Time: {current_sim_time:.2e} seconds")

            # Plot the first contour plot
            contour1 = axs[0, 0].contourf(X_data, Y_data, rho_data, cmap='turbo')
            axs[0, 0].set_title('Density')
            fig.colorbar(contour1, ax=axs[0, 0])

            # Plot the second contour plot
            contour2 = axs[0, 1].contourf(X_data, Y_data, p_data, cmap='turbo')
            axs[0, 1].set_title('Pressure')
            fig.colorbar(contour2, ax=axs[0, 1])

            # Plot the third contour plot
            contour3 = axs[1, 0].contourf(X_data, Y_data, potential_data, cmap='turbo')
            axs[1, 0].set_title('potential')
            fig.colorbar(contour3, ax=axs[1, 0])

            # Plot the fourth contour plot
            contour4 = axs[1, 1].contourf(X_data, Y_data, U_data, cmap='turbo')
            axs[1, 1].set_title('U')
            fig.colorbar(contour4, ax=axs[1, 1])

            plt.tight_layout()
            plt.show()

except Exception as e:
    logger.error("Solver build failed: %s", e)
f.close()
