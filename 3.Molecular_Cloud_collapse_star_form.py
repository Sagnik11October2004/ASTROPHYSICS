'''THIS IS A CODE SIMULATING A COLLAPSING MOLECULAR CLOUD IN PYTHON USING DEDALUS LIBRARY. 
IT SOLVES THE CONTINUITY EQUATION , MOMENTUM EQUATIONs, ENERGY EQUATION,INTERNAL ENERGY EQUATION,RELATION BETWEEN PRESSURE AND INTERNAL ENERGY 
USING RUNGE KUTTA METHOD . 
IT ALSO SOLVES FOR THE GRAVITATIONAL ACCELERATION. '''



from numba import jit
import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

# Constants
c = 3e8   # speed of light
G = 6.67430e-11  # universal gravitational constant
Mstar = 1.989e30  # mass of gas cloud
Rstar = 10.7e17 # Dimensions of gas cloud
pi = np.pi   # pi
Nx =20  # no. of grid spacings in x
Ny = 20    # no. of grid spacings in y
Nz = 20     # no. of grid spacings in z
stop_sim_time = 9e15  # Simulation stops at
timestep = 1e3 # timestep
dtype = np.float64  # data type for arrays
gamma = 1.5  # average ratio of C_p/C_v for the gas(due to presence of H and traces of He)
dV=(Rstar/(Nx-1))**3

# Coordinates and distributor
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype)

# Bases
xbasis = d3.Chebyshev(coords['x'], size=Nx, bounds=(-Rstar, Rstar))
ybasis = d3.Chebyshev(coords['y'], size=Ny, bounds=(-Rstar, Rstar))
zbasis = d3.Chebyshev(coords['z'], size=Nz, bounds=(-Rstar, Rstar))

# Fields
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
r_vec = dist.VectorField(coords, name='r_vec', bases=(xbasis, ybasis, zbasis))
r_vec['g'][2] = z
r_vec['g'][1] = y
r_vec['g'][0] = x
radius = dist.Field(name='radius', bases=(xbasis, ybasis, zbasis))
radius['g'] = np.sqrt(x**2 + y**2 + z**2)

p = dist.Field(name='p', bases=(xbasis, ybasis, zbasis))  #pressure
rho = dist.Field(name='rho', bases=(xbasis, ybasis, zbasis))  #density

vx = dist.Field(name='vx', bases=(xbasis, ybasis, zbasis))  #vel. in x dir.
vy = dist.Field(name='vy', bases=(xbasis, ybasis, zbasis))  #vel. in y dir.
vz = dist.Field(name='vz', bases=(xbasis, ybasis, zbasis))  #vel. in z dir.
gx = dist.Field(name='gx', bases=(xbasis, ybasis, zbasis))  #acc. in x dir.
gy = dist.Field(name='gy', bases=(xbasis, ybasis, zbasis))  #acc. in y dir.
gz = dist.Field(name='gz', bases=(xbasis, ybasis, zbasis))  #acc. in z dir.
E = dist.Field(name='E', bases=(xbasis, ybasis, zbasis))    #total energy including gravitational potential
U = dist.Field(name='U', bases=(xbasis, ybasis, zbasis))    # internal energy

# Initial Conditions
vx['g'] = 0
vy['g'] = 0
vz['g'] = 0
rho['g'] = 0.0001*np.exp(3-(x/Rstar)**2 -(y/Rstar)**2 -(z/Rstar)**2)+0.001
p['g'] = 4.14e-12
U['g'] = (p['g'] / (gamma - 1))
E['g'] = U['g'] + 1.4
g=dist.Field(name='g', bases=(xbasis, ybasis, zbasis))  #stores enclosed mass
px=dist.Field(name='px', bases=(xbasis, ybasis, zbasis))
px['g']=x
py=dist.Field(name='py', bases=(xbasis, ybasis, zbasis))
py['g']=y
pz=dist.Field(name='pz', bases=(xbasis, ybasis, zbasis))
pz['g']=z
px=dist.Field(name='px', bases=(xbasis, ybasis, zbasis))
px['g']=x
py=dist.Field(name='py', bases=(xbasis, ybasis, zbasis))
py['g']=y
pz=dist.Field(name='pz', bases=(xbasis, ybasis, zbasis))
pz['g']=z


def group_and_sum_densities(rho, radius, Nx, Ny, Nz, px, py, pz):
    # Initialize a list to store grouped elements with their radius and density sum
    grouped_elements = []

    # Iterate through each element in the grid
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Calculate radius for the current element
                x_l = px['g'][i, j, k]
                y_l = py['g'][i, j, k]
                z_l = pz['g'][i, j, k]
                r = np.sqrt(x_l**2 + y_l**2 + z_l**2)
                
                # Evaluate density for the current element
                dens = rho(x=x_l, y=y_l, z=z_l).evaluate()['g'][0, 0, 0]
                
                # Append current element's radius, indices, and density to the list
                grouped_elements.append((r, i, j, k, dens))

    # Sort grouped elements based on radius (first element in tuple)
    grouped_elements_sorted = sorted(grouped_elements, key=lambda x: x[0]
   
    # Iterate through sorted grouped elements
    for idx, (r, i, j, k, dens) in enumerate(grouped_elements_sorted):
        sum_densities =0
        # Sum densities of groups with strictly smaller radius
        for prev_idx in range(idx):
            if grouped_elements_sorted[prev_idx][0] < r:
                sum_densities+= grouped_elements_sorted[prev_idx][4]
        g['g'][i,j,k]=sum_densities*dV




group_and_sum_densities(rho, radius, Nx, Ny, Nz, px, py, pz)

gx=(-G*g*px/(radius**3)).evaluate()
gy=(-G*g*py/(radius**3)).evaluate()
gz=(-G*g*pz/(radius**3)).evaluate()
# Problem Definition
problem1 = d3.IVP([rho, vx, vy, vz, E, U, p], namespace=locals())
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])
dz = lambda A: d3.Differentiate(A, coords['z'])

# Equations
equations1 = [
    "dt(rho) = -dx(rho * vx)-dy(rho * vy)-dz(rho * vz)",
    "dt(vx) = (rho * gx - dx(p) - dx(rho * vx * vx) - dy(rho * vx * vy) - dz(rho * vx * vz) + (dx(rho * vx) + dy(rho * vy) + dz(rho * vz)) * vx) / rho",
    "dt(vy) = (rho * gy - dy(p) - dx(rho * vy * vx) - dy(rho * vy * vy) - dz(rho * vy * vz) + (dx(rho * vx) + dy(rho * vy) + dz(rho * vz)) * vy) / rho",
    "dt(vz) = (rho * gz - dz(p) - dx(rho * vz * vx) - dy(rho * vz * vy) - dz(rho * vz * vz) + (dx(rho * vx) + dy(rho * vy) + dz(rho * vz)) * vz) / rho",
    "dt(E) = -dx((E + p) * vx) - dy((E + p) * vy) - dz((E + p) * vz) +rho * (gx * vx + gy * vy + gz * vz)",
    "U = E - 0.5 * rho * (vx * vx + vy * vy + vz * vz)",
    "p = (gamma - 1) * U"
]

for i, eq in enumerate(equations1):
    try:
        print(f"Adding equation {i}: {eq}")
        problem1.add_equation(eq)
    except Exception as e:
        logger.error("Error adding equation %d: %s", i, e)

current_sim_time = 0

try:
    solver = problem1.build_solver(d3.RK443)
    solver.stop_sim_time = stop_sim_time
    while solver.proceed:
        solver.step(timestep)
        
        group_and_sum_densities(rho, radius, Nx, Ny, Nz, px, py, pz)

        gx=(-G*g*px/(radius**3)).evaluate()
        gy=(-G*g*py/(radius**3)).evaluate()
        gz=(-G*g*pz/(radius**3)).evaluate()
        current_sim_time += timestep

        if solver.iteration % 10 == 0:
            print(f'Completed iteration {solver.iteration}')
            print(f"Min density: {np.min(rho['g'])}")
            print(f"Max density: {np.max(rho['g'])}")
            print(f"Min pressure: {np.min(p['g'])}")
            print(f"Max pressure: {np.max(p['g'])}")
            print(f"Min energy: {np.min(E['g'])}")
            print(f"Max energy: {np.max(E['g'])}")
            print(f"Max velx: {np.max(vx['g'])}")
            print(f"Max vely: {np.max(vy['g'])}")
            print(f"Max velz: {np.max(vz['g'])}")
            print(f"Max mass enclosed: {np.max(g['g'])}")
            print(f"Max Internal Energy: {np.max(U['g'])}")
            print(f"Min Internal Energy: {np.min(U['g'])}")
            
            rho_data = (rho(z=0).evaluate()['g']).reshape(Nx, Ny)
            p_data = (p(z=0).evaluate()['g']).reshape(Nx, Ny)
            g_data = (g(z=0).evaluate()['g']).reshape(Nx, Ny)
            gx_data = (gx(z=0).evaluate()['g']).reshape(Nx, Ny)
            gy_data = (gy(z=0).evaluate()['g']).reshape(Nx, Ny)
            gz_data = (gz(z=0).evaluate()['g']).reshape(Nx, Ny)
            energy_data = (E(z=0).evaluate()['g']).reshape(Nx, Ny)
            U_data = (U(z=0).evaluate()['g']).reshape(Nx, Ny)
            X_data = np.linspace(-Rstar, Rstar, Nx)
            Y_data = np.linspace(-Rstar, Rstar, Ny)            
            fig, axs = plt.subplots(2, 2, figsize=(5, 5))  # 2 row, 2 columns
            fig.suptitle(f"VALUE OF PARAMETERS IN Z=0 PLANE\nCurrent Simulation Time: {current_sim_time:.2e} seconds")
            # Plot the first contour plot
            contour1 = axs[0,0].contourf(X_data, Y_data, rho_data, cmap='turbo')
            axs[0,0].set_title('Density')
            fig.colorbar(contour1, ax=axs[0,0])  # Add color bar for the first plot

            # Plot the second contour plot
            contour2 = axs[0,1].contourf(X_data, Y_data, p_data, cmap='turbo')
            axs[0,1].set_title('pressure')
            fig.colorbar(contour2, ax=axs[0,1])  # Add color bar for the second plot
            # Plot the second contour plot          
            contour3 = axs[1,0].contourf(X_data, Y_data, g_data, cmap='turbo')
            axs[1,0].set_title('enclosed mass density')
            fig.colorbar(contour3, ax=axs[1,0])  # Add color bar for the third plot
            
            contour4 = axs[1,1].contourf(X_data, Y_data, U_data, cmap='turbo')
            axs[1,1].set_title('Internal energy')
            fig.colorbar(contour4, ax=axs[1,1])  # Add color bar for the third plot
                       
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            # Adjust layout and display the plot
            plt.tight_layout()
            
            plt.show()
            
except Exception as e:
    logger.error("Solver build failed: %s", e)
