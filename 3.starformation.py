'''THIS IS A CODE SIMULATING A COLLAPSING MOLECULAR CLOUD IN PYTHON USING DEDALUS LIBRARY. 
IT SOLVES THE CONTINUITY EQUATION , MOMENTUM EQUATIONs, ENERGY EQUATION,INTERNAL ENERGY EQUATION,RELATION BETWEEN PRESSURE AND INTERNAL ENERGY 
USING RUNGE KUTTA METHOD . 
IT ALSO SOLVES THE POISSON EQUATION GOVERNING THE GRAVITATIONAL POTENTIAL OF THE MOLECULAR CLOUD
BY USING IT AS AN IVP INSTEAD OF A BVP. 
THIS  IS DONE BY FINDING THE STABLE POTENTIAL FOR THE DENSITY DISTRIBUTION FUNCTION ON A NON-STANDARD TIME SCALE AND
ESTIMATING WHEN THE CHANGE IN THE POTENTIAL AT A POINT WITH TIME IS LOWER THAN A CERTAIN TOLERANCE VALUE.'''




import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

# Constants
c=3e8   #speed of light
G = 6.67430e-11  #universal gravitational constant
Mstar = 1.989e34  #mass of gas cloud
Rstar = 10.7e14   #Dimenisons of gas cloud
pi = np.pi   #pi
Nx = 20      #no. of grid spacings in x
Ny = 20      #no. of grid spacings in y
Nz = 20      #no. of grid spacings in z
stop_sim_time = 3e8   #Simulation stops at
timestep = 1e3  #timestep
dtype = np.float64  #data type for arrays
gamma=1.5  #average ratio of C_p/C_v for the gas(due to presence of H and traces of He)
tol=1e-5   #value of tolerance

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

p = dist.Field(name='p', bases=(xbasis, ybasis, zbasis))
rho = dist.Field(name='rho', bases=(xbasis, ybasis, zbasis))
vx= dist.Field( name='vel', bases=(xbasis, ybasis, zbasis))
vy= dist.Field( name='vel', bases=(xbasis, ybasis, zbasis))
vz= dist.Field( name='vel', bases=(xbasis, ybasis, zbasis))
potential = dist.Field(name='potential', bases=(xbasis, ybasis, zbasis))
gx=dist.Field(name='gx', bases=(xbasis, ybasis, zbasis))
gy=dist.Field(name='gy', bases=(xbasis, ybasis, zbasis))
gz=dist.Field(name='gz', bases=(xbasis, ybasis, zbasis))
E = dist.Field(name='E', bases=(xbasis, ybasis, zbasis))
U=dist.Field(name='U', bases=(xbasis, ybasis, zbasis))

# Initial Conditions
vx['g'] = 0
vy['g'] = 0
vz['g'] = 0
rho['g'] = 0.001
p['g']=4.14e-12
U=(p/(gamma-1)).evaluate()
E=(U+1.4).evaluate()

# Initial potential calculation and setting gradient of potential as g
potential= ((-2/3) * G * rho * pi * (3 * (Rstar**2) - radius**2)).evaluate()
gx['g']=(d3.Gradient(potential).evaluate())['g'][0,:,:,:]
gy['g']=(d3.Gradient(potential).evaluate())['g'][1,:,:,:]
gz['g']=(d3.Gradient(potential).evaluate())['g'][2,:,:,:]

# Problem Definition
problem1 = d3.IVP([rho, vx,vy,vz ,E,U,p], namespace=locals())
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])
dz = lambda A: d3.Differentiate(A, coords['z'])
# Equations
equations1 = [
    "dt(rho) = -dx(rho * vx)-dy(rho * vy)-dz(rho * vz)",
    "dt(vx) = (-rho*gx-dx(p)-dx(rho*vx*vx)-dy(rho*vx*vy)-dz(rho*vx*vz)+(dx(rho*vx)+dy(rho*vy)+dz(rho*vz))*vx)/rho",
    "dt(vy) = (-rho*gy-dy(p)-dx(rho*vy*vx)-dy(rho*vy*vy)-dz(rho*vy*vz)+(dx(rho*vx)+dy(rho*vy)+dz(rho*vz))*vy)/rho",
    "dt(vz) = (-rho*gz-dz(p)-dx(rho*vz*vx)-dy(rho*vz*vy)-dz(rho*vz*vz)+(dx(rho*vx)+dy(rho*vy)+dz(rho*vz))*vz)/rho",
    "dt(E)=-dx((E+p)*vx)-dy((E+p)*vy)-dz((E+p)*vz)-rho*(gx*vx+gy*vy+gz*vz)",
    "U= E-0.5*rho*(vx*vx+vy*vy+vz*vz)",
    "p=(gamma-1)*U"
    
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
        #Finding stable solution of poisson equation
        sub_pot = dist.Field(name='sub_pot', bases=(xbasis, ybasis, zbasis))
        sub_pot['g'] = potential['g']
        problem2 = d3.IVP([potential], namespace=locals())
        problem2.add_equation('dt(potential) + lap(potential) = 4 * pi * G * rho')
        solver2 = problem2.build_solver(d3.RK443)
        solver2.stop_sim_time = 1e4
        print("SOLVING FOR POTENTIAL")
        while solver2.proceed:
            solver2.step(5)
            arrays_equal = potential['g'] == sub_pot['g']
            
            if (np.allclose(potential['g'], sub_pot['g'], atol=tol) or np.allclose(sub_pot['g'],potential['g'],  atol=tol)):

                break
            else:
                sub_pot['g'] = potential['g']
                
        print("COMPLETED SOLVING FOR POTENTIAL")
        gx['g']=(d3.Gradient(potential).evaluate())['g'][0,:,:,:]
        gy['g']=(d3.Gradient(potential).evaluate())['g'][1,:,:,:]
        gz['g']=(d3.Gradient(potential).evaluate())['g'][2,:,:,:]
        current_sim_time += timestep

        #printing the output
        if solver.iteration % 30== 0:
            print('Completed iteration {}'.format(solver.iteration))
            print(f"Min density: {np.min(rho['g'])}")
            print(f"Potential Max: {np.max(potential['g'])}")
            print(f"Min pressure:{np.min(p['g'])}")
            print(f"Min energy:{np.min(E['g'])}")
            rho_data=((rho(z=0).evaluate()['g']).reshape(Nx,Ny))
            p_data=((p(z=0).evaluate()['g']).reshape(Nx,Ny))
            potential_data=((potential(z=0).evaluate()['g']).reshape(Nx,Ny))
            energy_data=((E(z=0).evaluate()['g']).reshape(Nx,Ny))
            X_data=np.linspace(-Rstar,Rstar,Nx)
            Y_data=np.linspace(-Rstar,Rstar,Ny)
           
            fig, axs = plt.subplots(2, 2, figsize=(6, 6))  # 2 row, 2 columns
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
            contour3 = axs[1,0].contourf(X_data, Y_data, potential_data, cmap='turbo')
            axs[1,0].set_title('Potential')
            fig.colorbar(contour3, ax=axs[1,0])  # Add color bar for the third plot
            
            contour4 = axs[1,1].contourf(X_data, Y_data, energy_data, cmap='turbo')
            axs[1,1].set_title('Energy')
            fig.colorbar(contour4, ax=axs[1,1])  # Add color bar for the third plot
                       
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            # Adjust layout and display the plot
            plt.tight_layout()
            
            plt.show()
            
except Exception as e:
    logger.error("Solver build failed: %s", e)
