'''THIS IS A CODE SIMULATING A COLLAPSING MOLECULAR CLOUD IN PYTHON USING DEDALUS LIBRARY. 
IT SOLVES THE CONTINUITY EQUATION , MOMENTUM EQUATION , USES IDEAL GAS LAW TO FIND PRESSURE, TEMPERATURE EQUATION WHICH HAS BEEN 
DERIVED FROM THE ENERGY EQUATION USING RUNGE KUTTA METHOD . IT ALSO SOLVES THE POISSON EQUATION GOVERNING 
THE GRAVITATIONAL POTENTIAL OF THE MOLECULAR CLOUD BY USING IT AS AN IVP INSTEAD OF A BVP. 
THIS  IS DONE BY FINDING THE STABLE POTENTIAL FOR THE DENSITY DISTRIBUTION FUNCTION ON A NON-STANDARD TIME SCALE AND
ESTIMATING WHEN THE CHANGE IN THE POTENTIAL AT A POINT WITH TIME IS LOWER THAN A CERTAIN TOLERANCE VALUE.'''




import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

# Constants
c=3e8
G = 6.67430e-11
c_v = 6.44
sigma = 5.6704e-8
Mstar = 1.989e30
Rstar = 6.957e10
pi = np.pi
eta = 0.02
Nx = 20
Ny = 20
Nz = 20
k = 1e-6
mew = 2*1.67e-24
tau_cr=1e-25
tau_pr=1e-34
tau_cool=1e-34
stop_sim_time = 3e8
timestep = 1e1
dtype = np.float64
gamma=1.5

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
vel = dist.VectorField(coords, name='vel', bases=(xbasis, ybasis, zbasis))
potential = dist.Field(name='potential', bases=(xbasis, ybasis, zbasis))
T = dist.Field(name='T', bases=(xbasis, ybasis, zbasis))

# Initial Conditions
vel['g'] = 0
T['g'] = 100
rho['g'] = 10
p['g'] = rho['g'] * 8.3144 * T['g'] / mew

# Initial potential calculation
potential= ((-2/3) * G * rho * pi * (3 * (Rstar**2) - radius**2)).evaluate()

# Problem Definition
problem1 = d3.IVP([rho, vel,p,T], namespace=locals())

# Equations
equations = [
    "dt(rho) = -div(rho * vel)",
    "dt(vel) = -(grad(p) / rho) + eta * lap(vel)/rho   + (div(vel * rho) * vel / rho) - (((grad(0.5 * (rho**2) * dot(vel, vel))) - rho * cross(vel, curl(rho * vel))) / (rho**2))-grad(potential)",
    "p = rho * 8.3144 * T / (1.67e-2)",
    "dt(T)=-T*(div(vel))/(p)-dot(vel,(grad(T)))+(1/(rho*c_v))*((rho/mew)*(tau_cr+tau_pr-tau_cool)-((rho/mew))*(tau_cool))+(dot(vel,vel)/c_v)*(lap(T))+0.001*rho*(1.43e-6)/c_v"
  
]

for i, eq in enumerate(equations):
    try:
        print(f"Adding equation {i}: {eq}")
        problem1.add_equation(eq)
    except Exception as e:
        logger.error("Error adding equation %d: %s", i, e)




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
            solver2.step(10)
            arrays_equal = potential['g'] == sub_pot['g']
            
            if (np.allclose(potential['g'], sub_pot['g'], atol=1e-4) or np.allclose(sub_pot['g'],potential['g'],  atol=1e-4)):

                break
            else:
                sub_pot['g'] = potential['g']
                
        print("DONE SOLVING FOR POTENTIAL")
        if solver.iteration % 20== 0:
            print('Completed iteration {}'.format(solver.iteration))
            max_rho = np.max(rho['g'])
            min_rho = np.min(rho['g'])
            print(f"Max rho: {max_rho}")
            print(f"Min rho: {min_rho}")
            print(f"Potential Min: {np.min(potential['g'])}")
            print(f"Potential Max: {np.max(potential['g'])}")
            print(f"Max T: {np.max(T['g'])}")
            print(f"Min T: {np.min(T['g'])}")
            print(f"Min pressure:{np.min(p['g'])}")
            rho_data=((rho(z=0).evaluate()['g']).reshape(Nx,Ny))
            T_data=((T(z=0).evaluate()['g']).reshape(Nx,Ny))
            potential_data=((potential(z=0).evaluate()['g']).reshape(Nx,Ny))
            X_data=np.linspace(-Rstar,Rstar,Nx)
            Y_data=np.linspace(-Rstar,Rstar,Ny)
            plt.figure(figsize=(8, 6))
            fig, axs = plt.subplots(1, 3, figsize=(12, 6))  # 1 row, 2 columns
           
            # Plot the first contour plot
            contour1 = axs[0].contourf(X_data, Y_data, rho_data, cmap='turbo')
            axs[0].set_title('rho')
            fig.colorbar(contour1, ax=axs[0])  # Add color bar for the first plot

            # Plot the second contour plot
            contour2 = axs[1].contourf(X_data, Y_data, T_data, cmap='turbo')
            axs[1].set_title('Temperature')
            fig.colorbar(contour2, ax=axs[1])  # Add color bar for the second plot
            # Plot the second contour plot          
            contour3 = axs[2].contourf(X_data, Y_data, potential_data, cmap='turbo')
            axs[2].set_title('Potential')
            fig.colorbar(contour3, ax=axs[2])  # Add color bar for the third plot
            
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            # Adjust layout and display the plot
            plt.tight_layout()
            plt.show()


           

        
            
except Exception as e:
    logger.error("Solver build failed: %s", e)
