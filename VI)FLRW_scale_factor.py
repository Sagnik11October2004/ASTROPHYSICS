import numpy as np
import matplotlib.pyplot as plt

# Rescaling H to express the age in billions of years
H0 = 70  # Hubble constant in km/s/Mpc
Mpc = 3.085677581e19  # 1 Megaparsec in km
km = 1.0  # 1 km
Gyr = 3.1536e16  # 1 Gyr in seconds

# Rescaled Hubble constant
H_0 = (H0 * km * Gyr) / Mpc

def MNL(Omega_0, t):
    """
    Function for a universe with two components: matter and a negative cosmological constant.
    
    Parameters:
        Omega_0 (float): Matter density parameter.
        t (array-like): Array of time values.
    
    Returns:
        t (array): Adjusted time values (avoiding t > t_c).
        a (array): Scale factor values.
        age (float): Age of the universe (Gyr).
        t_c (float): Time of the Big Crunch (Gyr).
    """
    if Omega_0 <= 1:
        raise ValueError("Omega_0 must be greater than 1 for a closed universe.")
    
    # Time of the Big Crunch (a = 0)
    t_c = (2 * np.pi / (3 * H_0)) * (1 / np.sqrt(Omega_0 - 1))
    
    # Consider only times less than t_c
    mask = t < t_c
    t = t[mask]
    
    # Scale factor
    a = (np.sqrt(Omega_0 / (Omega_0 - 1)) * np.sin((3 / 2) * H_0 * t * np.sqrt(Omega_0 - 1)))**(2 / 3)
    
    # Age of the universe
    age = (2 / (3 * H_0 * np.sqrt(Omega_0 - 1))) * np.arcsin(np.sqrt((Omega_0 - 1) / Omega_0))
    
    return t, a, age, t_c

def Time(x, m, r, l):
    """
    Function to compute the time derivative of the scale factor.
    
    Parameters:
        x (float): Scale factor.
        m (float): Matter density parameter.
        r (float): Radiation density parameter.
        l (float): Cosmological constant density parameter.
    
    Returns:
        t (float): Time derivative at scale factor x.
    """
    Omega_k = 1 - (m + r + l)
    Omega_R = r / (x**2)
    Omega_M = m / x
    Omega_C = l * x**2
    rho = Omega_R + Omega_M + Omega_C + Omega_k
    
    return (1 / H_0) * (1 / np.sqrt(rho))

def RK4(num_steps, a0, af, m, r, l):
    """
    4th-order Runge-Kutta integrator for solving the scale factor evolution.
    
    Parameters:
        num_steps (int): Number of integration steps.
        a0 (float): Initial scale factor.
        af (float): Final scale factor.
        m (float): Matter density parameter.
        r (float): Radiation density parameter.
        l (float): Cosmological constant density parameter.
    
    Returns:
        t (array): Time values.
        a (array): Scale factor values.
        Age (float): Age of the universe when a = 1.
    """
    Age = 0
    t = np.zeros(num_steps + 1)
    a = np.zeros(num_steps + 1)
    a[0] = a0
    da = abs(af - a0) / num_steps

    for i in range(num_steps):
        tk1 = Time(a[i], m, r, l)
        tk2 = Time(a[i] + da / 2, m, r, l)
        tk3 = Time(a[i] + da / 2, m, r, l)
        tk4 = Time(a[i] + da, m, r, l)
        t[i + 1] = t[i] + (da / 6) * (tk1 + 2 * tk2 + 2 * tk3 + tk4)
        a[i + 1] = a[i] + da

        # Record the universe's age when a = 1
        if abs(a[i] - 1) < da:
            Age = t[i]

    print(f"The age of the universe with m={m}, r={r}, l={l} is: {Age:.3f} Gyr")
    return t, a, Age

# Number of points and initial conditions
n = 10000
a_0 = 1e-12
a_f = 2.5

# Positive cosmological constant scenarios
print('For a positive cosmological constant:')
t0, a0, age0 = RK4(n, a_0, a_f, 0.3, 0, 0.7)
t1, a1, age1 = RK4(n, a_0, a_f, 1, 0, 0)
t2, a2, age2 = RK4(n, a_0, a_f, 0, 0, 1)
t3, a3, age3 = RK4(n, a_0, a_f, 0, 0, 0)
t4, a4, age4 = RK4(n, a_0, a_f, 0, 1, 0)

# Closed universe with a negative cosmological constant
print('For a negative cosmological constant (no radiation):')
tp = np.linspace(0, 30, n)
t5, a5, age5, t_c = MNL(2, tp)

print(f"The age of the universe is: {age5:.3f} Gyr")
print(f"The end of the universe (Big Crunch) will occur at: {t_c:.3f} Gyr")

# Plotting results
plt.figure(figsize=(10, 6))
plt.title('Evolution of the Scale Factor')
plt.plot(t0 - age0, a0, label='$\Lambda CDM: \Omega_{\Lambda}=0.7, \Omega_{M}=0.3$')
plt.plot(t1 - age1, a1, label='Einstein-de Sitter: $ \Omega_{M}=1$')
plt.plot(t2 - age2, a2, label='De Sitter: $\Omega_{\Lambda}=1$')
plt.plot(t3 - age3, a3, label='Empty Universe: $\Omega_{k}=1$')
plt.plot(t4 - age4, a4, label='Radiation Dominated: $\Omega_{R}=1$')
plt.plot(t5 - age5, a5, label='Closed Universe: $\Omega_{M}=2, \Omega_{\Lambda}=-1$')

plt.xlabel('Time [Gyr]')
plt.ylabel('Scale Factor $a(t)$')
plt.xlim(-20, 30)
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()
