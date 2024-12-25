#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General Relativity Tensor Calculator

This script calculates various geometric tensors for the Schwarzschild metric in General Relativity:
- Christoffel symbols
- Riemann curvature tensor
- Ricci tensor
- Scalar curvature

Originally created by Thomas Stinglhamber (De La Physique)
Translated and enhanced with additional documentation
"""

import sympy as sp
import numpy as np
from pprint import pprint

# Define symbolic variables for calculations
# These represent coordinates and physical constants in the spacetime metric
t = sp.Symbol('t')      # time coordinate
r = sp.Symbol('r')      # radial coordinate
theta = sp.Symbol('theta')  # polar angle
phi = sp.Symbol('phi')     # azimuthal angle
G = sp.Symbol('G')      # gravitational constant
M = sp.Symbol('M')      # mass of the central object
u = sp.Symbol('u')      # auxiliary variable
v = sp.Symbol('v')      # auxiliary variable
K = sp.Symbol('K')      # auxiliary constant
p_1 = sp.Symbol('p_1')  # parameter 1
p_2 = sp.Symbol('p_2')  # parameter 2
p_3 = sp.Symbol('p_3')  # parameter 3
p_4 = sp.Symbol('p_4')  # parameter 4
C = sp.Symbol('C')      # integration constant

# Define the Schwarzschild metric
# This represents the geometry of spacetime around a spherically symmetric mass
g = [[(1-(2*G*M)/r), 0, 0, 0],
     [0, (1/(1-2*G*M/r)), 0, 0],
     [0, 0, r**2, 0],
     [0, 0, 0, r**2 * sp.sin(theta)**2]]

# Coordinate system variables in order: (t, r, θ, φ)
coordinates = [t, r, theta, phi]

# File path for LaTeX output
Path = "LaTeX_GR_4x4.txt"  # Change this to your desired output path
GenerateLatex = True       # Set to False to skip LaTeX generation

# Initialize output file
f = open(Path, "w")

# Convert metric to matrix form and calculate its inverse
g2 = sp.Matrix(([g[0][0], g[1][0], g[2][0], g[3][0]],
                [g[0][1], g[1][1], g[2][1], g[3][1]],
                [g[0][2], g[1][2], g[2][2], g[3][2]],
                [g[3][0], g[3][1], g[3][2], g[3][3]]))
g_inv = g2.inv()

# Display the metric
print("\u0332".join("Metric: "))
pprint(g)
print()

def calculate_christoffel_4x4():
    """
    Calculate the Christoffel symbols for the 4D metric.
    
    The Christoffel symbols represent the connection coefficients that describe
    how vectors change when parallel transported along curves in the manifold.
    
    Returns:
        list: Four 4x4 matrices containing the Christoffel symbols for each coordinate
    """
    # Initialize arrays for storing Christoffel symbols
    christo = [[0]*4 for _ in range(4)]
    christo_t = [[0]*4 for _ in range(4)]
    christo_r = [[0]*4 for _ in range(4)]
    christo_theta = [[0]*4 for _ in range(4)]
    christo_phi = [[0]*4 for _ in range(4)]
    
    print("\u0332".join("Christoffel Symbols: "))
    # Calculate each component of the Christoffel symbols
    for m in range(4):
        for l in range(4):
            for i in range(4):
                for j in range(4):
                    # Christoffel symbol formula implementation
                    christo[i][j] = 1/2 * g_inv[l,m]*(
                        sp.diff(g[l][i], coordinates[j]) +
                        sp.diff(g[j][l], coordinates[i]) -
                        sp.diff(g[i][j], coordinates[l]))
                    
                    if christo[i][j] != 0:
                        print(f"Γ|{coordinates[m]}{coordinates[i]}{coordinates[j]}: {sp.simplify(christo[i][j])}")
                        
                        # Store components by coordinate type
                        if m == 0:
                            christo_t[i][j] = christo[i][j]
                        elif m == 1:
                            christo_r[i][j] = christo[i][j]
                        elif m == 2:
                            christo_theta[i][j] = christo[i][j]
                        elif m == 3:
                            christo_phi[i][j] = christo[i][j]
    
    christoffel_symbols = [christo_t, christo_r, christo_theta, christo_phi]
    print()
    
    # Display the results
    for i in range(4):
        print(f"Christoffel symbols for {coordinates[i]}:")
        pprint(christoffel_symbols[i])
        print()
    
    return christoffel_symbols

def calculate_riemann_tensor(christoffel):
    """
    Calculate the Riemann curvature tensor.
    
    The Riemann tensor describes the curvature of spacetime by measuring
    how vectors change when parallel transported around closed loops.
    
    Args:
        christoffel: List of Christoffel symbols calculated earlier
        
    Returns:
        list: Four 4x4x4 matrices representing the Riemann tensor components
    """
    # Initialize component matrices
    riemann_components = {
        f"R_l{l}n{n}": [[0]*4 for _ in range(4)]
        for l in range(4) for n in range(4)
    }
    
    print("\u0332".join("Riemann Tensor:"))
    
    # Calculate Riemann tensor components
    for l in range(4):
        for i in range(4):
            for j in range(4):
                for n in range(4):
                    # Initialize temporary storage
                    temp_component = 0
                    
                    # Calculate each component using the Riemann tensor formula
                    for m in range(4):
                        temp_component += (
                            sp.diff(christoffel[l][i][j], coordinates[n]) -
                            sp.diff(christoffel[l][i][n], coordinates[j]) +
                            christoffel[m][i][j] * christoffel[l][m][n] -
                            christoffel[m][i][n] * christoffel[l][m][j]
                        )
                    
                    if temp_component != 0:
                        print(f"R|{coordinates[l]}{coordinates[i]}{coordinates[j]}{coordinates[n]}: {sp.simplify(temp_component)}")
                        riemann_components[f"R_l{l}n{n}"][j][i] = temp_component
    
    # Organize components into final structure
    riemann_tensor = [[riemann_components[f"R_l{l}n{n}"] for n in range(4)] for l in range(4)]
    
    # Display results
    print()
    for i in range(4):
        print(f"Riemann tensor components for {coordinates[i]}:")
        pprint(riemann_tensor[i])
        print()
    
    return riemann_tensor

def calculate_ricci_tensor(riemann):
    """
    Calculate the Ricci tensor by contracting the Riemann tensor.
    
    The Ricci tensor represents the trace of the Riemann tensor and describes
    the average curvature in each direction.
    
    Args:
        riemann: The previously calculated Riemann tensor
        
    Returns:
        list: 4x4 matrix representing the Ricci tensor
    """
    ricci = [[0]*4 for _ in range(4)]
    ricci_temp = [[0]*4 for _ in range(4)]
    
    print("\u0332".join("Ricci Tensor:"))
    
    # Calculate Ricci tensor components through contraction
    for l in range(4):
        for i in range(4):
            for j in range(4):
                ricci_temp[i][j] = riemann[l][i][j][l]
                if ricci_temp[i][j] != 0:
                    ricci[i][j] = ricci_temp[i][j]
    
    pprint(ricci)
    print()
    return ricci

def calculate_scalar_curvature(ricci):
    """
    Calculate the scalar curvature (Ricci scalar).
    
    The scalar curvature is a single number representing the total curvature
    of spacetime at each point, obtained by contracting the Ricci tensor.
    
    Args:
        ricci: The previously calculated Ricci tensor
        
    Returns:
        sympy expression: The scalar curvature
    """
    scalar = [[0]*4 for _ in range(4)]
    
    print("\u0332".join("Scalar Curvature:"))
    
    # Calculate scalar curvature components
    for i in range(4):
        for j in range(4):
            scalar[i][j] = g_inv[i,j] * ricci[i][j]
    
    # Sum all components to get final scalar curvature
    total = sum(scalar[i][j] for i in range(4) for j in range(4))
    print(total)
    return total

def generate_latex_output(christoffel, riemann, ricci, scalar_curvature):
    """
    Generate LaTeX output for all calculated tensors.
    
    Creates a formatted LaTeX document containing all tensor components
    in a mathematically precise notation.
    """
    def format_matrix(matrix):
        """Helper function to format matrices for LaTeX"""
        lines = str(matrix).replace('[', '').replace(']', '').replace(' - ','-').replace(' + ','+').splitlines()
        output = [r' $$\begin{pmatrix}']
        output += ['  ' + ' ,& '.join(l.split()) + r'\\' for l in lines]
        output += [r'\end{pmatrix}$$']
        return '\n'.join(output)
    
    def format_tensor(tensor):
        """Helper function to format tensors for LaTeX"""
        output = [r'$$ \begin{pmatrix}']
        for i in range(len(tensor)):
            output += [r' \begin{pmatrix}']
            for j in range(len(tensor[i])):
                lines = str(tensor[i][j]).replace('[', '').replace(']', '').replace(' - ','-').replace(' + ','+').splitlines()
                output += ['  ' + ', & '.join(l.split()) + r'\\' for l in lines]
            output += [r'\end{pmatrix}\\']
        output += [r'\end{pmatrix}$$']
        return '\n'.join(output)
    
    # Write LaTeX document structure
    print(r'\setlength\parindent{0pt}', file=f)
    print(r'\section{General Relativity 4x4 Tensors}', file=f)
    print(r'\subsection{Metric}', file=f)
    print(format_matrix(np.array(g)), '\n', file=f)
    
    # Write Christoffel symbols
    print(r'\subsection{Christoffel Symbols}', file=f)
    for m in range(4):
        for i in range(4):
            for j in range(4):
                if christoffel[m][i][j] != 0:
                    print(f'$\Gamma^{{{coordinates[m]}}}_{{{coordinates[i]}{coordinates[j]}}}'
                          f"= {christoffel[m][i][j]}$\\\\", file=f)
    print('\n', file=f)
    
    # Write Riemann tensor components
    print(r'\subsection{Riemann Tensor}', file=f)
    for l in range(4):
        for i in range(4):
            for j in range(4):
                for n in range(4):
                    if riemann[l][i][j][n] != 0:
                        print(f'$R^{{{coordinates[l]}}}_{{{coordinates[i]}{coordinates[j]}{coordinates[n]}}}'
                              f"= {riemann[l][i][j][n]}$\\\\", file=f)
    print('\n', file=f)
    
    # Write Ricci tensor components
    print(r'\subsection{Ricci Tensor}', file=f)
    for l in range(4):
        for i in range(4):
            for j in range(4):
                if riemann[l][i][j][l] != 0:
                    print(f'$R\mathrm{{icci}}_{{{coordinates[i]}{coordinates[j]}}}'
                          f"= {riemann[l][i][j][l]}$\\\\", file=f)
    print('\n', file=f)
    
    # Write scalar curvature
    print(r'\paragraph{Scalar Curvature} ', '\n', f'$${scalar_curvature}$$', file=f)
    
    f.close()
    
    # Post-process the LaTeX file to improve formatting
    with open(Path, 'r') as f1:
        latex_content = f1.read()
    
    # Replace symbolic notation with proper LaTeX notation
    replacements = {
        'theta ': r' \theta ',
        'phi ': r' \phi ',
        '**': '^',
        'cos(theta)': r'\cos(\theta)',
        'sin(theta)': r'\sin(\theta)',
        'sin(phi)': r'\sin(\phi)',
        'cos(phi)': r'\cos(\phi)',
        '*': '',
        '1.00000000000000': '1',
        '1.0/': '1/',
        '1.0+': '1+',
        '1.0-': '1-',
        '1.0': ''
    }
    
    for old, new in replacements.items():
        latex_content = latex_content.replace(old, new)
    
    with open(Path, 'w') as f2:
        f2.write(latex_content)
    
    print('\n', 'LaTeX file generated successfully')

# Main execution
if __name__ == "__main__":
    # Calculate all tensor components
    christoffel_symbols = calculate_christoffel_4x4()
    riemann_tensor = calculate_riemann_tensor(christoffel_symbols)
    ricci_tensor = calculate_ricci_tensor(riemann_tensor)
    scalar_curvature = calculate_scalar_curvature(ricci_tensor)
    
    # Generate LaTeX output if requested
    if GenerateLatex:
        generate_latex_output(christoffel_symbols, riemann_tensor, ricci_tensor, scalar_curvature)
