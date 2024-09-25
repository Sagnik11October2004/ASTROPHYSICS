using StaticArrays
using NearestNeighbors
using Statistics
using BenchmarkTools
using Random
using LinearAlgebra
using Plots

const G = 6.67430e-11  # gravitational constant
const k_B = 1.380649e-23  # Boltzmann constant
const gamma = 5/3  # adiabatic index for ideal gas
const ν = 0.1  # Viscosity coefficient

mutable struct Particle
    position::SVector{3, Float64}
    velocity::SVector{3, Float64}
    mass::Float64
    density::Float64
    pressure::Float64
    potential::Float64
    internal_energy::Float64
    h::Float64  
end

mutable struct SPHSimulation
    particles::Vector{Particle}
    h::Float64
    cell_size::Float64
    grid::Dict{Tuple{Int, Int, Int}, Vector{Int}}
    cell_com::Dict{Tuple{Int, Int, Int}, SVector{3, Float64}}
    cell_mass::Dict{Tuple{Int, Int, Int}, Float64}
    total_energy::Float64

    function SPHSimulation(particles::Vector{Particle}, h::Float64)
        cell_size = h * 2
        new(particles, h, cell_size, Dict(), Dict(), Dict(), 0.0)
    end
end

function wendland_c4_kernel(r::Float64, h::Float64)
    q = r / h
    if q >= 2.0
        return 0.0
    else
        return (495.0 / (32.0 * π * h^3)) * (1 - q/2)^6 * (35/3 * q^2 + 6*q + 1)
    end
end

function wendland_c4_kernel_derivative(r::Float64, h::Float64)
    q = r / h
    if q >= 2.0
        return 0.0
    else
        return (-495.0 / (4.0 * π * h^4)) * q * (1 - q/2)^5 * (5*q + 1)
    end
end

function hash_position(sim::SPHSimulation, pos::SVector{3, Float64})
    grid_pos = floor.(Int, pos ./ sim.cell_size)
    return (grid_pos[1], grid_pos[2], grid_pos[3])
end

function build_hash_map!(sim::SPHSimulation)
    empty!(sim.grid)
    empty!(sim.cell_com)
    empty!(sim.cell_mass)
    for (i, p) in enumerate(sim.particles)
        cell = hash_position(sim, p.position)
        if !haskey(sim.grid, cell)
            sim.grid[cell] = Int[]
            sim.cell_com[cell] = SVector{3, Float64}(0.0, 0.0, 0.0)
            sim.cell_mass[cell] = 0.0
        end
        push!(sim.grid[cell], i)
        sim.cell_com[cell] += p.position * p.mass
        sim.cell_mass[cell] += p.mass
    end
    for cell in keys(sim.cell_com)
        sim.cell_com[cell] /= sim.cell_mass[cell]
    end
end

function get_neighbor_cells(cell::Tuple{Int, Int, Int})
    return [(cell[1]+dx, cell[2]+dy, cell[3]+dz) for dx in -1:1, dy in -1:1, dz in -1:1]
end

function update_smoothing_length!(sim::SPHSimulation)
    ρ_ref = mean(p.density for p in sim.particles)  # Reference density based on current average
    for particle in sim.particles
        particle.h = sim.h * (ρ_ref / particle.density)^(1/3)  # Update smoothing length
    end
end

function compute_gravitational_force(sim::SPHSimulation, i::Int)
    particle = sim.particles[i]
    h_i = particle.h  # Use particle-specific smoothing length
    cell = hash_position(sim, particle.position)
    neighbor_cells = get_neighbor_cells(cell)
    force = SVector{3, Float64}(0.0, 0.0, 0.0)
    
    # Compute force from particles in neighboring cells
    for nc in neighbor_cells
        if haskey(sim.grid, nc)
            for j in sim.grid[nc]
                if i != j
                    r = sim.particles[j].position - particle.position
                    r_mag = norm(r)
                    h_j = sim.particles[j].h  # Use neighbor-specific smoothing length
                    h_avg = (h_i + h_j) / 2  # Average smoothing length between particles
                    if r_mag < h_avg  # Prevent singularity
                        r_mag = h_avg
                    end
                    force += G * particle.mass * sim.particles[j].mass * r / r_mag^3
                end
            end
        end
    end
    
    # Compute force from non-neighboring cells using their COM
    for (other_cell, com) in sim.cell_com
        if !(other_cell in neighbor_cells)
            r = com - particle.position
            r_mag = norm(r)
            if r_mag < h_i  # Prevent singularity
                r_mag = h_i
            end
            force += G * particle.mass * sim.cell_mass[other_cell] * r / r_mag^3
        end
    end
    
    return force
end

function calculate_potential!(sim::SPHSimulation)
    build_hash_map!(sim)
    for i in 1:length(sim.particles)
        particle = sim.particles[i]
        cell = hash_position(sim, particle.position)
        neighbor_cells = get_neighbor_cells(cell)
        potential = 0.0
        
        # Compute potential from particles in neighboring cells
        for nc in neighbor_cells
            if haskey(sim.grid, nc)
                for j in sim.grid[nc]
                    if i != j
                        r = norm(particle.position - sim.particles[j].position)
                        if r < particle.h  # Use particle-specific smoothing length
                            r = particle.h
                        end
                        potential -= G * sim.particles[j].mass / r
                    end
                end
            end
        end
        
        # Compute potential from non-neighboring cells using their COM
        for (other_cell, com) in sim.cell_com
            if !(other_cell in neighbor_cells)
                r = norm(particle.position - com)
                if r < particle.h  # Use particle-specific smoothing length
                    r = particle.h
                end
                potential -= G * sim.cell_mass[other_cell] / r
            end
        end
        
        particle.potential = potential
    end
end

function calculate_density!(sim::SPHSimulation)
    build_hash_map!(sim)
    for i in 1:length(sim.particles)
        particle = sim.particles[i]
        cell = hash_position(sim, particle.position)
        neighbor_cells = get_neighbor_cells(cell)
        density = 0.0
        h_i = particle.h  # Use particle-specific smoothing length
        for nc in neighbor_cells
            if haskey(sim.grid, nc)
                for j in sim.grid[nc]
                    r = norm(particle.position - sim.particles[j].position)
                    h_j = sim.particles[j].h  # Use neighbor-specific smoothing length
                    h_avg = (h_i + h_j) / 2  # Average smoothing length between particles
                    density += sim.particles[j].mass * wendland_c4_kernel(r, h_avg)
                end
            end
        end
        particle.density = density
    end
end

function calculate_pressure!(sim::SPHSimulation)
    for particle in sim.particles
        particle.pressure = (gamma - 1) * particle.density * particle.internal_energy
    end
end

function compute_pressure_force(sim::SPHSimulation, i::Int)
    particle = sim.particles[i]
    h_i = particle.h  # Use particle-specific smoothing length
    cell = hash_position(sim, particle.position)
    neighbor_cells = get_neighbor_cells(cell)
    force = SVector{3, Float64}(0.0, 0.0, 0.0)
    
    for nc in neighbor_cells
        if haskey(sim.grid, nc)
            for j in sim.grid[nc]
                if i != j
                    r = particle.position - sim.particles[j].position
                    r_mag = norm(r)
                    h_j = sim.particles[j].h  # Use neighbor-specific smoothing length
                    h_avg = (h_i + h_j) / 2  # Average smoothing length between particles
                    if r_mag < 2 * h_avg
                        grad_W = r / r_mag * wendland_c4_kernel_derivative(r_mag, h_avg)
                        force -= (particle.pressure / particle.density^2 + 
                                  sim.particles[j].pressure / sim.particles[j].density^2) * 
                                 sim.particles[j].mass * grad_W
                    end
                end
            end
        end
    end
    
    return force
end

function update_particle!(particle::Particle, acceleration::SVector{3, Float64}, dt::Float64)
    particle.velocity += acceleration * dt
    particle.position += particle.velocity * dt
end

function calculate_total_energy(sim::SPHSimulation)
    kinetic_energy = sum(0.5 * p.mass * norm(p.velocity)^2 for p in sim.particles)
    potential_energy = sum(0.5 * p.mass * p.potential for p in sim.particles)
    internal_energy = sum(p.mass * p.internal_energy for p in sim.particles)
    return kinetic_energy + potential_energy + internal_energy
end

function calculate_cfl_timestep(sim::SPHSimulation, C::Float64, max_dt::Float64)
    min_dt = 1e5
    for particle in sim.particles
        c_i = sqrt(gamma * abs(particle.pressure) / particle.density)  # Sound speed
        denom = norm(particle.velocity) + c_i  # Velocity + sound speed
        dt_i = particle.h / denom  # CFL condition
        min_dt = max(min_dt, dt_i)
    end
    return min(C * min_dt, max_dt) 
end



# Viscous force using Brookshaw's method for Laplacian
function compute_viscous_force(sim::SPHSimulation, i::Int)
    particle = sim.particles[i]
    h_i = particle.h  # Use particle-specific smoothing length
    cell = hash_position(sim, particle.position)
    neighbor_cells = get_neighbor_cells(cell)
    viscous_force = SVector{3, Float64}(0.0, 0.0, 0.0)

    for nc in neighbor_cells
        if haskey(sim.grid, nc)
            for j in sim.grid[nc]
                if i != j
                    # Position and velocity difference between particles
                    r = particle.position - sim.particles[j].position
                    v_ij = particle.velocity - sim.particles[j].velocity
                    r_mag = norm(r)
                    h_j = sim.particles[j].h  # Use neighbor-specific smoothing length
                    h_avg = (h_i + h_j) / 2  # Average smoothing length between particles

                    if r_mag < 2 * h_avg
                        # Gradient of the kernel function
                        grad_W = (r / r_mag) * wendland_c4_kernel_derivative(r_mag, h_avg)
                        # Element-wise viscous force contribution using Brookshaw's method
                        viscous_force += sim.particles[j].mass * ν .* (v_ij / (r_mag^2 + h_avg^2)) .* grad_W
                    end
                end
            end
        end
    end
    
    return viscous_force
end

function cooling_rate(ρ::Float64, T::Float64)
    return ρ^2 * sqrt(T)  # Example cooling function, can be modified
end

# Function definition
function compute_viscous_heating(sim::SPHSimulation, i::Int)
    particle = sim.particles[i]
    h_i = particle.h
    cell = hash_position(sim, particle.position)
    neighbor_cells = get_neighbor_cells(cell)
    viscous_heating = 0.0

    for nc in neighbor_cells
        if haskey(sim.grid, nc)
            for j in sim.grid[nc]
                if i != j
                    r = particle.position - sim.particles[j].position
                    v_ij = particle.velocity - sim.particles[j].velocity
                    r_mag = norm(r)
                    h_j = sim.particles[j].h
                    h_avg = (h_i + h_j) / 2

                    if r_mag < 2 * h_avg
                        grad_W = (r / r_mag) * wendland_c4_kernel_derivative(r_mag, h_avg)
                        viscous_heating += sim.particles[j].mass * ν * norm(v_ij)^2 / (r_mag^2 + h_avg^2)
                    end
                end
            end
        end
    end

    return viscous_heating
end

# Function call (in update_internal_energy!)
function update_internal_energy!(sim::SPHSimulation, dt::Float64)
    for (i, particle) in enumerate(sim.particles)
        # Calculate temperature based on internal energy and density
        T = particle.internal_energy / (1.5 * k_B)
        
        # Radiative cooling rate
        cooling = cooling_rate(particle.density, T)
        
        # Viscous heating rate
        visc_heating = compute_viscous_heating(sim, i)
        
        # Update internal energy with both cooling and heating
        particle.internal_energy += dt * (visc_heating - cooling)
    end
end
# Modified simulation step to include viscous grav_force
function simulate_step!(sim::SPHSimulation, dt::Float64, max_dt::Float64)
    calculate_potential!(sim)
    calculate_density!(sim)
    update_smoothing_length!(sim)  # Update the smoothing lengths dynamically
    calculate_pressure!(sim)

    # Apply CFL condition
    C = 0.2  # Courant safety factor
    dt = calculate_cfl_timestep(sim, C, max_dt)
    println("CFL time step: $dt")

    for i in 1:length(sim.particles)
        particle = sim.particles[i]
        grav_force = compute_gravitational_force(sim, i)
        pressure_force = compute_pressure_force(sim, i)
        viscous_force = compute_viscous_force(sim, i)  # Viscous force based on Brookshaw's method
        acceleration = (grav_force + pressure_force + viscous_force) / particle.mass
        update_particle!(particle, acceleration, dt)
    end

    # Update internal energy with cooling and heating
    update_internal_energy!(sim, dt)

    # Update total energy
    sim.total_energy = calculate_total_energy(sim)
    return dt  # Return the actual timestep used for the step
end


function run_simulation!(sim::SPHSimulation, total_time::Float64, dt::Float64, max_dt::Float64)
    t = 0.0
    N = length(sim.particles)
    num_particles_to_plot = min(100, N)
    sampled_indices = rand(1:N, num_particles_to_plot)
    while t < total_time
        println("Time: ", t)
        dt = simulate_step!(sim, dt, max_dt)  # Update the step with CFL-limited timestep
        visualize_particles(sampled_indices, sim)
        t += dt
    end
end

function initialize_simulation(N::Int, sphere_radius::Float64, h::Float64, total_mass::Float64, initial_temperature::Float64)
    particles = Vector{Particle}(undef, N)
    particle_mass = total_mass / N

    for i in 1:N
        # Generate random position within a sphere
        r = sphere_radius * cbrt(rand())
        θ = acos(2 * rand() - 1)
        φ = 2π * rand()
        
        x = r * sin(θ) * cos(φ)
        y = r * sin(θ) * sin(φ)
        z = r * cos(θ)
        
        position = SVector{3, Float64}(x, y, z)
        
        # Assign initial velocities 
        velocity = -1e1 * normalize(SVector{3, Float64}(x, y, z))  # Arbitrary radial velocity
        
        # Initialize internal energy based on the temperature
        internal_energy = 1.5 * k_B * initial_temperature / (gamma - 1)
        
        # Create each particle with its properties
        particles[i] = Particle(position, velocity, particle_mass, 0.0, 0.0, 0.0, internal_energy, h)
    end
    
    # Create the SPHSimulation object 
    sim = SPHSimulation(particles, h)
    
    # Calculate initial potential and energy of the system
    calculate_potential!(sim)
    sim.total_energy = calculate_total_energy(sim)
    
    return sim
end


function visualize_particles(sampled_indices, sim::SPHSimulation)
    x = [sim.particles[i].position[1] for i in sampled_indices]
    y = [sim.particles[i].position[2] for i in sampled_indices]
    z = [sim.particles[i].position[3] for i in sampled_indices]
    densities = [sim.particles[i].density for i in sampled_indices]
    potentials = [sim.particles[i].potential for i in sampled_indices]
    internal_energies = [sim.particles[i].internal_energy for i in sampled_indices]
    pressures = [sim.particles[i].pressure for i in sampled_indices]
    sml = [sim.particles[i].h for i in sampled_indices]
    p_vel = [norm(sim.particles[i].velocity) for i in sampled_indices]
    
    p1 = scatter(x, y, z, marker_z=densities, color=:viridis, markersize=3,
                 xlabel="X", ylabel="Y", zlabel="Z", title="Particle Distribution (Density)",
                 legend=false, camera=(30, 30),colorbar=true, colorbar_title="Density")

    p2 = scatter(x, y, z, marker_z=potentials, color=:plasma, markersize=3,
                 xlabel="X", ylabel="Y", zlabel="Z", title="Particle Distribution (Potential)",
                 legend=false, camera=(30, 30),  colorbar=true, colorbar_title="Potential")

    p3 = scatter(x, y, z, marker_z=internal_energies, color=:plasma, markersize=3,
                 xlabel="X", ylabel="Y", zlabel="Z", title="Particle Distribution (Internal Energy)",
                 legend=false, camera=(30, 30),  colorbar=true, colorbar_title="Internal Energy")

    p4 = scatter(x, y, z, marker_z=pressures, color=:plasma, markersize=3,
                 xlabel="X", ylabel="Y", zlabel="Z", title="Particle Distribution (Pressure)",
                 legend=false, camera=(30, 30), colorbar=true, colorbar_title="Pressure")
    
    p5 = scatter(x, y, z, marker_z=sml, color=:plasma, markersize=3,
                 xlabel="X", ylabel="Y", zlabel="Z", title="Particle Distribution (h)",
                 legend=false, camera=(30, 30), colorbar=true, colorbar_title="Pressure")

    p6 = scatter(x, y, z, marker_z=p_vel, color=:plasma, markersize=3,
                 xlabel="X", ylabel="Y", zlabel="Z", title="Particle Distribution (velocity)",
                 legend=false, camera=(30, 30), colorbar=true, colorbar_title="velocity")

    display(plot(p1, p2, p3, p4, p5, p6, layout=(3,2), size=(1200, 1200),dpi=150))
    sleep(0.1)
end

# Main simulation parameters
N = 10000  # Number of particles
sphere_radius = 1e13  # light years in meters
h = sphere_radius / 10  # Initial smoothing length
total_mass = 2e33  # Total mass 
total_time = 1e12  # Simulation time in seconds
max_dt = 1e6  # Maximum allowed time step in seconds
initial_temperature = 100.0  # Initial temperature in Kelvin

# Initialize simulation
sim = initialize_simulation(N, sphere_radius, h, total_mass, initial_temperature)

# Run simulation with viscosity
@time run_simulation!(sim, total_time, max_dt, max_dt)
