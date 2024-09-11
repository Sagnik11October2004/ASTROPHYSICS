using StaticArrays
using NearestNeighbors
using Statistics
using BenchmarkTools
using Random
using LinearAlgebra

const G = 6.67430e-11  # gravitational constant

struct Particle
    position::SVector{3, Float64}
    mass::Float64
end

struct Group
    particles::Vector{Int}
    com::SVector{3, Float64}
    total_mass::Float64
end

struct ParticleSystem
    particles::Vector{Particle}
    hierarchical_structure::Vector{Group}
end

function calculate_average_interparticle_distance(particles::Vector{Particle}, sample_size::Int=1000)
    N = length(particles)
    total_distance = 0.0
    count = 0
    rng = MersenneTwister(42)  # For reproducibility

    for _ in 1:sample_size
        i = rand(rng, 1:N)
        j = rand(rng, 1:N)
        if i != j
            total_distance += norm(particles[i].position - particles[j].position)
            count += 1
        end
    end

    return total_distance / count
end


function calculate_density(particles::Vector{Particle})
    total_mass = sum(p.mass for p in particles)
    positions = [p.position for p in particles]
    min_pos = minimum(reduce(hcat, positions), dims=2)
    max_pos = maximum(reduce(hcat, positions), dims=2)
    volume = prod(max_pos .- min_pos)
    return total_mass / volume
end

function create_hierarchical_structure(particles::Vector{Particle}, max_group_size::Int, min_distance::Float64)
    N = length(particles)
    positions = [p.position for p in particles]
    tree = KDTree(reduce(hcat, positions))

    structure = Group[]
    unassigned = Set(1:N)

    while !isempty(unassigned)
        seed = pop!(unassigned)
        group = [seed]
        idxs = inrange(tree, particles[seed].position, min_distance)

        for idx in idxs
            if idx in unassigned && length(group) < max_group_size
                push!(group, idx)
                delete!(unassigned, idx)
            end
        end

        group_particles = particles[group]
        total_mass = sum(p.mass for p in group_particles)
        com = sum(p.position * p.mass for p in group_particles) / total_mass

        push!(structure, Group(group, com, total_mass))
    end

    return structure
end


function ParticleSystem(particles::Vector{Particle})
    avg_distance = calculate_average_interparticle_distance(particles)
    density = calculate_density(particles)
    
    # Adjust max_group_size based on density, but clamp it to a reasonable range
    max_group_size = Int(clamp((32 * (density / 1e-20)), 1, 1000))
    
    # Adjust min_distance based on average interparticle distance
    min_distance = avg_distance * 0.1
    
    hierarchical_structure = create_hierarchical_structure(particles, max_group_size, min_distance)
    ParticleSystem(particles, hierarchical_structure)
end

function compute_potential_hierarchical(system::ParticleSystem, i::Int)
    potential = 0.0
    particle = system.particles[i]
    particle_group = findfirst(group -> i in group.particles, system.hierarchical_structure)

    for (g, group) in enumerate(system.hierarchical_structure)
        if g == particle_group
            for j in group.particles
                if i != j
                    r = norm(particle.position - system.particles[j].position)
                    potential -= G * particle.mass * system.particles[j].mass / r
                end
            end
        else
            r = norm(particle.position - group.com)
            potential -= G * particle.mass * group.total_mass / r
        end
    end

    return potential
end

function compute_potential_brute_force(system::ParticleSystem, i::Int)
    potential = 0.0
    particle = system.particles[i]

    for (j, other) in enumerate(system.particles)
        if i != j
            r = norm(particle.position - other.position)
            potential -= G * particle.mass * other.mass / r
        end
    end

    return potential
end

function compare_methods(system::ParticleSystem)
    N = length(system.particles)

    hierarchical_potentials = @btime [compute_potential_hierarchical($system, i) for i in 1:$N]
    brute_force_potentials = @btime [compute_potential_brute_force($system, i) for i in 1:$N]

    relative_errors = abs.(hierarchical_potentials .- brute_force_potentials) ./ abs.(brute_force_potentials)

    println("Number of particles: ", N)
    println("Number of groups: ", length(system.hierarchical_structure))
    println("Max relative error: ", maximum(relative_errors))
    println("Mean relative error: ", mean(relative_errors))
    println("Median relative error: ", median(relative_errors))
end

function generate_particles(n::Int)
    rng = MersenneTwister(42)  # For reproducibility
    [Particle(SVector{3, Float64}(rand(rng, 3) .* 2e12 .- 1e12), rand(rng) * 9e29 + 1e29) for _ in 1:n]
end

# Test with different numbers of particles
for n in [100, 1000, 5000, 10000, 20000, 30000,50000,100000]
    particles = generate_particles(n)
    system = ParticleSystem(particles)
    println("\nTesting with $n particles:")
    compare_methods(system)
end
