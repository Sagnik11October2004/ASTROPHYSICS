using DifferentialEquations
using LinearAlgebra
using Plots
using ForwardDiff
using NLsolve

# Constants and parameters
const α = 0.01
const a = 1.0
const c = 3e8
const G = 6.67e-11
const γ = 1.4
const M = 8.5e36
const r_min = 2.54e13
const r_max = 1.0e15
const nr = 600

# Create logarithmically spaced radial grid
r = exp.(range(log(r_min), log(r_max), length=nr))

# Improved finite difference function using 4th order central differences
function finite_difference(y::Vector{Float64}, x::Vector{Float64})
    n = length(y)
    dy = similar(y)
    
    # 4th order central difference for interior points
    for i in 3:n-2
        dy[i] = (-y[i+2] + 8y[i+1] - 8y[i-1] + y[i-2]) / (12(x[i+1] - x[i-1]))
    end
    
    # 2nd order forward difference for the first two points
    dy[1] = (-3y[1] + 4y[2] - y[3]) / (2(x[2] - x[1]))
    dy[2] = (-2y[1] - 3y[2] + 6y[3] - y[4]) / (6(x[3] - x[1]))
    
    # 2nd order backward difference for the last two points
    dy[n-1] = (y[n-3] - 6y[n-2] + 3y[n-1] + 2y[n]) / (6(x[n] - x[n-2]))
    dy[n] = (y[n-2] - 4y[n-1] + 3y[n]) / (2(x[n] - x[n-1]))
    
    return dy
end

# Function defining the system of equations
function accretion_disk_equations!(residual, u, p)
    ρ, v_r, Ω, P, T, cs, H, τ_rϕ, F, j, U, κ = [u[i,:] for i in 1:12]
    
    dv_r_dr = finite_difference(v_r, r)
    dP_dr = finite_difference(P, r)
    dΩ_dr = finite_difference(Ω, r)
    dρ_dr = finite_difference(ρ, r)
    dU_dr = finite_difference(U, r)
    
    Ṁ = 4π .* r .* H .* ρ .* v_r
    j₀ = Ω .* r.^2
    X₀ = 4π .* r.^2 .* H .* τ_rϕ - Ṁ .* (j .- j₀)
    
    residual[1,:] = ρ .* v_r .* dv_r_dr + ρ .* dP_dr ./ ρ - ρ .* (Ω.^2 .- G*M ./ r.^3) .* r
    residual[2,:] = Ṁ .* (dU_dr + P ./ ρ.^2 .* dρ_dr) - 4π .* r.^2 .* H .* τ_rϕ .* dΩ_dr - 4π .* r .* F
    residual[3,:] = Ṁ .- 4π .* r .* H .* ρ .* v_r
    residual[4,:] = Ṁ .* (j .- j₀) .- 4π .* r.^2 .* H .* τ_rϕ .+ X₀
    residual[5,:] = F .- a * c * T.^4 ./ (κ .* ρ .* H)
    residual[6,:] = (cs ./ (r .* sqrt.(G * M ./ r.^3))) .- H ./ r
    residual[7,:] = τ_rϕ .- ρ .* (α .* cs .* H) .* r .* dΩ_dr
    residual[8,:] = P .- ρ .* T
    residual[9,:] = cs .- sqrt.(γ .* P ./ ρ)
    residual[10,:] = j .- Ω .* r.^2
    residual[11,:] = U .- P ./ (ρ .* (γ - 1))
    residual[12,:] = κ .- 0.1 .* ρ .* T.^2
end

# Initial guess
function initial_guess()
    ρ = 1e-8 .* (r ./ r_min).^(-3/2)
    v_r = -1e-3 .* (r ./ r_min).^(-1/2)
    Ω = sqrt.(G * M ./ r.^3)
    P = 1e-6 .* (r ./ r_min).^(-5/2)
    T = 1e4 .* (r ./ r_min).^(-3/4)
    cs = sqrt.(γ .* P ./ ρ)
    H = cs ./ Ω
    τ_rϕ = α .* P
    F = 3G * M .* Ṁ ./ (8π .* r.^3) .* (1 .- sqrt.(r_min ./ r))
    j = Ω .* r.^2
    U = P ./ (ρ .* (γ - 1))
    κ = 0.1 .* ρ .* T.^2
    
    return vcat(ρ', v_r', Ω', P', T', cs', H', τ_rϕ', F', j', U', κ')
end

# Solve the system using NLsolve
function solve_disk()
    u0 = initial_guess()
    
    function f!(residual, u)
        accretion_disk_equations!(residual, u, nothing)
    end
    
    sol = nlsolve(f!, u0, method=:newton, ftol=1e-8, iterations=1000)
    
    if converged(sol)
        return sol.zero
    else
        error("Failed to converge")
    end
end

# Solve the system
solution = solve_disk()

# Extract solutions
ρ, v_r, Ω, P, T, cs, H, τ_rϕ, F, j, U, κ = [solution[i,:] for i in 1:12]

# Plotting
plot(layout=(6,2), size=(900, 1500), legend=false)
params = [ρ, v_r, Ω, P, T, cs, H, τ_rϕ, F, j, U, κ]
labels = ["ρ", "v_r", "Ω", "P", "T", "cs", "H", "τ_rϕ", "F", "j", "U", "κ"]

for i in 1:12
    plot!(r, params[i], subplot=i, xlabel="r", ylabel=labels[i], title=labels[i], xscale=:log10, yscale=:log10)
end

savefig("accretion_disk_plots.png")
display(plot!(size=(900, 1500)))

# Print results
println("Steady-State Solutions:")
for (param, label) in zip(params, labels)
    println("$label = ", param)
end

# Calculate and print derived quantities
Ṁ = 4π .* r .* H .* ρ .* v_r
println("Mass accretion rate Ṁ = ", Ṁ)

t_visc = r.^2 ./ (α .* cs .* H)
println("Viscous timescale t_visc = ", t_visc)

Q_plus = 9/4 .* α .* P .* Ω
println("Viscous heating rate Q_plus = ", Q_plus)

Q_minus = F
println("Radiative cooling rate Q_minus = ", Q_minus)

τ = κ .* ρ .* H
println("Optical depth τ = ", τ)
