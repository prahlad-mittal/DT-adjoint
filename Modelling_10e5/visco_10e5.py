from gadopt import *

rmin, rmax, ncells, nlayers = 1.22, 2.22, 256, 64
mesh1d = CircleManifoldMesh(ncells, radius=rmin, degree=2)  # construct a circle mesh
mesh = ExtrudedMesh(mesh1d, layers=nlayers, extrusion_type='radial')  # extrude into a cylinder
mesh.cartesian = False  #new change
bottom_id, top_id = "bottom", "top"

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.

z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")

Ra = Constant(1e5)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

time = 0.0  # Initial time
delta_t = Constant(1e-7)  # Initial time-step
timesteps = 20000  # Maximum number of timesteps
t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)
steady_state_tolerance = 1e-8  # Used to determine if solution has reached a steady state.

X = SpatialCoordinate(mesh)
T = Function(Q, name="Temperature")
r = sqrt(X[0]**2 + X[1]**2)
T.interpolate(rmax - r + 0.02*cos(4*atan2(X[1], X[0])) * sin((r - rmin) * pi))

#equations
gamma_T = Constant(ln(100))  # temperature sensitivity of viscosity
mu_star = Constant(0.001)      # effective viscosity at high stresses
sigma_y = Constant(1.0)        # yield stress
epsilon = sym(grad(u))  # Strain-rate
epsii = sqrt(inner(epsilon, epsilon) + 1e-10)  # 2nd invariant (with tolerance to ensure stability)
mu_lin = exp(-gamma_T * T)  # temperature-dependent linear component
mu_plast = mu_star + (sigma_y / epsii) # Plastic component of rheological formulation
mu_expr = (2. * mu_lin * mu_plast) / (mu_lin + mu_plast) # Harmonic mean of linear and plastic components
#viscosity function
mu = Function(Q, name="Viscosity")
mu.interpolate(mu_expr)

Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)

Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])

stokes_bcs = {
    bottom_id: {'un': 0},
    top_id: {'un': 0},
}

temp_bcs = {
    bottom_id: {'T': 1.0},
    top_id: {'T': 0.0},
}

output_file = VTKFile("output_hr_new.pvd")
ref_file = VTKFile('reference_state_hr_new.pvd')
output_frequency = 20

plog = ParameterLog('params_hr_new.log', mesh)
plog.log_str("timestep time dt maxchange u_rms nu_base nu_top energy avg_T T_min T_max")

gd = GeodynamicalDiagnostics(z, T, bottom_id, top_id, quad_degree=6)  #new
 
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)

stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu,
                            constant_jacobian=True,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             near_nullspace=Z_near_nullspace)

for timestep in range(0, timesteps):

    # Write output:
    if timestep % output_frequency == 0:
        output_file.write(*z.subfunctions, T, mu)

    if timestep != 0:
        dt = t_adapt.update_timestep()
    else:
        dt = float(delta_t)
    time += dt

    # Solve Stokes sytem:
    stokes_solver.solve()

    # Temperature system:
    energy_solver.solve()

    # Compute diagnostics:
    f_ratio = rmin/rmax
    top_scaling = 1.3290170684486309  # log(f_ratio) / (1.- f_ratio)
    bot_scaling = 0.7303607313096079  # (f_ratio * log(f_ratio)) / (1.- f_ratio)
    nusselt_number_top = gd.Nu_top() * top_scaling
    nusselt_number_base = gd.Nu_bottom() * bot_scaling
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))

    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((T - energy_solver.T_old)**2 * dx))
    
    # Log diagnostics:
    plog.log_str(f"{timestep} {time} {float(delta_t)} {maxchange} {gd.u_rms()} "
                 f"{nusselt_number_base} {nusselt_number_top} "
                 f"{energy_conservation} {gd.T_avg()} {gd.T_min()} {gd.T_max()} ")

    # Leave if steady-state has been achieved:
    if maxchange < steady_state_tolerance:
        log("Steady-state achieved -- exiting time-step loop")
        break


plog.close()

with CheckpointFile("Final_State_hr_new.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(T, name="Temperature")
    final_checkpoint.save_function(z, name="Stokes")
    final_checkpoint.save_function(mu, name="Viscosity")
