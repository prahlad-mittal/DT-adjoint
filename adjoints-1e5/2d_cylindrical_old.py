from gadopt import *
from gadopt.inverse import *
import numpy as np
import inspect
# Open the checkpoint file and subsequently load the mesh:
rmin, rmax = 1.22, 2.22
with CheckpointFile("../forward/Final_State.h5", "r") as forward_check:
    mesh = forward_check.load_mesh("firedrake_default_extruded")
    mesh.cartesian = False
    bottom_id, top_id = "bottom", "top"    
    T = forward_check.load_function(mesh, "Temperature")
    dtopo_obs = forward_check.load_function(mesh, "Observed_DT")
    mu_observed = forward_check.load_function(mesh, "Observed_Viscosity")
    T_average = forward_check.load_function(mesh, "Average_Temperature")
    mu_av = forward_check.load_function(mesh, "Average_Viscosity")    
    u_obs = forward_check.load_function(mesh, "Velocity")

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q1 = FunctionSpace(mesh, "CG", 1) #scalar space for functions
Z = MixedFunctionSpace([V, W])  # Mixed function space.

tape = get_working_tape()
tape.clear_tape()
print(tape.get_blocks())

z = Function(Z)  # A field over the mixed function space Z
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")

# We next specify the important constants for this problem, and set up the approximation.
Ra = Constant(5e4)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

time = 0.0  # Initial time
delta_t = Constant(5e-6)  # Initial time-step
timesteps = 10  # Maximum number of timesteps
t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)


# For our inverse study we need to determine the viscosity,  mu_control.
# This will be the parameter that we pretend we don't know and
# we try to invert for. So let's assume that mu is the layer average of the forward model, to start.

mu_average = Function(Q1, name="Average_Viscosity").project(mu_av)

mu0_bc_base = DirichletBC(Q1, 0.01, bottom_id)
mu0_bc_top = DirichletBC(Q1, 1., top_id)
# Combine the boundary conditions:
mu0_bcs = [mu0_bc_base, mu0_bc_top]

# control viscosity define
mu_control = Function(Q1, name="Viscosity_Control").assign(mu_average)
control = Control(mu_control)

# And apply through projection of mu_average:
# We next evaluate the viscosity, which depends on temperature:
mu0 = Function(Q1, name="viscosity_0").project(mu_control, bcs=mu0_bcs)

# mu_actual
# bcs
mu_bc_base = DirichletBC(Q, 0.01, bottom_id)
mu_bc_top = DirichletBC(Q, 1., top_id)
# Combine the boundary conditions:  
mu_bcs = [mu_bc_base, mu_bc_top]
mu = Function(Q, name="Viscosity").project(mu0, bcs=mu_bcs)

# We also set up a dynamic topography field for visualisation:
dtopo = Function(Q1, name="Dynamic_Topography")
deltarho_g = Constant(1e3) #for scaling
dtopo_error = Function(Q1, name="Dynamic_Topography_Error")


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

# +
stokes_solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)

stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu, solver_parameters=stokes_solver_parameters,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace)
#                             near_nullspace=Z_near_nullspace)

# Define the surface force solver

surface_force_solver = BoundaryNormalStressSolver(stokes_solver, top_id)
u_misfit = 0.0
dt_misfit = 0.0

for timestep in range(0, timesteps): 

    mu.interpolate(mu)  
    
    # Update the temperature field
    energy_solver.solve()
    
    # Update the stokes solver
    stokes_solver.solve()
    
    # Surface force:
    surface_force = surface_force_solver.solve()
    
    # Update fields 
    dtopo.interpolate((surface_force/deltarho_g))
    dtopo_error.interpolate(dtopo_obs - dtopo)

    dt_misfit += assemble((dtopo - dtopo_obs)**2 * ds_t)

    u_misfit += assemble(dot(u - u_obs, u - u_obs) * ds_t)



# So up until here everything is exactly like the forward code (albeit without time).
# Now at this point, we define our misfit: the difference between model and `observation`

# Form the objective function, between model and `data` and add some smoothing
surface_misfit = dt_misfit / timesteps

mu_misfit = assemble(0.5 * (mu - mu_observed) ** 2 * dx)

print("surface misfit=", surface_misfit)
log(f"velocity misfit={u_misfit}")
log(f"mu misfit={mu_misfit}")

alpha_u = 50.0
# alpha_s = 100.0
# alpha_d = 6000.0
alpha_s = 0
alpha_d = 0

# Define the component terms of the overall objective functional
damping = assemble((mu0 - mu_average) ** 2 * dx)
norm_damping = assemble(mu_average**2 * dx)
smoothing = assemble(dot(grad(mu0 - mu_average ), grad(mu0 -mu_average)) * dx)
norm_smoothing = assemble(dot(grad(dtopo_obs), grad(dtopo_obs)) * dx)
norm_obs = assemble(dtopo_obs**2 * dx)
norm_u_surface = assemble(dot(u_obs, u_obs) * ds_t)

smoother= (
        alpha_u * (norm_obs * u_misfit / norm_u_surface / timesteps) +
        alpha_s * (norm_obs * smoothing / norm_smoothing) +
        alpha_d * (norm_obs * damping / norm_damping)
    )

objective = ( surface_misfit + smoother )

# print individual terms 
log(f"Surface vel: {alpha_u* (norm_obs * u_misfit / norm_u_surface/ timesteps)}")
log(f"damping: {alpha_d* (norm_obs * damping / norm_damping)}")
log(f"smoothing: {alpha_s* (norm_obs * smoothing / norm_smoothing)}")

# Using the definition of our objective function we can define the reduced functional
reduced_functional = ReducedFunctional(objective, control)
log(f"\n\nReduced functional: {reduced_functional(mu_control)}")
log(f"Objective: {objective}\n\n")

# Having the reduced functional one can easily compute the derivative
der_func = reduced_functional.derivative(options={"riesz_representation": "L2"})
der_func.rename("derivative")

# Visualising the derivative
VTKFile("derivative-visualisation.pvd").write(*z.subfunctions, u_obs, T, mu_observed, mu, dtopo_obs, dtopo, dtopo_error, der_func)

# Performing taylor test
Delta_mu = Function(mu_control.function_space(), name="Delta_Temperature")
Delta_mu.dat.data[:] = np.random.random(Delta_mu.dat.data.shape) * 0.1

# Perform the Taylor test to verify the gradients
minconv = taylor_test(reduced_functional, mu_control, Delta_mu)

# ------------------------------inversion-------------------------

# Now perform inversion:
solution_pvd = VTKFile("solutions.pvd")

def callback():
    solution_pvd.write(mu_control.block_variable.checkpoint)
    mu_inv_misfit = assemble(
        (mu_control.block_variable.checkpoint - mu_observed) ** 2 * dx
    )
    log(f"Terminal Viscosity Misfit: {mu_inv_misfit}")



mu_lb = Function(mu_control.function_space(), name="Lower_bound_mu").assign(1e-2)
mu_ub = Function(mu_control.function_space(), name="Upper_bound_mu").assign(1.0)

minimisation_problem = MinimizationProblem(reduced_functional, bounds=(mu_lb, mu_ub))

# Adjust minimisation parameters:
minimisation_parameters["Status Test"]["Iteration Limit"] = 200

optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
)
optimiser.add_callback(callback)
optimiser.run()

# Now look at final solution:
optimiser.rol_solver.rolvector.dat[0].rename("Final_Solution")
with CheckpointFile("final_solution.h5", mode="w") as fi:
    fi.save_mesh(mesh)
    fi.save_function(optimiser.rol_solver.rolvector.dat[0])

VTKFile("final_solution.pvd").write(optimiser.rol_solver.rolvector.dat[0])
