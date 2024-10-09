from gadopt import *

#solver parameters (with changes)
newton_stokes_solver_parameters = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_max_it": 100,
    "snes_atol": 1e-3,
    "snes_rtol": 1e-2,
    "snes_stol": 1e-3,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_converged_reason": None,
    "fieldsplit_0": {
        "ksp_converged_reason": None,
    },
    "fieldsplit_1": {
        "ksp_converged_reason": None,
    },
}

nx, ny = 120, 120  # Number of cells in x and y directions.
mesh = UnitSquareMesh(nx, ny, quadrilateral=True)  # Square mesh generated via firedrake
mesh.cartesian = True
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.
Q2 = FunctionSpace(mesh, "CG", 2) #Viscosity function space (scalar)

z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")

X = SpatialCoordinate(mesh)
T = Function(Q, name="Temperature")
T.interpolate((1.0-X[1]) + (0.05*cos(pi*X[0])*sin(pi*X[1])))

Ra = Constant(1e4)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

time = 0.0  # Initial time
delta_t = Constant(1e-6)  # Initial time-step
timesteps = 10000  # Maximum number of timesteps
t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.2)
steady_state_tolerance = 1e-5  # Used to determine if solution has reached a steady state.

checkpoint_file = CheckpointFile("final_checkpoint_cartesian.h5", "w")
checkpoint_file.save_mesh(mesh)
checkpoint_file.save_function(T, name="Temperature", idx=0)

# viscosity equations
mu_lin = 2.0
gamma_T = Constant(ln(50)) 
mu_lin *= exp(-gamma_T * T)  
mu = conditional(mu_lin > 0.1, mu_lin, 0.1)
#viscosity function
mu_function = Function(Q2, name="Viscosity")

Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

stokes_bcs = {
    bottom_id: {'uy': 0},
    top_id: {'uy': 0},
    left_id: {'ux': 0},
    right_id: {'ux': 0},
}

temp_bcs = {
    bottom_id: {'T': 1.0},
    top_id: {'T': 0.0},
}

energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)

stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu = mu,
                             solver_parameters=newton_stokes_solver_parameters,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             )

output_file = VTKFile("output_cartesian.pvd")
dump_period = 25


for timestep in range(0, timesteps):
    stokes_solver.solve()
    energy_solver.solve()

    #saving simulation
    if timestep % dump_period == 0 or timestep == timesteps-1:
        mu_function.interpolate(mu)
        output_file.write(*z.subfunctions, T, mu_function)
        print("timestep = ", timestep)

    dt = t_adapt.update_timestep()
    time += dt


    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((T - energy_solver.T_old)**2 * dx))
    
    # Leave if steady-state has been achieved:
    if maxchange < steady_state_tolerance:
        log("Steady-state achieved -- exiting time-step loop")
        break

# close checkpoint file
checkpoint_file.save_function(T, name="Temperature", idx = timestep)
checkpoint_file.save_function(mu_function, name="Viscosity", idx = timestep)
checkpoint_file.close()
