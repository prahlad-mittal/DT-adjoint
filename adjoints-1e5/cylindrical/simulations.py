from gadopt import *
import numpy as np

#solver parameters (with changes)
newton_stokes_solver_parameters = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_max_it": 80,
    "snes_atol": 1e-8,
    "snes_rtol": 1e-5,
    "snes_stol": 1e-10,
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


#set up mesh
rmin, rmax, ncells, nlayers = 1.22, 2.22, 256, 64
mesh1d = CircleManifoldMesh(ncells, radius=rmin, degree=2)  # construct a circle mesh
mesh = ExtrudedMesh(mesh1d, layers=nlayers, extrusion_type='radial')  # extrude into a cylinder
mesh.cartesian = False  #new change
bottom_id, top_id = "bottom", "top"

# Set up geometry:
rmax_earth = 6370  # Radius of Earth [km]
rmin_earth = rmax_earth - 2900  # Radius of CMB [km]
r_410_earth = rmax_earth - 410  # 410 radius [km]
r_660_earth = rmax_earth - 660  # 660 raidus [km]
r_410 = rmax - (rmax_earth - r_410_earth) / (rmax_earth - rmin_earth)
r_660 = rmax - (rmax_earth - r_660_earth) / (rmax_earth - rmin_earth)

# Set up function spaces for the Q2Q1 pair
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q1 = FunctionSpace(mesh, "CG", 1)  # Average temperature function space (scalar, P1)
Q2 = FunctionSpace(mesh, "CG", 2) #Viscosity function space (scalar)
# DQ1 = FunctionSpace(mesh, "DG", 2) #lin Viscosity function space (scalar)
# DQ2 = FunctionSpace(mesh, "DG", 2) #plast Viscosity function space (scalar)
Z = MixedFunctionSpace([V, W])

#initialised temperature
X = SpatialCoordinate(mesh)
r = sqrt(X[0]**2 + X[1]**2)
T_f = Function(Q, name="T").interpolate(rmax - r + 0.02*cos(4*atan2(X[1], X[0])) * sin((r - rmin) * pi))
T_check = conditional(T_f > 0, T_f, 0)
T = Function(Q, name="Temperature").interpolate(T_check)

# Test functions and functions to hold solutions:
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p

#velocity and pressure functions
u_func, p_func = z.subfunctions
u_func.rename("Velocity")
p_func.rename("Pressure")

#rayleigh number
Ra = Constant(1e5) 
approximation = BoussinesqApproximation(Ra)

# Define time stepping parameters:
time = 0.0  # Initial time
timesteps = 20000 # Maximum number of timesteps
delta_t = Constant(1e-7)  # Constant time step
t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)
steady_state_tolerance = 1e-8  # Used to determine if solution has reached a steady state.

#average temperature fn
Taverage = Function(Q1, name="Average Temperature")
# Calculate the layer average of the initial state
averager = LayerAveraging(mesh, np.linspace(rmin, rmax, nlayers * 2), quad_degree=6)
averager.extrapolate_layer_average(Taverage, averager.get_layer_average(T))

checkpoint_file = CheckpointFile("final_checkpoint_1e5_mu(t).h5", "w")
checkpoint_file.save_mesh(mesh)
checkpoint_file.save_function(Taverage, name="Average Temperature", idx=0)
checkpoint_file.save_function(T, name="Temperature", idx=0)

# viscosity equations
mu_lin = 10.0
gamma_T = Constant(ln(100))  # temperature sensitivity of viscosity
# mu_star = Constant(0.1)      # effective viscosity at high stresses
# sigma_y = Constant(1e6)             # yield stress
# epsilon = sym(grad(u))  # Strain-rate
# epsii = sqrt(inner(epsilon, epsilon) + 1e-10)  # 2nd invariant (with tolerance to ensure stability)
mu_lin *= exp(-gamma_T * T)  # temperature-dependent linear component
# mu_plast = mu_star + (sigma_y / epsii) # Plastic component of rheological formulation
# mu_expr = 2 * (mu_lin * mu_plast) / (mu_lin + mu_plast) # Harmonic mean of linear and plastic components
mu = conditional(mu_lin > 0.1, mu_lin, 0.1)
#viscosity function
mu_function = Function(Q2, name="Viscosity")

#nullspaces
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)
Z_near_nullspace = create_stokes_nullspace(
    Z, closed=False, rotational=True, translations=[0, 1]
)

# Free-slip velocity boundary condition on all sides
stokes_bcs = {
    "bottom": {"un": 0},
    "top": {"un": 0},
}
temp_bcs = {
    "bottom": {"T": 1.0},
    "top": {"T": 0.0},
}

gd = GeodynamicalDiagnostics(z, T, bottom_id, top_id, quad_degree=6)  #new

energy_solver = EnergySolver(
    T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs
)

stokes_solver = StokesSolver(
    z,
    T,
    approximation,
    mu=mu,
    bcs=stokes_bcs,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
    near_nullspace=Z_near_nullspace,
    solver_parameters=newton_stokes_solver_parameters,
)

# Create output file and select output_frequency
output_file = VTKFile("output_mu(t)_1e5.pvd")
dump_period = 40


# Now perform the time loop:
for timestep in range(0, timesteps):
    stokes_solver.solve()
    energy_solver.solve()
    
    #timestepping
    if timestep != 0:
        dt = t_adapt.update_timestep()
    else:
        dt = float(delta_t)
    time += dt

    #saving simulation
    if timestep % dump_period == 0 or timestep == timesteps-1:
        mu_function.interpolate(mu)
        output_file.write(*z.subfunctions, T, mu_function)
        print("timestep =", timestep)
    

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
print("timestep = ", timestep)
