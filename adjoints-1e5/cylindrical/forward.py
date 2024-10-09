# final timestep =  15234
from firedrake import *
from gadopt import *
import numpy as np

#set up mesh
rmin, rmax, ncells, nlayers = 1.22, 2.22, 256, 64
bottom_id, top_id = "bottom", "top"

with CheckpointFile("final_checkpoint_1e5_mu(t).h5", 'r') as final_checkpoint:
    mesh = final_checkpoint.load_mesh("firedrake_default_extruded")
    mesh.cartesian = False
    
    T = final_checkpoint.load_function(mesh, "Temperature", idx = 15234)
    mu = final_checkpoint.load_function(mesh, "Viscosity", idx = 15234)
    Taverage = final_checkpoint.load_function(mesh, "Average Temperature", idx =0)

#solving for the Dynamic Topography (DT1)

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q1 = FunctionSpace(mesh, "CG", 1)  # Dynamic topography
Z = MixedFunctionSpace([V, W])  # Mixed function space.

# Paverage = Function(Q1, name="Average Pressure")
# # Calculate the layer average of the initial state
# averager_pressure = LayerAveraging(mesh, np.linspace(rmin, rmax, nlayers * 2), quad_degree=6)
# averager_pressure.extrapolate_layer_average(Paverage, averager_pressure.get_layer_average(p_load))

z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p

#velocity and pressure functions
u_func, p_func = z.subfunctions
u_func.rename("Velocity")
p_func.rename("Pressure")

Ra = Constant(1e5)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)

Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])

#bcs 
stokes_bcs = {
    bottom_id: {'un': 0},
    top_id: {'un': 0},
}

temp_bcs = {
    bottom_id: {'T': 1.0},
    top_id: {'T': 0.0},
}

stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu,
                            constant_jacobian=True,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             near_nullspace=Z_near_nullspace)

surface_force_solver = BoundaryNormalStressSolver(stokes_solver, top_id)

# At this point we have all the solver objects we need, we first solve for
# velocity, and then surface force (or surface dynamic topography)

# Solve Stokes sytem:
stokes_solver.solve()

#get the stresses
surface_force = surface_force_solver.solve()

#dynamic topography
deltarho_g = Constant(1e3) #delta rho = 100, g = 10
dt_actual = Function(Q1, name="Actual DT")
dt_actual.interpolate((surface_force / deltarho_g))

# And here we visualise it and write the fields out

VTKFile("dt_actual_mu_t.pvd").write(*z.subfunctions, T, dt_actual, mu)
with CheckpointFile("dt_actual_mu_t.h5", mode="w") as file:
    file.save_mesh(mesh)
    file.save_function(dt_actual, name="Actual DT")
    file.save_function(T, name="Temperature")
    file.save_function(mu, name="Viscosity")
