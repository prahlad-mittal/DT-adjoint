# final timestep = 2797
from firedrake import *
from gadopt import *
import numpy as np

left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs
nx, ny = 120, 120  # Number of cells in x and y directions.
with CheckpointFile("final_checkpoint_cartesian.h5", 'r') as final_checkpoint:
    mesh = final_checkpoint.load_mesh(name="firedrake_default")
    mesh.cartesian = True
    
    T = final_checkpoint.load_function(mesh, "Temperature", idx = 2797)
    mu = final_checkpoint.load_function(mesh, "Viscosity", idx = 2797)
    
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q1 = FunctionSpace(mesh, "CG", 1)  # DT scalar function
Z = MixedFunctionSpace([V, W])  # Mixed function space.

z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")

Ra = Constant(1e4)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

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

stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             )

surface_force_solver = BoundaryNormalStressSolver(stokes_solver, top_id)

# Solve Stokes sytem:
stokes_solver.solve()
surface_force = surface_force_solver.solve()


#dynamic topography
deltarho_g = Constant(1e3) #delta rho = 100, g = 10
dt_actual = Function(Q1, name="Actual DT")
dt_actual.interpolate((surface_force / deltarho_g))

with CheckpointFile("dt_actual_cartesian.h5", mode="w") as file:
    file.save_mesh(mesh)
    file.save_function(dt_actual, name="Actual DT")
    file.save_function(T, name="Temperature")
    file.save_function(mu, name="Viscosity")
