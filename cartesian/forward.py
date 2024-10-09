# Generating the synthetic forward problem (reference twin)
# ===========
# This is the forward code for generating the synthetic test
# In this example using the example which is strongly based on our `base_case`, we
# generate a synthetic field of dynamic topography at domain surface boundary

from gadopt import *

nx, ny = 80, 80  # Number of cells in x and y directions.
mesh = UnitSquareMesh(nx, ny, quadrilateral=True)  # Square mesh generated via firedrake
mesh.cartesian = True
X = SpatialCoordinate(mesh)  # Spatial coordinates x = X[0], y = X[1]
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q1 = FunctionSpace(mesh, "CG", 1)  # DT scalar function
Z = MixedFunctionSpace([V, W])  # Mixed function space.

z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")

Ra = Constant(1e5)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)
T_f = Function(Q, name="T").interpolate((1.0-X[1]) + (0.05*cos(pi*X[0])*sin(pi*X[1])))
T_check = conditional(T_f > 0, T_f, 0)
T = Function(Q, name="Temperature").interpolate(T_check)
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

# Here we define the logarithmic scale of viscosity
# and define mu which will be passed on to StokesSolver

mu_power = Function(W, name="mu power").interpolate(-2*exp(-(X[1] - 0.8)**2/0.005))
mu_f = Function(W, name="mu f").interpolate(10 ** mu_power)
mu_check = conditional(mu_f > 0.1, mu_f, 0.1)
mu = Function(W, name="mu").interpolate(mu_check)

# +
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
                             constant_jacobian=True)

surface_force_solver = BoundaryNormalStressSolver(stokes_solver, top_id)

# At this point we have all the solver objects we need, we first solve for
# velocity, and then surface force (or surface dynamic topography)

# Solve Stokes sytem:
stokes_solver.solve()
surface_force = surface_force_solver.solve()

#dynamic topography
deltarho_g = Constant(1e3) #delta rho = 100, g = 10
dt_actual = Function(Q1, name="Actual DT")
dt_actual.interpolate((surface_force / deltarho_g))

# And here we visualise it and write the fields out

VTKFile("forward-example.pvd").write(*z.subfunctions, T, dt_actual, mu)
with CheckpointFile("forward_checkpoint.h5", mode="w") as fi:
    fi.save_mesh(mesh)
    fi.save_function(dt_actual, name="Actual DT")
    fi.save_function(mu, name = "mu")
    fi.save_function(T, name = "Temperature")
    
