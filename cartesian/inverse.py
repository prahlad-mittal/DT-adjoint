import numpy as np
from gadopt import *
from gadopt.inverse import *

left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

with CheckpointFile("forward_checkpoint.h5", mode="r") as fi:
    mesh = fi.load_mesh(name="firedrake_default")
    mesh.cartesian = True
    dt_actual = fi.load_function(mesh, "Actual DT")
    mu_actual = fi.load_function(mesh, "mu")


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
mu_control = Function(W, name="control log viscosity").assign(0.0)
control = Control(mu_control)

kx = 1.0  # Conductivity in the x direction
ky = 0.0  # Conductivity in the y direction (set to zero for no diffusion)

# Construct the anisotropic conductivity tensor
# This tensor will have non-zero values only for the x component
ex = as_vector((1, 0))  # Unit vector in the x direction
ey = as_vector((0, 1))  # Unit vector in the y direction

K = kx * outer(ex, ex) + ky * outer(ey, ey)

smoother = DiffusiveSmoothingSolver(function_space=W, wavelength=1.0, K=K)
mu = Function(W, name="viscosity")
mu.project(10 ** smoother.action(mu_control))
approximation = BoussinesqApproximation(Ra)

X = SpatialCoordinate(mesh)
T_f = Function(Q, name="T").interpolate((1.0-X[1]) + (0.05*cos(pi*X[0])*sin(pi*X[1])))
T_check = conditional(T_f > 0, T_f, 0)
T = Function(Q, name="Temperature").interpolate(T_check)

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

# compute `model` dynamic topography
surface_force = surface_force_solver.solve()

#dynamic topography
deltarho_g = Constant(1e3) #delta rho = 100, g = 10
dt_iso = Function(Q1, name="Iso DT")
dt_iso.interpolate((surface_force / deltarho_g))

# form the objective function, between model and `data`
objective = assemble(0.5 * (dt_actual - dt_iso) ** 2 * ds(top_id))

print("J=", objective)

# Defining the reduced functional
reduced_functional = ReducedFunctional(objective, controls=control)
# Callback function for writing out the solution's visualisation
solution_pvd = VTKFile("solutions_example.pvd")


def callback():
    solution_pvd.write(mu_control.block_variable.checkpoint)


# Perform a bounded nonlinear optimisation where temperature
# is only permitted to lie in the range [0, 1]
mu_lb = Function(mu_control.function_space(), name="Lower bound viscosity")
mu_ub = Function(mu_control.function_space(), name="Upper bound viscosity")
mu_lb.assign(-2.0)
mu_ub.assign(2.0)

minimisation_problem = MinimizationProblem(reduced_functional, bounds=(mu_lb, mu_ub))

# # Adjust minimisation parameters
# minimisation_parameters["Iteration Limit"] = 200

optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
)
optimiser.add_callback(callback)
optimiser.run()

optimiser.rol_solver.rolvector.dat[0].rename("Final Solution")
mu_final = Function(W, name="viscosity")
mu_final.interpolate(10 ** (optimiser.rol_solver.rolvector.dat[0]))

with CheckpointFile("final_soln_example.h5", mode="w") as fi:
    fi.save_mesh(mesh)
    fi.save_function(optimiser.rol_solver.rolvector.dat[0])
    fi.save_function(mu_final)
    
VTKFile("final_solution_example.pvd").write(mu_final)
