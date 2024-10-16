import numpy as np
from gadopt import *
from gadopt.inverse import *

left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

with CheckpointFile("dt_actual_cartesian.h5", mode="r") as fi:
    mesh = fi.load_mesh(name="firedrake_default")
    mesh.cartesian = True

    T = fi.load_function(mesh, "Temperature")
    dt_actual = fi.load_function(mesh, "Actual DT")
    mu_actual = fi.load_function(mesh, "Viscosity")


V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.
Q1 = FunctionSpace(mesh, "CG", 1)  # DT scalar function

z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")

Ra = Constant(1e4)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

mu_control = Function(W, name="control log viscosity").assign(0.0)
control = Control(mu_control)

#kx = 1.0  # Conductivity in the x direction
kx = 1.0
ky = 0.0  # Conductivity in the y direction (set to zero for no diffusion)

# Construct the anisotropic conductivity tensor
# This tensor will have non-zero values only for the x component
ex = as_vector((1, 0))  # Unit vector in the x direction
ey = as_vector((0, 1))  # Unit vector in the y direction

K = kx * outer(ex, ex) + ky * outer(ey, ey)

smoother = DiffusiveSmoothingSolver(function_space=W, wavelength=1.0, K=K)
mu_iso = Function(W, name="isoviscosity")
mu_iso.project((10 ** smoother.action(mu_control))/4)

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

stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu_iso,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             )

surface_force_solver = BoundaryNormalStressSolver(stokes_solver, top_id)

# Solve Stokes sytem:
stokes_solver.solve()

# compute `model` dynamic topography
surface_force = surface_force_solver.solve()

#isoviscous DT
deltarho_g = Constant(1e3) #delta rho = 100, g = 10
dt_iso = Function(Q1, name="Isoviscous DT")
dt_iso.interpolate((surface_force / deltarho_g))

# #cost function calculation (J)
# objective_func = assemble(0.5 * (dt_actual - dt_iso) ** 2 * ds(top_id))   #J (Cost function)
# print ("cost function = ", objective_func)

# # Tikhonov Regularisation
# dt_misfit = assemble((dt_actual - dt_iso) ** 2 * ds(top_id))  #L2 regularisation
# alpha = Constant(1e-4)
# reg_term = assemble(dot(grad(mu_actual - mu), grad(mu_actual-mu)) * dx)
# objective_func = dt_misfit + alpha * reg_term

# print ("J = ", objective_func)

# #reduced functional (adjoints)
# reduced_functional = ReducedFunctional(objective_func, controls=control)  #dJ/d(mu)
# print ("dJ/d(mu) ", reduced_functional)

# #Calculate the gradient and see sensitivity
# grad = reduced_functional.derivative(options={"riesz_representation": "L2"}) #see gradient
# grad.rename("gradient func")

# #visualisations 
# VTKFile("gradient_cartesian.pvd").write(*z.subfunctions, T, mu_iso, mu_actual, dt_iso, dt_actual, grad)

# # Performing taylor test
# Delta_mu = Function(mu_control.function_space(), name="Delta_Temperature")
# Delta_mu.dat.data[:] = np.random.random(Delta_mu.dat.data.shape)

# # Perform the Taylor test to verify the gradients
# minconv = taylor_test(reduced_functional, mu_control, Delta_mu)

# ---------------------------------------------------optimisation---------
# # Callback function for writing out the solution's visualisation
# solution_pvd = VTKFile("solutions_cartesian.pvd")

# #optimisation part
# def callback():
#     solution_pvd.write(mu_control.block_variable.checkpoint)


# Perform a bounded nonlinear optimisation where temperature
# is only permitted to lie in the range [0, 1]
# mu_lb = Function(mu_control.function_space(), name="Lower bound viscosity")
# mu_ub = Function(mu_control.function_space(), name="Upper bound viscosity")
# mu_lb.assign(-1.0)
# mu_ub.assign(0)

# minimisation_problem = MinimizationProblem(reduced_functional, bounds=(mu_lb, mu_ub))

# optimiser = LinMoreOptimiser(
#         minimisation_problem,
#         minimisation_parameters,
#     )

# optimiser.add_callback(callback)
# optimiser.run()

# optimiser.rol_solver.rolvector.dat[0].rename("Final Solution")
# mu_final = Function(W, name="inv viscosity")
# mu_final.interpolate(10 ** (optimiser.rol_solver.rolvector.dat[0]))

# with CheckpointFile("final_solution_cartesian.h5", mode="w") as fi:
#     fi.save_mesh(mesh)
#     fi.save_function(optimiser.rol_solver.rolvector.dat[0])
#     fi.save_function(mu_final)
    
# VTKFile("final_solution_cartesian.pvd").write(mu_final, optimiser.rol_solver.rolvector.dat[0])

dt_misfit = assemble((dt_actual - dt_iso) ** 2 * ds(top_id))  #L2 regularisation
misfit_dt = []
misfit_mu = []
reg_mu = []
reg_term = assemble(dot(grad(mu_actual - mu_iso), grad(mu_actual-mu_iso)) * dx)

# Perform a bounded nonlinear optimisation where temperature
# is only permitted to lie in the range [0, 1]
mu_lb = Function(mu_control.function_space(), name="Lower bound viscosity")
mu_ub = Function(mu_control.function_space(), name="Upper bound viscosity")
mu_lb.assign(-1.0)
mu_ub.assign(0)

# # Adjust minimisation parameters
# minimisation_parameters["Status Test"]["Iteration Limit"] = 2

for i in range (0, 6):

    # Tikhonov Regularisation
    
    alpha = (10 ** (-1 * i))
    objective_func = dt_misfit + alpha * reg_term
    print ("alpha = ", alpha)
    print ("J = ", objective_func)

    #reduced functional (adjoints)
    reduced_functional = ReducedFunctional(objective_func, controls=control)  #dJ/d(mu)
    print ("dJ/d(mu) ", reduced_functional)

    minimisation_problem = MinimizationProblem(reduced_functional, bounds=(mu_lb, mu_ub))

    optimiser = LinMoreOptimiser(
            minimisation_problem,
            minimisation_parameters,
        )

    optimiser.run()

    optimiser.rol_solver.rolvector.dat[0].rename("Final Solution")
    mu_inv = Function(W, name="inv viscosity")
    mu_inv.interpolate(10 ** (optimiser.rol_solver.rolvector.dat[0]))

    #dt_inv
    stokes_solver2 = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu_inv,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             )

    surface_force_solver2 = BoundaryNormalStressSolver(stokes_solver2, top_id)

    # Solve Stokes sytem:
    stokes_solver2.solve()

    # compute `inverse` dynamic topography
    surface_force2 = surface_force_solver2.solve()

    dt_inv = Function(Q1, name="Inverse DT")
    dt_inv.interpolate((surface_force2 / deltarho_g))

    misfit_dt_inv = assemble((dt_actual - dt_inv) ** 2 * ds(top_id))
    misfit_dt.append(misfit_dt_inv)

    misfit_mu_inv = assemble((mu_actual - mu_inv) ** 2 * dx)
    misfit_mu.append(misfit_mu_inv)

    reg_inv = assemble(dot(grad(mu_actual - mu_inv), grad(mu_actual-mu_inv)) * dx)
    reg_mu.append(reg_inv)
    
    print("dt_inv_misfit = ", misfit_dt_inv)    
    print("mu_inv_misfit = ", misfit_mu_inv)
    print("Reg term = ", reg_inv)

misfit_dt = np.array(misfit_dt)
misfit_mu = np.array(misfit_mu)
reg_mu = np.array(reg_mu)

print("Misfit DT", misfit_dt)
print("Misfit mu", misfit_mu)
print("Reg Terms", reg_mu)

import matplotlib.pyplot as plt 

# Plotting the L-Curve
plt.figure(figsize=(16, 12))
plt.loglog(reg_mu, misfit_dt, marker='o', linestyle='-')
plt.title('L-Curve')
plt.xlabel('Regularization Term (||L(μ_actual - μ_0)||^2)')
plt.ylabel('Data Misfit (||d_obs - d_pred||^2)')
plt.grid(True)
plt.show()





