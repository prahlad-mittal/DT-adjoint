#control woth smoothing
#DT2 (wrong viscosity)
#solving for the Dynamic Topography (DT2)

from gadopt import *
from gadopt.inverse import *

#load previous stuff - mesh, T, DT1
bottom_id, top_id = "bottom", "top"
with CheckpointFile("dt_actual_mu_t.h5", 'r') as final_checkpoint:
    mesh = final_checkpoint.load_mesh("firedrake_default_extruded")
    mesh.cartesian = False
    
    #viscosity as control
    T = final_checkpoint.load_function(mesh, "Temperature")
    dt_actual = final_checkpoint.load_function(mesh, "Actual DT")
    mu_actual = final_checkpoint.load_function(mesh, "Viscosity")

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q2 = FunctionSpace(mesh, "CG", 2) #Dynamic Topogrpahy function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.

z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")

Ra = Constant(1e5)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)


#Smoothing 
X = SpatialCoordinate(mesh)
x = X[0]
y = X[1]
er = as_vector((x/sqrt(x**2 + y**2), y/sqrt(x**2 + y**2)))  
etheta = as_vector((-y/sqrt(x**2 + y**2), x/sqrt(x**2 + y**2))) 

kr = 1.0  # radial 
ktheta = 0.0  # angular 

# Conductivity  
K = kr * outer(er, er) + ktheta * outer(etheta, etheta)

smoother = DiffusiveSmoothingSolver(function_space=W, wavelength=1.0, K=K)

#viscosity as control d(mu)
mu_control = Function(W, name="control").assign(0.0)
control = Control(mu_control)

#viscosity (isoviscous)
mu_iso = Function(W, name="Isoviscosity")
mu_iso.project(10 ** smoother.action(mu_control))

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

stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu_iso,
                            constant_jacobian=True,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             near_nullspace=Z_near_nullspace)

surface_force_solver = BoundaryNormalStressSolver(stokes_solver, top_id)

# At this point we have all the solver objects we need, we first solve for
# velocity, and then surface force (or surface dynamic topography)

# Solve Stokes sytem:
stokes_solver.solve()

#surface stress
surface_force = surface_force_solver.solve()

#isoviscous DT
deltarho_g = Constant(1e3) #delta rho = 100, g = 10
dt_iso = Function(Q2, name="Isoviscous DT")
dt_iso.interpolate((surface_force / deltarho_g))

#store the data till now
with CheckpointFile("DT_file_mu_t_smoothing.h5", mode="w") as file:
    file.save_mesh(mesh)
    file.save_function(dt_actual, name="DT_actual")
    file.save_function(dt_iso, name="DT_isoviscous")
    file.save_function(T, name="Temperature")
    file.save_function(mu_iso, name="Isoviscosity")
    file.save_function(mu_actual, name="Viscosity")


#cost function calculation (J)
objective_func = assemble(0.5 * (dt_actual - dt_iso) ** 2 * ds_t)   #J (Cost function)
print ("cost function = ", objective_func)

#Calculate the gradient and see sensitivity
grad_func = reduced_functional.derivative(options={"riesz_representation": "L2"}) #see gradient
grad_func.rename("gradient func")

# #visualisations 
# VTKFile("gradient_mu_t_smoothing.pvd").write(T, mu_iso, mu_actual, dt_iso, dt_actual, grad_func)

# Performing taylor test
Delta_mu = Function(mu_iso.function_space(), name="Delta_Temperature")
Delta_mu.dat.data[:] = np.random.random(Delta_mu.dat.data.shape)

# Perform the Taylor test to verify the gradients
minconv = taylor_test(reduced_functional, mu_iso, Delta_mu)



#i cant run the following part of the code.....---------------------------------------------------------------------------

# Callback function for writing out the solution's visualisation
optimiser_pvd = VTKFile("optimiser.pvd")
def callback():
    optimiser_pvd.write(mu_control.block_variable.checkpoint)
    

# Perform a bounded nonlinear optimisation 
mu_lb = Function(mu_control.function_space(), name="Lower bound viscosity")
mu_ub = Function(mu_control.function_space(), name="Upper bound viscosity")
mu_lb.assign(0.4)
mu_ub.assign(2.0)

minimisation_problem = MinimizationProblem(reduced_functional, bounds=(mu_lb, mu_ub))

# Adjust minimisation parameters
minimisation_parameters["Status Test"]={
        "Gradient Tolerance": 1e-6,
        "Iteration Limit": 25,
    }

optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
    checkpoint_dir="optimisation_checkpoint"
)

optimiser.add_callback(callback)
optimiser.run()



