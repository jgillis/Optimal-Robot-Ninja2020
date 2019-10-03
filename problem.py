from rockit import *
from numpy import sin, cos, pi, exp, linspace
from casadi import sum1, Function, vertcat, horzcat

ocp = Ocp(T=FreeTime(1))

N = 50 # Number of control intervals
L = 0.4 # Wheel geometry

# Model: differential drive
v1 = ocp.control() # Wheel speed 1
v2 = ocp.control() # Wheel speed 2

x     = ocp.state()
y     = ocp.state()
theta = ocp.state()
state = vertcat(x,y,theta)

v = (v1+v2)/2
ocp.set_der(x, v*cos(theta))
ocp.set_der(y, v*sin(theta))
ocp.set_der(theta, (2/L)*(v2-v1))

max_meas = 1000



sigma       = 0.1
Ghat        = 0.15
umin       = -0.2
umax       = 0.2
ommin       = -pi
ommax       = pi

# Symbolic parameters of the optimization (not optimized)
#measp       = ocp.parameter(max_meas,2)
xbeginp     = ocp.parameter(3,1)
xfinalp     = ocp.parameter(3,1)

# Boundary conditions
ocp.subject_to(ocp.at_t0(state)==xbeginp)
ocp.subject_to(ocp.at_tf(state)==xfinalp)

# Path constraints
ocp.subject_to(umin <= (v1 <= umax))
ocp.subject_to(umin <= (v2 <= umax))

ocp.subject_to(v >= 0)
ocp.subject_to(ommin <= (ocp.der(theta) <= ommax))


# Obstacles
sx = sigma
sy = sigma
#g = sum1(exp(-(x-measp[:,0])**2/(2*sx**2)-(y-measp[:,1])**2./(2*sy**2)))
#costf = Function('costf',[measp,x,y],[g])
#ocp.subject_to(costf(measp,x,y)<=Ghat)

ocp.add_objective(ocp.T)

ocp.solver('ipopt')

ocp.method(MultipleShooting(N=N))

ocp.set_initial(theta,pi/2)
ocp.set_initial(v1,0.01)
ocp.set_initial(v2,0.01)

ocp.set_value(xbeginp, vertcat(0,0,pi/2))
ocp.set_value(xfinalp, vertcat(0.02,0.1,pi/2))

# Simulate an obstacle
ts = linspace(0,2*pi,max_meas)
meas_val = horzcat(0.1*cos(ts)-0.1,0.1*sin(ts))

#ocp.set_value(measp, meas_val)
#ocp.set_value(measp, meas_val)

sol = ocp.solve()


from pylab import *
xsol = sol.sample(x,grid='integrator',refine=10)
ysol = sol.sample(y,grid='integrator',refine=10)

figure()
_,xsol = sol.sample(x,grid='control')
_,ysol = sol.sample(y,grid='control')
plot(xsol,ysol,'o')
_,xsol = sol.sample(x,grid='integrator',refine=10)
_,ysol = sol.sample(y,grid='integrator',refine=10)
plot(xsol,ysol)

axis("equal")

figure()
ts, thetasol = sol.sample(theta,grid='integrator',refine=10)
ts, v1sol = sol.sample(v1,grid='integrator',refine=10)
ts, v2sol = sol.sample(v2,grid='integrator',refine=10)

plot(ts,v1sol,label="v1")
plot(ts,v2sol,label="v2")
plot(ts,xsol,label="x")
plot(ts,ysol,label="y")
plot(ts,thetasol,label="theta")
legend()



#xx, yy = np.meshgrid(linspace(-0.1,0.1,10), linspace(-0.1,0.1,20))
#costf(measp,x,y)

show(block=True)


