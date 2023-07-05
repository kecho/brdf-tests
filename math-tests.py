import numpy as np
import matplotlib.pyplot as plt
import math


def sg(z,lamb):
    return np.exp(lamb*z - lamb)

def sg_int(lamb):
    return 2.0 * math.pi * (1.0/lamb) * (1.0 - np.exp(-2.0*lamb))

def sg_int_omega(lamb):
    return 2.0 * math.pi * (1.0/lamb) * (1.0 - np.exp(-lamb))

def create_normal(theta, phi):
    return np.array([np.sin(phi)*np.sin(theta), np.cos(phi)*np.sin(theta), np.cos(theta)])

def invent_basis_from_normal(n):
    s = -1.0 if 0.0 > n[2] else 1.0
    a0 = 1.0 / (s + n[2])
    a1 = n[0] * n[1] * a0
    t = np.array([1.0 + s * n[0] * n[0] * a0, s * a1, -s * n[0] ])
    b = np.array([a1, s + n[1] * n[1] * a0, -n[1]])
    return np.array([t, b, n]);

def numerically_verify_int(samples, thetaVals):
    int_val = 0
    for i in range(0,samples, 1):
        r_val = 0
        for j in range(0, samples, 1):
            #r_val += 2.0 * math.pi * math.sin(thetaVals[i]) * sg(math.cos(thetaVals[i]), lamb)
            r_val += 2.0 * math.pi * math.sin(thetaVals[i]) * math.cos(thetaVals[i])
        r_val /= samples
        int_val += math.pi/2.0*r_val
    int_val /= samples
    print("an integral: " + str(sg_int_omega(lamb)))
    print("rat: " + str(int_val))


samples = 32
idxX, idxY = np.indices((samples,samples))
thetaVals, phiVals = np.linspace(0,math.pi/2,samples), np.linspace(0,2.0*math.pi,samples)
thetas, phis = np.meshgrid(thetaVals, phiVals)
x, y, z = np.sin(phis)*np.sin(thetas), np.cos(phis)*np.sin(thetas), np.cos(thetas)
#thetas = np.ravel(thetas)
#phis = np.ravel(phis)
#x, y, z = np.sin(phis)*np.sin(thetas), np.cos(phis)*np.sin(thetas), np.cos(thetas)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim3d(-1.0, 1.0)
ax.set_ylim3d(-1.0, 1.0)
ax.set_zlim3d(-1.0, 1.0)

lamb = 4
sg_v = sg(z, lamb)# - sg(-1,lamb)
x = sg_v * x
y = sg_v * y
z = sg_v * z
ax.scatter(x, y, z, marker='o')
ax.plot_trisurf(np.ravel(x), np.ravel(y), np.ravel(z), cmap='viridis')

quad_w = 2
quad_h = 3
quad_r = 3
quad_theta = 0.25 * math.pi
quad_phi  = 0.25 * math.pi
quad_basis = invent_basis_from_normal(create_normal(quad_theta, quad_phi))
q0 = quad_basis[0] * (-quad_w) + quad_basis[1] * (-quad_h) + quad_basis[2] * quad_r
q1 = quad_basis[0] * (quad_w)  + quad_basis[1] * (-quad_h) + quad_basis[2] * quad_r
q2 = quad_basis[0] * (-quad_w) + quad_basis[1] * (quad_h)  + quad_basis[2] * quad_r
q3 = quad_basis[0] * (quad_w)  + quad_basis[1] * (quad_h)  + quad_basis[2] * quad_r

ax.scatter(np.array([q0, q1, q2, q3]), marker='^')

plt.show()


numerically_verify_int(samples, thetaVals)

