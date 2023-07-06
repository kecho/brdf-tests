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
    n_x, n_y, n_z = n
    s  = -1.0 if 0.0 > n_z else  1.0
    a0 = -1.0 / (s + n_z)
    a1 = n_x * n_y * a0
    t = np.array([ 1.0 + s * n_x * n_x * a0, s * a1, -s * n_x ])
    b = np.array([ a1, s + n_y * n_y * a0, -n_y ])
    return np.array([t, b, n])

def quad_sample_points(theta, phi, halfwidth, halfheight, c, axis_samples):
    n = create_normal(theta, phi)
    bi, bj, bn = invent_basis_from_normal(n)
    i_range, j_range = np.linspace(-halfwidth, halfwidth, axis_samples), np.linspace(-halfheight, halfheight, axis_samples)
    i, j = np.meshgrid(i_range, j_range)
    x, y, z = c[0] + i * bi[0] + j * bj[0], c[1] + i * bi[1] + j * bj[1], c[2] + i * bi[2] + j * bj[2]
    return (np.ravel(x), np.ravel(y), np.ravel(z))

def quad_edge_points(q_theta, q_phi, q_halfwidth, q_halfheight, q_c):
    (x, y, z) = quad_sample_points(q_theta, q_phi, q_halfwidth, q_halfheight, q_c, 2)
    return (x, y, z)

def numerically_verify_sg_int(samples, thetaVals):
    int_val = 0
    for i in range(0,samples, 1):
        r_val = 0
        for j in range(0, samples, 1):
            r_val += 2.0 * math.pi * math.sin(thetaVals[i]) * sg(math.cos(thetaVals[i]), lamb)
        r_val /= samples
        int_val += math.pi/2.0*r_val
    int_val /= samples
    print("analytical integral: " + str(sg_int_omega(lamb)))
    print("monte carlo integral: " + str(int_val))

def hemisphere_cos_quad_analytical_integral(q_theta, q_phi, q_halfwidth, q_halfheight, q_c):
    q_x, q_y, q_z = quad_edge_points(q_theta, q_phi, q_halfwidth, q_halfheight, q_c)
    t = (q_x[2], q_y[2], q_z[2])
    q_x[2], q_y[2], q_z[2] = q_x[3], q_y[3], q_z[3]
    q_x[3], q_y[3], q_z[3] = t

    norms = np.sqrt(q_x*q_x + q_y*q_y + q_z*q_z)
    q_nx, q_ny, q_nz = (q_x, q_y, q_z)/norms
    sum = 0
    for i in range(0,4,1):
        n_i = (i + 1) % 4
        v0 = [q_nx[i], q_ny[i], q_nz[i]]
        v1 = [q_nx[n_i], q_ny[n_i], q_nz[n_i]]
        ang = np.arccos(np.dot(v0, v1))
        print(ang)
        sum += ang*np.cross(v0, v1)[2]

    return sum  / (2.0 * math.pi)
    
def numerically_verify_quad_cos_int(samples, thetaVals, phiVals, q_theta, q_phi, q_halfwidth, q_halfheight, q_c):
    int_val = 0
    n = create_normal(q_theta, q_phi)
    bi, bj, bn = invent_basis_from_normal(n)
    
    for i in range(0,samples, 1):
        r_val = 0
        for j in range(0, samples, 1):
            dir_x, dir_y, dir_z = create_normal(thetaVals[i], phiVals[j])
            r_val += 2.0 * math.pi * math.sin(thetaVals[i]) * hits_plane(bn, bi, bj, q_c, q_halfwidth, q_halfheight, dir_x, dir_y, dir_z) * dir_z 
        r_val /= samples
        int_val += math.pi/2.0*r_val
    int_val /= samples
    print("monte carlo integral: " + str(int_val/math.pi))
    print("analytical integral: " + str(hemisphere_cos_quad_analytical_integral(q_theta, q_phi, q_halfwidth, q_halfheight, q_c)))

def numerically_verify_quad_sg_int(samples, thetaVals, phiVals, q_theta, q_phi, q_halfwidth, q_halfheight, q_c, lamb):
    int_val = 0
    n = create_normal(q_theta, q_phi)
    bi, bj, bn = invent_basis_from_normal(n)
    
    for i in range(0,samples, 1):
        r_val = 0
        for j in range(0, samples, 1):
            dir_x, dir_y, dir_z = create_normal(thetaVals[i], phiVals[j])
            r_val += 2.0 * math.pi * math.sin(thetaVals[i]) * sg(dir_z, lamb) 
        r_val /= samples
        int_val += math.pi/2.0*r_val
    int_val /= samples
    print("monte carlo integral: " + str(int_val/math.pi))
    print("analytical integral: " + str(hemisphere_cos_quad_analytical_integral(q_theta, q_phi, q_halfwidth, q_halfheight, q_c)))

def hits_plane(plane_n, plane_u, plane_v, plane_c, plane_halfwidth, plane_halfheight, dir_x, dir_y, dir_z):
    d = plane_n[0] * plane_c[0] + plane_n[1] * plane_c[1] + plane_n[2] * plane_c[2] 
    nDotD = plane_n[0]*dir_x + plane_n[1]*dir_y + plane_n[2]*dir_z
    dp = d/nDotD
    p_x = dp*dir_x - plane_c[0]
    p_y = dp*dir_y - plane_c[1]
    p_z = dp*dir_z - plane_c[2]
    ii = p_x*plane_v[0] + p_y*plane_v[1] + p_z*plane_v[2]
    jj = p_x*plane_u[0] + p_y*plane_u[1] + p_z*plane_u[2]
    return (nDotD>=0.000) * (np.abs(jj) <= plane_halfwidth) * (np.abs(ii) <= plane_halfheight)

def hemisphere_plane_points(q_theta, q_phi, q_halfwidth, q_halfheight, q_c, x, y, z):
    q_n = create_normal(q_theta, q_phi)
    q_b = invent_basis_from_normal(q_n)
    it_hits = hits_plane(q_n, q_b[0], q_b[1], q_c, q_halfwidth, q_halfheight, x, y, z)
    return it_hits*(x, y, z)

samples = 500
idxX, idxY = np.indices((samples,samples))
thetaVals, phiVals = np.linspace(0,math.pi/2,samples), np.linspace(0,2.0*math.pi,samples)
thetas, phis = np.meshgrid(thetaVals, phiVals)
x, y, z = np.sin(phis)*np.sin(thetas), np.cos(phis)*np.sin(thetas), np.cos(thetas)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim3d(-5.0, 5.0)
ax.set_ylim3d(-5.0, 5.0)
ax.set_zlim3d(-5.0, 5.0)

lamb = 4
sg_v = sg(z, lamb)# - sg(-1,lamb)
#x = sg_v * x
#y = sg_v * y
#z = sg_v * z

#ax.scatter(x, y, z, marker='o')
#ax.plot_trisurf(np.ravel(x), np.ravel(y), np.ravel(z), cmap='viridis')

q_samples = 9
q_theta = math.pi*0.5*0.1#0.3 * math.pi * 0.5
q_phi = 0.1*2.0 * math.pi
q_c = (4, 7, 4)
q_halfwidth =  10
q_halfheight =  10

quad_edge_points(q_theta, q_phi, q_halfwidth, q_halfheight, q_c)
q_x, q_y, q_z = quad_sample_points(q_theta, q_phi, q_halfwidth, q_halfheight, q_c, q_samples)
q_leninv = (1.0/(np.sqrt(q_x * q_x + q_y*q_y + q_z*q_z)))
q_cx, q_cy, q_cz = ((q_x, q_y, q_z)*q_leninv)
q_px, q_py, q_pz = sg(q_cz, lamb) * (q_cx, q_cy, q_cz)
sp_x, sp_y, sp_z = hemisphere_plane_points(q_theta, q_phi, q_halfwidth, q_halfheight, q_c, x, y, z)
e_x, e_y, e_z = quad_edge_points(q_theta, q_phi, q_halfwidth, q_halfheight, q_c)
e_norms = np.sqrt(e_x*e_x + e_y*e_y + e_z*e_z)
e_nx, e_ny, e_nz = (e_x, e_y, e_z) / e_norms

#ax.scatter(q_x, q_y, q_z, marker='^')
#ax.scatter(e_x, e_y, e_z, marker='o', color='blue')
#ax.scatter(e_nx, e_ny, e_nz, marker='o', color='blue')
ax.scatter(e_nx[0], e_ny[0], e_nz[0], marker='o', color='red')
ax.scatter(e_nx[1], e_ny[1], e_nz[1], marker='o', color='green')
ax.scatter(e_nx[3], e_ny[3], e_nz[3], marker='o', color='blue')
ax.scatter(e_nx[2], e_ny[2], e_nz[2], marker='o', color='orange')
#ax.scatter(q_cx, q_cy, q_cz, marker='^')

"""
ax.plot_trisurf(np.ravel(q_x), np.ravel(q_y), np.ravel(q_z), color='red')
#ax.scatter(sp_x, sp_y, sp_z, marker='^')
ax.plot_trisurf(np.ravel(sp_x), np.ravel(sp_y), np.ravel(sp_z), color='green')
#ax.scatter(q_cx, q_cy, q_cz, marker='^', color='blue')
#ax.scatter(q_px, q_py, q_pz, marker='^')
ax.plot_trisurf(np.ravel(q_px), np.ravel(q_py), np.ravel(q_pz), color='blue')
#ax.plot_trisurf(np.ravel(q_cx), np.ravel(q_cy), np.ravel(q_cz))
numerically_verify_sg_int(samples, thetaVals)
"""
#plt.show()


numerically_verify_quad_cos_int(samples, thetaVals, phiVals, q_theta, q_phi, q_halfwidth, q_halfheight, q_c)
#numerically_verify_quad_sg_int(samples, thetaVals, phiVals, q_theta, q_phi, q_halfwidth, q_halfheight, q_c, lamb)


