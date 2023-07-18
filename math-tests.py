import numpy as np
import matplotlib.pyplot as plt
import math


def sg(z,lamb):
    return np.exp(lamb*z - lamb)

def sg_int(lamb):
    return 2.0 * math.pi * (1.0/lamb) * (1.0 - np.exp(-2.0*lamb))

def sg_int_omega(lamb):
    return 2.0 * math.pi * (1.0/lamb) * (1.0 - np.exp(-lamb))

def sg_hem(lamb, costheta0, costheta1):
    return 2.0 * math.pi * (1.0/lamb) * (np.exp(lamb*(costheta0 - 1)) - np.exp(lamb*(costheta1 - 1)))

def sg_hem_range(lamb, costheta0, costheta1):
    return (1.0/lamb) * (np.exp(lamb*(costheta0 - 1)) - np.exp(lamb*(costheta1 - 1)))

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

def transform_sg_to_cos(dir_x, dir_y, dir_z, lamb):
    norms = np.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
    q_nx, q_ny, q_nz = (dir_x, dir_y, dir_z)/norms

    norm_base = np.sqrt(dir_x*dir_x + dir_y*dir_y)
    b_x, b_y = (dir_x, dir_y)/norm_base

    sg_val = sg(q_nz, lamb)
    new_z = sg_val
    new_z_inv = np.sqrt(1.0 - new_z*new_z)
    b_x, b_y = new_z_inv * (b_x, b_y)
    return (b_x, b_y, new_z)
    

def hemisphere_sg_quad_analytical_integral(q_theta, q_phi, q_halfwidth, q_halfheight, q_c, lamb):
    q_x, q_y, q_z = quad_edge_points(q_theta, q_phi, q_halfwidth, q_halfheight, q_c)
    t = (q_x[2], q_y[2], q_z[2])
    q_x[2], q_y[2], q_z[2] = q_x[3], q_y[3], q_z[3]
    q_x[3], q_y[3], q_z[3] = t

    b_x, b_y, b_z = transform_sg_to_cos(q_x, q_y, q_z, lamb)

    sum = 0
    for i in range(0,4,1):
        n_i = (i + 1) % 4
        v0 = [b_x[i], b_y[i], b_z[i]]
        v1 = [b_x[n_i], b_y[n_i], b_z[n_i]]
        ang = np.arccos(np.dot(v0, v1))
        sum += ang*np.cross(v0, v1)[2]

    return sum  / (2.0 * math.pi)

def hemisphere_sg_quad_analytical_integral2(q_theta, q_phi, q_halfwidth, q_halfheight, q_c, lamb):
    q_x, q_y, q_z = quad_edge_points(q_theta, q_phi, q_halfwidth, q_halfheight, q_c)
    t = (q_x[2], q_y[2], q_z[2])
    q_x[2], q_y[2], q_z[2] = q_x[3], q_y[3], q_z[3]
    q_x[3], q_y[3], q_z[3] = t

    norms = np.sqrt(q_x*q_x + q_y*q_y + q_z*q_z)
    q_nx, q_ny, q_nz = (q_x, q_y, q_z)/norms

    max_z = max(q_nz[0], max(q_nz[1], max(q_nz[2], q_nz[3])))
    min_z = min(q_nz[0], min(q_nz[1], min(q_nz[2], q_nz[3])))

    sid_norm = np.sqrt(q_nx*q_nx + q_ny*q_ny)
    b_x = q_nx/sid_norm
    phis = np.arccos(b_x)

    max_phi = max(phis[0],max(phis[1],max(phis[2], phis[3])))
    min_phi = min(phis[0],min(phis[1],min(phis[2], phis[3])))
    del_phi = max_phi - min_phi

    return sg_hem(lamb, max_z, min_z) * (del_phi/(2.0*math.pi))/math.pi

def hemisphere_sg_quad_analytical_integral3(q_theta, q_phi, q_halfwidth, q_halfheight, q_c, lamb):

    q_x, q_y, q_z = quad_edge_points(q_theta, q_phi, q_halfwidth, q_halfheight, q_c)
    t = (q_x[2], q_y[2], q_z[2])
    q_x[2], q_y[2], q_z[2] = q_x[3], q_y[3], q_z[3]
    q_x[3], q_y[3], q_z[3] = t
    q_norms = np.sqrt(q_x*q_x + q_y*q_y + q_z*q_z)
    b_x, b_y, b_z = (q_x, q_y, q_z)/q_norms

    sum = 0
    for i in range(0,4,1):
        n_i = (i + 1) % 4
        v0 = [b_x[i], b_y[i], b_z[i]]
        v1 = [b_x[n_i], b_y[n_i], b_z[n_i]]
        angs0 = (v0[2], np.arccos(v0[0]) if v0[1] > 0 else (math.pi * 2.0) - np.arccos(v0[0]))
        angs1 = (v1[2], np.arccos(v1[0]) if v1[1] > 0 else (math.pi * 2.0) - np.arccos(v1[0]))
        cos_theta_top = max(angs0[0], angs1[0])
        cos_theta_bottom = min(angs0[0], angs1[0])
        ar = (v1[1] - v0[1]) * sg_hem(lamb, 1.0, cos_theta_top) - 0.6*(v1[1] - v0[1])*sg_hem(lamb, cos_theta_bottom, cos_theta_top)
        print(ar)
        sum += ar
    return sum / math.pi 

def numerically_verify_quad_sg_int(samples, thetaVals, phiVals, q_theta, q_phi, q_halfwidth, q_halfheight, q_c, lamb):
    int_val = 0
    n = create_normal(q_theta, q_phi)
    bi, bj, bn = invent_basis_from_normal(n)
    
    for i in range(0,samples, 1):
        r_val = 0
        for j in range(0, samples, 1):
            dir_x, dir_y, dir_z = create_normal(thetaVals[i], phiVals[j])
            r_val += 2.0 * math.pi * math.sin(thetaVals[i]) * sg(dir_z, lamb) * hits_plane(n, bi, bj, q_c,q_halfwidth, q_halfheight,dir_x, dir_y, dir_z)
        r_val /= samples
        int_val += math.pi/2.0*r_val
    int_val /= samples
    print("monte carlo integral: " + str(int_val/math.pi))
    #print("analytical integral: " + str(hemisphere_sg_quad_analytical_integral(q_theta, q_phi, q_halfwidth, q_halfheight, q_c, lamb)))
    #print("analytical integral2: " + str(hemisphere_sg_quad_analytical_integral2(q_theta, q_phi, q_halfwidth, q_halfheight, q_c, lamb)))
    print("analytical integral3: " + str(hemisphere_sg_quad_analytical_integral3(q_theta, q_phi, q_halfwidth, q_halfheight, q_c, lamb)))

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

"""
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

lamb = 2.0
sg_v = sg(z, lamb)# - sg(-1,lamb)
#x = sg_v * x
#y = sg_v * y
#z = sg_v * z

#ax.scatter(x, y, z, marker='o')
#ax.plot_trisurf(np.ravel(x), np.ravel(y), np.ravel(z), cmap='viridis')

q_samples = 9
q_theta = math.pi*0.5*0.3#0.3 * math.pi * 0.5
q_phi = 0# * 2.0 * math.pi
q_c = (8, 8, 4)
q_halfwidth = 10
q_halfheight = 10
"""

"""
quad_edge_points(q_theta, q_phi, q_halfwidth, q_halfheight, q_c)
q_x, q_y, q_z = quad_sample_points(q_theta, q_phi, q_halfwidth, q_halfheight, q_c, q_samples)
q_leninv = (1.0/(np.sqrt(q_x * q_x + q_y*q_y + q_z*q_z)))
q_cx, q_cy, q_cz = ((q_x, q_y, q_z)*q_leninv)
q_px, q_py, q_pz = sg(q_cz, lamb) * (q_cx, q_cy, q_cz)
sp_x, sp_y, sp_z = hemisphere_plane_points(q_theta, q_phi, q_halfwidth, q_halfheight, q_c, x, y, z)
e_x, e_y, e_z = quad_edge_points(q_theta, q_phi, q_halfwidth, q_halfheight, q_c)
e_norms = np.sqrt(e_x*e_x + e_y*e_y + e_z*e_z)
e_nx, e_ny, e_nz = (e_x, e_y, e_z) / e_norms
"""

#ax.scatter(q_x, q_y, q_z, marker='^')
#t_x, t_y, t_z = transform_sg_to_cos(q_x, q_y, q_z, lamb)
#ax.scatter(t_x, t_y, t_z, marker='o')

#ax.scatter(e_x, e_y, e_z, marker='o', color='blue')
#ax.scatter(e_nx, e_ny, e_nz, marker='o', color='blue')
#ax.scatter(e_nx[0], e_ny[0], e_nz[0], marker='o', color='red')
#ax.scatter(e_nx[1], e_ny[1], e_nz[1], marker='o', color='green')
#ax.scatter(e_nx[3], e_ny[3], e_nz[3], marker='o', color='blue')
#ax.scatter(e_nx[2], e_ny[2], e_nz[2], marker='o', color='orange')
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
#numerically_verify_quad_cos_int(samples, thetaVals, phiVals, q_theta, q_phi, q_halfwidth, q_halfheight, q_c)
#plt.show()


def vec_len(v):
    x, y, z = v
    return np.sqrt(x*x + y*y + z*z)

def vec_len2(v):
    x, y = v
    return np.sqrt(x*x + y*y)

def jacobian_slice_theta(v):
    print(v[2])
    v = v/vec_len(v)
    zb = v
    yb = np.cross(v, [-1,0,0])
    yb = yb/vec_len(yb)
    xb = np.cross(yb, zb)
    steps = 256
    a = np.empty(steps, dtype=float)
    b = np.empty(steps, dtype=float)
    dt = np.empty(steps, dtype=float)
    dt2 = np.empty(steps, dtype=float)
    f_x = np.empty(steps, dtype=float)
    e = np.empty(steps, dtype=float)
    e2 = np.empty(steps, dtype=float)
    tang = np.empty(steps, dtype=float)
    rad = np.empty(steps, dtype=float)
    qq,pp = np.empty(steps, dtype=float), np.empty(steps, dtype=float)
    qq2,pp2 = np.empty(steps, dtype=float), np.empty(steps, dtype=float)
    ssx,ssy = np.empty(steps, dtype=float), np.empty(steps, dtype=float)
    tangsX, tangsY = np.empty(steps, dtype=float), np.empty(steps, dtype=float)
    s = 0
    s2 = 0
    cosTheta, sinTheta = np.cos(2.0*np.pi/steps), np.sin(2.0*np.pi/steps)
    for i in range(0,steps,1):
        tp = (i + 0.5 - 1.0)/steps
        t = (i + 0.5)/steps
        ang =   t * np.pi + 0.5 * np.pi
        angp =  tp * np.pi + 0.5 * np.pi
        tvp = np.cos(angp) * xb + np.sin(angp) * yb
        tv = np.cos(ang) * xb + np.sin(ang) * yb

        p_tvp = (tvp[0], tvp[1])
        p_tv  = (tv[0], tv[1])

        qq[i] = p_tvp[0]
        pp[i] = p_tvp[1]
        qq2[i] = np.cos(ang)*v[2]
        pp2[i] = np.sin(ang)

        p_tvp = p_tvp/vec_len2(p_tvp)
        p_tv = p_tv/vec_len2(p_tv)

        a[i] = ang/np.pi
        b[i] = ang
        dt[i] = np.arccos(np.dot(p_tvp, p_tv))
        f_x[i] = sg_hem_range(13.0, 1.0, tv[2])

        rad[i] = np.sqrt(np.sin(ang)**2 + (v[2]*np.cos(ang))**2)
        tang[i] = np.sqrt(np.cos(ang)**2 + (v[2]*np.sin(ang))**2)* np.pi / steps
        tangsX[i] = np.sin(ang)*v[2] * np.pi / steps
        tangsY[i] = -np.cos(ang)# * np.pi * 2.0 / steps
        #dt2[i] =  np.arctan(tang[i]/rad[i])
        #dt2[i] = tang[i]/np.sqrt(tang[i]*tang[i] + rad[i]*rad[i])
        #dt2[i] = tang[i]#/rad[i]
        err = ((1.0 - v[2])*np.cos(ang)*np.cos(ang) + v[2])**0.25
        dt2[i] = tang[i]/rad[i]

        ssx[i] = p_tv[0]
        ssy[i] = p_tv[1]

        e[i] = s
        e2[i] = s2 
        s += f_x[i]*dt[i]
        s2 += f_x[i]*dt2[i]

    fig, ax = plt.subplots()
    #ax.set_aspect('equal')
    #for i in range(0, steps, 1):
    #    ax.plot([qq[i], qq[i]+tangsX[i]], [pp[i], pp[i] + tangsY[i]])
    #ax.plot(a,f_x, color='red')
    #ax.plot(a,tang,color='blue')
    ax.plot(a,e,color='red')
    ax.plot(a,e2,color='green')
    #ax.plot(a,e,color='blue')
    #ax.plot(a,e2,color='green')
    #ax.scatter(qq, pp)
    #ax.scatter(ssx, ssy)
    #ax.scatter(qq2, pp2)
    ax.grid()
    plt.show()

jacobian_slice_theta(np.array([-0.5, 0.0, 0.5])) 
