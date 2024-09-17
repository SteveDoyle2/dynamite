"""
# https://www.sciencedirect.com/topics/engineering/sideslip-angle
beta = arctan(Vt / ut)
"""

import numpy as np

def traj_6dof(Vt, beta, alpha, phi, theta, psi, p, q, r, pN, pE, h):
    tan(alpha) = w / u
    sin(beta) = v / Vt
    q_psf = 0.5 * rho * Vt ** 2
    qS = q_psf * S_ft2
    qSc = qS * c
    qSb = qS * b

    CL = CLa * alpha + CLb * beta
    CD = CDo + k*CL**2
    Cy = 0.
    Cx = CD*cos(alpha) + CL*sin(alpha)
    Cy = 0.
    Cz = CL*sin(alpha) + CD*cos(alpha)

    Gamma = Ix*Iz - Ixz**2
    c1 = ((Iy - Iz)*Iz - Ixz**2) / Gamma
    c2 = (Ix - Iy + Iz) * Ixz / Gamma
    c3 = Iz / Gamma
    c4 = Ixz / Gamma
    c5 = (Iz - Ix) / Iy
    c6 = Ixz / Iy
    c7 = 1 / (Iy * Gamma)
    c8 = (Jx*(Ix - Iy) + Ixz**2) / Gamma
    c9 = Ix / Gamma

    Fx = qS * Cx
    Fy = qS * Cy
    Fz = qS * Cz

    L = qSb * Cll
    M = qSc * Clm
    N = qSb * Cln

    # force equations
    u_dot =  r*v - q*w - g0*sin(theta) + Fx/mass
    v_dot = -r*u + p*w + g0*sin(phi)*cos(theta) + Fy/mass
    w_dot =  q*u - p*v + g0*cos(phi)*cos(theta) + Fz/mass

    # kinematic equations
    phi_dot   = p+tan(theta)*(q*sin(phi) + r*cos(phi))
    theta_dot = q*cos(phi) - r*sin(phi)
    psi_dot   = (q*sin(phi) + r*cos(phi)) / cos(theta)

    # moment equations
    p_dot = (c1*r + c2*p) * q + c3*L + c4*N
    q_dot = c5*p*r - c6*(p**2 - r**2) + c7*M
    r_dot = (c8*p - c2*r)*q + c4*L + c9*N

    # navigation equations
    pN_dot = (
        U*cos(theta)*cos(phi) +
        V*(-cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(phi)) +
        W*(sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(phi))
    )
    pE_dot = (
        U*cos(theta)*sin(phi) +
        V*(cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi)) +
        W*(-sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(phi))
    )
    h_dot = (
        U*sin(theta) - V*sin(phi)*cos(theta) - W*cos(phi)*cos(theta)
    )

    Vt_dot = (u*u_dot + v*v_dot + w*w_dot) / Vt
    beta_dot = (v_dot*Vt - v*Vt_dot) / Vt**2*cos(beta)
    alpha_dot = (u*w_dot - w*u_dot) / (u**2 + w**2)
    return Vt_dot, beta_dot, alpha_dot, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot, pN_dot, pE_dot, h_dot

def traj_3dof(Vt, beta, alpha, phi, theta, psi, p, q, r, pN, pE, h):
    q_psf = 0.5 * rho * Vt ** 2
    qS = q_psf * S_ft2
    qSc = qS * c

    Gamma = Ix*Iz - Ixz**2
    c7 = 1 / (Iy * Gamma)

    Fx = qS * Cx
    Fz = qS * Cz

    M = qSc * Clm

    # force equations
    u_dot = -q*w - g0*sin(theta) + Fx/mass
    w_dot =  q*u + g0*cos(theta) + Fz/mass

    # kinematic equations
    theta_dot = q

    # moment equations
    p_dot = 0.
    q_dot = c7*M
    r_dot = 0.

    # navigation equations
    pN_dot = U*cos(theta) + W*sin(theta)
    pE_dot = U*cos(theta) + V
    h_dot = U*sin(theta) - W*cos(theta)

    Vt_dot = (u*u_dot+ w*w_dot) / Vt
    beta_dot = (v_dot*Vt - v*Vt_dot) / Vt**2*cos(beta)
    alpha_dot = (u*w_dot - w*u_dot) / (u**2 + w**2)
    return Vt_dot, beta_dot, alpha_dot, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot, pN_dot, pE_dot, h_dot

def main():
    p = 1.
    q = 1.
    r = 1.
    alpha = radians(10.)
    beta = 0.
    theta = np.radians(10)
    phi = np.radians(0.)
    psi = np.radians(0.)
    Vt = 500.
    traj_3dof(Vt, alpha, theta, q, pN, pE, h)
    traj_6dof(Vt, beta, alpha, phi, theta, psi, p, q, r, pN, pE, h)

if __name__ == '__main__':
    S_ft2 = 1.
    c = 1.
    b = 1.

    Ixz = 0
    Ix = 0.2
    Iy = 1.
    Iz = Iy

    CLa = 0.8 * 2*np.pi
    CLb = -0.3 * 2*np.pi
    CDo = 0.001
    AR = 10.
    e = 0.8
    k = 1 / (np.pi * AR * e)
    main()
