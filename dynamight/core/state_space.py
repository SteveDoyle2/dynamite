import numpy as np

"""
x(t) - state vector
y(t) - output
u(t) - input/control
A    - (n x n) state/system matrix
B    - (n x p) input matrix
C    - (q x n) output matrix
D    - (q x p) feedthrough/feedforward matrix

Continuous time-invariant
  xd(t)  = A*x(t) + B*u(t)
  y(t)   = C*x(t) + D*u(t)
Continuous time-variant
  xd(t)  = A(t)*x(t) + B(t)*u(t)
  y(t)   = C(t)*x(t) + D(t)*u(t)
Explicit discrete time-invariant
  x(k+1) = A*x(k) + B*u(k)
  y(k)   = C*x(k) + D*u(k)
Explicit discrete time-variant
  x(k+1) = A(k)*x(k) + B(k)*u(k)
  y(k)   = C(k)*x(k) + D(k)*u(k)

Laplace domain of continuous time-invariant
  s*X(s) - x0 = A*X(s) + B*U(s)
  Y(s)        = C*X(s) + D*U(s)

TF:
  s*X(s) - x0 = A*X(s) + B*U(s)
  s*X(s) - A*X(s)  = x0 + B*U(s)
  (s8I - A) * X(s) = x0 + B*U(s)
  X(s) = (s*I - A)^-1 * (x0 + B*U(s))

  Y(s) = C*X(s) + D*U(s)
  Y(s) = C*(s*I - A)^-1 * (x0 + B*U(s))  + D*U(s)

  let x0=0:
  Y(s) = C*(s*I - A)^-1 * B*U(s)  + D*U(s)
  Y(s) = (C*(s*I - A)^-1 * B  + D)*U(s)
  Y(s) = G(s) * U(s)
  G(s) = C*(s*I - A)^-1 * B  + D

let G(s) =    n1*s^3 + n2*s^2 + n3*s + n4
           ----------------------------------
           s^4 + d1*s^3 + d2*s^2 + d3*s + d4

let:
  m*ydd = u(t) - b*yd(t) - k*y(t)
  ydd = u(t)/m - b/m*yd(t) - k/m*y(t)
  x1 - position
  x2 - velocity
  y  - new position
  :: x2 = x1d

  ydd = u(t)/m -  b/m*yd(t) - k/m*y(t)
  ydd = u(t)/m - (b/m*yd(t) + k/m*y(t))

  comparing:
    ydd = u(t)/m - (b/m*yd(t) + k/m*y(t))
    x(t) = A*x(t) + B*u(t)

  x is the state vector:
     x = [x1(t)]
         [x2(t)]

  y(t) is the output, so:
    y(t) = x1(t)
    y = [1, 0] * x(t)

  comparing
    ydd = -(b/m*yd(t) + k/m*y(t)) + 1/m*u(t)
    x(t) = A*x(t)                 + B*u(t)
  we get:
     B = [0, 1/m].T

  ydd = -(b/m*yd(t) + k/m*y(t)) + B*u(t)

  x1d = x2
  x2d = -(b/m*x2 + k/m*x1)      + B*u(t)

  in matrix form:
    [x1d] = [0,       1] [x1(t)] + [ 0 ] u(t)
    [x2d]   [-k/m, -b/m] [x2(t)]   [1/m]
    [y] = [1, 0] [x1(t)]
                 [x2(t)]

  xd(t) = [0,       1] x(t) + [ 0 ] u(t)
          [-k/m, -b/m]        [1/m]


   A = [0,       1]
       [-k/m, -b/m]

   B = [ 0 ]
       [1/m]

   C = [1, 0]
   D = []

pendulum:
 m*l*thetadd(t) = -m*l*g*sin(theta(t)) - k*l*thetad(t)

 shortening:
   ml^2*thetadd = -mlg*sin(theta) - kl*thetad
   thetadd = -g/l*sin(theta) - k/(ml)*thetad

 let:
   x1 = theta
   x2 = x1d
   x2d = thetadd
   x = [x1]
       [x2]
   y = [x2d] = [0, 1] [x]

 so:
   x2d = -g*sin(x1) - k/m*x2
   xd(t)  = A*x(t) + B*u(t)

   x = [x1d(t)] = [x2(t)                  ]
       [x2d(t)]   [-g*sin(x1) - k/(ml)*x2 ]

   [y] = [1, 0] [x1(t)]
                [x2(t)]

"""
def controllability(A: np.ndarray,
                    B: np.ndarray):
    controllability = np.hstack([B, A @ B])
    return controllability

def observability(A: np.ndarray,
                  C: np.ndarray):
    observability = np.vstack([C, C @ A])
    return observability

def specs(A: np.ndarray,
          B: np.ndarray,
          C: np.ndarray,
          D: np.ndarray, ):
    controllability(A, B)
    observability(A, C)

