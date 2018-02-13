"""Butcher tableaux for some popular Runge Kutta methods.

See https://en.wikipedia.org/wiki/List_of_Runge–Kutta_methods for more details.
"""


from collections import namedtuple
import numpy as np


butcher = namedtuple('Butcher', 'name A b c')

# partitioned half-explicit methods
butcher_phem = namedtuple('Butcher', 'name A b c A_hat')

r3 = 3**0.5
r6 = 6**0.5
r15 = 15**0.5




####################
# EXPLICIT METHODS #
####################

euler_a = np.array([[0]])
euler_b = np.array([1])
euler_c = np.array([0])

Euler = butcher('Forward Euler', euler_a, euler_b, euler_c)


em_a = np.zeros((2, 2))
em_a[1, 0] = 1/2
em_b = np.array([0, 1])
em_c = np.array([0, 1/2])

ExplicitMidpoint = butcher('Explicit midpoint', em_a, em_b, em_c)


ralston_a = np.zeros((2, 2))
ralston_a[1, 0] = 2/3
ralston_b = np.array([1/4, 3/4])
ralston_c = np.array([0, 2/3])

Ralston = butcher('Ralston', ralston_a, ralston_b, ralston_c)


kutta_a = np.zeros((3, 3))
kutta_a[1, 0] = 1/2
kutta_a[2, :2] = [-1, 2]
kutta_b = np.array([1/6, 2/3, 1/6])
kutta_c = np.array([0, 1/2, 1])

Kutta3 = butcher('Kutta3', kutta_a, kutta_b, kutta_c)


rk4_a = np.zeros((4, 4))
rk4_a[1, 0] = rk4_a[2, 1] = 1/2
rk4_a[3, 2] = 1
rk4_b = np.array([1/6, 1/3, 1/3, 1/6])
rk4_c = np.array([0, 1/2, 1/2, 1])

RK4 = butcher('RK4', rk4_a, rk4_b, rk4_c)


pho4_a = np.zeros((4, 4))
pho4_a[1, 0] = 1/3
pho4_a[2, :2] = [-1/3, 1]
pho4_a[3, :3] = [1, -1, 1]
pho4_b = np.array([1/8, 3/8, 3/8, 1/8])
pho4_c = np.array([0, 1/3, 2/3, 1])
pho4_a_hat = np.array([[0, 0, 0, 0, 0],
                       [1/8, 3/8, 0, 0, 0],
                       [161/1024, 147/512, 441/1024, 0, 0],
                       [1/8, 3/8, 3/8, 1/8, 0],
                       [693/5000, 1701/5000, 243/625, 81/1250, -81/2500]
                      ])

PHO4 = butcher_phem('3/8', pho4_a, pho4_b, pho4_c, pho4_a_hat)



rkf_a = np.zeros((6, 6))
rkf_a[1, 0] = 1/4
rkf_a[2, :2] = [3/32, 9/32]
rkf_a[3, :3] = [1932/2197, -7200/2197, 7296/2197]
rkf_a[4, :4] = [439/216, -8, 3680/513, -845/4104]
rkf_a[5, :5] = [-8/27, 2, -3544/2565, 1859/4104, -11/40]
rkf_b = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
rkf_c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])

RKF = butcher('RKF', rkf_a, rkf_b, rkf_c)


ck_a = np.zeros((6, 6))
ck_a[1, 0] = 1/5
ck_a[2, :2] = [3/40, 9/40]
ck_a[3, :3] = [3/10, -9/10, 6/5]
ck_a[4, :4] = [-11/54, 5/2, -70/27, 35/27]
ck_a[5, :5] = [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]
ck_b = np.array([2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4])
ck_c = np.array([0, 1/5, 3/10, 3/5, 1, 7/8])

CK = butcher('Cash-Karp', ck_a, ck_b, ck_c)


dopri_a = np.zeros((7, 7))
dopri_a[1, 0] = 1/5
dopri_a[2, :2] = [3/40, 9/40]
dopri_a[3, :3] = [44/45, -56/15, 32/9]
dopri_a[4, :4] = [19372/6561, -25360/2187, 64448/6561, -212/729]
dopri_a[5, :5] = [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
dopri_a[6, :6] = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
dopri_b = dopri_a[-1]
dopri_c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])

dopri_a_hat = np.zeros((7, 7))
dopri_a_hat[:6] = dopri_a[1:]
dopri_a_hat[6] = [-18611506045861/19738176307200,
                  59332529/14479296,
                  -2509441598627/893904224850,
                  2763523204159/3289696051200,
                  -41262869588913/116235927142400,
                  46310205821/287848404480,
                  -3280/75413
                 ]

DOPRI5 = butcher_phem('DOPRI5', dopri_a, dopri_b, dopri_c, dopri_a_hat)









####################
# IMPLICIT METHODS #
####################



be_a = np.array([[1]])
be_b = np.array([1])
be_c = np.array([1])

BackwardEuler = butcher('Backward Euler', be_a, be_b, be_c)



#########################
# Gauss-Legendre family #
#########################

im_a = np.array([[1/2]])
im_b = np.array([1])
im_c = np.array([1/2])

ImplicitMidpoint = butcher('Implicit midpoint', im_a, im_b, im_c)


gl4_a = np.array([[1/4, 1/4 - r3/6],
                  [1/4 + r3/6, 1/4]])
gl4_b = np.array([0.5, 0.5])
gl4_c = np.array([1/2 - r3/6, 1/2 + r3/6])

Gauss4 = butcher('Gauss-Legendre (4)', gl4_a, gl4_b, gl4_c)


gl6_a = np.array([[5/36, 2/9 - r15/15, 5/36 - r15/30],
                  [5/36 + r15/24, 2/9, 5/36 - r15/24],
                  [5/36 + r15/30, 2/9 + r15/15, 5/36]])
gl6_b = np.array([5/18, 4/9, 5/18])
gl6_c = np.array([1/2 - r15/10, 1/2, 1/2 + r15/10])

Gauss6 = butcher('Gauss-Legendre (6)', gl6_a, gl6_b, gl6_c)



################
# Radau family #
################

rad3IA_a = np.array([[1/4, -1/4],
                     [1/4, 5/12]])
rad3IA_b = np.array([1/4, 3/4])
rad3IA_c = np.array([0, 2/3])

RadauIA_3 = butcher('Radau IA (3)', rad3IA_a, rad3IA_b, rad3IA_c)


rad5IA_a = np.array([[1/9, (-1-r6)/18, (-1+r6)/18],
                     [1/9, 11/45 + 7*r6/360, 11/45 - 43*r6/360],
                     [1/9, 11/45 + 43*r6/360, 11/45 - 7*r6/360]])
rad5IA_b = np.array([1/9, 4/9 + r6/36, 4/9 - r6/36])
rad5IA_c = np.array([0, 3/5 - r6/10, 3/5 + r6/10])

RadauIA_5 = butcher('Radau IA (5)', rad5IA_a, rad5IA_b, rad5IA_c)


rad3IIA_a = np.array([[5/12, -1/12],
                      [3/4, 1/4]])
rad3IIA_b = rad3IIA_a[-1]
rad3IIA_c = np.array([1/3, 1])

RadauIIA_3 = butcher('Radau IIA (3)', rad3IIA_a, rad3IIA_b, rad3IIA_c)


rad5IIA_a = np.array([[11/45 - 7*r6/360, 37/225 - 169*r6/1800, -2/225 + r6/75],
                      [37/225 + 169*r6/1800, 11/45 + 7*r6/360, -2/225 - r6/75],
                      [4/9 - r6/36, 4/9 + r6/36, 1/9]])
rad5IIA_b = rad5IIA_a[-1]
rad5IIA_c = np.array([2/5 - r6/10, 2/5 + r6/10, 1])

RadauIIA_5 = butcher('Radau IIA (5)', rad5IIA_a, rad5IIA_b, rad5IIA_c)



##################
# Lobatto family #
##################

lob2IIIA_a = np.array([[0, 0],
                       [1/2, 1/2]])
lob2IIIA_b = lob2IIIA_a[-1]
lob2IIIA_c = np.array([0, 1])

LobattoIIIA_2 = butcher('Lobatto IIIA (2)', lob2IIIA_a, lob2IIIA_b, lob2IIIA_c)


lob4IIIA_a = np.array([[0, 0, 0],
                       [5/24, 1/3, -1/24],
                       [1/6, 2/3, 1/6]])
lob4IIIA_b = lob4IIIA_a[-1]
lob4IIIA_c = np.array([0, 1/2, 1])

LobattoIIIA_4 = butcher('Lobatto IIIA (4)', lob4IIIA_a, lob4IIIA_b, lob4IIIA_c)


lob2IIIC_a = np.array([[1/2, -1/2],
                       [1/2, 1/2]])
lob2IIIC_b = lob2IIIC_a[-1]
lob2IIIC_c = np.array([0, 1])

LobattoIIIC_2 = butcher('Lobatto IIIC (2)', lob2IIIC_a, lob2IIIC_b, lob2IIIC_c)


lob4IIIC_a = np.array([[1/6, -1/3, 1/6],
                       [1/6, 5/12, -1/12],
                       [1/6, 2/3, 1/6]])
lob4IIIC_b = lob4IIIC_a[-1]
lob4IIIC_c = np.array([1, 1/2, 1])

LobattoIIIC_4 = butcher('Lobatto IIIC (4)', lob4IIIC_a, lob4IIIC_b, lob4IIIC_c)
