from collections import namedtuple
import numpy as np


np.seterr(over='raise')

g = 9.81
Pendulum = namedtuple('Pendulum', 'm x y u v')


class DoublePendulum:
    """Double pendulum consisting of two massless bodies and a point masses on
    their ends.

    Attributes:
        M     (np.array, (4, 4)): mass matrix
        M_inv (np.array, (4, 4)): inverse of mass matrix
        f     (np.array, (4, 1)): vector of gravitational forces
        y_0   (np.array, (8,)):   initial positions and velocities
        z_0   (np.array, (2, 1)): initial Lagrange multipliers
    """

    def __init__(self, body_1, body_2):
        """Initializes double pendulum instance.

        Args:
            body_1, body_2 (namedtuple):
                m: point mass on the end of the body
                x: x-coordinate of the point mass
                y: y-coordinate of the point mass
                u: velocity in x-direction of the point mass
                v: velocity in y-direction of the point mass
        """
        self._check_consistency(body_1, body_2)
        self.M = np.diag([body_1.m, body_2.m, body_1.m, body_2.m])
        self.M_inv = np.linalg.inv(self.M)
        self.f = -g * np.array([0, 0, body_1.m, body_2.m]).reshape((4, 1))
        self.y_0 = np.array([body_1.x, body_2.x, body_1.y, body_2.y,
                             body_1.u, body_2.u, body_1.v, body_2.v])
        self.z_0 = self.get_z(self.y_0)
        self._b1 = body_1
        self._b2 = body_2

    def get_z(self, y):
        """Calculate Lagrange multipliers for `y`.
        See AP98, p. 248.

        Args:
            y (np.array, (8,)): <x1, x2, y1, y2, u1, u2, v1, v2>

        Returns:
            np.array, (2,)
        """
        G = self.get_G(y)
        g_uu = self.get_g_uu(y)
        G_Mi = G.dot(self.M_inv)

        A = G_Mi.dot(G.T)
        b = G_Mi.dot(self.f) + g_uu
        return np.linalg.solve(A, b).reshape(2,)

    def get_dy(self, y):
        """Calculate `y' = dy/dt` by solving the system
                [[M  G.T]    [[du]    [[ f  ]
                 [G   0 ]] @  [z ]] =  [g_uu]]
        See HW96, p. 465 for more details.

        Args:
            y (np.array, (8,)): <x1, x2, y1, y2, u1, u2, v1, v2>

        Returns:
            np.array, (8,): <dx1, dx2, dy1, dy2, du1, du2, dv1, dv2>
        """
        G = self.get_G(y)
        g_uu = self.get_g_uu(y)

        A = np.zeros((6, 6))
        b = np.zeros((6, 1))

        A[:4, :4] = self.M
        A[:4, -2:] = G.T
        A[-2:, :4] = G

        b[:4] = self.f
        b[-2:] = -1 * g_uu

        x = np.linalg.solve(A, b)
        return np.concatenate((y[-4:], x[:4, 0]))

    @staticmethod
    def get_G(y):
        """Calculate matrix `G = grad(g)`, where `g` are algebraic constraints.

        Args:
            y (np.array, (8,)): <x1, x2, y1, y2, u1, u2, v1, v2>

        Returns:
            np.array, (2, 4)
        """
        x1, x2, y1, y2 = y[:4]
        return np.array([[x1, 0, y1, 0],
                         [-(x2-x1), (x2-x1), -(y2-y1), (y2-y1)]])

    @staticmethod
    def get_g_uu(y):
        """Calculate vector `g_uu = G' @ u`, needed to solve double pendulum
        as index 1 DAE system. (See HW96, p. 465)

        Args:
            y (np.array, (8,)): <x1, x2, y1, y2, u1, u2, v1, v2>

        Returns:
            np.array, (2, 1)

        """
        u1, u2, v1, v2 = y[-4:]
        return np.array([[u1**2 + v1**2],
                         [(u2 - u1)**2 + (v2 - v1)**2]])

    @staticmethod
    def _check_consistency(body_1, body_2):
        """Checks if the initial values satisfy an index 2 constraint."""

        if body_1.x * body_1.u + body_1.y * body_1.v != 0:
            raise InconsistentInitialValues("First pendulum has inconsistent "
                "initial values. Try setting velocities to zero.")

        if ((body_2.x - body_1.x) * (body_2.u - body_1.u) +
            (body_2.y - body_1.y) * (body_2.v - body_1.v)) != 0:
            raise InconsistentInitialValues("Second pendulum has inconsistent "
                "initial values. Try setting velocities to zero.")


class InconsistentInitialValues(Exception):
    pass


def __run_basic_example():
    p1 = Pendulum(m=5, x=1.5, y=-2, u=10, v=0)
    p2 = Pendulum(m=15, x=5.5, y=-5, u=0, v=20)
    dp = DoublePendulum(p1, p2)
    help(dp)
    print(dp.M)
    print(dp.M_inv)
    print(dp.f)
    print(dp.y_0)
    print(dp.z_0)
    print(dp.get_dy(dp.y_0))


if __name__ == '__main__':
    __run_basic_example()
