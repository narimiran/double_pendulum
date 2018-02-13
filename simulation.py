import random

from pendulum import Pendulum, DoublePendulum
from RK_DAE_solver import ERK_DAE1
from methods import Euler, ExplicitMidpoint, RK4, DOPRI5


def random_initial_values(p_mag, m_mag=5):
    """Returns tuple of random initial values of mass `m`, and positional
    coordinates `x` and `y`.

    Prefers initial positions above the fixed end.
    Function `round` is used to get reproducible simulation results.

    Args:
        p_mag (float): magnitude of the maximum allowed position coordinate
        m_mag (float): magnitude of the highest allowed mass

    Returns:
        tuple (m, x, y)
    """
    m = round(random.random() * m_mag + 1, 2)
    x = round(random.random() * p_mag * random.choice([-1, 1]), 2)
    y = round((random.random() - 0.2) * p_mag, 2)
    return (m, x, y)


def create_random_example(p1_mag=4, p2_mag=6, m1_mag=5, m2_mag=5):
    """Creates a random double pendulum example.

    Positions and masses are chosen via `random_initial_values` function,
    velocities in both directions are zero.

    Args:
        p1_mag, p2_mag (float): magnitudes of the maximum allowed position
            coordinate for each pendulum
        m1_mag, m2_mag (float): magnitude of the highest allowed mass

    Returns:
        `DoublePendulum` class instance
    """
    p1_0 = random_initial_values(p1_mag, m1_mag)
    p2_0 = random_initial_values(p2_mag, m2_mag)

    p1 = Pendulum(*p1_0, u=0, v=0)
    p2 = Pendulum(*p2_0, u=0, v=0)
    dp = DoublePendulum(p1, p2)
    return dp


def create_perturbations(number, ex=None, amount=1e-6):
    """Creates perturbations of DoublePendulum example by changing the
    initial conditions by `amount` in each direction for both pendulums.

    Args:
        number (int): number of perturbations
        ex (`DoublePendulum` class instance): if not defined, a random example
            is created.
        amount (float): amount of change we want to introduce to the `y`
            coordinate of the second pendulum.

    Returns:
        list of `DoublePendulum` class instances
    """
    if ex is None:
        ex = create_random_example()
    p1 = ex._b1
    p2 = ex._b2

    examples = []
    for n in range(number):
        p1_per = Pendulum(p1.m, (p1.x - n*amount), (p1.y + n*amount), p1.u, p1.v)
        p2_per = Pendulum(p2.m, (p2.x + n*amount), (p2.y - n*amount), p2.u, p2.v)
        examples.append(DoublePendulum(p1_per, p2_per))
    return examples


def simulate(example, method=RK4, duration=30, step_size=0.001):
    """Solves `example` with an explicit Runge-Kutta method.

    Args:
        example (class instance): the example we want to solve, e.g. `DoublePendulum`
        method (namedtuple): explicit Runge-Kutta method defined in `methods.py`
        duration (int): total duration of the simulation
        step_size (float): size of a time-step. Too high value might produce
            unstable results.

    Returns:
        `ERK_DAE1` class instance with the results in `ys` and `zs` attributes.
    """
    return ERK_DAE1(example, method, duration, step_size).solve()


def simulate_multiple_methods(example, methods, duration=30, step_size=0.001):
    """Solve one example with different methods.

    Args:
        example (class instance): the example we want to solve, e.g. `DoublePendulum`
        methods (list of namedtuple): list of explicit Runge-Kutta methods
            defined in `methods.py`
        duration (int): total duration of the simulation
        step_size (float): size of a time-step. Too high value might produce
            unstable results.

    Returns:
        list of `ERK_DAE1` class instances
    """
    return [simulate(example, method, duration, step_size) for method in methods]


def simulate_multiple_examples(examples, method=RK4, duration=30, step_size=0.001):
    """Solve multiple examples. Usually used in conjunction with
    `create_perturbations` function.

    Args:
        examples (list of class instances): list of examples we want to solve,
            e.g. `DoublePendulum`
        method (namedtuple): explicit Runge-Kutta method defined in `methods.py`
        duration (int): total duration of the simulation
        step_size (float): size of a time-step. Too high value might produce
            unstable results.

    Returns:
        list of `ERK_DAE1` class instances
    """
    return [simulate(ex, method, duration, step_size) for ex in examples]


def __run_basic_example():
    rsys = create_random_example()
    r1 = simulate(rsys, method=Euler, duration=15)
    print(r1.ys[:3])
    print(r1)

    exes = create_perturbations(5, amount=1e-3)
    for e in exes:
        print(e.y_0)
        print(e.z_0)

    mtd = [Euler, DOPRI5, ExplicitMidpoint]
    mms = simulate_multiple_methods(rsys, mtd, duration=10, step_size=0.01)
    for mm in mms:
        print(mm.name)
        print(mm.ys[-1])


if __name__ == '__main__':
    __run_basic_example()
