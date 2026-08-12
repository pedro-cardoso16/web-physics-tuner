import numpy as np
import time
import matplotlib.pyplot as plt

from typing import Callable
from numpy.typing import ArrayLike, NDArray


class Reference:
    def __init__(self, **kwargs) -> None:
        for key, val in kwargs.items():
            setattr(self, key, val)


class Constraint:
    def __init__(self, func, reference: Reference) -> None:
        self.func: Callable[[], float] = func
        # self.target: Particle = target
        self.reference = reference

    def compute_force(self) -> np.ndarray:
        kwargs = vars(self.reference)
        return self.func(**kwargs) # type: ignore


class Particle:
    def __init__(
        self,
        m: float | None = None,
        x: ArrayLike | None = None,
        v: ArrayLike | None = None,
        a: ArrayLike | None = None,
    ) -> None:
        self.m = m  # mass
        self.x: np.ndarray = np.array(x)  # current position
        self.v: np.ndarray = np.array(v)  # current velocity
        self.a: np.ndarray = np.array(a)  # current acceleration
        self.f: np.ndarray = np.zeros_like(self.x)
        self.xp: np.ndarray | None = None

        self.constraints: list[Constraint] = []

    def compute_acceleration(self, net_force: np.ndarray | None = None) -> np.ndarray:
        """Compute acceleration

        Args:
            net_force (np.ndarray | None): net force acting on the particle. If
                `None`, will compute the force based on the constraints. Defaults to `None`
        """
        if net_force is None:
            net_force = self.compute_force()
            if net_force is None:
                return np.array([])

        if self.m is None:
            raise ValueError(
                "The particle's mass is not defined, please set a positive value."
            )

        a = net_force / self.m

        self.a = a
        return a

    def compute_velocity(self, dt) -> np.ndarray:
        self.v = (self.x - self.xp) / dt

        return self.v

    def update_vars(self, dt) -> None:
        if self.xp is None:
            self.compute_acceleration()
            self.xp = self.x[:]
            self.x = self.x + self.v * dt + 0.5 * self.a * dt**2
            self.v = self.compute_velocity(dt)

            return

        self.a = self.compute_acceleration()

        # Calculate next position
        xn = stromer(self.x, self.xp, self.a, dt)

        # Update the previous position
        self.xp[:] = self.x[:]

        # change current position to the next one
        self.x = xn

        self.v = self.compute_velocity(dt)

    def compute_force(self) -> np.ndarray | None:
        # raise NotImplementedError("Particle instance must implement compute_force()")
        if self.constraints is None:
            return

        net_force = 0

        for constraint in self.constraints:
            net_force += constraint.compute_force()

        self.f[:] = net_force  
        return self.f


class Simulation:
    def __init__(self, particles: list[Particle]) -> None:
        self.particles = particles
        self.dt = 0.0001

    def run(self, n: int | None = None) -> None:
        i = 0
        while True:
            if n is not None and i > n:
                break

            # Compute all particles' accelerations
            for particle in self.particles:
                particle.compute_acceleration()

            # Update all the positions once the accelerations are computed
            for particle in self.particles:
                particle.update_vars(self.dt)

            # for particle in self.particles:
            #     ref = None
            #     try:
            #         ref = vars(particle.constraints[0].reference)
            #     except:
            #         pass
            #     print(
            #         f"id {i}:",
            #         "x =",
            #         particle.x,
            #         "| v =",
            #         particle.v,
            #         "f =",
            #         particle.f,
            #         # "ref =",
            #         # ref,
            #     )
            #     print()

            i += 1


constraint_list = []
particles: dict[str, Particle] = {}


def elastic_force(
    x1: np.ndarray, x2: np.ndarray, k: float, dr: float, d_min: float = 0.0001
) -> np.ndarray:

    dx = x2 - x1
    d = np.linalg.norm(dx)

    # select between itself and the minimal distance, avoids force explosion.
    d = np.max((d, d_min))

    dxu = dx / d  # dx unitary vector
    f = dxu * (k * (d - dr))  # force

    return f


def gravitational_force(m: float, g: np.ndarray = np.array((0.0, 9.8))):
    if m <= 0:
        raise ValueError("The mass (m) must be a positive number.")

    return m * g


def stromer(x: NDArray, xp: NDArray, a: NDArray, dt: float) -> np.ndarray:
    """Computes next stromer position xn

    Args:
        x (float): current position of the object
        xp (float): previous object's position
        a (float): current acceleration
        dt (float): time step
    """
    xn = (2 * x - xp) + a * dt**2

    return xn


def main() -> None:

    # Create the particles:
    particle1 = Particle(
        1.0,
        [0.0, 0.0],
        [0.0, 0.0],
    )

    particle2 = Particle(
        1.0,
        [1.0, 0.0],
        [0.0, 0.0],
    )

    ref = Reference(x1=particle2.x, x2=particle1.x, k=1, dr=0.95)

    constraint = Constraint(elastic_force, reference=ref)

    print(particle2.x)
    # return

    particle2.constraints.append(constraint)

    simulation = Simulation([particle1, particle2])
    xt = []
    yt = []
    start_time = time.perf_counter()

    while True:
        simulation.run(n=1)
        end_time = time.perf_counter()

        xt.append(particle2.x[0])
        yt.append(particle2.x[1])

        print(round(end_time - start_time,2), end="                 \r")
        if (end_time - start_time) > 100:
            break

    plt.figure()
    plt.plot(xt)
    plt.plot(yt)
    plt.show()


if __name__ == "__main__":
    main()
