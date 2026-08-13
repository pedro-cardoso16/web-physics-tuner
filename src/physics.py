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
        return self.func(**kwargs)  # type: ignore


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
        self.v: np.ndarray = np.array(v) if v is not None else np.zeros_like(x)  # current velocity
        self.a: np.ndarray = (
            np.array(a) if a is not None else np.zeros_like(x)
        )  # current acceleration
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
        self.v[:] = (self.x - self.xp) / dt

        return self.v

    def update_vars(self, dt) -> None:
        if self.xp is None:
            self.xp = self.x[:]
            self.x = self.x + self.v * dt + 0.5 * self.a * dt**2
            self.v = self.compute_velocity(dt)
            return

        # Calculate next position using Störmer-Verlet
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
    def __init__(self, particles: list[Particle] = []) -> None:
        self.particles = particles
        self.dt = 0.0001

    def run(self, n: int | None = None) -> None:
        i = 0
        while True:
            if n is not None and i >= n:
                break

            # Vectorized acceleration computation
            # To make this truly fast, we need to move away from per-particle loops
            # and use NumPy's vectorization across the entire particle set.
            
            # First pass: compute accelerations
            for particle in self.particles:
                particle.compute_acceleration()

            # Second pass: update variables
            for particle in self.particles:
                particle.update_vars(self.dt)

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


def make_elastic_constraint(
    particle1: Particle, particle2: Particle, k: float, dr: float, d_min: float = 0.0001
) -> Constraint:
    """
    Create an elastic force constraint from two Particle instances.

    Automatically extracts positions from particles and creates a Reference + Constraint.

    Args:
        particle1: First Particle instance
        particle2: Second Particle instance
        k: Spring constant
        dr: Rest distance
        d_min: Minimum distance to avoid force explosion

    Returns:
        Constraint: Ready-to-use constraint for particles
    """
    # Store particles directly so we always access current positions
    ref = Reference(x1=particle1, x2=particle2, k=k, dr=dr, d_min=d_min)
    
    # Create wrapper function that extracts positions on-demand
    def elastic_force_wrapper(**kwargs):
        p1 = kwargs['x1']
        p2 = kwargs['x2']
        return elastic_force(p1.x if isinstance(p1, Particle) else p1,
                            p2.x if isinstance(p2, Particle) else p2,
                            kwargs['k'], kwargs['dr'], kwargs['d_min'])
    
    return Constraint(elastic_force_wrapper, reference=ref)


def gravitational_force(m: float, g: np.ndarray = np.array((0.0, 9.8))):
    if m <= 0:
        raise ValueError("The mass (m) must be a positive number.")

    return m * g


def make_gravitational_constraint(
    particle: Particle, g: np.ndarray = np.array((0.0, 9.8))
) -> Constraint:
    """
    Create a gravitational force constraint from a Particle instance.
    
    Args:
        particle: Particle instance
        g: Gravitational acceleration vector (default: (0, 9.8))
        
    Returns:
        Constraint: Ready-to-use constraint for the particle
    """
    ref = Reference(particle=particle, g=g)
    
    def gravitational_force_wrapper(**kwargs):
        p = kwargs['particle']
        return gravitational_force(p.m, kwargs['g'])
    
    return Constraint(gravitational_force_wrapper, reference=ref)


def dampening_force(k: float, v: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(v)
    return -k * magnitude * v


def make_dampening_constraint(particle: Particle, k: float) -> Constraint:
    """
    Create a dampening force constraint from a Particle instance.
    
    Args:
        particle: Particle instance
        k: Dampening coefficient
        
    Returns:
        Constraint: Ready-to-use constraint for the particle
    """
    ref = Reference(particle=particle, k=k)
    
    def dampening_force_wrapper(**kwargs):
        p = kwargs['particle']
        return dampening_force(kwargs['k'], p.v)
    
    return Constraint(dampening_force_wrapper, reference=ref)


def rigid_connection_force(
    mass: float,
    pos: np.ndarray,
    velocity: np.ndarray,
    pivot_pos: np.ndarray,
    pivot_velocity: np.ndarray,
    d_fixed: float,
    dt: float,
) -> np.ndarray:
    """Rigid connection force

    This force fixates the distance between the particle and the pivot. It calculates
    the force assuming the pivot has infinite mass, that is all the force is applied to the
    particle.

    **This is like the elastic force with infinite elastic constant value.**

    Args:
        mass (float): Particle's mass.
        pos (np.ndarray): Particle's current position.
        velocity (np.ndarray): Particle's current velocity.
        pivot_pos (np.ndarray): Pivot's current position.
        pivot_velocity (np.ndarray): Pivot's current velocity.
        d_fixed (float): Fixed distance between the pivot and the particle.
        dt (float): Time interval into the future.

    Returns:
        f (np.ndarray): Net force required for position correction into the future.
            Doesn't take into account other external forces, only current particle
            and pivot's positions and velocities.
    """
    future_rel_pos = (pos - pivot_pos) + dt * (velocity - pivot_velocity)
    future_rel_pos_norm = np.linalg.norm(future_rel_pos)
    future_rel_pos_normalized = future_rel_pos / future_rel_pos_norm

    f = -(mass / dt**2) * (future_rel_pos_norm - d_fixed) * future_rel_pos_normalized
    return f


def make_rigid_connection_constraint(
    particle: Particle, pivot_particle: Particle, d_fixed: float, dt: float
) -> Constraint:
    """
    Create a rigid connection force constraint from two Particle instances.
    
    Args:
        particle: The particle to apply force to
        pivot_particle: The pivot/fixed particle
        d_fixed: Fixed distance between particles
        dt: Time interval
        
    Returns:
        Constraint: Ready-to-use constraint
    """
    ref = Reference(particle=particle, pivot_particle=pivot_particle, d_fixed=d_fixed, dt=dt)
    
    def rigid_connection_wrapper(**kwargs):
        p = kwargs['particle']
        pp = kwargs['pivot_particle']
        return rigid_connection_force(
            p.m, p.x, p.v,
            pp.x, pp.v,
            kwargs['d_fixed'], kwargs['dt']
        )
    
    return Constraint(rigid_connection_wrapper, reference=ref)


def rope_force(
    mass: float,
    pos: np.ndarray,
    velocity: np.ndarray,
    pivot_pos: np.ndarray,
    pivot_velocity: np.ndarray,
    d_max: float,
    dt: float,
) -> np.ndarray:
    """Rope connection force

    This force fixates the maximum distance between the particle and the pivot.
    It works exactly as [`rigid_connection_force`](rigit_connection_force) when
    the distance is predicted to be above `d_max` `dt` seconds into the future.
    It simply returns a zero-valued force otherwise.

    Args:
        mass (float): Particle's mass.
        pos (np.ndarray): Particle's current position.
        velocity (np.ndarray): Particle's current velocity.
        pivot_pos (np.ndarray): Pivot's current position.
        pivot_velocity (np.ndarray): Pivot's current velocity.
        d_max (float): Max distance between the pivot and the particle.
            Equivalent to a rope's length.
        dt (float): Time interval into the future.

    Returns:
        f (np.ndarray): Net force required for position correction into the future.
            Doesn't take into account other external forces, only current particle
            and pivot's positions and velocities.
    """
    future_rel_pos = (pos - pivot_pos) + dt * (velocity - pivot_velocity)
    future_rel_pos_norm = np.linalg.norm(future_rel_pos)

    if future_rel_pos_norm <= d_max:
        return np.zeros_like(pos)

    future_rel_pos_normalized = future_rel_pos / future_rel_pos_norm

    f = -(mass / dt**2) * (future_rel_pos_norm - d_max) * future_rel_pos_normalized
    return f


def make_rope_constraint(
    particle: Particle, pivot_particle: Particle, d_max: float, dt: float
) -> Constraint:
    """
    Create a rope force constraint from two Particle instances.
    
    Args:
        particle: The particle to apply force to
        pivot_particle: The pivot/fixed particle
        d_max: Maximum distance (rope length)
        dt: Time interval
        
    Returns:
        Constraint: Ready-to-use constraint
    """
    ref = Reference(particle=particle, pivot_particle=pivot_particle, d_max=d_max, dt=dt)
    
    def rope_wrapper(**kwargs):
        p = kwargs['particle']
        pp = kwargs['pivot_particle']
        return rope_force(
            p.m, p.x, p.v,
            pp.x, pp.v,
            kwargs['d_max'], kwargs['dt']
        )
    
    return Constraint(rope_wrapper, reference=ref)


def set_constraint(constraint_func: Callable, **kwargs) -> Constraint:
    reference = Reference()

    for key in constraint_func.__annotations__.keys():
        setattr(reference, key, kwargs[key])

    return Constraint(constraint_func, reference)

def add_constraint_to_particle(particle: Particle, *constraint: Constraint) -> None:
    particle.constraints.extend(constraint)


def stromer(x: NDArray, xp: NDArray, a: NDArray, dt: float) -> np.ndarray:
    """Computes next stromer position `xn`.

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

        print(round(end_time - start_time, 2), end="                 \r")
        if (end_time - start_time) > 100:
            break

    plt.figure()
    plt.plot(xt)
    plt.plot(yt)
    plt.show()


if __name__ == "__main__":
    main()
