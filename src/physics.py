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
        self.v: np.ndarray = (
            np.array(v) if v is not None else np.zeros_like(x)
        )  # current velocity
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
        self.dt = 0.001

        # Vectorized state arrays
        self.num_particles = len(particles)
        if self.num_particles > 0:
            self.pos = np.array([p.x for p in particles], dtype=np.float64)
            self.vel = np.array([p.v for p in particles], dtype=np.float64)
            self.acc = np.array([p.a for p in particles], dtype=np.float64)
            self.prev_pos = np.array(
                [p.xp if p.xp is not None else p.x for p in particles], dtype=np.float64
            )
            self.masses = np.array([p.m for p in particles], dtype=np.float64).reshape(
                -1, 1
            )
        else:
            self.pos = np.array([], dtype=np.float64)
            self.vel = np.array([], dtype=np.float64)
            self.acc = np.array([], dtype=np.float64)
            self.prev_pos = np.array([], dtype=np.float64)
            self.masses = np.array([], dtype=np.float64)

        # Vectorized constraint caches
        self.elastic_owner_indices = np.array([], dtype=np.int32)
        self.elastic_indices_a = np.array([], dtype=np.int32)
        self.elastic_indices_b = np.array([], dtype=np.int32)
        self.elastic_k = np.array([], dtype=np.float64)
        self.elastic_dr = np.array([], dtype=np.float64)

        self.gravity_indices = np.array([], dtype=np.int32)
        self.gravity_vecs = np.array([], dtype=np.float64)

        self.dampening_indices = np.array([], dtype=np.int32)
        self.dampening_k = np.array([], dtype=np.float64)

    def build_vectorized_constraints(self) -> None:
        """Compiles individual Particle constraints into vectorized NumPy arrays."""
        e_owner, e_a, e_b, e_k, e_dr = [], [], [], [], []
        g_idx, g_vec = [], []
        d_idx, d_k = [], []

        for idx, p in enumerate(self.particles):
            for c in p.constraints:
                ref = c.reference
                # Check for elastic constraints
                if (
                    hasattr(ref, "x1")
                    and hasattr(ref, "x2")
                    and hasattr(ref, "k")
                    and hasattr(ref, "dr")
                ):
                    x1_attr = getattr(ref, "x1", None)
                    x2_attr = getattr(ref, "x2", None)
                    if isinstance(x1_attr, Particle) and isinstance(x2_attr, Particle):
                        try:
                            x1_idx = self.particles.index(x1_attr)
                            x2_idx = self.particles.index(x2_attr)
                            # The resulting force always applies to the
                            # constraint's owner (idx) - regardless of
                            # whether the owner happens to be x1 or x2.
                            e_owner.append(idx)
                            e_a.append(x1_idx)
                            e_b.append(x2_idx)
                            e_k.append(getattr(ref, "k"))
                            e_dr.append(getattr(ref, "dr"))
                        except ValueError:
                            pass
                # Check for gravitational constraints
                elif hasattr(ref, "g") and hasattr(ref, "particle"):
                    g_idx.append(idx)
                    g_vec.append(getattr(ref, "g"))
                # Check for dampening constraints
                elif (
                    hasattr(ref, "k")
                    and hasattr(ref, "particle")
                    and not hasattr(ref, "x1")
                ):
                    d_idx.append(idx)
                    d_k.append(getattr(ref, "k"))

        self.elastic_owner_indices = np.array(e_owner, dtype=np.int32)
        self.elastic_indices_a = np.array(e_a, dtype=np.int32)
        self.elastic_indices_b = np.array(e_b, dtype=np.int32)
        self.elastic_k = np.array(e_k, dtype=np.float64)
        self.elastic_dr = np.array(e_dr, dtype=np.float64)

        self.gravity_indices = np.array(g_idx, dtype=np.int32)
        self.gravity_vecs = np.array(g_vec, dtype=np.float64)

        self.dampening_indices = np.array(d_idx, dtype=np.int32)
        self.dampening_k = np.array(d_k, dtype=np.float64)

    def run(self, n: int | None = None) -> None:
        if self.num_particles == 0:
            return

        # Particles with no constraints stay frozen (0 acceleration)
        fixed_mask = np.array([len(p.constraints) == 0 for p in self.particles])

        i = 0
        while True:
            if n is not None and i >= n:
                break

            # 1. Vectorized force computation - calls the same force functions
            #    used by individual constraints, just with batched inputs.
            net_forces = np.zeros((self.num_particles, 2), dtype=np.float64)

            if self.gravity_indices.size > 0:
                net_forces[self.gravity_indices] += gravitational_force(
                    self.masses[self.gravity_indices], self.gravity_vecs
                )

            if self.dampening_indices.size > 0:
                net_forces[self.dampening_indices] += dampening_force(
                    self.dampening_k[:, np.newaxis], self.vel[self.dampening_indices]
                )

            if self.elastic_indices_a.size > 0:
                f_elastic = elastic_force(
                    self.pos[self.elastic_indices_a],
                    self.pos[self.elastic_indices_b],
                    self.elastic_k[:, np.newaxis],
                    self.elastic_dr[:, np.newaxis],
                )
                np.add.at(net_forces, self.elastic_owner_indices, f_elastic)

            # 2. Vectorized acceleration: a = F / m
            self.acc = net_forces / self.masses
            self.acc[fixed_mask] = 0

            # 3. Vectorized Stormer-Verlet integration (reuses `stromer`)
            next_pos = stromer(self.pos, self.prev_pos, self.acc, self.dt)
            self.prev_pos = self.pos.copy()
            self.pos = next_pos

            # 4. Vectorized velocity: v = (x - xp) / dt
            self.vel = (self.pos - self.prev_pos) / self.dt

            i += 1

        # Sync the vectorized state back to the particle objects once, at the end
        for idx, particle in enumerate(self.particles):
            particle.x = self.pos[idx]
            particle.v = self.vel[idx]
            particle.a = self.acc[idx]
            particle.xp = self.prev_pos[idx]


# constraint_list = []
# particles: dict[str, Particle] = {}


def elastic_force(
    x1: np.ndarray,
    x2: np.ndarray,
    k: float | np.ndarray,
    dr: float | np.ndarray,
    d_min: float | np.ndarray = 0.0001,
) -> np.ndarray:

    dx = x2 - x1
    d = np.linalg.norm(dx, axis=-1, keepdims=True)

    # select between itself and the minimal distance, avoids force explosion.
    d = np.maximum(d, d_min)

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
        p1 = kwargs["x1"]
        p2 = kwargs["x2"]
        return elastic_force(
            p1.x if isinstance(p1, Particle) else p1,
            p2.x if isinstance(p2, Particle) else p2,
            kwargs["k"],
            kwargs["dr"],
            kwargs["d_min"],
        )

    return Constraint(elastic_force_wrapper, reference=ref)


def gravitational_force(m: float | np.ndarray, g: np.ndarray = np.array((0.0, 9.8))):
    if np.any(np.asarray(m) <= 0):
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
        p = kwargs["particle"]
        return gravitational_force(p.m, kwargs["g"])

    return Constraint(gravitational_force_wrapper, reference=ref)


def dampening_force(k: float | np.ndarray, v: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(v, axis=-1, keepdims=True)
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
        p = kwargs["particle"]
        return dampening_force(kwargs["k"], p.v)

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
    ref = Reference(
        particle=particle, pivot_particle=pivot_particle, d_fixed=d_fixed, dt=dt
    )

    def rigid_connection_wrapper(**kwargs):
        p = kwargs["particle"]
        pp = kwargs["pivot_particle"]
        return rigid_connection_force(
            p.m, p.x, p.v, pp.x, pp.v, kwargs["d_fixed"], kwargs["dt"]
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
    ref = Reference(
        particle=particle, pivot_particle=pivot_particle, d_max=d_max, dt=dt
    )

    def rope_wrapper(**kwargs):
        p = kwargs["particle"]
        pp = kwargs["pivot_particle"]
        return rope_force(p.m, p.x, p.v, pp.x, pp.v, kwargs["d_max"], kwargs["dt"])

    return Constraint(rope_wrapper, reference=ref)


def torsion_spring_force(
    theta0: float,   # Target angle in range [0, 2*pi]
    k: float,        # Stiffness coefficient
    v1: np.ndarray,  # Vector from Center to Node A (A - B)
    v2: np.ndarray,  # Vector from Center to Node C (C - B)
    epsilon: float = 1e-4  # Security threshold to avoid division by zero
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    if len1 < epsilon or len2 < epsilon:
        return np.zeros(2), np.zeros(2), np.zeros(2)
        
    v1_normalized = v1 / len1
    v2_normalized = v2 / len2

    # 1. Native NumPy dot and cross products for 2D angle tracking
    dot_product = np.dot(v1_normalized, v2_normalized)
    cross_product = np.cross(v1_normalized, v2_normalized)
    
    # Compute true counter-clockwise angle in range [0, 2*pi]
    theta = np.arctan2(cross_product, dot_product)
    if theta < 0:
        theta += 2 * np.pi

    # 2. Linear delta theta calculation
    delta_theta = theta - theta0
    delta_theta_sign = np.sign(delta_theta)

    # 3. Central Force Vector along the bisector
    bisector = v1_normalized + v2_normalized
    bisector_len = np.linalg.norm(bisector)

    if bisector_len < epsilon:
        # Fallback if segments are completely opposite (180 degrees)
        direction = np.array([-v1_normalized[1], v1_normalized[0]])
    else:
        direction = bisector / bisector_len

    # Central restoring force vector
    central_magnitude = 2 * k * delta_theta * np.cos(delta_theta / 2)
    central_force = -delta_theta_sign * central_magnitude * direction

    # 4. Outer Forces perpendicular to their respective segments
    v1_perpendicular = np.array([-v1_normalized[1], v1_normalized[0]])
    v2_perpendicular = np.array([v2_normalized[1], -v2_normalized[0]])

    # Scale with 1/length to maintain proper torque leverage
    torque_scalar = delta_theta_sign * k * delta_theta
    
    outer_force_1 = (torque_scalar / len1) * v1_perpendicular
    outer_force_2 = (torque_scalar / len2) * v2_perpendicular

    return central_force, outer_force_1, outer_force_2


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
