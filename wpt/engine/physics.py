from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


def _tensor(value: Any, *, dtype: torch.dtype = torch.float64) -> Tensor:
    """Convert values without copying or detaching an existing tensor."""
    if isinstance(value, Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _index(value: Any, device: torch.device) -> Tensor:
    return torch.as_tensor(value, dtype=torch.long, device=device)


def _stack_values(values: list[Any], *, dtype: torch.dtype, device: torch.device) -> Tensor:
    if not values:
        return torch.empty(0, dtype=dtype, device=device)
    tensors = [_tensor(value, dtype=dtype).to(device) for value in values]
    return torch.stack([value.reshape(()) for value in tensors])


class Reference:
    def __init__(self, **kwargs) -> None:
        for key, val in kwargs.items():
            setattr(self, key, val)


class Constraint:
    def __init__(self, func: Callable, reference: Reference) -> None:
        self.func = func
        self.reference = reference

    def compute_force(self) -> Tensor:
        return self.func(**vars(self.reference))


class Particle:
    def __init__(self, m=None, x=None, v=None, a=None) -> None:
        if x is None:
            raise ValueError("Particle position x must be provided.")
        self.m = m
        self.x = _tensor(x)
        self.v = _tensor(v if v is not None else torch.zeros_like(self.x))
        self.a = _tensor(a if a is not None else torch.zeros_like(self.x))
        self.f = torch.zeros_like(self.x)
        self.xp: Tensor | None = None
        self.vp = self.v.clone()
        self.constraints: list[Constraint] = []

    def compute_acceleration(self, net_force: Tensor | None = None) -> Tensor:
        if net_force is None:
            net_force = self.compute_force()
            if net_force is None:
                return torch.empty(0, dtype=self.x.dtype, device=self.x.device)
        if self.m is None:
            raise ValueError("The particle's mass is not defined, please set a positive value.")
        self.a = net_force / _tensor(self.m, dtype=self.x.dtype)
        return self.a

    def compute_velocity(self, dt: float) -> Tensor:
        self.vp = self.v.clone()
        self.v = (self.x - self.xp) / dt
        return self.v

    def update_vars(self, dt: float, dtp: float | None = None) -> None:
        if self.xp is None:
            self.xp = self.x.clone()
            self.x = self.x + self.v * dt + 0.5 * self.a * dt**2
        else:
            self.xp, self.x = self.x, stromer(self.x, self.xp, self.a, dt, dtp)
        self.v = self.compute_velocity(dt)

    def compute_force(self) -> Tensor | None:
        if not self.constraints:
            return None
        force = torch.zeros_like(self.x)
        for constraint in self.constraints:
            force = force + constraint.compute_force()
        self.f = force
        return force


class Simulation:
    def __init__(self, particles: list[Particle] | None = None) -> None:
        self.dt = 0.001
        self.dtp: float | None = None
        self.particles = particles or []

    @property
    def particles(self) -> list[Particle]:
        return self.__particles

    @particles.setter
    def particles(self, value: list[Particle]) -> None:
        self.__particles = value
        self.__build_vectorized_params()

    def __build_vectorized_params(self) -> None:
        self.num_particles = len(self.particles)
        if self.num_particles:
            device = self.particles[0].x.device
            self.pos = torch.stack([p.x for p in self.particles])
            self.vel = torch.stack([p.v for p in self.particles])
            self.prev_vel = torch.stack([p.vp for p in self.particles])
            self.acc = torch.stack([p.a for p in self.particles])
            self.prev_pos = torch.stack([
                p.xp if p.xp is not None else p.x - p.v * self.dt
                for p in self.particles
            ])
            self.masses = torch.stack([
                _tensor(p.m, dtype=self.pos.dtype).reshape(())
                for p in self.particles
            ]).reshape(-1, 1)
        else:
            device = torch.device("cpu")
            self.pos = torch.empty((0, 2), dtype=torch.float64, device=device)
            self.vel = self.pos.clone()
            self.prev_vel = self.pos.clone()
            self.acc = self.pos.clone()
            self.prev_pos = self.pos.clone()
            self.masses = torch.empty((0, 1), dtype=torch.float64, device=device)

        empty_i = torch.empty(0, dtype=torch.long, device=device)
        empty_f = torch.empty(0, dtype=self.pos.dtype, device=device)
        self.elastic_owner_indices = empty_i
        self.elastic_indices_a = empty_i
        self.elastic_indices_b = empty_i
        self.elastic_k = empty_f
        self.elastic_dr = empty_f
        self.elastic_dampening_k = empty_f
        self.gravity_indices = empty_i
        self.gravity_vecs = torch.empty((0, 2), dtype=self.pos.dtype, device=device)
        self.dampening_indices = empty_i
        self.dampening_k = empty_f
        self.torsion_central_indices = empty_i
        self.torsion_outer1_indices = empty_i
        self.torsion_outer2_indices = empty_i
        self.torsion_theta0 = empty_f
        self.torsion_k = empty_f
        self.torsion_epsilon = empty_f
        self.rigid_owner_indices = empty_i
        self.rigid_pivot_indices = empty_i
        self.rigid_d_fixed = empty_f
        self.rigid_dt = empty_f
        self.rope_owner_indices = empty_i
        self.rope_pivot_indices = empty_i
        self.rope_d_max = empty_f
        self.rope_dt = empty_f

    def build_vectorized_constraints(self) -> None:
        e_owner, e_a, e_b, e_k, e_dr, e_kd = [], [], [], [], [], []
        g_idx, g_vec, d_idx, d_k = [], [], [], []
        tc, to1, to2, tt, tk, te = [], [], [], [], [], []
        ro, rp, rdf, rdt, rpo, rpp, rdm, rdtp = [], [], [], [], [], [], [], []
        seen: set[int] = set()
        for idx, particle in enumerate(self.particles):
            for constraint in particle.constraints:
                ref = constraint.reference
                if all(hasattr(ref, x) for x in ("central_particle", "outer_particle_1", "outer_particle_2", "theta0")):
                    if id(ref) in seen:
                        continue
                    try:
                        tc.append(self.particles.index(ref.central_particle))
                        to1.append(self.particles.index(ref.outer_particle_1))
                        to2.append(self.particles.index(ref.outer_particle_2))
                    except ValueError:
                        continue
                    seen.add(id(ref))
                    tt.append(ref.theta0); tk.append(ref.k); te.append(getattr(ref, "epsilon", 1e-4))
                elif all(hasattr(ref, x) for x in ("particle", "pivot_particle", "d_fixed", "dt")):
                    try: pivot = self.particles.index(ref.pivot_particle)
                    except ValueError: continue
                    ro.append(idx); rp.append(pivot); rdf.append(ref.d_fixed); rdt.append(ref.dt)
                elif all(hasattr(ref, x) for x in ("particle", "pivot_particle", "d_max", "dt")):
                    try: pivot = self.particles.index(ref.pivot_particle)
                    except ValueError: continue
                    rpo.append(idx); rpp.append(pivot); rdm.append(ref.d_max); rdtp.append(ref.dt)
                elif all(hasattr(ref, x) for x in ("x1", "x2", "k", "dr")):
                    if isinstance(ref.x1, Particle) and isinstance(ref.x2, Particle):
                        try:
                            e_owner.append(idx); e_a.append(self.particles.index(ref.x1)); e_b.append(self.particles.index(ref.x2))
                        except ValueError: continue
                        e_k.append(ref.k); e_dr.append(ref.dr); e_kd.append(ref.k_damp if ref.k_damp is not None else 0.0)
                elif hasattr(ref, "g") and hasattr(ref, "particle"):
                    g_idx.append(idx); g_vec.append(ref.g)
                elif hasattr(ref, "k") and hasattr(ref, "particle") and not hasattr(ref, "x1"):
                    d_idx.append(idx); d_k.append(ref.k)

        device, dtype = self.pos.device, self.pos.dtype
        def inds(values): return torch.as_tensor(values, dtype=torch.long, device=device)
        def vals(values): return _stack_values(values, dtype=dtype, device=device)
        def vectors(values):
            if not values:
                return torch.empty((0, 2), dtype=dtype, device=device)
            return torch.stack([_tensor(value, dtype=dtype).to(device).reshape(2) for value in values])
        self.elastic_owner_indices, self.elastic_indices_a, self.elastic_indices_b = inds(e_owner), inds(e_a), inds(e_b)
        self.elastic_k, self.elastic_dr, self.elastic_dampening_k = vals(e_k), vals(e_dr), vals(e_kd)
        self.gravity_indices, self.gravity_vecs = inds(g_idx), vectors(g_vec)
        self.dampening_indices, self.dampening_k = inds(d_idx), vals(d_k)
        self.torsion_central_indices, self.torsion_outer1_indices, self.torsion_outer2_indices = inds(tc), inds(to1), inds(to2)
        self.torsion_theta0, self.torsion_k, self.torsion_epsilon = vals(tt), vals(tk), vals(te)
        self.rigid_owner_indices, self.rigid_pivot_indices = inds(ro), inds(rp)
        self.rigid_d_fixed, self.rigid_dt = vals(rdf), vals(rdt)
        self.rope_owner_indices, self.rope_pivot_indices = inds(rpo), inds(rpp)
        self.rope_d_max, self.rope_dt = vals(rdm), vals(rdtp)

    def run(self, n: int | None = None) -> None:
        if not self.num_particles:
            return
        fixed = torch.tensor([not p.constraints for p in self.particles], device=self.pos.device)
        i = 0
        while n is None or i < n:
            forces = torch.zeros_like(self.pos)
            if self.gravity_indices.numel():
                forces = forces.index_add(0, self.gravity_indices, gravitational_force(self.masses[self.gravity_indices], self.gravity_vecs))
            if self.dampening_indices.numel():
                forces = forces.index_add(0, self.dampening_indices, dampening_force(self.dampening_k[:, None], self.vel[self.dampening_indices]))
            if self.elastic_indices_a.numel():
                f = elastic_force(self.pos[self.elastic_indices_a], self.pos[self.elastic_indices_b], self.elastic_k[:, None], self.elastic_dr[:, None], k_damp=self.elastic_dampening_k[:, None], v=self.vel[self.elastic_indices_a] - self.vel[self.elastic_indices_b])
                forces = forces.index_add(0, self.elastic_owner_indices, f)
            if self.torsion_central_indices.numel():
                f0, f1, f2 = torsion_spring_force(self.torsion_theta0[:, None], self.torsion_k[:, None], self.pos[self.torsion_outer1_indices] - self.pos[self.torsion_central_indices], self.pos[self.torsion_outer2_indices] - self.pos[self.torsion_central_indices], self.torsion_epsilon[:, None])
                for indices, force in ((self.torsion_central_indices, f0), (self.torsion_outer1_indices, f1), (self.torsion_outer2_indices, f2)):
                    forces = forces.index_add(0, indices, force)
            if self.rigid_owner_indices.numel():
                forces = forces.index_add(0, self.rigid_owner_indices, rigid_connection_force(self.masses[self.rigid_owner_indices], self.pos[self.rigid_owner_indices], self.vel[self.rigid_owner_indices], self.pos[self.rigid_pivot_indices], self.vel[self.rigid_pivot_indices], self.rigid_d_fixed[:, None], self.rigid_dt[:, None]))
            if self.rope_owner_indices.numel():
                forces = forces.index_add(0, self.rope_owner_indices, rope_force(self.masses[self.rope_owner_indices], self.pos[self.rope_owner_indices], self.vel[self.rope_owner_indices], self.pos[self.rope_pivot_indices], self.vel[self.rope_pivot_indices], self.rope_d_max[:, None], self.rope_dt[:, None]))
            acceleration = torch.where(fixed[:, None], torch.zeros_like(forces), forces / self.masses)
            next_pos = stromer(self.pos, self.prev_pos, acceleration, self.dt, self.dtp)
            self.dtp = self.dt
            self.prev_pos, self.pos = self.pos, next_pos
            self.prev_vel, self.vel = self.vel, (self.pos - self.prev_pos) / self.dt
            self.acc = acceleration
            i += 1
        for idx, particle in enumerate(self.particles):
            particle.x, particle.v, particle.vp, particle.a, particle.xp = self.pos[idx], self.vel[idx], self.prev_vel[idx], self.acc[idx], self.prev_pos[idx]

    def clear(self) -> None:
        self.particles.clear()
        self.__build_vectorized_params()


def gravitational_force(m, g=(0.0, 9.8)):
    mass = _tensor(m)
    if torch.any(mass <= 0):
        raise ValueError("The mass (m) must be a positive number.")
    return mass * _tensor(g).reshape(-1, 2) if mass.ndim > 1 else mass * _tensor(g)


def make_gravitational_constraint(particle, g=(0.0, 9.8)):
    ref = Reference(particle=particle, g=g)
    return Constraint(lambda **kw: gravitational_force(kw["particle"].m, kw["g"]), ref)


def dampening_force(k, v):
    velocity = _tensor(v)
    return -_tensor(k).reshape(-1, 1) * torch.linalg.vector_norm(velocity, dim=-1, keepdim=True) * velocity


def make_dampening_constraint(particle, k):
    ref = Reference(particle=particle, k=k)
    return Constraint(lambda **kw: dampening_force(kw["k"], kw["particle"].v), ref)


def elastic_force(x1, x2, k, dr, d_min=1e-16, d_max=float("1e300"), max_force=1e6, k_damp=None, v=None):
    x1, x2 = _tensor(x1), _tensor(x2)
    dx = x2 - x1
    distance = torch.linalg.vector_norm(dx, dim=-1, keepdim=True)
    safe_distance = distance.clamp(min=d_min, max=d_max)
    unit = dx / safe_distance
    magnitude = (_tensor(k) * (safe_distance - _tensor(dr))).clamp(-max_force, max_force)
    force = unit * magnitude
    if v is not None and k_damp is not None:
        if torch.any(_tensor(k_damp) < 0):
            raise ValueError("k_damp must be a non-negative value.")
        velocity = v() if callable(v) else _tensor(v)
        projection = unit * (velocity * unit).sum(dim=-1, keepdim=True)
        force = force - _tensor(k_damp).reshape(-1, 1) * projection
    return force


def make_elastic_constraint(particle1, particle2, k, dr, d_min=1e-16, k_damp=None):
    ref = Reference(x1=particle1, x2=particle2, k=k, dr=dr, d_min=d_min, k_damp=k_damp)
    def wrapper(**kw):
        p1, p2 = kw["x1"], kw["x2"]
        return elastic_force(p1.x, p2.x, kw["k"], kw["dr"], kw["d_min"], k_damp=kw["k_damp"], v=lambda: p1.v - p2.v)
    return Constraint(wrapper, ref)


def rigid_connection_force(mass, pos, velocity, pivot_pos, pivot_velocity, d_fixed, dt):
    relative = _tensor(pos) - _tensor(pivot_pos) + _tensor(dt) * (_tensor(velocity) - _tensor(pivot_velocity))
    norm = torch.linalg.vector_norm(relative, dim=-1, keepdim=True).clamp_min(1e-16)
    return -_tensor(mass) / _tensor(dt).square() * (norm - _tensor(d_fixed)) * relative / norm


def make_rigid_connection_constraint(particle, pivot_particle, d_fixed, dt):
    ref = Reference(particle=particle, pivot_particle=pivot_particle, d_fixed=d_fixed, dt=dt)
    return Constraint(lambda **kw: rigid_connection_force(kw["particle"].m, kw["particle"].x, kw["particle"].v, kw["pivot_particle"].x, kw["pivot_particle"].v, kw["d_fixed"], kw["dt"]), ref)


def rope_force(mass, pos, velocity, pivot_pos, pivot_velocity, d_max, dt):
    relative = _tensor(pos) - _tensor(pivot_pos) + _tensor(dt) * (_tensor(velocity) - _tensor(pivot_velocity))
    norm = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
    slack = norm <= _tensor(d_max)
    safe = torch.where(slack, torch.ones_like(norm), norm)
    force = -_tensor(mass) / _tensor(dt).square() * (norm - _tensor(d_max)) * relative / safe
    return torch.where(slack, torch.zeros_like(force), force)


def make_rope_constraint(particle, pivot_particle, d_max, dt):
    ref = Reference(particle=particle, pivot_particle=pivot_particle, d_max=d_max, dt=dt)
    return Constraint(lambda **kw: rope_force(kw["particle"].m, kw["particle"].x, kw["particle"].v, kw["pivot_particle"].x, kw["pivot_particle"].v, kw["d_max"], kw["dt"]), ref)


def torsion_spring_force(theta0, k, v1, v2, epsilon=1e-4):
    v1, v2 = _tensor(v1), _tensor(v2)
    single = v1.ndim == 1
    if single:
        v1, v2 = v1[None], v2[None]
    theta0, k, epsilon = (_tensor(x).reshape(-1, 1) for x in (theta0, k, epsilon))
    length1 = torch.linalg.vector_norm(v1, dim=1, keepdim=True)
    length2 = torch.linalg.vector_norm(v2, dim=1, keepdim=True)
    degenerate = (length1 < epsilon) | (length2 < epsilon)
    l1, l2 = length1.clamp_min(1.0), length2.clamp_min(1.0)
    u1, u2 = v1 / l1, v2 / l2
    dot = (u1 * u2).sum(dim=1, keepdim=True)
    cross = (u1[:, 0] * u2[:, 1] - u1[:, 1] * u2[:, 0]).abs().reshape(-1, 1)
    theta = torch.atan2(cross, dot)
    delta = theta - theta0
    sign = torch.sign(delta)
    bisector = u1 + u2
    bl = torch.linalg.vector_norm(bisector, dim=1, keepdim=True)
    fallback = torch.stack((-u1[:, 1], u1[:, 0]), dim=1)
    direction = torch.where((bl < epsilon), fallback, bisector / bl.clamp_min(1.0))
    central = -sign * (2 * k * delta * torch.cos(delta / 2)) * direction
    perp1 = torch.stack((-u1[:, 1], u1[:, 0]), dim=1)
    perp2 = torch.stack((u2[:, 1], -u2[:, 0]), dim=1)
    scalar = sign * k * delta
    outer1, outer2 = scalar / l1 * perp1, scalar / l2 * perp2
    mask = ~degenerate
    central, outer1, outer2 = (torch.where(mask, force, torch.zeros_like(force)) for force in (central, outer1, outer2))
    return tuple(force[0] for force in (central, outer1, outer2)) if single else (central, outer1, outer2)


def make_torsion_spring_constraint(central_particle, outer_particle_1, outer_particle_2, theta0, k, epsilon=1e-4, **kwargs):
    ref = Reference(central_particle=central_particle, outer_particle_1=outer_particle_1, outer_particle_2=outer_particle_2, theta0=theta0, k=k, epsilon=epsilon, **kwargs)
    cache = {}
    def compute():
        result = torsion_spring_force(ref.theta0, ref.k, outer_particle_1.x - central_particle.x, outer_particle_2.x - central_particle.x, ref.epsilon)
        return result
    def wrapper(index):
        return lambda **_: compute()[index]
    return tuple(Constraint(wrapper(i), ref) for i in range(3))


def set_constraint(constraint_func: Callable, **kwargs) -> Constraint:
    reference = Reference(**{key: kwargs[key] for key in constraint_func.__annotations__})
    return Constraint(constraint_func, reference)


def add_constraint_to_particle(particle, *constraints) -> None:
    particle.constraints.extend(constraints)


def stromer(x, xp, a, dt, dtp=None):
    if dtp is None or dt == dtp:
        return 2 * x - xp + a * dt**2
    return x + (x - xp) * (dt / dtp) + (a / 2) * (dt + dtp) * dt


def randomize_particle_property(particle, property, r, dx=None, dy=None) -> None:
    theta = 2 * math.pi * torch.rand((), device=particle.x.device, dtype=particle.x.dtype)
    value = getattr(particle, property)
    value += r * torch.stack((torch.cos(theta), torch.sin(theta)))
    if property == "x":
        particle.xp = particle.x.clone()
