import pygame as pg
from physics import *
from typing import Literal, Iterable

PARTICLE_RADIUS = 3

# Color Defs
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
pg.init()


def draw_dots(*coords, screen: pg.Surface, **kwargs):
    default_kwargs = {
        "color": BLUE,
        "radius": PARTICLE_RADIUS,
    }

    kwargs = default_kwargs | kwargs

    for coord in coords:
        pg.draw.circle(screen, WHITE, coord, PARTICLE_RADIUS)


def draw_particle(particle: Particle, screen: pg.Surface) -> None:
    pg.draw.circle(screen, WHITE, tuple(particle.x), PARTICLE_RADIUS)


def draw_connection(particle1, particle2, screen: pg.Surface) -> None:
    pg.draw.polygon(screen, WHITE, (particle1.x, particle2.x), width=1)


def draw_connections(
    *coords,
    screen: pg.Surface,
    mode: Literal["sequential", "all"] = "sequential",
    **kwargs
):
    default_kwargs = {
        "color": WHITE,
        "width": 1,
    }

    kwargs = default_kwargs | kwargs

    n_coords = len(coords)

    if mode == "sequential":
        for i in range(n_coords - 1):
            pg.draw.polygon(
                screen, WHITE, (coords[i], coords[i + 1]), width=kwargs["width"]
            )

        return

    if mode == "all":
        for i in range(n_coords - 1):
            for j in range(i + 1, n_coords):
                pg.draw.polygon(
                    screen, WHITE, (coords[i], coords[j]), width=kwargs["width"]
                )
        return


def run_engine(simulation: Simulation, width: int = 800, height: int = 500, **kwargs):
    # start_time = time.perf_counter()
    running = True
    accumulator = 0.0

    connections: set[frozenset[Particle]] = kwargs.get("connections", set())

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        screen.fill(BLACK)
        frame_time = clock.tick(60) / 1000.0
        accumulator += frame_time

        while accumulator > simulation.dt:
            simulation.run(n=1)
            accumulator -= simulation.dt

        # Draw particles
        # for particle in simulation.particles:
        #     draw_particle(particle, screen)

        positions = (particle.x for particle in simulation.particles)
        draw_dots(*positions, screen=screen)

        for c in connections:
            c = tuple(c)
            draw_connection(c[0], c[1], screen=screen)

        pg.display.flip()

    pg.quit()


def run_engine_with_predefined_chain_path(coords: Iterable, dts: list, **kwargs):
    # start_time = time.perf_counter()
    running = True
    accumulator = 0.0

    while running:
        for c, dt in zip(coords, dts):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

            if not running:
                break

            screen.fill(BLACK)
            frame_time = clock.tick(60) / 1000.0
            accumulator += frame_time

            while accumulator > dt:
                accumulator -= dt

            draw_connected_chain(*c, screen=screen)

            pg.display.flip()


        if not kwargs.get("loop", False) or not running:
            break

    pg.quit()


from collections.abc import Iterable

# import pygame as pg


def run_engine_with_multiple_predefined_chain_paths(
    coords: list[Iterable], dts: list, **kwargs
):
    """
    Runs the Pygame engine, drawing any number of predefined chain paths
    simultaneously, synchronized by frame delta-times.

    Args:
        coords: A list of coordinate iterables, e.g., [coords1, coords2, coords3]
        dts: A list of frame delta-times (seconds)
        kwargs: Optional arguments (e.g., loop=True)
    """
    running = True
    accumulator = 0.0

    while running:
        # zip(*coords, dts) unpacks the list of coordinates to match your dts
        for *chains, dt in zip(*coords, dts):

            # Handle Pygame events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                    break  # Break out of the event loop

            if not running:
                break
                        
            # Clear screen and draw background
            screen.fill(BLACK)

            # Tick the clock and accumulate time
            frame_time = clock.tick(kwargs.get('framerate', 60)) / 1000.0
            accumulator += frame_time

            # Step the simulation accumulator
            while accumulator > dt:
                accumulator -= dt

            # Draw every active chain for this frame
            for chain in chains:
                draw_connected_chain(*chain, screen=screen)

            pg.display.flip()

        if not kwargs.get("loop", False) or not running:
            break

    pg.quit()


def zoom_transform(*coords: ArrayLike, zoom_factor: float | ArrayLike):

    new_coords = np.array(coords) * zoom_factor

    return new_coords


def draw_connected_chain(*coords: ArrayLike, screen: pg.Surface, **kwargs):
    draw_dots(*coords, screen=screen, **kwargs)
    draw_connections(*coords, screen=screen, mode="sequential")


# Setup Screen Configuration
WIDTH, HEIGHT = 800, 500
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Particle Connections")
clock = pg.time.Clock()

if __name__ == "__main__":
    # Create the particles:
    center = (WIDTH / 2, HEIGHT / 2)
    particle1 = Particle(
        1.0,
        center + np.array([0.0, 0.0]),
        [0.0, 0.0],
    )

    particle2 = Particle(
        10.0,
        center + np.array([100.0, 0.0]),
        [0.0, 0.0],
    )

    particle3 = Particle(
        10.0,
        center + np.array([120.0, 10.0]),
        [0.0, 0.0],
    )

    particle4 = Particle(
        10.0,
        center + np.array([120.0, 20.0]),
        [0.0, -20.0],
    )

    from presets import *

    simulation = Simulation()

    # ref = Reference(x1=particle2.x, x2=particle1.x, k=50, dr=100)
    # ref_g = Reference(m=1.0, g=np.array((0, 500)))
    # ref_dampening = Reference(v=particle2.v, k=0.01)
    # ref_rigid_conn = Reference(
    #     mass=particle2.m,
    #     pos=particle2.x,
    #     velocity=particle2.v,
    #     pivot_pos=particle1.x,
    #     pivot_velocity=particle1.v,
    #     d_fixed=100,
    #     dt=0.0001,
    # )

    # ref_rope_conn = Reference(
    #     mass=particle2.m,
    #     pos=particle2.x,
    #     velocity=particle2.v,
    #     pivot_pos=particle1.x,
    #     pivot_velocity=particle1.v,
    #     d_max=100,
    #     dt=0.0001,
    # )

    constraint_gravity = make_gravitational_constraint(particle2, np.array((0, 100)))
    dampening_constraint = make_dampening_constraint(particle2, 0.02)
    # rope_constraint = make_rope_constraint(particle2, particle1, 100, 0.0001)
    elastic_constraint_21 = make_elastic_constraint(particle2, particle1, 100, 200)
    elastic_constraint_32 = make_elastic_constraint(particle3, particle2, 100, 40)
    elastic_constraint_23 = make_elastic_constraint(particle2, particle3, 100, 40)
    elastic_constraint_42 = make_elastic_constraint(particle4, particle2, 100, 40)

    particle2.constraints.extend(
        (
            constraint_gravity,
            dampening_constraint,
            elastic_constraint_21,
            elastic_constraint_23,
        )
    )
    particle3.constraints.append(elastic_constraint_32)
    particle4.constraints.append(elastic_constraint_42)

    simulation = Simulation([particle1, particle2, particle3, particle4])

    # add_mesh_2d(simulation, (WIDTH / 2, HEIGHT / 2), 1, 1, 20, 100, m=3, dr=100)

    # add_mesh_2d(
    #     simulation, top_left=(100, 100), w=3, h=1, step=20.0, k=20, m=1, dt=0.0001
    # )
    start_time = time.perf_counter()

    running = True

    accumulator = 0.0
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        # 2. Clear Screen / Draw Background
        screen.fill(BLACK)

        frame_time = clock.tick(60) / 1000.0

        accumulator += frame_time

        while accumulator > simulation.dt:
            simulation.run(n=1)
            accumulator -= simulation.dt

        # for particle in simulation.particles:
        #     draw_particle(particle, screen)

        # draw_connection(particle1, particle2, screen)
        # draw_connection(particle3, particle2, screen)
        # draw_connection(particle4, particle3, screen)

        draw_connected_chain(
            particle1.x, particle2.x, particle3.x, particle4.x, screen=screen
        )
        pg.display.flip()
