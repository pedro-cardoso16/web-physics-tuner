import pygame as pg
from physics import *

PARTICLE_RADIUS = 3

# Color Defs
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

pg.init()


def draw_particle(particle: Particle, screen: pg.Surface) -> None:
    pg.draw.circle(screen, WHITE, tuple(particle.x), PARTICLE_RADIUS)


def draw_connection(particle1, particle2, screen: pg.Surface) -> None:
    pg.draw.polygon(screen, WHITE, (particle1.x, particle2.x), width=1)


def run_engine(simulation: Simulation, width: int = 800, height: int = 500):
    # start_time = time.perf_counter()
    running = True
    accumulator = 0.0

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

        # Draw connections
        # for i in range(1, len(particles)):
        #     p = particles[i]
        #     col = i // h
        #     row = i % h
        #     if col > 0:
        #         draw_connection(p, particles[(col - 1) * h + row], screen)
        #     if row > 0:
        #         draw_connection(p, particles[col * h + (row - 1)], screen)

        # Draw particles
        for particle in simulation.particles:
            draw_particle(particle, screen)

        pg.display.flip()

    pg.quit()
    


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

        for particle in simulation.particles:
            draw_particle(particle, screen)

        draw_connection(particle1, particle2, screen)
        draw_connection(particle3, particle2, screen)
        draw_connection(particle4, particle3, screen)
        pg.display.flip()
