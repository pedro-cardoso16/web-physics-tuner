import pygame as pg
from physics import *

PARTICLE_RADIUS = 2

# Color Defs
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

pg.init()


def draw_particle(particle: Particle, screen: pg.Surface) -> None:
    pg.draw.circle(screen, WHITE, tuple(particle.x), PARTICLE_RADIUS)

def draw_connection(particle1, particle2, screen:pg.Surface) -> None:
    # vec = pg.Vector2(particle2.x - particle1.x)
    # vec_normalized = vec.normalize()
    # vec_normalized_rotated = vec.normalize().rotate(angle=90)
    # rect = pg.Rect(particle1.x[0], particle1.x[1], vec.length(), 1)
    # pos_a
    # pos_b
    # pos_c
    # pos_d

    pg.draw.polygon(screen,WHITE, (particle1.x, particle2.x), width=1)

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
        1.0,
        center + np.array([100.0, 0.0]),
        [0.0, 30.0],
    )

    ref = Reference(x1=particle2.x, x2=particle1.x, k=5, dr=80)

    constraint = Constraint(elastic_force, reference=ref)

    particle2.constraints.append(constraint)

    simulation = Simulation([particle1, particle2])

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
        pg.display.flip()
