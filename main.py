import pygame
import numpy as np
from pygame import Vector2
import sys
from collections import defaultdict
import time

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = 400
PARTICLE_RADIUS = 4
PARTICLE_MASS = 0.5
REST_DENSITY = 1.2
PRESSURE_CONSTANT = 100
SURFACE_TENSION = 0.05
VISCOSITY = 0.01
GRAVITY = Vector2(0, 100)
BASE_TIME_STEP = 1/60
MAX_PARTICLES = 800
MOUSE_INFLUENCE_RADIUS = 20
REMOVAL_RADIUS = 30

class NotificationText:
    def __init__(self, text, duration=0.5):
        self.text = text
        self.start_time = time.time()
        self.duration = duration
        
    def should_remove(self):
        return time.time() - self.start_time > self.duration
        
    def get_alpha(self):
        remaining = self.duration - (time.time() - self.start_time)
        return int(255 * (remaining / self.duration))

class PhysicsBox:
    def __init__(self, pos, size=Vector2(40, 40)):
        self.pos = Vector2(pos)
        self.size = size
        self.vel = Vector2(0, 0)
        self.force = Vector2(0, 0)
        self.mass = 5.0
        
    def update(self, time_step):
        self.vel += self.force / self.mass * time_step
        self.vel *= 0.98  # Damping
        self.pos += self.vel * time_step
        self.force = Vector2(0, 100 * self.mass)  # Reset force with gravity
        
        # Improved boundary collision
        margin = 2
        if self.pos.x < margin:
            self.pos.x = margin
            self.vel.x = abs(self.vel.x) * 0.5
        elif self.pos.x > WINDOW_SIZE - self.size.x - margin:
            self.pos.x = WINDOW_SIZE - self.size.x - margin
            self.vel.x = -abs(self.vel.x) * 0.5
            
        if self.pos.y < margin:
            self.pos.y = margin
            self.vel.y = abs(self.vel.y) * 0.5
        elif self.pos.y > WINDOW_SIZE - self.size.y - margin:
            self.pos.y = WINDOW_SIZE - self.size.y - margin
            self.vel.y = -abs(self.vel.y) * 0.5

class Particle:
    __slots__ = ['pos', 'vel', 'force', 'density', 'pressure']
    
    def __init__(self, pos, vel=Vector2(0, 0)):
        self.pos = Vector2(pos)
        self.vel = Vector2(vel)
        self.force = Vector2(0, 0)
        self.density = 0
        self.pressure = 0
        
    def update(self, time_step):
        # Semi-implicit Euler integration
        self.vel += (self.force / max(self.density, 0.1)) * time_step
        self.vel *= 0.98
        self.pos += self.vel * time_step
        
        # Improved boundary conditions with corner handling
        margin = PARTICLE_RADIUS * 2
        damping = 0.7
        corner_margin = margin * 1.5  # Larger margin for corners
        
        # Check if in corner regions
        in_left = self.pos.x < margin
        in_right = self.pos.x > WINDOW_SIZE - margin
        in_top = self.pos.y < margin
        in_bottom = self.pos.y > WINDOW_SIZE - margin
        
        # Corner handling with stronger damping
        if (in_left and in_top) or (in_left and in_bottom) or \
           (in_right and in_top) or (in_right and in_bottom):
            corner_damping = 0.5  # Stronger damping in corners
            if abs(self.vel.x) > abs(self.vel.y):
                self.vel.x *= -corner_damping
            else:
                self.vel.y *= -corner_damping
            # Push away from corner
            if in_left:
                self.pos.x = corner_margin
            if in_right:
                self.pos.x = WINDOW_SIZE - corner_margin
            if in_top:
                self.pos.y = corner_margin
            if in_bottom:
                self.pos.y = WINDOW_SIZE - corner_margin
        else:
            # Regular boundary handling
            if in_left:
                self.pos.x = margin
                if self.vel.x < 0:
                    self.vel.x *= -damping
            elif in_right:
                self.pos.x = WINDOW_SIZE - margin
                if self.vel.x > 0:
                    self.vel.x *= -damping
                    
            if in_top:
                self.pos.y = margin
                if self.vel.y < 0:
                    self.vel.y *= -damping
            elif in_bottom:
                self.pos.y = WINDOW_SIZE - margin
                if self.vel.y > 0:
                    self.vel.y *= -damping
                    self.vel.x *= 0.95

class FluidSimulation:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Fluid Simulation")
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 16)
        
        self.particles = []
        self.boxes = []
        self.notifications = []
        self.time_step = BASE_TIME_STEP
        self.substeps = 1
        
        # Initial particles
        center = Vector2(WINDOW_SIZE/2, WINDOW_SIZE/3)
        radius = 50
        for _ in range(200):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius)
            pos = center + Vector2(r * np.cos(angle), r * np.sin(angle))
            self.particles.append(Particle(pos))

    def add_notification(self, text):
        self.notifications.append(NotificationText(text))

    def update_notifications(self):
        self.notifications = [n for n in self.notifications if not n.should_remove()]
        
    def draw_notifications(self):
        y_offset = 10
        for notification in self.notifications:
            text_surface = self.font.render(notification.text, True, (255, 255, 255))
            text_surface.set_alpha(notification.get_alpha())
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25

    def add_box(self, pos):
        self.boxes.append(PhysicsBox(pos))

    def remove_particles_at(self, pos, radius):
        self.particles = [p for p in self.particles 
                         if p.pos.distance_to(pos) > radius]

    def handle_box_particle_collision(self, box, particle, time_step):
        # Check if particle is inside box
        if (box.pos.x <= particle.pos.x <= box.pos.x + box.size.x and
            box.pos.y <= particle.pos.y <= box.pos.y + box.size.y):
            
            # Find closest edge
            dx_left = particle.pos.x - box.pos.x
            dx_right = (box.pos.x + box.size.x) - particle.pos.x
            dy_top = particle.pos.y - box.pos.y
            dy_bottom = (box.pos.y + box.size.y) - particle.pos.y
            
            min_dist = min(dx_left, dx_right, dy_top, dy_bottom)
            
            # Push particle out and transfer momentum
            if min_dist == dx_left:
                particle.pos.x = box.pos.x
                if particle.vel.x > 0:
                    box.force.x += particle.vel.x * particle.density
                    particle.vel.x *= -0.7
            elif min_dist == dx_right:
                particle.pos.x = box.pos.x + box.size.x
                if particle.vel.x < 0:
                    box.force.x += particle.vel.x * particle.density
                    particle.vel.x *= -0.7
            elif min_dist == dy_top:
                particle.pos.y = box.pos.y
                if particle.vel.y > 0:
                    box.force.y += particle.vel.y * particle.density
                    particle.vel.y *= -0.7
            else:  # dy_bottom
                particle.pos.y = box.pos.y + box.size.y
                if particle.vel.y < 0:
                    box.force.y += particle.vel.y * particle.density
                    particle.vel.y *= -0.7

    def calculate_density_pressure(self):
        h = PARTICLE_RADIUS * 4
        
        for p in self.particles:
            p.density = 0
            for other in self.particles:
                r = p.pos.distance_to(other.pos)
                if r < h:
                    factor = (1 - (r/h)**2)**3
                    p.density += PARTICLE_MASS * factor
            
            p.pressure = PRESSURE_CONSTANT * (p.density - REST_DENSITY)

    def calculate_forces(self):
        h = PARTICLE_RADIUS * 4
        
        for p in self.particles:
            p.force = GRAVITY * p.density
            
            for other in self.particles:
                if p == other:
                    continue
                    
                r = p.pos.distance_to(other.pos)
                if r < h and r > 0:
                    dir = (p.pos - other.pos) / r
                    
                    pressure_force = -dir * (p.pressure + other.pressure) / (2 * other.density) * (1 - r/h)**2
                    p.force += pressure_force * PARTICLE_MASS
                    
                    rel_vel = (other.vel - p.vel)
                    viscosity_force = rel_vel * VISCOSITY * (1 - r/h)
                    p.force += viscosity_force * PARTICLE_MASS
                    
                    if r < h * 0.5:
                        surface_force = dir * SURFACE_TENSION * (1 - r/(h * 0.5))**2
                        p.force += surface_force

    def solve_constraints(self):
        h = PARTICLE_RADIUS * 4
        for _ in range(4):
            for i, p1 in enumerate(self.particles):
                for j, p2 in enumerate(self.particles[i+1:], i+1):
                    r = p1.pos.distance_to(p2.pos)
                    if r < h and r > 0:
                        delta = (h - r) * 0.5
                        dir = (p2.pos - p1.pos) / r
                        p1.pos -= dir * delta
                        p2.pos += dir * delta

    def add_particles_at_mouse(self):
        if len(self.particles) >= MAX_PARTICLES:
            return
            
        mouse_pos = Vector2(pygame.mouse.get_pos())
        for _ in range(2):
            offset = Vector2(
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5)
            )
            pos = mouse_pos + offset
            vel = offset * 2
            self.particles.append(Particle(pos, vel))

    def apply_mouse_force(self, mouse_pos):
        for p in self.particles:
            dist = p.pos.distance_to(mouse_pos)
            if dist < MOUSE_INFLUENCE_RADIUS:
                force_dir = (mouse_pos - p.pos).normalize()
                force_strength = 200 * (1 - dist/MOUSE_INFLUENCE_RADIUS)
                p.force += force_dir * force_strength

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 3:  # Right click
                        self.add_box(Vector2(pygame.mouse.get_pos()))
                elif event.type == pygame.MOUSEWHEEL:
                    speed_factor = 1.2 if event.y > 0 else 0.8
                    self.time_step = max(BASE_TIME_STEP * 0.1, 
                                       min(BASE_TIME_STEP * 5.0, 
                                           self.time_step * speed_factor))
                    self.add_notification(f"Simulation Speed: {self.time_step/BASE_TIME_STEP:.1f}x")
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFTBRACKET:
                        self.substeps = max(1, self.substeps - 1)
                        self.add_notification(f"Substeps: {self.substeps}")
                    elif event.key == pygame.K_RIGHTBRACKET:
                        self.substeps = min(8, self.substeps + 1)
                        self.add_notification(f"Substeps: {self.substeps}")

            self.screen.fill((0, 0, 0))
            
            # Mouse interaction
            mouse_buttons = pygame.mouse.get_pressed()
            mouse_pos = Vector2(pygame.mouse.get_pos())
            if mouse_buttons[0]:  # Left click
                self.add_particles_at_mouse()
            elif mouse_buttons[1]:  # Middle click
                self.remove_particles_at(mouse_pos, REMOVAL_RADIUS)
            if pygame.mouse.get_focused():
                self.apply_mouse_force(mouse_pos)
            
            # Physics updates with substeps
            substep_dt = self.time_step / self.substeps
            for _ in range(self.substeps):
                self.calculate_density_pressure()
                self.calculate_forces()
                self.solve_constraints()
                
                # Update particles
                for p in self.particles:
                    p.update(substep_dt)
                
                # Update boxes and handle collisions
                for box in self.boxes:
                    box.update(substep_dt)
                    for particle in self.particles:
                        self.handle_box_particle_collision(box, particle, substep_dt)
            
            # Draw particles
            for p in self.particles:
                density_factor = min(1.0, p.density / 2.0)
                color = (
                    int(density_factor * 255),
                    0,
                    int((1 - density_factor) * 255)
                )
                pygame.draw.circle(self.screen, color, p.pos, PARTICLE_RADIUS)
            
            # Draw boxes
            for box in self.boxes:
                pygame.draw.rect(self.screen, (150, 150, 150), 
                               (box.pos.x, box.pos.y, box.size.x, box.size.y))
            
            # Update and draw notifications
            self.update_notifications()
            self.draw_notifications()
            
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    simulation = FluidSimulation()
    simulation.run()
