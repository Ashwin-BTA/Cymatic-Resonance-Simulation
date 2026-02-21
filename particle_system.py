import numpy as np

class ParticleSystem:
    # Let's jump to 100,000 particles. 
    def __init__(self, num_particles=100000): 
        self.num_particles = num_particles
        
        # Position arrays
        self.x = np.random.uniform(-1, 1, self.num_particles)
        self.y = np.random.uniform(-1, 1, self.num_particles)
        
        # Velocity arrays (This is the 180IQ shift)
        self.vx = np.zeros(self.num_particles)
        self.vy = np.zeros(self.num_particles)

    def update(self, vibration_map, current_rms):
        res = vibration_map.shape[0]

        grid_x = np.clip(np.int32((self.x + 1) * (res - 1) / 2), 0, res - 1)
        grid_y = np.clip(np.int32((self.y + 1) * (res - 1) / 2), 0, res - 1)

        grad_y, grad_x = np.gradient(vibration_map)
        p_grad_x = grad_x[grid_y, grid_x]
        p_grad_y = grad_y[grid_y, grid_x]
        p_amplitude = vibration_map[grid_y, grid_x]

        # --- 1. DENSITY & WEIGHT CALCULATION ---
        # Create a blank map and count how many particles are in each pixel
        flat_indices = grid_y * res + grid_x
        counts = np.bincount(flat_indices, minlength=res*res)
        p_density = counts[flat_indices]
        
        # HALVED the weight penalty so the sand isn't quite as heavy
        p_weight = 1.0 + (p_density * 0.05)

        # --- 2. APPLY FORCES ---
        friction = 0.88 # Slightly looser so they can slide around the lines
        acceleration = 0.75 # Relaxed the vacuum force slightly
        
        self.vx = (self.vx * friction) - (p_grad_x * acceleration)
        self.vy = (self.vy * friction) - (p_grad_y * acceleration)

        # Base explosion force from the audio beat

        base_vibration = 0.005 + (current_rms * 0.04) 
        dynamic_vibration = base_vibration / p_weight
        
        self.vx += np.random.uniform(-1, 1, self.num_particles) * (p_amplitude * dynamic_vibration)
        self.vy += np.random.uniform(-1, 1, self.num_particles) * (p_amplitude * dynamic_vibration)

        self.x += self.vx
        self.y += self.vy

        out_of_bounds_x = (self.x < -1) | (self.x > 1)
        out_of_bounds_y = (self.y < -1) | (self.y > 1)
        
        # Changed from -0.5 to -1.0 so they bounce off the wall with full energy
        self.vx[out_of_bounds_x] *= -1.0 
        self.vy[out_of_bounds_y] *= -1.0
        
        # Kept strictly at 0.99 to push them off the exact edge and back into the gradient
        self.x = np.clip(self.x, -0.99, 0.99) 
        self.y = np.clip(self.y, -0.99, 0.99)

        return self.x, self.y, self.vx, self.vy

    def reset_positions(self):
        self.x = np.random.uniform(-1, 1, self.num_particles)
        self.y = np.random.uniform(-1, 1, self.num_particles)
        self.vx.fill(0)
        self.vy.fill(0)