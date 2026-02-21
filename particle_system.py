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
       # --- 1. DENSITY & WEIGHT CALCULATION (OPTIMIZED) ---
        flat_indices = grid_y * res + grid_x
        counts = np.bincount(flat_indices, minlength=res*res)
        p_density = counts[flat_indices]
        
        # THE FIX: Calculate the raw weight, but cap it at 2.5 maximum.
        # This prevents the particles from creating inescapable black holes!
        raw_weight = 1.0 + (p_density * 0.05)
        p_weight = np.clip(raw_weight, 1.0, 2.5) 

        # --- 2. APPLY FORCES (BALANCED FOR 25 FPS) ---
        # Increased friction closer to 1.0 to account for more frames per second
        friction = 0.92 
        
        # Decreased direct pull and curl power so total movement per second stays the same
        direct_pull = 0.44 
        curl_power = 0.28 
        
        force_x = (p_grad_x * direct_pull) + (p_grad_y * curl_power)
        force_y = (p_grad_y * direct_pull) - (p_grad_x * curl_power)
        
        self.vx = (self.vx * friction) - force_x
        self.vy = (self.vy * friction) - force_y

        # Scaled down slightly so the 25 FPS drum hits maintain the same total shatter energy
        base_vibration = 0.008 + (current_rms * 0.048) 
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