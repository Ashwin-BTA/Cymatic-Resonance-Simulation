import numpy as np
import matplotlib.pyplot as plt

class ChladniPlate:
    def __init__(self, resolution=500):
        """
        Initializes the 2D plate.
        - resolution: How many calculation points on the grid (500x500 is a good balance of detail and speed).
        """
        self.resolution = resolution
        # Create a 2D coordinate grid ranging from -1 to 1
        x = np.linspace(-1, 1, self.resolution)
        y = np.linspace(-1, 1, self.resolution)
        self.X, self.Y = np.meshgrid(x, y)

    def calculate_resonance(self, freq):
        """
        Calculates the vibration strength across the entire plate.
        Updated with aggressive scaling for highly dynamic shapes.
        """
        if freq <= 0:
            return np.zeros((self.resolution, self.resolution))

        # --- THE FIX: Aggressive Frequency Scaling ---
        # We divide by a much smaller number and add a higher baseline.
        # Now, 44 Hz becomes n = 3.1 (a complex shape) instead of 0.4.
        # 440 Hz becomes n = 13.0 (a highly intricate, dense web).
        n = (freq / 40.0) + 2.0  
        m = (freq / 30.0) + 4.0  
        
        # We also introduce a slight non-linear tweak to make the morphing 
        # between notes feel more chaotic and less uniform.
        n += np.sin(freq / 100.0) 
        
        # The Chladni Plate Equation
        term1 = np.cos(n * np.pi * self.X) * np.cos(m * np.pi * self.Y)
        term2 = np.cos(m * np.pi * self.X) * np.cos(n * np.pi * self.Y)
        
        Z = term1 - term2
        
        # Increased the exponent to pack the sand tighter
        return np.abs(Z) ** 2.5

# --- Test the module ---
if __name__ == "__main__":
    plate = ChladniPlate(resolution=500)
    
    # Let's test it with a standard A4 note (440 Hz)
    test_freq = 440.0
    
    print(f"Calculating resonance map for {test_freq} Hz...")
    vibration_map = plate.calculate_resonance(test_freq)
    
    # Visualize the math
    plt.figure(figsize=(6, 6))
    # Using a colormap where dark represents 0 vibration (where sand will go)
    plt.imshow(vibration_map, cmap='hot', origin='lower')
    plt.title(f"Chladni Resonance Map ({test_freq} Hz)")
    plt.colorbar(label='Vibration Amplitude')
    plt.show()