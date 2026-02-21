import pygame
import sys
import numpy as np

# Import our custom modules from Days 1-3
from audio_processor import extract_frequencies
from resonance_engine import ChladniPlate
from particle_system import ParticleSystem

# --- Configuration ---
# --- Configuration ---
FPS =25
WINDOW_SIZE = 800
AUDIO_FILE = "Hz.wav"
NUM_PARTICLES = 100000

def main():
    print("Initializing Simulation... Please wait.")
    
    # 1. Pre-process the audio (Day 1)
    try:
        # Note the new rms_data variable here
        frequencies, rms_data, track_length = extract_frequencies(AUDIO_FILE, target_fps=FPS)
    except Exception as e:
        print(f"Error loading audio: {e}")

        return

    # 2. Initialize the Engines (Day 2 & Day 3)
    plate = ChladniPlate(resolution=WINDOW_SIZE)
    particles = ParticleSystem(num_particles=NUM_PARTICLES)

    # 3. Initialize Pygame (The Stage)
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Chladni Resonance Simulation")
    clock = pygame.time.Clock()

    # Load and play the audio
    pygame.mixer.music.load(AUDIO_FILE)
    pygame.mixer.music.play()

    running = True
    while running:
        # --- 1. Event Handling (Input) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- 2. Synchronization (Timekeeping) ---
        # Get the current song time in milliseconds
        current_time_ms = pygame.mixer.music.get_pos()
        
        # If the song finishes, get_pos() returns -1
        if current_time_ms == -1:
            print("Audio finished. Ending simulation.")
            break

        # Convert milliseconds to our current frame index based on 30 FPS
        current_frame = int((current_time_ms / 1000.0) * FPS)

        # Prevent index out of bounds if the math rounds up at the very end of the track
        if current_frame >= len(frequencies):
            current_frame = len(frequencies) - 1

        current_freq = frequencies[current_frame]
        current_rms = rms_data[current_frame]

        # --- 3. Physics Updates (The Brain & The Sand) ---
        # Calculate the 2D map for the exact frequency playing right now
        vibration_map = plate.calculate_resonance(current_freq)
        
        # Update the 6000 sand grains based on the newly calculated map
        p_x, p_y, p_vx, p_vy = particles.update(vibration_map, current_rms)

        # --- 4. Rendering (Drawing to Screen) ---
        # --- 4. Rendering (Drawing to Screen) ---
       # --- 4. Rendering (The 180IQ Array Broadcast) ---
        # --- 4. Rendering (Kinetic Color Mapping) ---
        screen.fill((10, 10, 15)) 

        # Mathematically map coordinates strictly to the visible 0-799 pixel range
        px = np.clip(np.int32((p_x + 1) * ((WINDOW_SIZE - 1) / 2)), 0, WINDOW_SIZE - 1)
        py = np.clip(np.int32((p_y + 1) * ((WINDOW_SIZE - 1) / 2)), 0, WINDOW_SIZE - 1)

        pvx, pvy = p_vx, p_vy 

        # (Keep your speed calculation and color mapping exactly the same below this)
        speed = np.sqrt(pvx**2 + pvy**2)
        speed_norm = np.clip(speed * 3.0, 0, 1)

        colors = np.zeros((len(px), 3), dtype=np.uint8)
        colors[:, 0] = 255 - (speed_norm * 205) # Red channel
        colors[:, 1] = 215 - (speed_norm * 215) # Green channel
        colors[:, 2] = speed_norm * 100



        # Lock the screen memory and give NumPy direct access
        # (Inside the Rendering loop)
        pixel_array = pygame.surfarray.pixels3d(screen)
        
        pixel_array[px, py] = colors
        
        px1 = np.clip(px + 1, 0, WINDOW_SIZE - 1)
        py1 = np.clip(py + 1, 0, WINDOW_SIZE - 1)
        
        # Thicker 2x2 grains
        pixel_array[px1, py] = colors
        pixel_array[px, py1] = colors
        pixel_array[px1, py1] = colors

        del pixel_array

        # HUD Text
        font = pygame.font.SysFont(None, 36)

       # HUD Text
        font = pygame.font.SysFont(None, 36)
        freq_text = font.render(f"Freq: {current_freq:.1f} Hz", True, (100, 255, 100))
        controls_text = font.render("SPACE: Play/Pause | R: Scramble", True, (150, 150, 150))
        
        # --- NEW: FPS Counter ---
        actual_fps = clock.get_fps()
        # Turns red if you are lagging, green if you are hitting your target
        fps_color = (255, 100, 100) if actual_fps < (FPS - 2) else (100, 255, 100)
        fps_text = font.render(f"FPS: {actual_fps:.1f} / {FPS}", True, fps_color)
        
        screen.blit(freq_text, (20, 20))
        screen.blit(controls_text, (20, 60))
        screen.blit(fps_text, (20, 100)) # Draws right below the controls

        pygame.display.flip()
        clock.tick(FPS)

        # Push the drawing to your monitor and cap the loop at 30 FPS
        pygame.display.flip()
        clock.tick(FPS)

    # Clean exit
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()