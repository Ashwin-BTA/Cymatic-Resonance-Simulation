import librosa
import numpy as np
import os

def extract_frequencies(audio_path, target_fps=45):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    # New cache name to force a re-calc with RMS included
    cache_file = f"{base_name}_{target_fps}fps_rms_cache.npy"
    
    if os.path.exists(cache_file):
        print(f"Cache found! Loading frequencies and volume from {cache_file}...")
        data = np.load(cache_file)
        synced_frequencies = data[0]
        synced_rms = data[1]
        duration = len(synced_frequencies) / target_fps
        return synced_frequencies, synced_rms, duration

    print(f"No cache found. Loading audio file: {audio_path}...")
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    print("Extracting fundamental frequencies (this may take a minute)...")
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    f0 = np.nan_to_num(f0, nan=0.0)

    print("Extracting volume (RMS)...")
    rms = librosa.feature.rms(y=y)[0]
    # Normalize RMS to a strict 0.0 to 1.0 scale so it's predictable in the physics engine
    rms = rms / np.max(rms)

    num_visual_frames = int(duration * target_fps)
    
    original_times_f0 = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
    visual_times = np.linspace(0, duration, num_visual_frames)
    synced_frequencies = np.interp(visual_times, original_times_f0, f0)

    # Interpolate the RMS array to match our 45 FPS framerate
    original_times_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    synced_rms = np.interp(visual_times, original_times_rms, rms)
    
    # Save both arrays into a single cache file
    print(f"Caching data to {cache_file}...")
    np.save(cache_file, np.array([synced_frequencies, synced_rms]))
    
    return synced_frequencies, synced_rms, duration