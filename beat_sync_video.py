import os
import random
import sys
from pathlib import Path

import librosa
import moviepy.editor as mvpy
import numpy as np
from scenedetect import (
    AdaptiveDetector,
    ContentDetector,
    HistogramDetector,
    ThresholdDetector,
    detect,
)

# Import transitions
sys.path.append("Transitions")  # Add Transitions folder to path
# Import configuration
from config import *
from Transitions.transitions import (
    blur_transition,
    dissolve,
    slide_transition,
    zoom_transition,
)


def try_all_detectors(video_path):
    """
    Try all detectors and return their results
    """
    detectors = {
        "histogram": HistogramDetector(
            threshold=HISTOGRAM_THRESHOLD, min_scene_len=MIN_SCENE_LEN
        ),
        "content": ContentDetector(
            threshold=CONTENT_THRESHOLD, min_scene_len=MIN_SCENE_LEN
        ),
        "adaptive": AdaptiveDetector(
            adaptive_threshold=ADAPTIVE_THRESHOLD, min_scene_len=MIN_SCENE_LEN
        ),
        "threshold": ThresholdDetector(
            threshold=THRESHOLD_DETECTOR, min_scene_len=MIN_SCENE_LEN
        ),
    }

    results = {}
    for name, detector in detectors.items():
        detected_scenes = detect(video_path, detector)
        scenes = []
        for scene in detected_scenes:
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            if end_time - start_time >= MIN_SCENE_DURATION:
                scenes.append((start_time, end_time))
        results[name] = (len(scenes), detector, scenes)
        print(f"{name.capitalize()} detector found {len(scenes)} scenes")

    return results


def choose_detector(tempo, bass_strength, onset_strength, video_path):
    """
    Choose the most appropriate scene detector based on audio characteristics and test results
    """
    print("\nTesting all detectors first...")
    detector_results = try_all_detectors(video_path)

    # Filter out detectors that found 0 scenes
    valid_detectors = {
        name: (count, detector, scenes)
        for name, (count, detector, scenes) in detector_results.items()
        if count > 0
    }

    if not valid_detectors:
        print("No valid detectors found, using default HistogramDetector")
        return HistogramDetector(threshold=0.03, min_scene_len=15), []

    # If only one detector found scenes, use it
    if len(valid_detectors) == 1:
        name = next(iter(valid_detectors))
        print(f"Only {name} detector found scenes ({valid_detectors[name][0]} scenes)")
        return valid_detectors[name][1], valid_detectors[name][2]

    # Remove detector with maximum scenes if it found significantly more scenes
    scene_counts = [count for count, _, _ in valid_detectors.values()]
    max_scenes = max(scene_counts)
    median_scenes = sorted(scene_counts)[len(scene_counts) // 2]

    if max_scenes > median_scenes * 2:  # If max is more than double the median
        valid_detectors = {
            name: (count, detector, scenes)
            for name, (count, detector, scenes) in valid_detectors.items()
            if count < max_scenes
        }

    print("\nSelecting from valid detectors based on audio characteristics...")

    # Strong bass with high onset strength suggests color changes
    if bass_strength > 20 and onset_strength > 0.8 and "histogram" in valid_detectors:
        print("Strong bass with high onset strength - using HistogramDetector")
        return valid_detectors["histogram"][1], valid_detectors["histogram"][2]

    # High tempo suggests fast cuts
    if tempo > 90 and "content" in valid_detectors:
        print("High tempo detected - using ContentDetector")
        return valid_detectors["content"][1], valid_detectors["content"][2]

    # Strong bass suggests fades
    if bass_strength > 20 and "threshold" in valid_detectors:
        print("Strong bass detected - using ThresholdDetector")
        return valid_detectors["threshold"][1], valid_detectors["threshold"][2]

    # Moderate onsets suggests varying content
    if onset_strength > 0.5 and "adaptive" in valid_detectors:
        print("Moderate onsets detected - using AdaptiveDetector")
        return valid_detectors["adaptive"][1], valid_detectors["adaptive"][2]

    # If no specific detector matches, use the one with median number of scenes
    scene_counts = [(count, name) for name, (count, _, _) in valid_detectors.items()]
    scene_counts.sort()
    median_detector_name = scene_counts[len(scene_counts) // 2][1]
    print(f"Using {median_detector_name} detector as median choice")
    return valid_detectors[median_detector_name][1], valid_detectors[
        median_detector_name
    ][2]


def detect_scenes(video_path, detector=None):
    """
    Detect scenes in the video using the provided or default detector
    """
    print("\nDetecting scenes in video...")

    # Use provided detector or default to HistogramDetector
    if detector is None:
        detector = HistogramDetector(threshold=0.05, min_scene_len=15)

    # Detect scenes with the chosen detector
    detected_scenes = detect(video_path, detector)

    # Convert scenes to time ranges
    all_scenes = []
    for scene in detected_scenes:
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        if end_time - start_time >= 0.5:  # Minimum scene duration
            all_scenes.append((start_time, end_time))

    print(f"Detected {len(all_scenes)} scenes")

    # If we don't get enough scenes, try more sensitive settings
    if len(all_scenes) < 5:
        print("Fewer scenes than expected, trying more sensitive settings...")
        if isinstance(detector, HistogramDetector):
            detector = HistogramDetector(threshold=0.03, min_scene_len=15)
        elif isinstance(detector, ContentDetector):
            detector = ContentDetector(threshold=20)
        elif isinstance(detector, AdaptiveDetector):
            detector = AdaptiveDetector(adaptive_threshold=2.0)
        elif isinstance(detector, ThresholdDetector):
            detector = ThresholdDetector(threshold=25)

        detected_scenes = detect(video_path, detector)
        current_scenes = []
        for scene in detected_scenes:
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            if end_time - start_time >= 0.5:
                current_scenes.append((start_time, end_time))

        if len(current_scenes) > len(all_scenes):
            all_scenes = current_scenes
            print(f"Found {len(all_scenes)} scenes with more sensitive threshold")

    all_scenes.sort()
    return all_scenes


def detect_beats(audio_path, sensitivity=1.0, algorithm="default"):
    """
    Detect beats in an audio file using librosa with adjustable sensitivity.
    Now includes downbeat detection for stronger beats.
    """
    y, sr = librosa.load(audio_path)

    # Get onset envelope for beat tracking
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, start_bpm=120, tightness=100 * sensitivity
    )

    # Convert to beat times for all algorithms
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if algorithm == "onset":
        # For onset, we'll use the raw beat times but filter by onset strength
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        onset_times = librosa.times_like(onset_strength, sr=sr, hop_length=512)

        # Find peaks in onset strength
        peaks = librosa.util.peak_pick(
            onset_strength,
            pre_max=20,
            post_max=20,
            pre_avg=20,
            post_avg=20,
            delta=0.2,
            wait=20,
        )

        # Get times of strong onsets
        strong_onset_times = onset_times[peaks]

        # Only keep beats that align with strong onsets
        valid_beats = []
        for beat in beat_times:
            # Find closest onset
            closest_onset = strong_onset_times[
                np.argmin(np.abs(strong_onset_times - beat))
            ]
            if np.abs(closest_onset - beat) < 0.05:  # Within 50ms
                valid_beats.append(beat)

        beat_times = np.array(valid_beats)

    elif algorithm == "downbeat":
        # Get harmonic and percussive components for better downbeat detection
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Compute mel spectrogram for harmonic part
        mel_spec = librosa.feature.melspectrogram(y=y_harmonic, sr=sr)

        # Get tempo-synced onset envelope
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=512, aggregate=np.median
        )

        # Detect downbeats using multiple features
        downbeat_strength = []

        for beat_time in beat_times:
            # Get frame index for the current beat
            frame_idx = librosa.time_to_frames(beat_time, sr=sr)
            if frame_idx >= len(onset_env):
                continue

            # Get features at this beat
            onset_str = onset_env[frame_idx]

            # Spectral flux (change in spectrum)
            if frame_idx > 0 and frame_idx < len(mel_spec[0]):
                spec_flux = np.sum(
                    np.diff(mel_spec[:, frame_idx - 1 : frame_idx + 1], axis=1)
                )
            else:
                spec_flux = 0

            # Bass strength at this moment
            bass_idx = frame_idx if frame_idx < mel_spec.shape[1] else -1
            bass_strength = np.sum(mel_spec[:10, bass_idx])  # First 10 mel bands

            # Combine features with weights tuned for hip-hop
            strength = (
                onset_str * 0.4  # Less weight on pure onset
                + spec_flux * 0.3  # More weight on spectral change
                + bass_strength * 0.3
            )  # More weight on bass

            downbeat_strength.append(strength)

        # Normalize strengths
        if downbeat_strength:
            downbeat_strength = np.array(downbeat_strength)
            downbeat_strength = (downbeat_strength - np.min(downbeat_strength)) / (
                np.max(downbeat_strength) - np.min(downbeat_strength)
            )

            # For hip-hop, we want more frequent changes, so take top 40% as downbeats
            threshold = np.percentile(downbeat_strength, 60)  # Top 40%
            downbeat_mask = downbeat_strength > threshold

            # Keep only the stronger beats
            beat_times = beat_times[downbeat_mask]

    print(f"Detected {len(beat_times)} beats at tempo {float(tempo):.2f} BPM")
    return beat_times, y, sr


def analyze_audio(audio_path):
    """
    Analyze audio to determine the best parameters for beat detection

    Parameters:
        audio_path (str): Path to the audio file

    Returns:
        tuple: (algorithm, sensitivity, tempo)
    """
    global AUDIO_FEATURES  # Add this line to store results globally

    print("Analyzing audio for beat detection...")
    y, sr = librosa.load(audio_path)

    # Analyze bass frequencies
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    bass_strength = np.mean(spec_contrast[0])

    # Analyze overall onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_strength = np.mean(onset_env)

    # Analyze percussive elements
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    percussive_strength = np.mean(np.abs(y_percussive))

    # Calculate tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)  # Convert to float for formatting

    # Choose algorithm and sensitivity based on audio characteristics
    algorithm = "downbeat"  # Always use downbeat detection as requested

    # Adjust sensitivity based on song characteristics
    # For faster songs, use higher sensitivity
    if tempo > 140:  # Very fast
        sensitivity = 1.3
    elif tempo > 120:  # Fast
        sensitivity = 1.2
    elif tempo > 100:  # Moderate-fast
        sensitivity = 1.1
    elif tempo > 85:  # Moderate
        sensitivity = 1.0
    else:  # Slow
        sensitivity = 0.9

    # Adjust based on bass strength - stronger bass needs higher sensitivity
    if bass_strength > 25:  # Very strong bass
        sensitivity += 0.2
    elif bass_strength > 20:  # Strong bass
        sensitivity += 0.1

    # Store audio analysis results in a global dictionary for transition selection
    AUDIO_FEATURES = {
        "tempo": tempo,
        "bass_strength": float(bass_strength),
        "onset_strength": float(onset_strength),
        "percussive_strength": float(percussive_strength),
    }

    # Print analysis results
    print("Audio analysis results:")
    print(f"Tempo: {tempo:.1f} BPM")
    print(f"Bass strength: {bass_strength:.1f}")
    print(f"Onset strength: {onset_strength:.1f}")
    print(f"Percussive strength: {percussive_strength:.1f}")
    print(f"Using {algorithm} detection")
    print(f"Sensitivity: {sensitivity:.1f}")

    return algorithm, sensitivity, tempo


def select_transition(beat_index, beat_times, audio_features, prev_clip, next_clip):
    """
    Select an appropriate transition type based on beat characteristics and audio features.
    """

    print(f"Selecting transition for beat {beat_index}")
    # Default transition (hard cut - no transition)
    transition_func = None
    duration = 0.0
    transition_name = "Hard Cut"
    reason = "Default"

    # Skip transition for the first few beats to establish rhythm
    if beat_index < 3:
        reason = "First few beats - establishing rhythm"
        print(f"Beat {beat_index}: Using {transition_name} ({reason})")
        return transition_func, duration

    # Get audio characteristics
    tempo = audio_features.get("tempo", 120)
    bass_strength = audio_features.get("bass_strength", 15)
    onset_strength = audio_features.get("onset_strength", 1.0)
    percussive_strength = audio_features.get("percussive_strength", 0.1)

    # Check if this is a downbeat (every 4th beat in common time signatures)
    is_downbeat = beat_index % 4 == 0

    # Determine transition duration based on tempo
    # Faster tempo = shorter transitions
    base_duration = 60 / tempo  # One beat duration
    max_duration = min(prev_clip.duration / 3, next_clip.duration / 3, 1.0)

    # For fast music, keep transitions shorter
    if tempo > 140:
        duration = min(base_duration * 0.6, max_duration)
        duration_reason = "Very fast tempo"
    elif tempo > 100:
        duration = min(base_duration * 0.8, max_duration)
        duration_reason = "Fast tempo"
    else:
        duration = min(base_duration, max_duration)
        duration_reason = "Moderate/slow tempo"

    # REDUCED skip chances
    if is_downbeat:
        transition_chance = 0.8  # 80% chance on downbeats (unchanged)
        chance_type = "downbeat"
    else:
        transition_chance = 0.7  # 70% chance on regular beats (increased from 40%)
        chance_type = "regular beat"

    if random.random() > transition_chance:
        reason = f"Random skip ({chance_type}, {transition_chance * 100:.0f}% chance)"
        print(f"Beat {beat_index}: Using {transition_name} ({reason})")
        return transition_func, duration

    # Detect phrase changes (every 16 beats)
    is_phrase_change = beat_index % 16 == 0

    # Audio feature-based transition selection
    if is_phrase_change and bass_strength > 20:
        # Major phrase changes with strong bass - dramatic transitions
        transition_options = [
            (zoom_transition, 0.5, "Zoom", "Phrase change - dramatic impact"),
            (slide_transition, 0.3, "Slide", "Phrase change - directional movement"),
            (blur_transition, 0.2, "Blur", "Phrase change - defocus effect"),
        ]
        beat_type = "Phrase change"
    elif bass_strength > 25 and onset_strength > 1.5:
        # Strong bass + sharp onset - high energy transitions
        transition_options = [
            (zoom_transition, 0.6, "Zoom", "High energy - dramatic impact"),
            (blur_transition, 0.3, "Blur", "High energy - defocus effect"),
            (slide_transition, 0.1, "Slide", "High energy - directional movement"),
        ]
        beat_type = "High energy moment"
    elif onset_strength > 2.0:
        # Sharp onset - prioritize blur transitions
        transition_options = [
            (blur_transition, 0.7, "Blur", "Sharp onset - defocus effect"),
            (dissolve, 0.2, "Dissolve", "Sharp onset - smooth blend"),
            (slide_transition, 0.1, "Slide", "Sharp onset - directional movement"),
        ]
        beat_type = "Sharp onset moment"
    elif is_downbeat and bass_strength > 20:
        # Strong downbeat - mix of transitions
        transition_options = [
            (zoom_transition, 0.4, "Zoom", "Strong downbeat - moderate impact"),
            (dissolve, 0.3, "Dissolve", "Strong downbeat - smooth blend"),
            (slide_transition, 0.2, "Slide", "Strong downbeat - directional movement"),
            (blur_transition, 0.1, "Blur", "Strong downbeat - defocus effect"),
        ]
        beat_type = "Strong downbeat"
    elif is_downbeat:
        # Regular downbeat - smoother transitions
        transition_options = [
            (dissolve, 0.5, "Dissolve", "Downbeat - smooth blend"),
            (blur_transition, 0.3, "Blur", "Downbeat - defocus effect"),
            (slide_transition, 0.2, "Slide", "Downbeat - subtle movement"),
        ]
        beat_type = "Normal downbeat"
    else:
        # Regular beats - mostly dissolves with occasional effects
        transition_options = [
            (dissolve, 0.7, "Dissolve", "Regular beat - smooth blend"),
            (blur_transition, 0.2, "Blur", "Regular beat - subtle defocus"),
            (slide_transition, 0.1, "Slide", "Regular beat - subtle movement"),
        ]
        beat_type = "Regular beat"

    # Choose transition weighted by their probability
    weights = [weight for _, weight, _, _ in transition_options]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Select transition based on weights
    selected_index = np.random.choice(len(transition_options), p=normalized_weights)
    transition_func, _, transition_name, reason = transition_options[selected_index]

    # For slide transitions, select direction based on musical features and context
    if transition_func == slide_transition:
        # Track consecutive slide count to prevent overuse
        if not hasattr(select_transition, "last_slide_beat"):
            select_transition.last_slide_beat = -4  # Initialize
            select_transition.consecutive_slides = 0

        # If we used a slide recently, consider switching to dissolve
        beats_since_last_slide = beat_index - select_transition.last_slide_beat
        if beats_since_last_slide < 4 or select_transition.consecutive_slides >= 2:
            print("  → Avoiding overuse of slides, switching to dissolve")
            transition_func = dissolve
            transition_name = "Dissolve"
            reason += " (slide avoided)"
            select_transition.consecutive_slides = 0
        else:
            select_transition.last_slide_beat = beat_index
            select_transition.consecutive_slides += 1

            # Choose direction based on musical features and phrase position
            if is_phrase_change:
                # Use dramatic vertical movements for phrase changes
                directions = ["bottom", "top"]
            elif bass_strength > 20 and onset_strength > 1.5:
                # Strong bass + sharp onset - use dramatic horizontal movement
                directions = ["left", "right"]
            elif percussive_strength > 0.15:
                # Strong percussive elements - use rhythmic horizontal movement
                directions = ["left", "right"]
            else:
                # Default - prefer horizontal movement for smoother flow
                directions = ["left", "right"] * 3 + ["top", "bottom"]

            direction = random.choice(directions)

            # Adjust duration based on direction and context
            if direction in ["left", "right"]:
                slide_duration_factor = 1.2  # Slightly longer for horizontal slides
            else:
                slide_duration_factor = 1.1  # Slightly longer for vertical slides

            duration = min(duration * slide_duration_factor, max_duration)
            transition_name = f"Slide ({direction})"
            transition_func = lambda c1, c2, d: slide_transition(
                c1, c2, d, direction=direction
            )

    # Print transition information
    print(
        f"Beat {beat_index}: Using {transition_name} - Duration: {duration:.2f}s ({reason})"
    )
    print(
        f"  → Audio stats: Tempo={tempo:.1f} BPM, Bass={bass_strength:.1f}, Onset={onset_strength:.1f}, Percussive={percussive_strength:.1f}"
    )

    print(f"  → Transition function: {transition_func}")

    return transition_func, duration


def create_video_from_segments(
    scenes,
    beat_times,
    video_path,
    audio_path,
    target_duration,
    audio_start_time,
    audio_features,
):
    """
    Create a beat-synchronized video from detected scenes and beats.

    Args:
        scenes: List of detected scenes
        beat_times: List of beat times in seconds
        video_path: Path to the source video
        audio_path: Path to the audio file
        target_duration: Target duration for the final video
        audio_start_time: Start time in the audio file (in seconds)
        audio_features: Dictionary containing audio analysis results

    Returns:
        Path to the output video file
    """
    video = mvpy.VideoFileClip(video_path)

    # Calculate average beat duration
    if len(beat_times) > 1:
        avg_beat_duration = np.mean(np.diff(beat_times))
    else:
        avg_beat_duration = 60 / audio_features.get("tempo", 120)

    print(f"\nAverage beat duration: {avg_beat_duration:.3f}s")

    # Track scene usage to ensure variety
    scene_usage = {i: 0 for i in range(len(scenes))}

    clips = []
    current_duration = 0
    beat_index = 0

    # Loop until we reach target duration or run out of beats
    # Add a small buffer to ensure we reach the target duration
    target_with_buffer = target_duration * 1.1
    while current_duration < target_with_buffer and beat_index < len(beat_times):
        # Select scene with least usage
        available_scenes = sorted(scene_usage.keys(), key=lambda k: scene_usage[k])

        if len(available_scenes) == 0:
            print("\nAll scenes used, resetting scene usage tracking")
            scene_usage = {i: 0 for i in range(len(scenes))}
            available_scenes = list(scene_usage.keys())

        scene_index = available_scenes[0]
        scene = scenes[scene_index]

        # Update scene usage
        scene_usage[scene_index] += 1

        # Get scene boundaries
        start_time, end_time = scene
        scene_duration = end_time - start_time

        # Ensure minimum clip duration
        beats_to_use = min(
            MAX_BEATS_PER_SCENE, max(1, int(scene_duration / avg_beat_duration))
        )

        # Make sure we don't go beyond available beats
        beats_to_use = min(beats_to_use, len(beat_times) - beat_index)

        if beats_to_use <= 0:
            break

        # Extract clip for each beat
        for i in range(beats_to_use):
            if beat_index + i >= len(beat_times):
                break

            # Determine clip start and end based on current beat
            if i == 0:
                clip_start = start_time
            else:
                # For subsequent beats in the same scene, start from a random point
                max_start = end_time - MIN_CLIP_DURATION
                min_start = start_time + (i - 1) * (scene_duration / beats_to_use)
                clip_start = min(
                    max_start,
                    max(
                        min_start,
                        start_time
                        + i * (scene_duration / beats_to_use)
                        - MIN_CLIP_DURATION,
                    ),
                )

            # Determine clip duration - aim to match the beat
            next_beat_index = beat_index + i + 1

            # Calculate the target end time for this clip
            if next_beat_index < len(beat_times):
                target_clip_duration = (
                    beat_times[next_beat_index] - beat_times[beat_index + i]
                )
            else:
                target_clip_duration = avg_beat_duration

            # Don't make clips too short
            target_clip_duration = max(target_clip_duration, MIN_CLIP_DURATION)

            # Ensure clip doesn't exceed scene boundary
            clip_end = min(end_time, clip_start + target_clip_duration * 1.5)

            # Ensure minimum clip duration
            if clip_end - clip_start < MIN_CLIP_DURATION:
                clip_end = min(end_time, clip_start + MIN_CLIP_DURATION)

            # Create the clip
            try:
                print(
                    f"\nProcessing scene {scene_index + 1}/{len(scenes)} at beat {beat_index + i}/{len(beat_times)}"
                )
                print(
                    f"  Scene time: {start_time:.2f}s - {end_time:.2f}s (duration: {scene_duration:.2f}s)"
                )
                print(
                    f"  Clip time: {clip_start:.2f}s - {clip_end:.2f}s (duration: {clip_end - clip_start:.2f}s)"
                )

                clip = video.subclip(clip_start, clip_end)

                # Apply transition if this is not the first clip
                if len(clips) > 0:
                    prev_clip = clips[-1]

                    print(f"  Selecting transition for segment {len(clips)}")
                    # Select appropriate transition
                    transition_func, transition_duration = select_transition(
                        beat_index + i, beat_times, audio_features, prev_clip, clip
                    )

                    # Apply transition if selected
                    if transition_func and transition_duration > 0:
                        try:
                            print(
                                f"  Applying transition with duration {transition_duration:.2f}s"
                            )
                            # Replace the last clip with the transition
                            clips[-1] = transition_func(
                                prev_clip, clip, transition_duration
                            )
                        except Exception as e:
                            print(f"  Transition failed: {str(e)}, using hard cut")
                            clips.append(clip)
                    else:
                        print("  No transition selected, using hard cut")
                        clips.append(clip)
                else:
                    print("  First clip - no transition")
                    clips.append(clip)

                # Update current duration
                current_duration += clip.duration
                print(
                    f"  Current total duration: {current_duration:.2f}s / {target_duration:.2f}s"
                )

            except Exception as e:
                print(f"Error creating clip: {str(e)}")
                continue

        # Move to the next beat
        beat_index += beats_to_use

    # Check if we have enough clips
    if not clips:
        print("No clips were created. Check video and beat detection.")
        return None

    print(f"\nCreated {len(clips)} segments, total duration: {current_duration:.1f}s")

    # Concatenate clips
    print("\nConcatenating clips...")
    final_video = mvpy.concatenate_videoclips(clips)

    # Trim to target duration if needed
    if final_video.duration > target_duration:
        final_video = final_video.subclip(0, target_duration)

    # Add music
    print("\nAdding music...")
    audio = mvpy.AudioFileClip(audio_path).subclip(
        audio_start_time, audio_start_time + target_duration
    )
    final_video = final_video.set_audio(audio)

    # Create output directory
    video_name = os.path.basename(video_path).split(".")[0]
    audio_name = os.path.basename(audio_path).split(".")[0]
    output_dir = f"test_videos/{video_name}_x_{audio_name}"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/{video_name}_synced_to_{audio_name}.mp4"

    # Write final video
    print(f"\nWriting final video to {output_path}...")
    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=OUTPUT_FPS,
        threads=OUTPUT_THREADS,
    )

    print(f"\nVideo created successfully: {output_path}")
    print(f"Final duration: {final_video.duration:.1f}s")

    # Clean up
    video.close()
    final_video.close()
    for clip in clips:
        try:
            clip.close()
        except:
            pass

    return output_path


def create_beat_synced_video(
    video_path,
    audio_path,
    output_path=None,
    sensitivity=None,
    algorithm=None,
    min_clip_duration=MIN_CLIP_DURATION,
    target_duration=None,
    audio_start_time=0.0,
):
    """
    Create a beat-synchronized video from a video file and audio file

    Parameters:
        video_path (str): Path to the input video file
        audio_path (str): Path to the input audio file
        output_path (str, optional): Path for the output video file
        sensitivity (float, optional): Beat detection sensitivity
        algorithm (str, optional): Beat detection algorithm
        min_clip_duration (float): Minimum duration for video clips
        target_duration (float, optional): Target duration for the output video
        audio_start_time (float): Start time in the audio file (in seconds)
    """
    global DETECTED_SCENES, DETECTED_BEATS  # Add global variables to store detection results

    # Generate output path if not provided
    if output_path is None:
        video_name = Path(video_path).stem
        audio_name = Path(audio_path).stem
        project_folder = os.path.join("test_videos", f"{video_name}_x_{audio_name}")
        os.makedirs(project_folder, exist_ok=True)
        output_path = os.path.join(
            project_folder, f"{video_name}_synced_to_{audio_name}.mp4"
        )

    try:
        # Analyze audio first to choose the appropriate detector
        if sensitivity is None or algorithm is None:
            print("\nAnalyzing audio for beat detection...")
            algorithm, sensitivity, tempo = analyze_audio(audio_path)

            # Get the audio characteristics for detector selection
            y, sr = librosa.load(audio_path, offset=audio_start_time)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            bass_strength = np.mean(spec_contrast[0])
            onset_strength = np.mean(onset_env)

            # Choose the appropriate detector
            detector, scenes = choose_detector(
                tempo, bass_strength, onset_strength, video_path
            )

        # First, detect scenes in the video using the chosen detector
        scenes = detect_scenes(video_path, detector)
        if not scenes:
            print("No scenes detected, using entire video as one scene")
            try:
                video = mvpy.VideoFileClip(video_path)
            except:
                print("Trying alternate video loading method...")
                video = mvpy.VideoFileClip(video_path, audio=False)
            scenes = [(0, video.duration)]
            video.close()

        # Store detected scenes globally
        DETECTED_SCENES = scenes

        # Get beat timestamps
        print("\nDetecting beats...")
        beat_times, audio_y, sr = detect_beats(audio_path, sensitivity, algorithm)

        # Adjust beat times based on audio start time
        if audio_start_time > 0:
            # Only keep beats after the start time and adjust their timestamps
            beat_times = beat_times[beat_times >= audio_start_time] - audio_start_time

        # Filter out beats that are too close together
        filtered_beats = [beat_times[0]] if len(beat_times) > 0 else []
        for beat in beat_times[1:]:
            if beat - filtered_beats[-1] >= min_clip_duration:
                filtered_beats.append(beat)
        beat_times = np.array(filtered_beats)

        # Store detected beats globally
        DETECTED_BEATS = beat_times

        print(f"Using {len(beat_times)} beats after filtering")

        # Create the beat-synced video using the new function
        return create_video_from_segments(
            scenes=scenes,
            beat_times=beat_times,
            video_path=video_path,
            audio_path=audio_path,
            target_duration=target_duration,
            audio_start_time=audio_start_time,
            audio_features=AUDIO_FEATURES,
        )

    except Exception as e:
        print(f"Error creating beat-synced video: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


# Define global variables to store detection results
AUDIO_FEATURES = {}
DETECTED_SCENES = []
DETECTED_BEATS = []

if __name__ == "__main__":
    create_beat_synced_video(
        video_path=VIDEO_PATH,
        audio_path=AUDIO_PATH,
        target_duration=TARGET_DURATION,
        audio_start_time=AUDIO_START_TIME,
    )
