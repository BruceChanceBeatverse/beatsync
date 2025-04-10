"""
Demo script to showcase all available transitions.
Run with: python -m Transitions.demo_transitions
"""

import os
import sys
from pathlib import Path
import argparse
import moviepy.editor as mvpy

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import transitions
from Transitions.transitions import (
    dissolve,
    zoom_transition,
    slide_transition,
    blur_transition,
    flash_transition,
    glitch_transition,
    circular_wipe_transition
)

def create_demo(video_path, output_path=None, clip_duration=2.0, transition_duration=1.0):
    """
    Create a demo video showcasing all available transitions.
    
    Args:
        video_path: Path to source video
        output_path: Path for output demo video (default: auto-generated)
        clip_duration: Duration of each clip in seconds
        transition_duration: Duration of each transition in seconds
    """
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"transitions_demo_{video_name}.mp4"
    
    print(f"Loading video from {video_path}...")
    video = mvpy.VideoFileClip(video_path)
    
    # Get total duration to ensure we don't exceed it
    total_duration = video.duration
    
    # Calculate the number of clips we can extract
    if clip_duration * 8 > total_duration:
        # If video is too short, reduce clip duration
        clip_duration = total_duration / 8
        print(f"Video is short, reducing clip duration to {clip_duration:.1f}s")
    
    # Extract multiple clips from different parts of the video
    clips = []
    # Calculate offsets to get clips from different parts of the video
    offsets = [
        0,  # Start of video
        total_duration * 0.15,
        total_duration * 0.3,
        total_duration * 0.45,
        total_duration * 0.6,
        total_duration * 0.75,
        total_duration * 0.9
    ]
    
    for offset in offsets:
        # Ensure we don't go past the end of the video
        if offset + clip_duration > total_duration:
            offset = total_duration - clip_duration
        
        clip = video.subclip(offset, offset + clip_duration)
        clips.append(clip)
    
    # Add text to identify each transition
    from moviepy.video.tools.drawing import color_clip
    from moviepy.video.VideoClip import TextClip
    
    transition_names = [
        "Dissolve",
        "Zoom",
        "Slide (Left)",
        "Blur",
        "Flash",
        "Glitch",
        "Circular Wipe"
    ]
    
    # Create clips with transition names
    labeled_clips = []
    
    for i, (clip, name) in enumerate(zip(clips, transition_names)):
        # Create text clip
        txt = TextClip(
            f"Transition {i+1}: {name}", 
            fontsize=24, 
            color='white',
            bg_color='black',
            font='Arial-Bold',
        ).set_position(('center', 20)).set_duration(clip.duration)
        
        # Add text to video clip
        labeled_clip = mvpy.CompositeVideoClip([clip, txt])
        labeled_clips.append(labeled_clip)
    
    # Apply all transitions
    print("Applying transitions...")
    
    # Define all transition functions
    transitions = [
        lambda c1, c2: dissolve(c1, c2, transition_duration),
        lambda c1, c2: zoom_transition(c1, c2, transition_duration),
        lambda c1, c2: slide_transition(c1, c2, transition_duration, direction='left'),
        lambda c1, c2: blur_transition(c1, c2, transition_duration),
        lambda c1, c2: flash_transition(c1, c2, transition_duration),
        lambda c1, c2: glitch_transition(c1, c2, transition_duration, intensity=0.7),
        lambda c1, c2: circular_wipe_transition(c1, c2, transition_duration)
    ]
    
    # Apply transitions between clips
    final_clips = [labeled_clips[0]]
    
    for i in range(1, len(labeled_clips)):
        prev_clip = labeled_clips[i-1] 
        curr_clip = labeled_clips[i]
        
        # Apply the corresponding transition
        transition = transitions[i-1]
        try:
            transitioned_clip = transition(prev_clip, curr_clip)
            # We only keep the second part of the transition
            # (since we're adding transitions sequentially)
            clip_with_transition = curr_clip
            final_clips.append(clip_with_transition)
        except Exception as e:
            print(f"Error applying {transition_names[i-1]} transition: {str(e)}")
            # Fall back to simple concatenation
            final_clips.append(curr_clip)
    
    # Concatenate all clips
    print("Concatenating clips...")
    from Transitions.transitions import chain_transitions
    
    # Use our chain_transitions function
    final_video = chain_transitions(labeled_clips, transitions)
    
    # Write final video
    print(f"Writing demo video to {output_path}...")
    final_video.write_videofile(output_path, codec='libx264', fps=24)
    
    # Clean up
    video.close()
    for clip in clips + labeled_clips:
        try:
            clip.close()
        except:
            pass
    
    print(f"Demo video created successfully: {output_path}")
    print(f"Duration: {final_video.duration:.1f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a demo video showcasing all transitions")
    parser.add_argument("video_path", help="Path to source video file")
    parser.add_argument("--output", "-o", help="Path for output demo video")
    parser.add_argument("--clip-duration", "-d", type=float, default=2.0, help="Duration of each clip in seconds")
    parser.add_argument("--transition-duration", "-t", type=float, default=1.0, help="Duration of each transition in seconds")
    
    args = parser.parse_args()
    
    create_demo(
        args.video_path,
        args.output,
        args.clip_duration,
        args.transition_duration
    ) 