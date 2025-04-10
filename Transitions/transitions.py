"""
Video transitions library implemented using MoviePy.
"""

from typing import Optional, Tuple, Callable
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip
from moviepy.video.fx.all import resize

def hard_cut(clip1: VideoFileClip, clip2: VideoFileClip) -> VideoFileClip:
    """
    Create a hard cut between two clips (no transition effect).
    
    Args:
        clip1: First video clip
        clip2: Second video clip
    
    Returns:
        VideoFileClip with the clips concatenated directly
    """
    # Make sure both clips have the same size
    if clip1.size != clip2.size:
        clip2 = clip2.resize(clip1.size)
    
    return concatenate_videoclips([clip1, clip2])

def dissolve(clip1: VideoFileClip, clip2: VideoFileClip, duration: float = 1.0) -> VideoFileClip:
    """
    Create a crossfade/dissolve transition between two clips.
    
    Args:
        clip1: First video clip
        clip2: Second video clip
        duration: Duration of the crossfade in seconds
    
    Returns:
        VideoFileClip with the crossfade transition
    """
    # Safety checks
    if duration > min(clip1.duration, clip2.duration) / 2:
        duration = min(clip1.duration, clip2.duration) / 2
        
    # Make sure both clips have the same size
    if clip1.size != clip2.size:
        clip2 = clip2.resize(clip1.size)
    
    # Prepare the transition segments
    clip1_end = clip1.subclip(max(0, clip1.duration - duration))
    clip2_start = clip2.subclip(0, min(duration, clip2.duration))
    
    # Apply simple crossfade
    clip1_fade = clip1_end.crossfadeout(duration)
    clip2_fade = clip2_start.crossfadein(duration)
    
    # Create simplified composite
    transition = CompositeVideoClip([clip1_fade, clip2_fade], size=clip1.size)
    
    # Combine with main clips
    clip1_main = clip1.subclip(0, max(0, clip1.duration - duration))
    clip2_rest = clip2.subclip(min(duration, clip2.duration))
    
    try:
        # First try with compose method
        result = concatenate_videoclips([clip1_main, transition, clip2_rest], method="compose")
    except Exception as e:
        print(f"Compose method failed in dissolve, using direct: {str(e)}")
        # Fallback to direct concatenation
        result = concatenate_videoclips([clip1_main, transition, clip2_rest])
        
    return result

def zoom_transition(
    clip1: VideoFileClip,
    clip2: VideoFileClip,
    duration: float = 1.0,
    zoom_factor: float = 2.0
) -> VideoFileClip:
    """
    Create a zoom transition between two clips.
    
    Args:
        clip1: First video clip
        clip2: Second video clip
        duration: Duration of the zoom transition in seconds
        zoom_factor: How much to zoom (2.0 = 200%)
    
    Returns:
        VideoFileClip with the zoom transition
    """
    # Safety checks
    if duration > min(clip1.duration, clip2.duration) / 2:
        duration = min(clip1.duration, clip2.duration) / 2
    
    # Make sure both clips have the same size
    if clip1.size != clip2.size:
        clip2 = clip2.resize(clip1.size)
    
    # Extract transition segments
    clip1_end = clip1.subclip(clip1.duration - duration, clip1.duration)
    clip2_start = clip2.subclip(0, duration)
    
    # Apply dynamic zoom
    clip1_zoomed = clip1_end.resize(lambda t: 1 + (zoom_factor-1) * (t/duration))
    clip2_zoomed = clip2_start.resize(lambda t: zoom_factor - (zoom_factor-1) * (t/duration))
    
    # Create crossfade between zoomed clips
    clip2_zoomed = clip2_zoomed.set_start(0).crossfadein(duration)
    zoom_transition = CompositeVideoClip([clip1_zoomed, clip2_zoomed]).subclip(0, duration)
    
    # Combine with main clips
    clip1_main = clip1.subclip(0, clip1.duration - duration)
    clip2_rest = clip2.subclip(duration, clip2.duration)
    
    try:
        # Try with compose method
        result = concatenate_videoclips([clip1_main, zoom_transition, clip2_rest], method="compose")
    except Exception as e:
        print(f"Zoom transition failed: {str(e)}, falling back to dissolve")
        return dissolve(clip1, clip2, duration)
        
    return result

def slide_transition(
    clip1: VideoFileClip,
    clip2: VideoFileClip,
    duration: float = 1.0,
    direction: str = 'left'
) -> VideoFileClip:
    """
    Create a slide/wipe transition between two clips.
    
    Args:
        clip1: First video clip
        clip2: Second video clip
        duration: Duration of the slide in seconds
        direction: Direction of the slide ('left', 'right', 'top', 'bottom')
    
    Returns:
        VideoFileClip with the slide transition
    """
    # Safety checks for parameters
    if direction not in ['left', 'right', 'top', 'bottom']:
        direction = 'left'  # Default to left if invalid direction
    
    if duration > min(clip1.duration, clip2.duration) / 2:
        duration = min(clip1.duration, clip2.duration) / 2
    
    # Make sure both clips have the same size and fps
    if clip1.size != clip2.size:
        clip2 = clip2.resize(clip1.size)
    
    w, h = clip1.size
    
    # Add easing function for smoother motion
    def ease_out_quad(t):
        """Quadratic ease-out function for smoother motion"""
        return 1 - (1 - t) * (1 - t)
    
    def get_slide_position(t):
        """Calculate position based on time and direction with easing."""
        # Use easing function to make the movement smoother
        progress = ease_out_quad(t / duration)
        
        if direction == 'left':
            return (int(w * (1 - progress)), 0)
        elif direction == 'right':
            return (int(-w * (1 - progress)), 0)
        elif direction == 'top':
            return (0, int(h * (1 - progress)))
        else:  # bottom
            return (0, int(-h * (1 - progress)))
    
    def get_clip2_initial_position():
        """Get the initial position for clip2."""
        if direction == 'left':
            return (w, 0)
        elif direction == 'right':
            return (-w, 0)
        elif direction == 'top':
            return (0, h)
        else:  # bottom
            return (0, -h)
    
    def get_clip2_position(t):
        """Calculate position for clip2 with easing."""
        progress = ease_out_quad(t / duration)
        initial_pos = get_clip2_initial_position()
        return (
            initial_pos[0] * (1 - progress),
            initial_pos[1] * (1 - progress)
        )
    
    try:
        # Create sliding clips
        clip1_end = clip1.subclip(clip1.duration - duration, clip1.duration)
        clip2_start = clip2.subclip(0, duration)
        
        # Set positions for sliding effect with easing
        clip1_slide = clip1_end.set_position(get_slide_position)
        clip2_slide = clip2_start.set_position(get_clip2_position)
        
        # Composite the clips
        slide_transition = CompositeVideoClip(
            [clip1_slide, clip2_slide],
            size=clip1.size
        ).subclip(0, duration)
        
        # Combine with main clips
        clip1_main = clip1.subclip(0, clip1.duration - duration)
        clip2_rest = clip2.subclip(duration, clip2.duration)
        
        return concatenate_videoclips(
            [clip1_main, slide_transition, clip2_rest],
            method="compose"
        )
    except Exception as e:
        print(f"Slide transition failed: {str(e)}, falling back to dissolve")
        return dissolve(clip1, clip2, duration)

def blur_transition(
    clip1: VideoFileClip,
    clip2: VideoFileClip,
    duration: float = 1.0,
    blur_radius: Tuple[int, int] = (21, 21)
) -> VideoFileClip:
    """
    Create a blur/defocus transition between two clips.
    
    Args:
        clip1: First video clip
        clip2: Second video clip
        duration: Duration of the blur transition in seconds
        blur_radius: Tuple of (width, height) for Gaussian blur kernel
    
    Returns:
        VideoFileClip with the blur transition
    """
    # Safety checks for parameters
    if duration > min(clip1.duration, clip2.duration) / 2:
        duration = min(clip1.duration, clip2.duration) / 2
    
    # Make sure both clips have the same size
    if clip1.size != clip2.size:
        clip2 = clip2.resize(clip1.size)
    
    try:
        def blur_frame(frame):
            return cv2.GaussianBlur(frame, blur_radius, 0)
        
        # Create blurred segments
        clip1_blur = clip1.subclip(clip1.duration - duration, clip1.duration).fl_image(blur_frame)
        clip2_blur = clip2.subclip(0, duration).fl_image(blur_frame)
        
        # Create crossfade between blurred segments
        clip1_blur = clip1_blur.crossfadeout(duration)
        clip2_blur = clip2_blur.crossfadein(duration).set_start(0)
        blur_crossfade = CompositeVideoClip([clip1_blur, clip2_blur]).subclip(0, duration)
        
        # Combine with main clips
        clip1_main = clip1.subclip(0, clip1.duration - duration)
        clip2_main = clip2.subclip(duration, clip2.duration)
        
        return concatenate_videoclips([clip1_main, blur_crossfade, clip2_main], method="compose")
    except Exception as e:
        print(f"Blur transition failed: {str(e)}, falling back to dissolve")
        return dissolve(clip1, clip2, duration)

def chain_transitions(clips: list[VideoFileClip], 
                     transitions: list[Callable[[VideoFileClip, VideoFileClip], VideoFileClip]]) -> VideoFileClip:
    """
    Chain multiple clips together using specified transitions.
    
    Args:
        clips: List of video clips to chain together
        transitions: List of transition functions to apply between clips
        
    Returns:
        VideoFileClip with all clips chained together using the specified transitions
    """
    if len(clips) < 2:
        return clips[0]
        
    if len(transitions) != len(clips) - 1:
        raise ValueError("Number of transitions must be one less than number of clips")
    
    result = clips[0]
    for i in range(len(clips) - 1):
        try:
            # Apply the specified transition
            result = transitions[i](result, clips[i + 1])
        except Exception as e:
            print(f"Warning: Transition {i} failed with error: {str(e)}")
            print("Falling back to hard cut")
            # Fall back to hard cut if the transition fails
            result = hard_cut(result, clips[i + 1])
            
    return result 