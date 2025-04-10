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

def flash_transition(
    clip1: VideoFileClip,
    clip2: VideoFileClip,
    duration: float = 0.5,
    flash_color: Tuple[int, int, int] = (255, 255, 255),
    flash_intensity: float = 1.0
) -> VideoFileClip:
    """
    Create a flash transition between two clips - perfect for beat drops and high-energy moments.
    
    Args:
        clip1: First video clip
        clip2: Second video clip
        duration: Duration of the flash transition in seconds
        flash_color: RGB color of the flash (default: white)
        flash_intensity: Maximum intensity of the flash (0.0-1.0)
    
    Returns:
        VideoFileClip with the flash transition
    """
    # Safety checks
    if duration > min(clip1.duration, clip2.duration) / 2:
        duration = min(clip1.duration, clip2.duration) / 2
    
    # Make sure both clips have the same size
    if clip1.size != clip2.size:
        clip2 = clip2.resize(clip1.size)
    
    # Ensure flash_intensity is within bounds
    flash_intensity = max(0.1, min(1.0, flash_intensity))
    
    try:
        # Extract transition segments
        clip1_end = clip1.subclip(clip1.duration - duration, clip1.duration)
        clip2_start = clip2.subclip(0, duration)
        
        # Create a flash frame
        w, h = clip1.size
        flash_frame = np.ones((h, w, 3), dtype=np.uint8)
        flash_frame[:, :, 0] = flash_color[0]  # R
        flash_frame[:, :, 1] = flash_color[1]  # G
        flash_frame[:, :, 2] = flash_color[2]  # B
        
        def flash_clip_factory(t):
            """Generate flash with varying opacity based on time"""
            # Flash intensity peaks in the middle of the transition
            # Uses a sinusoidal curve for smooth flash effect
            progress = t / duration
            flash_opacity = flash_intensity * np.sin(progress * np.pi)
            return flash_frame * flash_opacity
        
        # Create the flash clip with dynamic opacity
        from moviepy.video.VideoClip import ImageClip
        flash_clip = ImageClip(flash_frame).set_duration(duration)
        flash_clip = flash_clip.fl_image(lambda img: flash_clip_factory(flash_clip.time))
        
        # Apply crossfade between the original clips
        clip1_fade = clip1_end.crossfadeout(duration)
        clip2_fade = clip2_start.crossfadein(duration)
        
        # Composite all three layers
        transition = CompositeVideoClip([clip1_fade, clip2_fade, flash_clip.set_opacity(lambda t: np.sin(t/duration * np.pi))], 
                                       size=clip1.size)
        
        # Combine with main clips
        clip1_main = clip1.subclip(0, clip1.duration - duration)
        clip2_rest = clip2.subclip(duration, clip2.duration)
        
        return concatenate_videoclips([clip1_main, transition, clip2_rest], method="compose")
    except Exception as e:
        print(f"Flash transition failed: {str(e)}, falling back to dissolve")
        return dissolve(clip1, clip2, duration)

def glitch_transition(
    clip1: VideoFileClip,
    clip2: VideoFileClip,
    duration: float = 0.5,
    intensity: float = 0.5,
    n_glitches: int = 10
) -> VideoFileClip:
    """
    Create a digital glitch transition between two clips - great for electronic music.
    
    Args:
        clip1: First video clip
        clip2: Second video clip
        duration: Duration of the glitch transition in seconds
        intensity: Intensity of the glitch effect (0.0-1.0)
        n_glitches: Number of glitch pulses during the transition
    
    Returns:
        VideoFileClip with the glitch transition
    """
    # Safety checks
    if duration > min(clip1.duration, clip2.duration) / 2:
        duration = min(clip1.duration, clip2.duration) / 2
    
    # Make sure both clips have the same size
    if clip1.size != clip2.size:
        clip2 = clip2.resize(clip1.size)
    
    # Ensure intensity is within bounds
    intensity = max(0.1, min(1.0, intensity))
    n_glitches = max(3, min(20, n_glitches))
    
    try:
        # Extract transition segments
        clip1_end = clip1.subclip(clip1.duration - duration, clip1.duration)
        clip2_start = clip2.subclip(0, duration)
        
        # Create a glitch effect function
        def apply_glitch(get_frame, t):
            """Apply glitch effects based on time"""
            frame = get_frame(t).copy()
            h, w = frame.shape[:2]
            
            # Determine if this is a glitch moment
            # Create several glitch pulses throughout the duration
            is_glitch_moment = False
            for i in range(n_glitches):
                glitch_time = duration * i / n_glitches
                if abs(t - glitch_time) < duration / (n_glitches * 2):
                    is_glitch_moment = True
                    break
            
            if is_glitch_moment:
                # Apply RGB shift
                shift_amount = int(w * 0.02 * intensity)
                if shift_amount > 0:
                    # Shift red channel
                    frame[0:h, 0:w-shift_amount, 0] = frame[0:h, shift_amount:w, 0]
                    # Shift blue channel
                    frame[0:h, shift_amount:w, 2] = frame[0:h, 0:w-shift_amount, 2]
                
                # Create horizontal line glitches
                num_lines = int(10 * intensity)
                for _ in range(num_lines):
                    y = np.random.randint(0, h)
                    line_height = np.random.randint(1, max(2, int(h * 0.05 * intensity)))
                    color = np.random.randint(0, 256, 3)
                    frame[y:min(y+line_height, h), :] = color
                
                # Block displacement
                num_blocks = int(5 * intensity)
                for _ in range(num_blocks):
                    block_h = np.random.randint(10, max(11, int(h * 0.2 * intensity)))
                    block_w = np.random.randint(w // 8, w // 2)
                    
                    src_y = np.random.randint(0, h - block_h)
                    src_x = np.random.randint(0, w - block_w)
                    
                    dst_y = np.random.randint(0, h - block_h)
                    dst_x = np.random.randint(0, w - block_w)
                    
                    # Copy a block to another location
                    if np.random.random() < 0.5:  # 50% chance to apply
                        frame[dst_y:dst_y+block_h, dst_x:dst_x+block_w] = frame[src_y:src_y+block_h, src_x:src_x+block_w]
            
            # Gradually blend from clip1 to clip2
            blend_factor = t / duration
            
            # During glitch moments, we make the blend more erratic
            if is_glitch_moment:
                blend_factor = np.random.choice([0, 0.2, 0.5, 0.8, 1]) * blend_factor
            
            return frame
        
        # Apply glitch to both clips
        clip1_glitch = clip1_end.fl_image(lambda img: apply_glitch(clip1_end.get_frame, clip1_end.time))
        clip2_glitch = clip2_start.fl_image(lambda img: apply_glitch(clip2_start.get_frame, clip2_start.time))
        
        # Apply crossfade between the glitched clips
        clip1_glitch = clip1_glitch.crossfadeout(duration)
        clip2_glitch = clip2_glitch.crossfadein(duration).set_start(0)
        
        # Composite the glitched clips
        glitch_transition = CompositeVideoClip([clip1_glitch, clip2_glitch], size=clip1.size).subclip(0, duration)
        
        # Combine with main clips
        clip1_main = clip1.subclip(0, clip1.duration - duration)
        clip2_main = clip2.subclip(duration, clip2.duration)
        
        return concatenate_videoclips([clip1_main, glitch_transition, clip2_main], method="compose")
    except Exception as e:
        print(f"Glitch transition failed: {str(e)}, falling back to dissolve")
        return dissolve(clip1, clip2, duration)

def circular_wipe_transition(
    clip1: VideoFileClip,
    clip2: VideoFileClip,
    duration: float = 0.8,
    reverse: bool = False,
    center: Optional[Tuple[float, float]] = None
) -> VideoFileClip:
    """
    Create a circular wipe transition between two clips - an expanding/contracting circle reveals the second clip.
    
    Args:
        clip1: First video clip
        clip2: Second video clip
        duration: Duration of the circular wipe in seconds
        reverse: If True, circle contracts instead of expands (default: False)
        center: Optional center point of the circle as (x, y) in relative coordinates (0.0-1.0)
               If None, uses center of frame
    
    Returns:
        VideoFileClip with the circular wipe transition
    """
    # Safety checks
    if duration > min(clip1.duration, clip2.duration) / 2:
        duration = min(clip1.duration, clip2.duration) / 2
    
    # Make sure both clips have the same size
    if clip1.size != clip2.size:
        clip2 = clip2.resize(clip1.size)
    
    try:
        # Extract transition segments
        clip1_end = clip1.subclip(clip1.duration - duration, clip1.duration)
        clip2_start = clip2.subclip(0, duration)
        
        # Get clip dimensions
        w, h = clip1.size
        
        # Default to center of frame if not specified
        if center is None:
            center = (0.5, 0.5)
        
        # Convert relative coordinates to absolute
        center_x = int(center[0] * w)
        center_y = int(center[1] * h)
        
        # Maximum radius needed to cover the entire frame
        max_radius = int(np.sqrt(w**2 + h**2) / 2)
        
        def make_circular_mask(t):
            """Create a circular mask that reveals/hides based on time"""
            progress = t / duration
            
            # Create a meshgrid for coordinates
            y, x = np.ogrid[:h, :w]
            
            # Calculate distance from center for each pixel
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Calculate current radius based on progress
            if reverse:
                current_radius = max_radius * (1 - progress)
            else:
                current_radius = max_radius * progress
            
            # Create mask: 1 where second clip should be visible, 0 otherwise
            mask = dist_from_center <= current_radius
            
            # Convert to float array with values 0.0-1.0
            return mask.astype(float)
        
        def apply_circular_wipe(t):
            """Apply circular wipe effect to frames"""
            # Get the current frames from both clips
            frame1 = clip1_end.get_frame(t)
            frame2 = clip2_start.get_frame(t)
            
            # Create the mask for this moment in time
            mask = make_circular_mask(t)
            
            # Expand mask dimensions to match RGB
            mask = np.expand_dims(mask, axis=2)
            mask = np.repeat(mask, 3, axis=2)
            
            # Apply mask to blend the frames
            # Where mask is 1, show frame2; where mask is 0, show frame1
            result = frame1 * (1 - mask) + frame2 * mask
            
            return result.astype('uint8')
        
        # Create the circular wipe transition
        from moviepy.video.VideoClip import VideoClip
        wipe_clip = VideoClip(make_frame=apply_circular_wipe, duration=duration)
        
        # Combine with main clips
        clip1_main = clip1.subclip(0, clip1.duration - duration)
        clip2_rest = clip2.subclip(duration, clip2.duration)
        
        return concatenate_videoclips([clip1_main, wipe_clip, clip2_rest], method="compose")
    except Exception as e:
        print(f"Circular wipe transition failed: {str(e)}, falling back to dissolve")
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