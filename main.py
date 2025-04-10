import sys
import os
import argparse
from pathlib import Path
from beat_sync_video import create_beat_synced_video, try_all_detectors

def main():
    parser = argparse.ArgumentParser(description='Create beat-synchronized videos from video and audio files')
    
    # Input arguments
    parser.add_argument('--video', '-v', type=str, help='Path to input video file')
    parser.add_argument('--audio', '-a', type=str, help='Path to input audio file')
    parser.add_argument('--output', '-o', type=str, help='Path to output video file')
    
    # Processing parameters
    parser.add_argument('--target-duration', '-t', type=float, default=30.0, 
                        help='Target duration of output video in seconds (default: 30)')
    parser.add_argument('--min-scene-duration', type=float, default=0.5, 
                        help='Minimum duration of each scene in seconds (default: 0.5)')
    parser.add_argument('--max-scene-duration', type=float, default=5.0, 
                        help='Maximum duration of each scene in seconds (default: 5.0)')
    parser.add_argument('--transition-duration', type=float, default=0.5, 
                        help='Duration of transitions in seconds (default: 0.5)')
    parser.add_argument('--no-transitions', action='store_true', 
                        help='Disable transitions between clips')
    parser.add_argument('--resample-fps', type=int, help='Resample video to specific FPS')
    parser.add_argument('--random-seed', type=int, help='Random seed for reproducibility')
    
    # Parse arguments
    args = parser.parse_args()
    
    # File selection if arguments not provided
    if args.video is None:
        video_files = [f for f in os.listdir() if f.lower().endswith(('.mp4', '.mov', '.avi'))]
        if not video_files:
            print("No video files found in current directory")
            sys.exit(1)
        
        # Let user choose a video file
        print("\nAvailable video files:")
        for i, file in enumerate(video_files):
            print(f"{i+1}. {file}")
        
        while True:
            try:
                choice = int(input("\nSelect a video file (number): ")) - 1
                if 0 <= choice < len(video_files):
                    args.video = video_files[choice]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
    
    if args.audio is None:
        audio_files = [f for f in os.listdir() if f.lower().endswith(('.mp3', '.wav', '.ogg', '.aac'))]
        if not audio_files:
            print("No audio files found in current directory")
            sys.exit(1)
        
        # Let user choose an audio file
        print("\nAvailable audio files:")
        for i, file in enumerate(audio_files):
            print(f"{i+1}. {file}")
        
        while True:
            try:
                choice = int(input("\nSelect an audio file (number): ")) - 1
                if 0 <= choice < len(audio_files):
                    args.audio = audio_files[choice]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
    
    # Generate output path if not provided
    if args.output is None:
        video_name = Path(args.video).stem
        audio_name = Path(args.audio).stem
        project_folder = os.path.join("Output", f"{video_name}_x_{audio_name}")
        os.makedirs(project_folder, exist_ok=True)
        args.output = os.path.join(project_folder, f"{video_name}_synced_to_{audio_name}.mp4")
        print(f"\nOutput will be saved to: {args.output}")
    
    # Create the beat-synchronized video
    print("\n=== Starting Beat Sync Video Creation ===")
    print(f"Video: {args.video}")
    print(f"Audio: {args.audio}")
    print(f"Target Duration: {args.target_duration} seconds")
    
    try:
        output_path = create_beat_synced_video(
            video_path=args.video,
            audio_path=args.audio,
            output_path=args.output,
            target_duration=args.target_duration,
            min_scene_duration=args.min_scene_duration,
            max_scene_duration=args.max_scene_duration,
            transition_duration=args.transition_duration,
            resample_fps=args.resample_fps,
            use_transitions=not args.no_transitions,
            random_seed=args.random_seed
        )
        print(f"\n=== Video Created Successfully ===")
        print(f"Output: {output_path}")
    except Exception as e:
        print(f"\nError creating video: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 