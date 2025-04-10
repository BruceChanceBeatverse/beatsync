# Video and audio file paths
VIDEO_PATH = "/Users/brucechen/Downloads/VID_20250309_143320_00_795.mov"
AUDIO_PATH = "/Users/brucechen/Downloads/Archive/Skrillex, Sirah - Bangarang (feat. Sirah).mp3"

# Output settings
TARGET_DURATION = 45.0  # Target duration in seconds
AUDIO_START_TIME = 0.0

# Processing parameters
MIN_CLIP_DURATION = 0.5
MAX_BEATS_PER_SCENE = 4
MIN_SCENE_DURATION = 0.5
MIN_SCENE_LEN = 15
OUTPUT_THREADS = 4
OUTPUT_FPS = 30

# Scene detection parameters
HISTOGRAM_THRESHOLD = 0.05
CONTENT_THRESHOLD = 27
ADAPTIVE_THRESHOLD = 3.0
THRESHOLD_DETECTOR = 30

# Beat detection parameters
DEFAULT_START_BPM = 120  # Starting BPM for beat detection
BEAT_TIGHTNESS = 100  # Beat tracking tightness
DOWNBEAT_PERCENTILE = 60  # Percentile for downbeat detection (top 40% as downbeats)
