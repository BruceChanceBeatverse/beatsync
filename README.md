# BeatSync Video Editor

A powerful tool for creating beat-synchronized video edits that automatically match video clips to music beats.

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

### Introduction

BeatSync automatically creates engaging, beat-synchronized video edits by intelligently detecting scenes in your video and matching them to beats in your music. The tool applies appropriate transitions based on the music's energy and rhythm.

### Features

- Automatic scene detection using multiple detection algorithms
- Advanced beat detection with adjustable sensitivity
- Intelligent transition selection based on music characteristics
- Support for various transition types (dissolves, blur, zoom, slide)
- Customizable output parameters

### Prerequisites

- Python 3.12 or later
- Required Python packages (install via `pip install -r requirements.txt`):
  - librosa
  - moviepy
  - numpy
  - scenedetect

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/YOUR_USERNAME/beatsync.git
   cd beatsync
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. Configure settings in `config.py`:
   ```python
   # Video and audio file paths
   VIDEO_PATH = "/path/to/your/video.mp4"
   AUDIO_PATH = "/path/to/your/audio.mp3"

   # Output settings
   TARGET_DURATION = 45.0  # Target duration in seconds
   AUDIO_START_TIME = 0.0  # Start time in audio
   ```

2. Run the main script:
   ```
   python beat_sync_video.py
   ```

3. Find your output video in the `test_videos` directory.

### Configuration Options

The `config.py` file contains several parameters you can adjust:

- `VIDEO_PATH`: Path to your input video file
- `AUDIO_PATH`: Path to your input audio file
- `TARGET_DURATION`: Desired duration of output video (in seconds)
- `AUDIO_START_TIME`: Start time in the audio file (in seconds)
- `MIN_CLIP_DURATION`: Minimum duration for video clips
- `MAX_BEATS_PER_SCENE`: Maximum number of beats per scene
- Scene detection parameters: `HISTOGRAM_THRESHOLD`, `CONTENT_THRESHOLD`, etc.
- Beat detection parameters: `DEFAULT_START_BPM`, `BEAT_TIGHTNESS`, etc.

### How It Works

1. The script analyzes the audio to detect beats and music characteristics
2. Multiple scene detection methods are tested to find the best algorithm for your video
3. Video scenes are matched to detected beats
4. Transitions are selected based on music energy, rhythm, and context
5. Final video is rendered with the original audio track

---

<a name="chinese"></a>
## 中文

### 简介

BeatSync 自动创建引人入胜的、与节拍同步的视频编辑，通过智能检测视频中的场景并将其与音乐节拍匹配。该工具根据音乐的能量和节奏应用适当的转场效果。

### 功能特点

- 使用多种检测算法自动场景检测
- 具有可调节灵敏度的高级节拍检测
- 基于音乐特性的智能转场选择
- 支持各种转场类型（溶解、模糊、缩放、滑动）
- 可自定义输出参数

### 系统要求

- Python 3.12 或更高版本
- 所需 Python 包（通过 `pip install -r requirements.txt` 安装）：
  - librosa
  - moviepy
  - numpy
  - scenedetect

### 安装步骤

1. 克隆此仓库：
   ```
   git clone https://github.com/YOUR_USERNAME/beatsync.git
   cd beatsync
   ```

2. 安装所需依赖：
   ```
   pip install -r requirements.txt
   ```

### 使用方法

1. 在 `config.py` 中配置设置：
   ```python
   # 视频和音频文件路径
   VIDEO_PATH = "/path/to/your/video.mp4"
   AUDIO_PATH = "/path/to/your/audio.mp3"

   # 输出设置
   TARGET_DURATION = 45.0  # 目标持续时间（秒）
   AUDIO_START_TIME = 0.0  # 音频起始时间
   ```

2. 运行主脚本：
   ```
   python beat_sync_video.py
   ```

3. 在 `test_videos` 目录中找到输出视频。

### 配置选项

`config.py` 文件包含多个可调整的参数：

- `VIDEO_PATH`：输入视频文件的路径
- `AUDIO_PATH`：输入音频文件的路径
- `TARGET_DURATION`：输出视频的所需持续时间（秒）
- `AUDIO_START_TIME`：音频文件中的起始时间（秒）
- `MIN_CLIP_DURATION`：视频剪辑的最小持续时间
- `MAX_BEATS_PER_SCENE`：每个场景的最大节拍数
- 场景检测参数：`HISTOGRAM_THRESHOLD`、`CONTENT_THRESHOLD` 等
- 节拍检测参数：`DEFAULT_START_BPM`、`BEAT_TIGHTNESS` 等

### 工作原理

1. 脚本分析音频以检测节拍和音乐特性
2. 测试多种场景检测方法，为您的视频找到最佳算法
3. 将视频场景与检测到的节拍匹配
4. 根据音乐能量、节奏和上下文选择转场效果
5. 使用原始音轨渲染最终视频

