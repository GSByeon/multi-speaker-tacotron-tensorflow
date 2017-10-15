#!/bin/sh

# 1. Download and extract audio and texts
python -m datasets.jtbc.download

# 2. Split audios on silence
python -m audio.silence --audio_pattern "./datasets/jtbc/audio/*.wav" --method=pydub

# 3. Run Google Speech Recognition
python -m recognition.google --audio_pattern "./datasets/jtbc/audio/*.*.wav"

# 4. Run heuristic text-audio pair search (any improvement on this is welcome)
python -m recognition.alignment --recognition_path "./datasets/jtbc/recognition.json" --score_threshold=0.5

# 5. Remove intro music
rm datasets/jtbc/data/*.0000.npz
