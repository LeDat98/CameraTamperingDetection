# Camera Tampering Detection System

An automated camera tampering detection system that uses image analysis and statistical techniques to detect when a camera has been tampered with.

## Overview

This program monitors a webcam and detects signs of tampering such as:
- Camera being covered
- Camera being moved
- Sudden changes in lighting
- Other forms of interference

The system uses histogram comparison methods across different time periods to detect abnormal deviations from the camera's "normal state".

## System Requirements

- Python 3.6 or higher
- Working webcam
- Python libraries:
  - OpenCV (cv2)
  - NumPy
  - Matplotlib
  - collections (standard library)

## Installation

1. Clone or download this repository
2. Install the required libraries:
```
pip install opencv-python numpy matplotlib
```

## Usage

Run the main file:
```
python camera_tampering_detection_cpu.py
```

When the program runs:
- A "Combined Feed" window will display with webcam feed (left) and analysis chart (right)
- Press 'q' key to exit the program

## Important Parameters

The program uses the following parameters that you can adjust:

### 1. Buffer Sizes (In the `__init__` method)

```python
def __init__(self, short_term_size=3, long_term_size=36):
    self.short_term_pool = deque(maxlen=short_term_size)
    self.long_term_pool = deque(maxlen=long_term_size)
```

- **short_term_size**: Number of recent frames to analyze for short-term changes
  - Default: 3 (later adjusted to 3*5=15 in `process_webcam()`)
  - Higher values = slower response but fewer false alarms

- **long_term_size**: Number of reference histograms to establish "normal" state
  - Default: 36 (later adjusted to 3*60=180 in `process_webcam()`)
  - Higher values = longer learning time, more accuracy but slower startup

### 2. Detection Threshold (In the `detect_tampering` method)

```python
threshold = 1
return d_norm > threshold, d_norm
```

- **threshold**: Threshold to determine when tampering occurs
  - Default: 1
  - Lower values = more sensitive (might have false alarms)
  - Higher values = less sensitive (might miss some tampering)

### 3. Processing Frequency (In the `process_webcam` method)

```python
frame_interval = int(fps / 3)
```

- **frame_interval**: Number of frames to skip between analyses
  - Default: fps/3 (analyzes about 3 frames/second)
  - Lower values = analyze more frames, more sensitive but higher CPU usage
  - Higher values = analyze fewer frames, lower CPU usage but less sensitive

### 4. Histogram Sizes (Number of bins in histograms)

```python
hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])  # Chromaticity
hist = cv2.calcHist([stacked], [0, 1], None, [8, 8], [0, 256, 0, 256])  # L1R
hist = cv2.calcHist([grad_dir], [0], None, [16], [-np.pi, np.pi])  # Gradient
```

- Changing the number of bins affects the detail of analysis
- More bins = more detailed but resource-intensive and potentially noisy
- Fewer bins = more general, faster but might miss small details

## How It Works

1. **Initialization Phase**:
   - System collects data to establish the "normal state"
   - During this phase, "Initializing..." will be displayed

2. **Regular Detection**:
   - Analyzes current frames and compares with reference data
   - Calculates normalized dissimilarity (d_norm)
   - If d_norm > threshold, displays "TAMPERING DETECTED"

3. **Visual Display**:
   - Real-time chart displays d_norm values
   - Threshold is set at 1.0 on the chart

## Performance Tuning

- If CPU usage is too high: Increase `frame_interval`
- If detection is too sensitive (many false alarms): Increase `threshold`
- If not sensitive enough: Decrease `threshold`
- For faster initialization: Decrease `long_term_size`

## Advanced Features

The program uses 3 types of histograms to analyze different aspects of the image:

1. **Chromaticity Histogram**: Analyzes color distribution (hue and saturation)
2. **L1R Histogram**: Combines total intensity (L1) and color range (R)
3. **Gradient Histogram**: Analyzes structure and edge directions

The combination of these methods helps the system detect various types of tampering.

## Troubleshooting

- **Can't open camera**: Check webcam access permissions
- **Poor performance**: Adjust `frame_interval` higher
- **False alarms**: Increase `threshold` or `short_term_size`
- **Not detecting tampering**: Decrease `threshold`

## Contributing

Contributions are welcome! Please create an issue or pull request if you have improvements.

## License

MIT License

Copyright (c) 2025 LEDUCDAT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
