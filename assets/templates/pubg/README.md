# PUBG Template Images

Place hitmarker and kill icon template images here for template matching detection.

## Required files:
- `hitmarker.png` — The red/white hitmarker that appears on hit (crop from a screenshot, ~40x40 pixels)
- `kill_icon.png` — The skull/death icon that appears on a kill (crop from a screenshot, ~40x40 pixels)

## How to create templates:
1. Take a screenshot of PUBG during combat
2. Crop the hitmarker from the center of the screen (~40x40 pixels)
3. Crop the kill icon from the kill notification area
4. Save as PNG with transparent or black background
5. Place files in this directory

**Note:** If no templates are provided, the system falls back to color-based detection (red/white pattern analysis at screen center). Templates improve accuracy significantly.
