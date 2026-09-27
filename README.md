# ECE Senior Design: ADAS Stack

L2+ ADAS Stack for mini autonomous vehicle.

### Usage

#### Compile
```
cmake -B build
cmake --build build --config Release
```

#### Detect Lane

```
.\build\Release\perception_stack.exe detect_lane --config config/perception_config.json
```

#### Calibrate
```
.\build\Release\perception_stack.exe calibrate assets/calibration/WIN_20260926_19_24_20_Pro.jpg
```

