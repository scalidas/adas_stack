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
.\build\Release\adas_stack.exe detect_lane_demo --config config/perception_config.json
```

#### Intrinsic Calibrate
```
.\build\Release\adas_stack.exe instrinsic_calibration assets\calibration\phone_mounted_calibration.jpg
```

#### Extrinsic Calibrate
```
.\build\Release\adas_stack.exe extrinsic_calibration assets\calibration\phone_mounted_calibration.jpg
```

