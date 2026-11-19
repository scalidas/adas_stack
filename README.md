# ECE Senior Design: ADAS Stack

L2+ ADAS Stack for mini autonomous vehicle.

### Usage

#### Compile on Windows
```
cmake -B build
cmake --build build --config Release
```

#### Compile on Jetson
```
cmake -B build \
  -DADAS_WITH_TENSORRT=ON \
  -DTENSORRT_ROOT=/opt/TensorRT-10.x \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  .
  
cmake --build build -j$(nproc)
```

#### Compile with Ninja (Windows)

```
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release .
cmake --build build
```

#### Adas Demo (Jetson)
```
./build/adas_stack adas_demo ./config/adas_demo.json
```

#### Adas Demo (Windows)
```
.\build\Release\adas_stack.exe adas_demo ./config/adas_demo.json
```

#### Detect Lane Demo (Jetson)
```
./build/adas_stack detect_lane_demo ./config/live_lane_demo.json
```

#### Detect Lane Demo (Windows)

```
.\build\Release\adas_stack.exe detect_lane_demo --config config/perception_config.json
```

#### Extrinsic Calibrate (Jetson)
```
./build/adas_stack extrinsic_calibration assets\calibration\phone_mounted_calibration.jpg
```

#### Extrinsic Calibrate (Windows)
```
.\build\Release\adas_stack.exe extrinsic_calibration assets\calibration\phone_mounted_calibration.jpg
```

#### Intrinsic Calibrate
```
.\build\Release\adas_stack.exe instrinsic_calibration assets\calibration\phone_mounted_calibration.jpg
```

#### Object Demo
```
.\build\Release\adas_stack.exe cup_demo_original --config config/cup_demo_original.json
```
