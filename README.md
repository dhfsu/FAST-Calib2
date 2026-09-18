# FAST-Calib2

## LiDAR-Camera Extrinsic Calibration with Reflective Annular Targets

FAST-Calib2 extends [FAST-Calib](https://github.com/hku-mars/FAST-Calib) to LiDAR-camera modules that were previously hard to calibrate due to **low-quality point clouds**. With a custom-designed reflective annular calibration target, it enables robust center extraction on **large-spot solid-state and mechanical LiDARs**, including Mid360, Avia, Ouster, XT32, JT128, Airy, E1R, and Adaps Photonics Spad LiDAR.

**Key highlights include:**

1. A self-designed 3D reflective annular calibration target that avoids center extraction errors caused by hole-edge inflation and bleeding artifacts in previous circular-hole calibration boards.
2. A robust concentric-circle fitting method that uses the fixed inner and outer annulus radii as geometric constraints.
3. Automatic calibration board ROI extraction without manual pass-through tuning.
4. Geometry and radius quality checks for extracted annulus centers.
5. Single-scene and multi-scene LiDAR-camera extrinsic calibration without initial extrinsic parameters.

📬 For further assistance or inquiries, please feel free to contact Chunran Zheng at zhengcr@connect.hku.hk.

<p align="center">
  <img src="./pics/cover.jpg" width="100%">
  <font color=#a0a0a0 size=2>Mid360 calibration example.</font>
</p>

## 1. Prerequisites

PCL>=1.8, OpenCV>=4.0.

## 2. Calibration Target

FAST-Calib2 uses four reflective annuli and four visual markers on one board. The annuli are used by LiDAR center extraction, while the visual markers are used by the camera pipeline.

Materials:

- Board: PVC
- Reflective annulus stickers: 3M engineering-grade reflective film

<p align="center">
  <img src="./pics/FAST-Calib2-board.png" width="100%">
  <font color=#a0a0a0 size=2>Reflective annular calibration target and annotated dimensions.</font>
</p>

DIY Calibration Target Tips:

1. Fabricate the board based on the schematic. Ensure a minimum thickness of 1 cm to avoid bending.
2. Apply reflective annulus stickers to the designated ring positions on the fabricated board.

## 3. Method Overview

Both LiDAR pipelines first **locate the calibration board automatically**, fit the board plane, and align the plane to `Z=0`. Center extraction is then performed in the aligned board frame.

Solid-state LiDAR pipeline:

1. Extract high-reflectivity annulus points on the fitted board plane.
2. Cluster the extracted annulus points.
3. Fit robust single circles as the default center estimate.
4. Optionally extract annulus boundary points and fit fixed inner/outer radius concentric circles.
5. Select the best result by checking four-center geometry consistency against the known target geometry.

Mechanical LiDAR pipeline:

1. Use LiDAR ring order within each scan to find intensity transition points on the annulus boundary.
2. Try both interpolated boundary points and high-reflectivity-side boundary points.
3. Cluster the extracted boundary points.
4. Fit fixed inner/outer radius concentric circles.
5. Select the best result by checking four-center geometry consistency against the known target geometry.

The final quality checks include center-to-center geometry error and annulus radius consistency.

## 4. Run Examples

Prepare static acquisition data in the `calib_data` folder (Download the example data from [Google Drive](https://drive.google.com/drive/folders/1VnMCsGj3Gat7dxe6IION0SfS7jYNMw1g?usp=sharing)):

- rosbag containing point cloud messages
- corresponding image

Describe the LiDAR mounting in `config/qr_params.yaml`:

```yaml
lidar_forward_axis: "+x"
lidar_up_axis: "+z"
```

The axis values must be signed, perpendicular axes such as `+x` and `-y`.

Run single-scene calibration:

```bash
roslaunch fast_calib calib.launch
```

After collecting at least three scenes, run multi-scene joint calibration:

```bash
roslaunch fast_calib multi_calib.launch
```

Typical multi-scene target placement:

<p align="center">
  <img src="./pics/multi-scene.jpg" width="100%">
  <font color=#a0a0a0 size=2>Placement of the calibration target for multi-scene data collection: (a) facing forward, (b) oriented to the right, (c) oriented to the left.</font>
</p>

### Fisheye cameras

Both camera models are supported and selected with `camera_model` in `config/qr_params.yaml`:

| `camera_model` | Model | Coefficients |
| --- | --- | --- |
| `pinhole` (default) | pinhole + radial-tangential | `k1, k2, p1, p2` |
| `fisheye` | Kannala-Brandt / equidistant (`cv::fisheye`) | `k1, k2, k3, k4` |

`fisheye` uses the same convention as OpenCV's `cv::fisheye`, Kalibr's `equidistant` and
FAST-LIVO2's `EquidistantCamera`. `camera_model` also accepts `equidistant`,
`EquidistantCamera`, `kb4`, `kannala_brandt` and `opencv_fisheye` (case-insensitive).
Any other value is rejected at startup rather than being silently treated as a pinhole camera.

For a fisheye camera the image is used **raw**: ArUco corners are detected on the original
fisheye image, mapped onto a virtual pinhole plane, and only then used for the board pose.
LiDAR points are likewise projected into the raw fisheye image for the colored cloud.
Marker corners whose off-axis angle exceeds `fisheye_max_theta_deg` (default `89`) are
dropped, since the Kannala-Brandt model cannot be inverted reliably outside its monotonic
range — wide-angle lenses are usually already black there. Detection requires at least
`min_detected_markers` (default 3) markers to remain after that filtering.

Steps:

1. Calibrate the intrinsics with the bundled script (checkerboard with 6 x 9 inner corners,
   0.1 m squares by default) and take `fx, fy, cx, cy, k1..k4` from its
   `calib_output/equidistant.yaml`:

   ```bash
   python3 scripts/calibrate_fisheye_intrinsics.py \
       --image_glob "images/*.png" \
       --checkerboard_cols 6 --checkerboard_rows 9 --square_size 0.10 \
       --output_dir calib_output
   ```

2. Put those values into `config/qr_params.yaml` together with `camera_model: fisheye` and
   `camera_width`/`camera_height` set to the **raw** resolution the intrinsics were
   calibrated at (the run aborts if they do not match the input image). A worked example
   for a 4000x3000 fisheye camera is kept as a comment block in that file.

3. Run the calibration as usual. The result file then carries a FAST-LIVO2-ready camera
   block (`cam_model: EquidistantCamera` with `k1..k4`), while pinhole cameras keep writing
   `cam_model: Pinhole` with `cam_d0..cam_d3`.

Wide-angle lenses need the target to stay inside the valid image circle; a board placed in
the black corners of a fisheye frame cannot be calibrated. The `qr_detect.png` written to
the output folder shows the detected markers, the reprojected annulus centers and the board
axes, which is the quickest way to confirm that the fisheye intrinsics are correct.

## 5. Standalone LiDAR Center Extraction Test

<details>
<summary>Show Unit Test Usage</summary>

The repository also provides a LiDAR-only test tool for checking annulus center extraction before running full camera-LiDAR calibration. It does not use the camera at all, so `camera_model` does not affect it.

Load parameters:

```bash
rosparam load config/qr_params.yaml /
rosparam set /output_path "$(rospack find fast_calib)/output"
```

Run solid-state LiDAR data:

```bash
rosrun fast_calib lidar_center_test calib_data/avia/left.bag /livox/lidar solid
rosrun fast_calib lidar_center_test calib_data/avia/mid.bag /livox/lidar solid
rosrun fast_calib lidar_center_test calib_data/avia/right.bag /livox/lidar solid
```

Run mechanical LiDAR data:

```bash
rosrun fast_calib lidar_center_test calib_data/hesai-jt128/left.bag /lidar_points mech
rosrun fast_calib lidar_center_test calib_data/hesai-jt128/mid.bag /lidar_points mech
rosrun fast_calib lidar_center_test calib_data/hesai-jt128/right.bag /lidar_points mech
```

The test tool writes:

- `*_centers.txt`: extracted annulus center coordinates
- `*_debug_cloud.pcd`: board point cloud, annulus points, boundary points, and center markers for visualization

Debug PCD colors:

- Board points: intensity color map
- Annulus points: green
- Solid-LiDAR boundary points: red
- Centers: white spheres

</details>
