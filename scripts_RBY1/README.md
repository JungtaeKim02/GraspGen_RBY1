# GraspGen Inference Node for RBY1

This ROS 2 node performs 6-DOF grasp pose generation using the **GraspGen** framework. It is designed specifically for the **RBY1** robot (dual-arm setup) but is configurable for other configurations.

The node waits for a service trigger, captures a point cloud, runs the inference model, filters the results based on the target arm (Left/Right), and publishes valid grasp poses for motion planning.

## 🚀 Key Features
* **On-Demand Inference:** Triggered via a ROS service to save resources when not grasping.
* **Dual-Arm Support:** Filters grasp poses based on the requested arm (`left` or `right`) to ensure kinematic feasibility.
* **Geometric Filtering:** Removes grasp approaches that are difficult for the specific arm to reach using a tolerance parameter.
* **Frame Transformation:** Automatically transforms input point clouds to the robot's base frame before inference.

---

## ⚙️ Parameters & Configuration

You can configure these parameters in your launch file or via the command line (`ros2 run ... --ros-args -p param_name:=value`).

### 1. Model & Input Configuration

| Parameter | Type | Default Value | Intent & Usage |
| :--- | :--- | :--- | :--- |
| `config_path` | `string` | `.../graspgen_franka_panda.yml` | **Model Selection.** <br>Path to the GraspGen model configuration file. Change this if you train a new model or move the weight files. |
| `input_pc_topic`| `string` | `/grounded_sam/pointcloud/objects` | **Sensor Source.** <br>The topic name for the input `PointCloud2`. Change this if using a different camera or a segmented point cloud topic (e.g., from YOLO/SAM). |
| `robot_base_frame`| `string` | `link_torso_5` | **Reference Frame.** <br>The target frame for output poses. The input point cloud is transformed *into* this frame. <br>Use the frame that your motion planner (MoveIt) uses as the planning frame (e.g., `base_link`, `torso`). |

### 2. Inference Tuning

| Parameter | Type | Default Value | Intent & Usage |
| :--- | :--- | :--- | :--- |
| `top_k` | `int` | `100` | **Performance vs. Variety.** <br>The number of grasp candidates to publish, sorted by confidence score.<br>• **Increase** if MoveIt fails to find a valid path (gives the planner more options).<br>• **Decrease** to reduce computational load on the planner. |
| `z_offset` | `float` | `0.065063` | **Gripper Alignment.** <br>Offsets the grasp pose along the approach vector (Z-axis). <br>Use this to compensate for the distance between the GraspGen prediction point and the actual Tool Center Point (TCP) of your physical gripper. |

### 3. Kinematic Filtering (Crucial)

| Parameter | Type | Default Value | Intent & Usage |
| :--- | :--- | :--- | :--- |
| `y_filter_tolerance`| `float` | `1.0` | **Workspace Restriction.** <br>Used to filter out grasp approaches that are "awkward" for the selected arm.<br>• The node calculates the approach vector (local Z) relative to the base.<br>• **Lower value (e.g., 0.0)**: Strict filtering. Left arm will *only* grasp from the left side.<br>• **Higher value (e.g., 1.0)**: Loose filtering. Allows the arm to reach slightly across the center line.<br>• **Tuning:** If the robot tries to cross its arms or reach unreachable angles, **decrease** this value. |

---

## 📥 Inputs & Outputs

### Subscribed Topics
* **`[input_pc_topic]`** (`sensor_msgs/PointCloud2`): The 3D point cloud of the object(s) to be grasped.

### Published Topics
* **`/grasp_candidates`** (`geometry_msgs/PoseArray`): The final list of grasp poses transformed to `robot_base_frame`.
* **`/filtered_grasp_poses_viz`** (`geometry_msgs/PoseArray`): Same data as above, intended for Rviz visualization.

### Services
* **`/manipulation/graspgen/enable`** (`inha_interfaces/srv/GraspgenEnable`)
    * **Request:** `string arm_id` ("left" or "right")
    * **Response:** `bool success`, `string message`
    * **Description:** Activates the node. The node captures the *next* available point cloud message, processes it for the specified arm, and then automatically disables itself.

---

## 📝 Usage Example

### 1. Running the Node
```bash
ros2 run <your_package_name> graspgen_inference_node --ros-args -p y_filter_tolerance:=0.5 -p top_k:=50
