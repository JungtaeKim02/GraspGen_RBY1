#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import torch
import trimesh.transformations as tra
import os
import time

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, PoseStamped

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from easydict import EasyDict as edict

from inha_interfaces.srv import GraspgenEnable

class GraspGenInferenceNode(Node):
    def __init__(self):
        super().__init__('graspgen_inference_node')
        
        self.declare_parameter('config_path', '/home/GraspGen/GraspGenModels/checkpoints/graspgen_franka_panda.yml')
        self.declare_parameter('robot_base_frame', 'link_torso_5') 
        self.declare_parameter('top_k', 100)
        self.declare_parameter('z_offset', 0.065063)
        self.declare_parameter('input_pc_topic', '/grounded_sam/pointcloud/objects')
        self.declare_parameter('y_filter_tolerance', 1.0)
        
        self.config_path = self.get_parameter('config_path').value
        self.robot_base_frame = self.get_parameter('robot_base_frame').value
        self.top_k = self.get_parameter('top_k').value
        self.z_offset = self.get_parameter('z_offset').value
        self.input_topic = self.get_parameter('input_pc_topic').value
        self.y_tolerance = self.get_parameter('y_filter_tolerance').value
        

        self.is_enabled = False 
        self.target_arm = None 
        
        self.dq_flip_z_wxyz = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sampler = None
        self._load_model()
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.pc_sub = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.pointcloud_callback,
            qos
        )
        
        self.srv = self.create_service(
            GraspgenEnable, 
            '/manipulation/graspgen/enable', 
            self.enable_callback
        )
        
        self.candidates_pub = self.create_publisher(PoseArray, '/grasp_candidates', 10)
        self.viz_pub = self.create_publisher(PoseArray, '/filtered_grasp_poses_viz', 10)
        
        self.get_logger().info(f"✅ Node Ready. Tolerance={self.y_tolerance}. Waiting for Service Call...")

    def _load_model(self):
        if not os.path.exists(self.config_path):
            self.get_logger().error(f"Config missing: {self.config_path}")
            return
        try:
            cfg = load_grasp_cfg(self.config_path)
            self.sampler = GraspGenSampler(cfg)
            self.get_logger().info('Model loaded successfully.')
        except Exception as e:
            self.get_logger().error(f"Model load failed: {e}")

    def enable_callback(self, request, response):
        if request.arm_id in ['left', 'right']:
            self.is_enabled = True
            self.target_arm = request.arm_id
            
            response.success = True
            response.message = f"GraspGen Enabled for {self.target_arm} arm. Waiting for PointCloud..."
            self.get_logger().info(f"🔔 Service: ENABLED for [{self.target_arm}]. Ready to process next frame.")
        else:
            self.is_enabled = False
            response.success = False
            response.message = f"Invalid arm_id: '{request.arm_id}'. Use 'left' or 'right'."
            self.get_logger().warn(f"🔕 Service: Invalid Request ({request.arm_id}).")
            
        return response

    def pointcloud_callback(self, msg: PointCloud2):
        if not self.is_enabled:
            return
        
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                self.robot_base_frame,
                msg.header.frame_id,
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF Lookup Failed: {e}", throttle_duration_sec=2)
            return

        self.is_enabled = False 
        current_target = self.target_arm 
        self.get_logger().info(f"PointCloud received. Starting GraspGen Inference for [{current_target}]...")
        
        try:
            n_points = msg.width * msg.height
            pc_numpy = np.frombuffer(msg.data, dtype=np.float32).reshape(n_points, 3).copy()
        except Exception as e:
            self.get_logger().error(f"PC conversion error: {e}")
            return
            
        try:
            pc_torch = torch.from_numpy(pc_numpy).float().to(self.device)
            
            if pc_torch.shape[0] > 2048:
                sel = np.random.choice(pc_torch.shape[0], 2048, replace=False)
                pc_torch = pc_torch[sel]
            
            res = self.sampler.sample(pc_torch)
            
            if isinstance(res, dict):
                grasps_t = res.get("grasps") or res.get("poses")
                scores_t = res.get("scores") or res.get("conf")
            else:
                grasps_t, scores_t = res[0], res[1]

            grasps_np = grasps_t.detach().cpu().numpy()
            scores_np = scores_t.detach().cpu().numpy().reshape(-1)
            
            order = np.argsort(-scores_np)
            if self.top_k is not None:
                order = order[:self.top_k]
                
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        final_pose_array = self._create_pose_array(grasps_np, order, transform_stamped, current_target)
        
        self.candidates_pub.publish(final_pose_array)
        self.viz_pub.publish(final_pose_array)
        
        self.get_logger().info(f"✅ Done. Published {len(final_pose_array.poses)} poses. (State: Locked)")

    def _create_pose_array(self, grasps_np, order, transform, target_arm):
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()
        pose_array_msg.header.frame_id = self.robot_base_frame

        for i in order:
            pose_matrix = grasps_np[i]
            try:
                p = tra.translation_from_matrix(pose_matrix)
                qw, qx, qy, qz = tra.quaternion_from_matrix(pose_matrix)
                
                ps = PoseStamped()
                ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = float(p[0]), float(p[1]), float(p[2])
                ps.pose.orientation.w, ps.pose.orientation.x = float(qw), float(qx)
                ps.pose.orientation.y, ps.pose.orientation.z = float(qy), float(qz)
                
                transformed_ps = do_transform_pose(ps.pose, transform)
                
                q_curr = np.array([
                    transformed_ps.orientation.w, transformed_ps.orientation.x,
                    transformed_ps.orientation.y, transformed_ps.orientation.z
                ])
                
                q_new = tra.quaternion_multiply(q_curr, self.dq_flip_z_wxyz)
                norm = np.linalg.norm(q_new)
                if norm > 0: q_new = q_new / norm

                rot = tra.quaternion_matrix(q_new) 
                local_z_in_base = rot[:3, 2] 

                if target_arm == 'left':
                    if local_z_in_base[1] < -self.y_tolerance:
                        continue

                elif target_arm == 'right':
                    if local_z_in_base[1] > self.y_tolerance:
                        continue
                
                offset = local_z_in_base * self.z_offset
                transformed_ps.position.x += offset[0]
                transformed_ps.position.y += offset[1]
                transformed_ps.position.z += offset[2]
                
                transformed_ps.orientation.w, transformed_ps.orientation.x = float(q_new[0]), float(q_new[1])
                transformed_ps.orientation.y, transformed_ps.orientation.z = float(q_new[2]), float(q_new[3])
                
                pose_array_msg.poses.append(transformed_ps)

            except Exception as e:
                self.get_logger().error(f"Error: {e}")
                continue
                
        return pose_array_msg

def main(args=None):
    rclpy.init(args=args)
    node = GraspGenInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
