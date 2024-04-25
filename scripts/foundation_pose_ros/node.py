#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
import rospy
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ImageMsg
from message_filters import TimeSynchronizer

# from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber

# np.float = np.float64  # temp fix for following import
# import ros_numpy

from ros_numpy import numpify
from ros_numpy import msgify

# CNOS
from hydra import initialize
from hydra import compose
from hydra.utils import instantiate
import torch
import glob
from cnos.src.model.loss import Similarity
from omegaconf import OmegaConf
from cnos.src.utils.bbox_utils import CropResizePad
from cnos.src.model.utils import Detections
from PIL import Image


def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """
    q = np.empty((4,), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


class CnosROS:
    def __init__(
        self,
        config_dir,
        template_dir,
        num_max_dets,
        conf_threshold,
        stability_score_thresh,
    ) -> None:
        with initialize(version_base=None, config_path=config_dir):
            cfg = compose(config_name="run_inference.yaml")
        cfg_segmentor = cfg.model.segmentor_model
        if "fast_sam" in cfg_segmentor._target_:
            rospy.loginfo("Using FastSAM, ignore stability_score_thresh!")
        else:
            cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
        self.metric = Similarity()
        rospy.loginfo("Initializing model")
        # rospy.loginfo(cfg)
        self.model = instantiate(cfg.model)
        rospy.loginfo("Here!!!")

        self.num_max_dets = num_max_dets
        self.conf_threshold = conf_threshold
        self.stability_score_thresh = stability_score_thresh

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.descriptor_model.model = self.model.descriptor_model.model.to(device)
        self.model.descriptor_model.model.device = device
        # if there is predictor in the model, move it to device
        if hasattr(self.model.segmentor_model, "predictor"):
            self.model.segmentor_model.predictor.model = (
                self.model.segmentor_model.predictor.model.to(device)
            )
        else:
            self.model.segmentor_model.model.setup_model(device=device, verbose=True)
        rospy.loginfo(f"Moving models to {device} done!")

        rospy.loginfo("Initializing template")
        template_paths = glob.glob(f"{template_dir}/*.png")
        boxes, templates = [], []
        for path in template_paths:
            image = Image.open(path)
            boxes.append(image.getbbox())

            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            templates.append(image)

        templates = torch.stack(templates).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))

        processing_config = OmegaConf.create(
            {
                "image_size": 224,
            }
        )
        proposal_processor = CropResizePad(processing_config.image_size)
        templates = proposal_processor(images=templates, boxes=boxes).cuda()
        # save_image(templates, f"{template_dir}/cnos_results/templates.png", nrow=7)
        self.ref_feats = self.model.descriptor_model.compute_features(
            templates, token_name="x_norm_clstoken"
        )
        rospy.loginfo(f"Ref feats: {self.ref_feats.shape}")

    def run_inference(self, rgb):
        # run inference
        # rgb = Image.open(rgb_path).convert("RGB")
        detections = self.model.segmentor_model.generate_masks(np.array(rgb))
        detections = Detections(detections)
        decriptors = self.model.descriptor_model.forward(np.array(rgb), detections)

        # get scores per proposal
        scores = self.metric(decriptors[:, None, :], self.ref_feats[None, :, :])
        score_per_detection = torch.topk(scores, k=5, dim=-1)[0]
        score_per_detection = torch.mean(score_per_detection, dim=-1)

        # get top-k detections
        scores, index = torch.topk(score_per_detection, k=self.num_max_dets, dim=-1)
        detections.filter(index)

        # keep only detections with score > conf_threshold
        detections.filter(scores > self.conf_threshold)
        detections.add_attribute("scores", scores)
        detections.add_attribute("object_ids", torch.zeros_like(scores))

        detections.to_numpy()
        rospy.loginfo(detections)
        # save_path = f"{template_dir}/cnos_results/detection"
        # detections.save_to_file(0, 0, 0, save_path, "custom", return_results=False)
        # detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path + ".npz"])
        # save_json_bop23(save_path + ".json", detections)
        # vis_img = visualize(rgb, detections)
        # vis_img.save(f"{template_dir}/cnos_results/vis.png")
        return detections.masks


class FoundationalPoseROSNode:

    def __init__(self):
        code_dir = os.path.dirname(os.path.realpath(__file__))
        self.mesh_file = "/home/felipe/foundational_pose/catkin_ws/src/foundational_pose_ros/scripts/foundation_pose_ros/demo_data/mustard0/mesh/textured_simple.obj"
        self.est_refine_iter = 5
        self.track_refine_iter = 2
        self.debug = 0
        self.debug_dir = f"{code_dir}/debug"

        self.br = TransformBroadcaster()

        set_logging_format()
        set_seed(0)

        mesh = trimesh.load(self.mesh_file)

        os.system(
            f"rm -rf {self.debug_dir}/* && mkdir -p {self.debug_dir}/track_vis {self.debug_dir}/ob_in_cam"
        )

        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(mesh)
        self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0).reshape(
            2, 3
        )

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir=self.debug_dir,
            debug=self.debug,
            glctx=self.glctx,
        )
        self.K = None
        self.previous_pose = None

        self.cnos_est = CnosROS(
            "./cnos/configs",
            "/home/felipe/foundational_pose/catkin_ws/src/foundational_pose_ros/scripts/foundation_pose_ros/cnos/media/mustard/rendered/",
            1,
            0.5,
            0.5,
        )

        rospy.Subscriber("/rs_1/color/camera_info", CameraInfo, self.camera_info_cb)
        self.debug_pub = rospy.Publisher("pose_estimation", ImageMsg, queue_size=1)

        self.atss = TimeSynchronizer(
            [
                Subscriber("/rs_1/color/image_raw", ImageMsg),
                Subscriber("/rs_1/aligned_depth_to_color/image_raw", ImageMsg),
            ],
            queue_size=1,
        )
        self.atss.registerCallback(self.estimate)

    def camera_info_cb(self, camera_info_msg: CameraInfo):
        k_raw = camera_info_msg.K
        self.K = np.array(k_raw).reshape((3, 3))

    def generate_mask(self, color_array):
        mask = self.cnos_est.run_inference(color_array)
        return mask.reshape(mask.shape[1:]).astype(bool)

    def estimate(self, color: ImageMsg, depth: ImageMsg):
        if self.K is None:
            return
        assert color.header.stamp == depth.header.stamp
        # rospy.loginfo("got an rbg and depth img")

        color_array = numpify(color).astype(np.uint8)
        depth_array = numpify(depth).astype(np.float64) * 1e-3
        # rospy.loginfo(color_array.shape)
        # rospy.loginfo(depth_array.shape)

        # return
        pose = None
        if self.previous_pose is None:
            # mask = reader.get_mask(0).astype(bool)
            mask = self.generate_mask(color_array)
            # rospy.logwarn(f"{mask.max()} : {mask.min()}")
            # rospy.logwarn(mask.shape)
            # mask_img = mask.astype(np.uint8) * 255
            # rospy.logwarn(mask.shape)
            # debug_img = msgify(ImageMsg, mask_img, encoding="mono8")
            # self.debug_pub.publish(debug_img)
            pose = self.est.register(
                K=self.K,
                rgb=color_array,
                depth=depth_array,
                ob_mask=mask,
                iteration=self.est_refine_iter,
            )
            # return
            # rospy.loginfo("AAAAAAAAAAAAAAa")
        else:
            pose = self.est.track_one(
                rgb=color_array,
                depth=depth_array,
                K=self.K,
                iteration=self.track_refine_iter,
            )
        self.previous_pose = pose

        if pose is None:
            return

        center_pose = pose @ np.linalg.inv(self.to_origin)
        # center_pose = pose
        vis = draw_posed_3d_box(
            self.K, img=color_array, ob_in_cam=center_pose, bbox=self.bbox
        )
        vis = draw_xyz_axis(
            color_array,
            ob_in_cam=center_pose,
            scale=0.1,
            K=self.K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        debug_img = msgify(ImageMsg, vis, encoding="rgb8")
        self.debug_pub.publish(debug_img)

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "camera"
        t.child_frame_id = "object"
        q = quaternion_from_matrix(center_pose)
        # q = [0, 0, 0, 1]

        t.transform.translation.x = center_pose[0, 3]
        t.transform.translation.y = center_pose[1, 3]
        t.transform.translation.z = center_pose[2, 3]
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.br.sendTransform(t)


if __name__ == "__main__":
    rospy.init_node("foundation_pose")
    pose_estimator = FoundationalPoseROSNode()
    rospy.spin()
