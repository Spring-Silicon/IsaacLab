# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import carb
import warp as wp
from isaacsim.core.prims import XFormPrim
from isaacsim.core.version import get_version
from pxr import UsdGeom

from isaaclab.utils.warp.kernels import reshape_tiled_image

from ..sensor_base import SensorBase
from .camera import Camera

if TYPE_CHECKING:
    from .tiled_camera_cfg import TiledCameraCfg


def _reshape_tiled_image_cuda(
    tiled_image_buffer: wp.array, batched_image: wp.array, num_tiles_x: int
) -> None:
    """Copy a CUDA atlas into owned camera storage on the current Warp stream.

    A single tile already has the output layout, so it only needs a device copy.
    For RGBA atlases, reinterpret each aligned four-byte pixel as one uint32 to
    use the existing retile kernel with coalesced pixel-sized loads and stores.
    Neither path aliases the annotator's borrowed storage or changes its lifetime.
    Other dtypes keep the kernel's element-wise cast semantics.
    """
    count, height, width, channels = batched_image.shape
    source = tiled_image_buffer.flatten()
    if count == 1 and num_tiles_x == 1 and source.dtype == batched_image.dtype and batched_image.is_contiguous:
        # Ignore any trailing annotator padding, just as the retile kernel does.
        wp.copy(batched_image, source, count=batched_image.size)
        return

    destination = batched_image
    if (
        channels == 4
        and source.dtype == wp.uint8
        and destination.dtype == wp.uint8
        and destination.is_contiguous
        and source.ptr % 4 == 0
        and destination.ptr % 4 == 0
    ):
        source = wp.array(ptr=source.ptr, shape=(source.size // 4,), dtype=wp.uint32, device=source.device)
        destination = wp.array(
            ptr=destination.ptr, shape=(count, height, width, 1), dtype=wp.uint32, device=destination.device
        )
        channels = 1

    wp.launch(
        kernel=reshape_tiled_image,
        dim=(count, height, width),
        inputs=[source, destination, height, width, channels, num_tiles_x],
        device=batched_image.device,
    )


class TiledCamera(Camera):
    r"""The tiled rendering based camera sensor for acquiring the same data as the Camera class.

    This class inherits from the :class:`Camera` class but uses the tiled-rendering API to acquire
    the visual data. Tiled-rendering concatenates the rendered images from multiple cameras into a single image.
    This allows for rendering multiple cameras in parallel and is useful for rendering large scenes with multiple
    cameras efficiently.

    The following sensor types are supported:

    - ``"rgb"``: A 3-channel rendered color image.
    - ``"rgba"``: A 4-channel rendered color image with alpha channel.
    - ``"distance_to_camera"``: An image containing the distance to camera optical center.
    - ``"distance_to_image_plane"``: An image containing distances of 3D points from camera plane along camera's z-axis.
    - ``"depth"``: Alias for ``"distance_to_image_plane"``.
    - ``"normals"``: An image containing the local surface normal vectors at each pixel.
    - ``"motion_vectors"``: An image containing the motion vector data at each pixel.
    - ``"semantic_segmentation"``: The semantic segmentation data.
    - ``"instance_segmentation_fast"``: The instance segmentation data.
    - ``"instance_id_segmentation_fast"``: The instance id segmentation data.

    .. note::
        Currently the following sensor types are not supported in a "view" format:

        - ``"instance_segmentation"``: The instance segmentation data. Please use the fast counterparts instead.
        - ``"instance_id_segmentation"``: The instance id segmentation data. Please use the fast counterparts instead.
        - ``"bounding_box_2d_tight"``: The tight 2D bounding box data (only contains non-occluded regions).
        - ``"bounding_box_2d_tight_fast"``: The tight 2D bounding box data (only contains non-occluded regions).
        - ``"bounding_box_2d_loose"``: The loose 2D bounding box data (contains occluded regions).
        - ``"bounding_box_2d_loose_fast"``: The loose 2D bounding box data (contains occluded regions).
        - ``"bounding_box_3d"``: The 3D view space bounding box data.
        - ``"bounding_box_3d_fast"``: The 3D view space bounding box data.

    .. _replicator extension: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html#annotator-output
    .. _USDGeom Camera: https://graphics.pixar.com/usd/docs/api/class_usd_geom_camera.html

    .. versionadded:: v1.0.0

        This feature is available starting from Isaac Sim 4.2. Before this version, the tiled rendering APIs
        were not available.

    """

    cfg: TiledCameraCfg
    """The configuration parameters."""

    def __init__(self, cfg: TiledCameraCfg):
        """Initializes the tiled camera sensor.

        Args:
            cfg: The configuration parameters.

        Raises:
            RuntimeError: If no camera prim is found at the given path.
            RuntimeError: If Isaac Sim version < 4.2
            ValueError: If the provided data types are not supported by the camera.
        """
        isaac_sim_version = float(".".join(get_version()[2:4]))
        if isaac_sim_version < 4.2:
            raise RuntimeError(
                f"TiledCamera is only available from Isaac Sim 4.2.0. Current version is {isaac_sim_version}. Please"
                " update to Isaac Sim 4.2.0"
            )
        super().__init__(cfg)

    def __del__(self):
        """Unsubscribes from callbacks and detach from the replicator registry."""
        # unsubscribe from callbacks
        SensorBase.__del__(self)
        # detach from the replicator registry
        for annotator in self._annotators.values():
            annotator.detach(self.render_product_paths)

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        # message for class
        return (
            f"Tiled Camera @ '{self.cfg.prim_path}': \n"
            f"\tdata types   : {list(self.data.output.keys())} \n"
            f"\tsemantic filter : {self.cfg.semantic_filter}\n"
            f"\tcolorize semantic segm.   : {self.cfg.colorize_semantic_segmentation}\n"
            f"\tcolorize instance segm.   : {self.cfg.colorize_instance_segmentation}\n"
            f"\tcolorize instance id segm.: {self.cfg.colorize_instance_id_segmentation}\n"
            f"\tupdate period (s): {self.cfg.update_period}\n"
            f"\tshape        : {self.image_shape}\n"
            f"\tnumber of sensors : {self._view.count}"
        )

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        if not self._is_initialized:
            raise RuntimeError(
                "TiledCamera could not be initialized. Please ensure --enable_cameras is used to enable rendering."
            )
        # reset the timestamps
        SensorBase.reset(self, env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # reset the frame count
        self._frame[env_ids] = 0

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers.

        This function creates handles and registers the provided data types with the replicator registry to
        be able to access the data from the sensor. It also initializes the internal buffers to store the data.

        Raises:
            RuntimeError: If the number of camera prims in the view does not match the number of environments.
            RuntimeError: If replicator was not found.
        """
        carb_settings_iface = carb.settings.get_settings()
        if not carb_settings_iface.get("/isaaclab/cameras_enabled"):
            raise RuntimeError(
                "A camera was spawned without the --enable_cameras flag. Please use --enable_cameras to enable"
                " rendering."
            )

        import omni.replicator.core as rep

        # Initialize parent class
        SensorBase._initialize_impl(self)
        # Create a view for the sensor
        self._view = XFormPrim(self.cfg.prim_path, reset_xform_properties=False)
        self._view.initialize()
        # Check that sizes are correct
        if self._view.count != self._num_envs:
            raise RuntimeError(
                f"Number of camera prims in the view ({self._view.count}) does not match"
                f" the number of environments ({self._num_envs})."
            )

        # Create all env_ids buffer
        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)
        # Create frame count buffer
        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)

        # Convert all encapsulated prims to Camera
        for cam_prim_path in self._view.prim_paths:
            # Get camera prim
            cam_prim = self.stage.GetPrimAtPath(cam_prim_path)
            # Check if prim is a camera
            if not cam_prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"Prim at path '{cam_prim_path}' is not a Camera.")
            # Add to list
            sensor_prim = UsdGeom.Camera(cam_prim)
            self._sensor_prims.append(sensor_prim)

        # Create replicator tiled render product
        rp = rep.create.render_product_tiled(
            cameras=self._view.prim_paths, tile_resolution=(self.cfg.width, self.cfg.height)
        )
        self._render_product_paths = [rp.path]

        # RTX renders on the GPU even when the simulation device is the CPU. Annotators therefore always
        # deliver CUDA buffers (texture -> device buffer); CPU-device outputs are filled from device staging
        # buffers with one device-to-host copy each (see :meth:`_create_buffers`), instead of replicator's
        # direct texture -> host-buffer copy and a host-side retile.
        if self.device == "cpu":
            self._annotator_device = f"cuda:{carb_settings_iface.get('/renderer/activeGpu') or 0}"
        else:
            self._annotator_device = self.device

        # Define the annotators based on requested data types
        self._annotators = dict()
        for annotator_type in self.cfg.data_types:
            if annotator_type == "rgba" or annotator_type == "rgb":
                annotator = rep.AnnotatorRegistry.get_annotator(
                    "rgb", device=self._annotator_device, do_array_copy=False
                )
                self._annotators["rgba"] = annotator
            elif annotator_type == "depth" or annotator_type == "distance_to_image_plane":
                # keep depth for backwards compatibility
                annotator = rep.AnnotatorRegistry.get_annotator(
                    "distance_to_image_plane", device=self._annotator_device, do_array_copy=False
                )
                self._annotators[annotator_type] = annotator
            # note: we are verbose here to make it easier to understand the code.
            #   if colorize is true, the data is mapped to colors and a uint8 4 channel image is returned.
            #   if colorize is false, the data is returned as a uint32 image with ids as values.
            else:
                init_params = None
                if annotator_type == "semantic_segmentation":
                    init_params = {
                        "colorize": self.cfg.colorize_semantic_segmentation,
                        "mapping": json.dumps(self.cfg.semantic_segmentation_mapping),
                    }
                elif annotator_type == "instance_segmentation_fast":
                    init_params = {"colorize": self.cfg.colorize_instance_segmentation}
                elif annotator_type == "instance_id_segmentation_fast":
                    init_params = {"colorize": self.cfg.colorize_instance_id_segmentation}

                annotator = rep.AnnotatorRegistry.get_annotator(
                    annotator_type, init_params, device=self._annotator_device, do_array_copy=False
                )
                self._annotators[annotator_type] = annotator

        # Attach the annotator to the render product
        for annotator in self._annotators.values():
            annotator.attach(self._render_product_paths)

        # Create internal buffers
        self._create_buffers()

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        # Increment frame count
        self._frame[env_ids] += 1

        # update latest camera pose
        if self.cfg.update_latest_camera_pose:
            self._update_poses(env_ids)

        # Extract the flattened image buffer
        for data_type, annotator in self._annotators.items():
            # check whether returned data is a dict (used for segmentation)
            output = annotator.get_data()
            if isinstance(output, dict):
                tiled_data_buffer = output["data"]
                self._data.info[data_type] = output["info"]
            else:
                tiled_data_buffer = output

            # Annotators deliver CUDA buffers (see `_initialize_impl`). Keep them on the annotator device.
            if isinstance(tiled_data_buffer, np.ndarray):
                tiled_data_buffer = wp.array(tiled_data_buffer, device=self._annotator_device, dtype=wp.uint8)
            else:
                tiled_data_buffer = tiled_data_buffer.to(device=self._annotator_device)

            # process data for different segmentation types
            # Note: Replicator returns raw buffers of dtype uint32 for segmentation types
            #   so we need to convert them to uint8 4 channel images for colorized types
            if (
                (data_type == "semantic_segmentation" and self.cfg.colorize_semantic_segmentation)
                or (data_type == "instance_segmentation_fast" and self.cfg.colorize_instance_segmentation)
                or (data_type == "instance_id_segmentation_fast" and self.cfg.colorize_instance_id_segmentation)
            ):
                tiled_data_buffer = wp.array(
                    ptr=tiled_data_buffer.ptr,
                    shape=(*tiled_data_buffer.shape, 4),
                    dtype=wp.uint8,
                    device=self._annotator_device,
                )

            # For motion vectors, we only require the first two channels of the tiled buffer
            # Note: Not doing this breaks the alignment of the data (check: https://github.com/isaac-sim/IsaacLab/issues/2003)
            if data_type == "motion_vectors":
                tiled_data_buffer = tiled_data_buffer[:, :, :2].contiguous()

            if self._host_staging is None:
                _reshape_tiled_image_cuda(
                    tiled_data_buffer, wp.from_torch(self._data.output[data_type]), self._tiling_grid_shape()[0]
                )
                # alias rgb as first 3 channels of rgba
                if data_type == "rgba" and "rgb" in self.cfg.data_types:
                    self._data.output["rgb"] = self._data.output["rgba"][..., :3]
            else:
                self._read_back_to_host(data_type, tiled_data_buffer)

            # NOTE: The `distance_to_camera` annotator returns the distance to the camera optical center. However,
            #       the replicator depth clipping is applied w.r.t. to the image plane which may result in values
            #       larger than the clipping range in the output. We apply an additional clipping to ensure values
            #       are within the clipping range for all the annotators.
            if data_type == "distance_to_camera":
                self._data.output[data_type][
                    self._data.output[data_type] > self.cfg.spawn.clipping_range[1]
                ] = torch.inf
            # apply defined clipping behavior
            if (
                data_type == "distance_to_camera" or data_type == "distance_to_image_plane" or data_type == "depth"
            ) and self.cfg.depth_clipping_behavior != "none":
                self._data.output[data_type][torch.isinf(self._data.output[data_type])] = (
                    0.0 if self.cfg.depth_clipping_behavior == "zero" else self.cfg.spawn.clipping_range[1]
                )

    """
    Private Helpers
    """

    def _check_supported_data_types(self, cfg: TiledCameraCfg):
        """Checks if the data types are supported by the ray-caster camera."""
        # check if there is any intersection in unsupported types
        # reason: these use np structured data types which we can't yet convert to torch tensor
        common_elements = set(cfg.data_types) & Camera.UNSUPPORTED_TYPES
        if common_elements:
            # provide alternative fast counterparts
            fast_common_elements = []
            for item in common_elements:
                if "instance_segmentation" in item or "instance_id_segmentation" in item:
                    fast_common_elements.append(item + "_fast")
            # raise error
            raise ValueError(
                f"TiledCamera class does not support the following sensor types: {common_elements}."
                "\n\tThis is because these sensor types output numpy structured data types which"
                "can't be converted to torch tensors easily."
                "\n\tHint: If you need to work with these sensor types, we recommend using their fast counterparts."
                f"\n\t\tFast counterparts: {fast_common_elements}"
            )

    def _create_buffers(self):
        """Create buffers for storing data."""
        # create the data object
        # -- pose of the cameras
        self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_w_world = torch.zeros((self._view.count, 4), device=self._device)
        self._update_poses(self._ALL_INDICES)
        # -- intrinsic matrix
        self._data.intrinsic_matrices = torch.zeros((self._view.count, 3, 3), device=self._device)
        self._update_intrinsic_matrices(self._ALL_INDICES)
        self._data.image_shape = self.image_shape
        # -- output data
        data_dict = dict()
        if "rgba" in self.cfg.data_types or "rgb" in self.cfg.data_types:
            data_dict["rgba"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 4), device=self.device, dtype=torch.uint8
            ).contiguous()
        if "rgb" in self.cfg.data_types:
            # RGB is the first 3 channels of RGBA
            data_dict["rgb"] = data_dict["rgba"][..., :3]
        if "distance_to_image_plane" in self.cfg.data_types:
            data_dict["distance_to_image_plane"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.float32
            ).contiguous()
        if "depth" in self.cfg.data_types:
            data_dict["depth"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.float32
            ).contiguous()
        if "distance_to_camera" in self.cfg.data_types:
            data_dict["distance_to_camera"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.float32
            ).contiguous()
        if "normals" in self.cfg.data_types:
            data_dict["normals"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 3), device=self.device, dtype=torch.float32
            ).contiguous()
        if "motion_vectors" in self.cfg.data_types:
            data_dict["motion_vectors"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 2), device=self.device, dtype=torch.float32
            ).contiguous()
        if "semantic_segmentation" in self.cfg.data_types:
            if self.cfg.colorize_semantic_segmentation:
                data_dict["semantic_segmentation"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 4), device=self.device, dtype=torch.uint8
                ).contiguous()
            else:
                data_dict["semantic_segmentation"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.int32
                ).contiguous()
        if "instance_segmentation_fast" in self.cfg.data_types:
            if self.cfg.colorize_instance_segmentation:
                data_dict["instance_segmentation_fast"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 4), device=self.device, dtype=torch.uint8
                ).contiguous()
            else:
                data_dict["instance_segmentation_fast"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.int32
                ).contiguous()
        if "instance_id_segmentation_fast" in self.cfg.data_types:
            if self.cfg.colorize_instance_id_segmentation:
                data_dict["instance_id_segmentation_fast"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 4), device=self.device, dtype=torch.uint8
                ).contiguous()
            else:
                data_dict["instance_id_segmentation_fast"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.int32
                ).contiguous()

        # CPU-device outputs are filled from same-shaped device staging buffers (one retile on the GPU,
        # then one asynchronous device-to-host copy per output into pinned memory). On this path "rgb"
        # is a separate contiguous buffer holding exactly RGBA[..., :3], compacted on the GPU, so
        # consumers that copy RGB do not pay a strided host copy.
        self._host_staging = None
        if self._annotator_device != self.device:
            self._host_staging = dict()
            for name, value in data_dict.items():
                if name != "rgb":
                    self._host_staging[name] = torch.empty_like(value, device=self._annotator_device)
                    data_dict[name] = value.pin_memory()
            if "rgb" in data_dict:
                rgb_shape = (self._view.count, self.cfg.height, self.cfg.width, 3)
                self._host_staging["rgb"] = torch.empty(rgb_shape, device=self._annotator_device, dtype=torch.uint8)
                data_dict["rgb"] = torch.zeros(rgb_shape, dtype=torch.uint8).pin_memory()
            # Wait for the copies with a blocking-sync event: the thread sleeps until the GPU signals
            # completion instead of spin-waiting a core (CUDA's default may spin), same completion point.
            self._readback_done = torch.cuda.Event(blocking=True)

        self._data.output = data_dict
        self._data.info = dict()

    def _read_back_to_host(self, data_type: str, tiled_data_buffer: wp.array):
        """Unpack a CUDA atlas into device staging and copy it into the pinned CPU output.

        Everything is ordered on the current torch stream of the annotator device; the method returns only
        after the host buffers are complete, so CPU consumers observe this frame's data.
        """
        staging = self._host_staging[data_type]
        stream = torch.cuda.current_stream(staging.device)
        with wp.ScopedStream(wp.stream_from_torch(stream)):
            _reshape_tiled_image_cuda(tiled_data_buffer, wp.from_torch(staging), self._tiling_grid_shape()[0])
        with torch.cuda.stream(stream):
            self._data.output[data_type].copy_(staging, non_blocking=True)
            if data_type == "rgba" and "rgb" in self._host_staging:
                rgb = self._host_staging["rgb"]
                rgb.copy_(staging[..., :3])
                self._data.output["rgb"].copy_(rgb, non_blocking=True)
            self._readback_done.record(stream)
        self._readback_done.synchronize()

    def _tiled_image_shape(self) -> tuple[int, int]:
        """Returns a tuple containing the dimension of the tiled image."""
        cols, rows = self._tiling_grid_shape()
        return (self.cfg.width * cols, self.cfg.height * rows)

    def _tiling_grid_shape(self) -> tuple[int, int]:
        """Returns a tuple containing the tiling grid dimension."""
        cols = math.ceil(math.sqrt(self._view.count))
        rows = math.ceil(self._view.count / cols)
        return (cols, rows)

    def _create_annotator_data(self):
        # we do not need to create annotator data for the tiled camera sensor
        raise RuntimeError("This function should not be called for the tiled camera sensor.")

    def _process_annotator_output(self, name: str, output: Any) -> tuple[torch.tensor, dict | None]:
        # we do not need to process annotator output for the tiled camera sensor
        raise RuntimeError("This function should not be called for the tiled camera sensor.")

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._view = None
