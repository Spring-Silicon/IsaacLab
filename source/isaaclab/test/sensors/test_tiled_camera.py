# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

import os

from isaaclab.app import AppLauncher

# GPU under test (for example cuda:1 on a multi-GPU host) and extra Kit arguments; default: the first GPU.
DEVICE = os.environ.get("ISAACLAB_TEST_DEVICE", "cuda:0")

# launch omniverse app
simulation_app = AppLauncher(
    headless=True, enable_cameras=True, device=DEVICE, kit_args=os.environ.get("ISAACLAB_TEST_KIT_ARGS", "")
).app

"""Rest everything follows."""

import copy
import numpy as np
import random
import torch

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import omni.replicator.core as rep
import pytest
import warp as wp
from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim
from pxr import Gf, UsdGeom

# from Isaac Sim 4.2 onwards, pxr.Semantics is deprecated
try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg, TiledCamera, TiledCameraCfg
from isaaclab.sensors.camera.tiled_camera import _reshape_tiled_image_cuda
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.timer import Timer


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 10])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize(
    "channels,input_dtype,output_dtype",
    [(4, np.uint8, np.uint8), (1, np.float32, np.float32), (2, np.float32, np.float32),
     (3, np.float32, np.float32), (1, np.uint32, np.int32)],
)
def test_cuda_atlas_copy_preserves_pixels(count, offset, channels, input_dtype, output_dtype):
    """GPU copies preserve owned output, padding, alignment and signed segmentation IDs."""
    height, width = 3, 5
    columns = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / columns))
    shape = (rows * height, columns * width, channels)
    pixels = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape)
    if input_dtype == np.uint32:
        pixels += np.uint32(2**31)
    pixels = pixels.astype(input_dtype)
    if input_dtype == np.float32:
        pixels = pixels * np.float32(0.125) - np.float32(4.0)
    # Include trailing padding and exercise byte-misaligned RGBA pointer views.
    storage = np.concatenate((np.zeros(offset, dtype=input_dtype), pixels.reshape(-1), np.ones(3, dtype=input_dtype)))
    source_storage = wp.array(storage, device=DEVICE)
    source = source_storage[offset:]
    destination_shape = (count, height, width, channels)
    torch_dtype = torch.from_numpy(np.empty(0, dtype=output_dtype)).dtype
    destination_storage = torch.full((int(np.prod(destination_shape)) + offset,), 7, dtype=torch_dtype, device=DEVICE)
    destination = destination_storage[offset:].reshape(destination_shape)
    destination_ptr = destination.data_ptr()
    with wp.ScopedStream(wp.Stream(DEVICE)) as stream:
        _reshape_tiled_image_cuda(source, wp.from_torch(destination), columns)
        wp.synchronize_stream(stream)
    actual = destination.cpu().numpy()
    for camera in range(count):
        row, column = divmod(camera, columns)
        expected = pixels[row * height:(row + 1) * height, column * width:(column + 1) * width]
        np.testing.assert_array_equal(actual[camera], expected.astype(output_dtype))
    assert destination.data_ptr() == destination_ptr
    assert destination_ptr != source.ptr
    if offset:
        assert destination_storage[0].item() == 7
    source_storage.zero_()
    wp.synchronize_device(DEVICE)
    np.testing.assert_array_equal(destination.cpu().numpy(), actual)


@pytest.fixture(scope="function")
def setup_camera() -> tuple[sim_utils.SimulationContext, TiledCameraCfg, float]:
    """Fixture to set up and tear down the camera simulation environment."""
    camera_cfg = TiledCameraCfg(
        height=128,
        width=256,
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 4.0), rot=(0.0, 0.0, 1.0, 0.0), convention="ros"),
        prim_path="/World/Camera",
        update_period=0,
        data_types=["rgb", "distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    # Create a new stage
    stage_utils.create_new_stage()
    # Simulation time-step
    dt = 0.01
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=DEVICE)
    sim: sim_utils.SimulationContext = sim_utils.SimulationContext(sim_cfg)
    # populate scene
    _populate_scene()
    # load stage
    stage_utils.update_stage()
    yield sim, camera_cfg, dt
    # Teardown
    rep.vp_manager.destroy_hydra_textures("Replicator")
    sim._timeline.stop()
    sim.clear_all_callbacks()
    sim.clear_instance()


def test_single_camera_init(setup_camera):
    """Test single camera initialization."""
    sim, camera_cfg, dt = setup_camera
    # Create camera
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[0].GetPath().pathString == camera_cfg.prim_path
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (1, 3)
    assert camera.data.quat_w_ros.shape == (1, 4)
    assert camera.data.quat_w_world.shape == (1, 4)
    assert camera.data.quat_w_opengl.shape == (1, 4)
    assert camera.data.intrinsic_matrices.shape == (1, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for im_type, im_data in camera.data.output.items():
            if im_type == "rgb":
                assert im_data.shape == (1, camera_cfg.height, camera_cfg.width, 3)
                assert (im_data / 255.0).mean() > 0.0
            elif im_type == "distance_to_camera":
                assert im_data.shape == (1, camera_cfg.height, camera_cfg.width, 1)
                assert im_data.mean() > 0.0
    del camera


def test_render_product_disabled_has_no_frame(setup_camera):
    """A disabled render product, or one enabled again without a delivered frame, raises instead of a stale image.

    Renders without stepping deliver the frame after enabling, and it shows the current state: it matches a camera
    with the same pose that rendered on every step, not the frame read before disabling.
    """
    sim, camera_cfg, dt = setup_camera
    camera = TiledCamera(camera_cfg)
    reference_cfg = copy.deepcopy(camera_cfg)
    reference_cfg.prim_path = "/World/ReferenceCamera"
    reference = TiledCamera(reference_cfg)
    sim.reset()
    for _ in range(5):
        sim.step()
    camera.update(dt)
    assert camera.render_product_enabled
    assert not camera.frame_pending
    assert camera.data.output["rgb"].float().mean() > 0.0
    stale_depth = camera.data.output["distance_to_camera"].clone()
    frames = camera.frame.clone()

    camera.set_render_product_enabled(False)
    assert not camera.render_product_enabled
    assert not camera.frame_pending
    with pytest.raises(RuntimeError, match="has no frame"):
        camera.data
    # The objects fall while the render product is disabled.
    for _ in range(30):
        sim.step()
        camera.update(dt)
        with pytest.raises(RuntimeError, match="has no frame"):
            camera.data
    # Eager updates skip the read instead of raising.
    camera.update(dt, force_recompute=True)

    camera.set_render_product_enabled(True)
    assert camera.frame_pending
    with pytest.raises(RuntimeError, match="has no frame"):
        camera.data
    sim.step()
    camera.update(dt)
    reference.update(dt)
    renders = 0
    while camera.frame_pending:
        with pytest.raises(RuntimeError, match="has no frame"):
            camera.data
        assert renders < 10, "no frame delivered within 10 renders after enabling"
        sim.render()
        renders += 1
    camera.update(dt)
    reference.update(dt)
    assert camera.data.output["rgb"].float().mean() > 0.0
    torch.testing.assert_close(camera.frame, frames + 1)
    depth = camera.data.output["distance_to_camera"]
    reference_depth = reference.data.output["distance_to_camera"]
    assert depth.mean() > 0.0
    fresh_error = (depth - reference_depth).abs().mean().item()
    stale_error = (stale_depth - reference_depth).abs().mean().item()
    assert stale_error > 1.0e-3, "the scene did not change while the render product was disabled"
    assert fresh_error < 0.2 * stale_error, f"stale frame after enabling: {fresh_error=} {stale_error=} {renders=}"
    del camera, reference


def test_empty_annotator_output_has_no_frame(setup_camera):
    """An empty CUDA annotator buffer (null pointer, as after enabling a render product again) raises cleanly.

    The read must not reach the CUDA retile copy, must not count a frame, and must keep the data outdated so
    that the next read with a delivered frame fills it.
    """
    sim, camera_cfg, dt = setup_camera
    camera = TiledCamera(camera_cfg)
    sim.reset()
    for _ in range(5):
        sim.step()
    camera.update(dt)
    assert camera.data.output["rgb"].float().mean() > 0.0
    frames = camera.frame.clone()

    annotator = camera._annotators["rgba"]
    empty = wp.empty(0, dtype=wp.uint8, device=camera._annotator_device)
    assert empty.ptr is None
    annotator.get_data = lambda *args, **kwargs: empty
    camera.data.output["rgb"].zero_()
    sim.step()
    camera.update(dt)
    with pytest.raises(RuntimeError, match="returned no data"):
        camera.data
    torch.testing.assert_close(camera.frame, frames)

    del annotator.get_data
    assert camera.data.output["rgb"].float().mean() > 0.0
    torch.testing.assert_close(camera.frame, frames + 1)
    del camera


def test_depth_clipping_max(setup_camera):
    """Test depth max clipping."""
    sim, _, dt = setup_camera
    # get camera cfgs
    camera_cfg = TiledCameraCfg(
        prim_path="/World/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(2.5, 2.5, 6.0), rot=(-0.125, 0.362, 0.873, -0.302), convention="ros"),
        spawn=sim_utils.PinholeCameraCfg().from_intrinsic_matrix(
            focal_length=38.0,
            intrinsic_matrix=[380.08, 0.0, 467.79, 0.0, 380.08, 262.05, 0.0, 0.0, 1.0],
            height=540,
            width=960,
            clipping_range=(4.9, 5.0),
        ),
        height=540,
        width=960,
        data_types=["depth"],
        depth_clipping_behavior="max",
    )
    camera = TiledCamera(camera_cfg)

    # Play sim
    sim.reset()

    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    camera.update(dt)

    assert len(camera.data.output["depth"][torch.isinf(camera.data.output["depth"])]) == 0
    assert camera.data.output["depth"].min() >= camera_cfg.spawn.clipping_range[0]
    assert camera.data.output["depth"].max() <= camera_cfg.spawn.clipping_range[1]

    del camera


def test_depth_clipping_none(setup_camera):
    """Test depth none clipping."""
    sim, _, dt = setup_camera
    # get camera cfgs
    camera_cfg = TiledCameraCfg(
        prim_path="/World/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(2.5, 2.5, 6.0), rot=(-0.125, 0.362, 0.873, -0.302), convention="ros"),
        spawn=sim_utils.PinholeCameraCfg().from_intrinsic_matrix(
            focal_length=38.0,
            intrinsic_matrix=[380.08, 0.0, 467.79, 0.0, 380.08, 262.05, 0.0, 0.0, 1.0],
            height=540,
            width=960,
            clipping_range=(4.9, 5.0),
        ),
        height=540,
        width=960,
        data_types=["depth"],
        depth_clipping_behavior="none",
    )
    camera = TiledCamera(camera_cfg)

    # Play sim
    sim.reset()

    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    camera.update(dt)

    assert len(camera.data.output["depth"][torch.isinf(camera.data.output["depth"])]) > 0
    assert camera.data.output["depth"].min() >= camera_cfg.spawn.clipping_range[0]
    if len(camera.data.output["depth"][~torch.isinf(camera.data.output["depth"])]) > 0:
        assert (
            camera.data.output["depth"][~torch.isinf(camera.data.output["depth"])].max()
            <= camera_cfg.spawn.clipping_range[1]
        )

    del camera


def test_depth_clipping_zero(setup_camera):
    """Test depth zero clipping."""
    sim, _, dt = setup_camera
    # get camera cfgs
    camera_cfg = TiledCameraCfg(
        prim_path="/World/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(2.5, 2.5, 6.0), rot=(-0.125, 0.362, 0.873, -0.302), convention="ros"),
        spawn=sim_utils.PinholeCameraCfg().from_intrinsic_matrix(
            focal_length=38.0,
            intrinsic_matrix=[380.08, 0.0, 467.79, 0.0, 380.08, 262.05, 0.0, 0.0, 1.0],
            height=540,
            width=960,
            clipping_range=(4.9, 5.0),
        ),
        height=540,
        width=960,
        data_types=["depth"],
        depth_clipping_behavior="zero",
    )
    camera = TiledCamera(camera_cfg)

    # Play sim
    sim.reset()

    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    camera.update(dt)

    assert len(camera.data.output["depth"][torch.isinf(camera.data.output["depth"])]) == 0
    assert camera.data.output["depth"].min() == 0.0
    assert camera.data.output["depth"].max() <= camera_cfg.spawn.clipping_range[1]

    del camera


def test_multi_camera_init(setup_camera):
    """Test multi-camera initialization."""
    sim, camera_cfg, dt = setup_camera

    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for im_type, im_data in camera.data.output.items():
            if im_type == "rgb":
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
                for i in range(4):
                    assert (im_data[i] / 255.0).mean() > 0.0
            elif im_type == "distance_to_camera":
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
                for i in range(4):
                    assert im_data[i].mean() > 0.0
    del camera


def test_rgb_only_camera(setup_camera):
    """Test initialization with only RGB data type."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["rgb"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["rgba", "rgb"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        im_data = camera.data.output["rgb"]
        assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
        for i in range(4):
            assert (im_data[i] / 255.0).mean() > 0.0
    assert camera.data.output["rgb"].dtype == torch.uint8
    del camera


def test_data_types(setup_camera):
    """Test different data types for camera initialization."""
    sim, camera_cfg, dt = setup_camera
    # Create camera
    camera_cfg_distance = copy.deepcopy(camera_cfg)
    camera_cfg_distance.data_types = ["distance_to_camera"]
    camera_cfg_distance.prim_path = "/World/CameraDistance"
    camera_distance = TiledCamera(camera_cfg_distance)
    camera_cfg_depth = copy.deepcopy(camera_cfg)
    camera_cfg_depth.data_types = ["depth"]
    camera_cfg_depth.prim_path = "/World/CameraDepth"
    camera_depth = TiledCamera(camera_cfg_depth)
    camera_cfg_both = copy.deepcopy(camera_cfg)
    camera_cfg_both.data_types = ["distance_to_camera", "depth"]
    camera_cfg_both.prim_path = "/World/CameraBoth"
    camera_both = TiledCamera(camera_cfg_both)

    # Play sim
    sim.reset()

    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check if cameras are initialized
    assert camera_distance.is_initialized
    assert camera_depth.is_initialized
    assert camera_both.is_initialized

    # Check if camera prims are set correctly and that they are camera prims
    assert camera_distance._sensor_prims[0].GetPath().pathString == "/World/CameraDistance"
    assert isinstance(camera_distance._sensor_prims[0], UsdGeom.Camera)
    assert camera_depth._sensor_prims[0].GetPath().pathString == "/World/CameraDepth"
    assert isinstance(camera_depth._sensor_prims[0], UsdGeom.Camera)
    assert camera_both._sensor_prims[0].GetPath().pathString == "/World/CameraBoth"
    assert isinstance(camera_both._sensor_prims[0], UsdGeom.Camera)
    assert list(camera_distance.data.output.keys()) == ["distance_to_camera"]
    assert list(camera_depth.data.output.keys()) == ["depth"]
    assert list(camera_both.data.output.keys()) == ["depth", "distance_to_camera"]

    del camera_distance
    del camera_depth
    del camera_both


def test_depth_only_camera(setup_camera):
    """Test initialization with only depth."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["distance_to_camera"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["distance_to_camera"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        im_data = camera.data.output["distance_to_camera"]
        assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
        for i in range(4):
            assert im_data[i].mean() > 0.0
    assert camera.data.output["distance_to_camera"].dtype == torch.float
    del camera


def test_rgba_only_camera(setup_camera):
    """Test initialization with only RGBA."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["rgba"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["rgba"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 4)
            for i in range(4):
                assert (im_data[i] / 255.0).mean() > 0.0
    assert camera.data.output["rgba"].dtype == torch.uint8
    del camera


def test_distance_to_camera_only_camera(setup_camera):
    """Test initialization with only distance_to_camera."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["distance_to_camera"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["distance_to_camera"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
            for i in range(4):
                assert im_data[i].mean() > 0.0
    assert camera.data.output["distance_to_camera"].dtype == torch.float
    del camera


def test_distance_to_image_plane_only_camera(setup_camera):
    """Test initialization with only distance_to_image_plane."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["distance_to_image_plane"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["distance_to_image_plane"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
            for i in range(4):
                assert im_data[i].mean() > 0.0
    assert camera.data.output["distance_to_image_plane"].dtype == torch.float
    del camera


def test_normals_only_camera(setup_camera):
    """Test initialization with only normals."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["normals"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["normals"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
            for i in range(4):
                assert im_data[i].mean() > 0.0
    assert camera.data.output["normals"].dtype == torch.float
    del camera


def test_motion_vectors_only_camera(setup_camera):
    """Test initialization with only motion_vectors."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["motion_vectors"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["motion_vectors"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 2)
            for i in range(4):
                assert im_data[i].mean() != 0.0
    assert camera.data.output["motion_vectors"].dtype == torch.float
    del camera


def test_semantic_segmentation_colorize_only_camera(setup_camera):
    """Test initialization with only semantic_segmentation."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["semantic_segmentation"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["semantic_segmentation"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 4)
            for i in range(4):
                assert (im_data[i] / 255.0).mean() > 0.0
    assert camera.data.output["semantic_segmentation"].dtype == torch.uint8
    assert isinstance(camera.data.info["semantic_segmentation"], dict)
    del camera


def test_instance_segmentation_fast_colorize_only_camera(setup_camera):
    """Test initialization with only instance_segmentation_fast."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["instance_segmentation_fast"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["instance_segmentation_fast"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 4)
            for i in range(num_cameras):
                assert (im_data[i] / 255.0).mean() > 0.0
    assert camera.data.output["instance_segmentation_fast"].dtype == torch.uint8
    assert isinstance(camera.data.info["instance_segmentation_fast"], dict)
    del camera


def test_instance_id_segmentation_fast_colorize_only_camera(setup_camera):
    """Test initialization with only instance_id_segmentation_fast."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["instance_id_segmentation_fast"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["instance_id_segmentation_fast"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 4)
            for i in range(num_cameras):
                assert (im_data[i] / 255.0).mean() > 0.0
    assert camera.data.output["instance_id_segmentation_fast"].dtype == torch.uint8
    assert isinstance(camera.data.info["instance_id_segmentation_fast"], dict)
    del camera


def test_semantic_segmentation_non_colorize_only_camera(setup_camera):
    """Test initialization with only semantic_segmentation."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["semantic_segmentation"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera_cfg.colorize_semantic_segmentation = False
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["semantic_segmentation"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
            for i in range(num_cameras):
                assert im_data[i].to(dtype=float).mean() > 0.0
    assert camera.data.output["semantic_segmentation"].dtype == torch.int32
    assert isinstance(camera.data.info["semantic_segmentation"], dict)

    del camera


def test_instance_segmentation_fast_non_colorize_only_camera(setup_camera):
    """Test initialization with only instance_segmentation_fast."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["instance_segmentation_fast"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera_cfg.colorize_instance_segmentation = False
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["instance_segmentation_fast"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
            for i in range(num_cameras):
                assert im_data[i].to(dtype=float).mean() > 0.0
    assert camera.data.output["instance_segmentation_fast"].dtype == torch.int32
    assert isinstance(camera.data.info["instance_segmentation_fast"], dict)
    del camera


def test_instance_id_segmentation_fast_non_colorize_only_camera(setup_camera):
    """Test initialization with only instance_id_segmentation_fast."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["instance_id_segmentation_fast"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera_cfg.colorize_instance_id_segmentation = False
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["instance_id_segmentation_fast"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
            for i in range(num_cameras):
                assert im_data[i].to(dtype=float).mean() > 0.0
    assert camera.data.output["instance_id_segmentation_fast"].dtype == torch.int32
    assert isinstance(camera.data.info["instance_id_segmentation_fast"], dict)
    del camera


def test_all_annotators_camera(setup_camera):
    """Test initialization with all supported annotators."""
    sim, camera_cfg, dt = setup_camera
    all_annotator_types = [
        "rgb",
        "rgba",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation_fast",
        "instance_id_segmentation_fast",
    ]

    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = all_annotator_types
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert sorted(camera.data.output.keys()) == sorted(all_annotator_types)

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for data_type, im_data in camera.data.output.items():
            if data_type in ["rgb", "normals"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
            elif data_type in [
                "rgba",
                "semantic_segmentation",
                "instance_segmentation_fast",
                "instance_id_segmentation_fast",
            ]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 4)
                for i in range(num_cameras):
                    assert (im_data[i] / 255.0).mean() > 0.0
            elif data_type in ["motion_vectors"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 2)
                for i in range(num_cameras):
                    assert im_data[i].mean() != 0.0
            elif data_type in ["depth", "distance_to_camera", "distance_to_image_plane"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
                for i in range(num_cameras):
                    assert im_data[i].mean() > 0.0

    # access image data and compare dtype
    output = camera.data.output
    info = camera.data.info
    assert output["rgb"].dtype == torch.uint8
    assert output["rgba"].dtype == torch.uint8
    assert output["depth"].dtype == torch.float
    assert output["distance_to_camera"].dtype == torch.float
    assert output["distance_to_image_plane"].dtype == torch.float
    assert output["normals"].dtype == torch.float
    assert output["motion_vectors"].dtype == torch.float
    assert output["semantic_segmentation"].dtype == torch.uint8
    assert output["instance_segmentation_fast"].dtype == torch.uint8
    assert output["instance_id_segmentation_fast"].dtype == torch.uint8
    assert isinstance(info["semantic_segmentation"], dict)
    assert isinstance(info["instance_segmentation_fast"], dict)
    assert isinstance(info["instance_id_segmentation_fast"], dict)

    del camera


def test_all_annotators_low_resolution_camera(setup_camera):
    """Test initialization with all supported annotators."""
    sim, camera_cfg, dt = setup_camera
    all_annotator_types = [
        "rgb",
        "rgba",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation_fast",
        "instance_id_segmentation_fast",
    ]

    num_cameras = 2
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.height = 40
    camera_cfg.width = 40
    camera_cfg.data_types = all_annotator_types
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert sorted(camera.data.output.keys()) == sorted(all_annotator_types)

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for data_type, im_data in camera.data.output.items():
            if data_type in ["rgb", "normals"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
            elif data_type in [
                "rgba",
                "semantic_segmentation",
                "instance_segmentation_fast",
                "instance_id_segmentation_fast",
            ]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 4)
                for i in range(num_cameras):
                    assert (im_data[i] / 255.0).mean() > 0.0
            elif data_type in ["motion_vectors"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 2)
                for i in range(num_cameras):
                    assert im_data[i].mean() != 0.0
            elif data_type in ["depth", "distance_to_camera", "distance_to_image_plane"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
                for i in range(num_cameras):
                    assert im_data[i].mean() > 0.0

    # access image data and compare dtype
    output = camera.data.output
    info = camera.data.info
    assert output["rgb"].dtype == torch.uint8
    assert output["rgba"].dtype == torch.uint8
    assert output["depth"].dtype == torch.float
    assert output["distance_to_camera"].dtype == torch.float
    assert output["distance_to_image_plane"].dtype == torch.float
    assert output["normals"].dtype == torch.float
    assert output["motion_vectors"].dtype == torch.float
    assert output["semantic_segmentation"].dtype == torch.uint8
    assert output["instance_segmentation_fast"].dtype == torch.uint8
    assert output["instance_id_segmentation_fast"].dtype == torch.uint8
    assert isinstance(info["semantic_segmentation"], dict)
    assert isinstance(info["instance_segmentation_fast"], dict)
    assert isinstance(info["instance_id_segmentation_fast"], dict)

    del camera


def test_all_annotators_non_perfect_square_number_camera(setup_camera):
    """Test initialization with all supported annotators."""
    sim, camera_cfg, dt = setup_camera
    all_annotator_types = [
        "rgb",
        "rgba",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation_fast",
        "instance_id_segmentation_fast",
    ]

    num_cameras = 11
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = all_annotator_types
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert sorted(camera.data.output.keys()) == sorted(all_annotator_types)

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for data_type, im_data in camera.data.output.items():
            if data_type in ["rgb", "normals"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
            elif data_type in [
                "rgba",
                "semantic_segmentation",
                "instance_segmentation_fast",
                "instance_id_segmentation_fast",
            ]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 4)
                for i in range(num_cameras):
                    assert (im_data[i] / 255.0).mean() > 0.0
            elif data_type in ["motion_vectors"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 2)
                for i in range(num_cameras):
                    assert im_data[i].mean() != 0.0
            elif data_type in ["depth", "distance_to_camera", "distance_to_image_plane"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
                for i in range(num_cameras):
                    assert im_data[i].mean() > 0.0

    # access image data and compare dtype
    output = camera.data.output
    info = camera.data.info
    assert output["rgb"].dtype == torch.uint8
    assert output["rgba"].dtype == torch.uint8
    assert output["depth"].dtype == torch.float
    assert output["distance_to_camera"].dtype == torch.float
    assert output["distance_to_image_plane"].dtype == torch.float
    assert output["normals"].dtype == torch.float
    assert output["motion_vectors"].dtype == torch.float
    assert output["semantic_segmentation"].dtype == torch.uint8
    assert output["instance_segmentation_fast"].dtype == torch.uint8
    assert output["instance_id_segmentation_fast"].dtype == torch.uint8
    assert isinstance(info["semantic_segmentation"], dict)
    assert isinstance(info["instance_segmentation_fast"], dict)
    assert isinstance(info["instance_id_segmentation_fast"], dict)

    del camera


def test_all_annotators_instanceable(setup_camera):
    """Test initialization with all supported annotators on instanceable assets."""
    sim, camera_cfg, dt = setup_camera
    all_annotator_types = [
        "rgb",
        "rgba",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation_fast",
        "instance_id_segmentation_fast",
    ]

    num_cameras = 10
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform", translation=(0.0, i, 0.0))

    # Create a stage with 10 instanceable cubes, where each camera points to one cube
    stage = stage_utils.get_current_stage()
    for i in range(10):
        # Remove objects added to stage by default
        stage.RemovePrim(f"/World/Objects/Obj_{i:02d}")
        # Add instanceable cubes
        prim_utils.create_prim(
            f"/World/Cube_{i}",
            "Xform",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            translation=(0.0, i, 5.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
            scale=(5.0, 5.0, 5.0),
        )
        prim = stage.GetPrimAtPath(f"/World/Cube_{i}")
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr()
        sem.CreateSemanticDataAttr()
        sem.GetSemanticTypeAttr().Set("class")
        sem.GetSemanticDataAttr().Set("cube")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.height = 120
    camera_cfg.width = 80
    camera_cfg.data_types = all_annotator_types
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera_cfg.offset.pos = (0.0, 0.0, 5.5)
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert sorted(camera.data.output.keys()) == sorted(all_annotator_types)

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Simulate physics
    for _ in range(2):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for data_type, im_data in camera.data.output.items():
            if data_type in ["rgb", "normals"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
            elif data_type in [
                "rgba",
                "semantic_segmentation",
                "instance_segmentation_fast",
                "instance_id_segmentation_fast",
            ]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 4)
                # semantic_segmentation has mean 0.43
                # rgba has mean 0.38
                # instance_segmentation_fast has mean 0.42
                # instance_id_segmentation_fast has mean 0.55-0.62
                for i in range(num_cameras):
                    assert (im_data[i] / 255.0).mean() > 0.2
            elif data_type in ["motion_vectors"]:
                # motion vectors have mean 0.2
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 2)
                for i in range(num_cameras):
                    assert (im_data[i].abs().mean()) > 0.15
            elif data_type in ["depth", "distance_to_camera", "distance_to_image_plane"]:
                # depth has mean 2.7
                # distance_to_image_plane has mean 3.1
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
                for i in range(num_cameras):
                    assert im_data[i].mean() > 2.5

    # access image data and compare dtype
    output = camera.data.output
    info = camera.data.info
    assert output["rgb"].dtype == torch.uint8
    assert output["rgba"].dtype == torch.uint8
    assert output["depth"].dtype == torch.float
    assert output["distance_to_camera"].dtype == torch.float
    assert output["distance_to_image_plane"].dtype == torch.float
    assert output["normals"].dtype == torch.float
    assert output["motion_vectors"].dtype == torch.float
    assert output["semantic_segmentation"].dtype == torch.uint8
    assert output["instance_segmentation_fast"].dtype == torch.uint8
    assert output["instance_id_segmentation_fast"].dtype == torch.uint8
    assert isinstance(info["semantic_segmentation"], dict)
    assert isinstance(info["instance_segmentation_fast"], dict)
    assert isinstance(info["instance_id_segmentation_fast"], dict)

    del camera


def test_throughput(setup_camera):
    """Test tiled camera throughput."""
    sim, camera_cfg, dt = setup_camera
    # create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.height = 480
    camera_cfg.width = 640
    camera = TiledCamera(camera_cfg)

    # Play simulator
    sim.reset()

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Simulate physics
    for _ in range(5):
        # perform rendering
        sim.step()
        # update camera
        with Timer(f"Time taken for updating camera with shape {camera.image_shape}"):
            camera.update(dt)
        # Check image data
        for im_type, im_data in camera.data.output.items():
            if im_type == "rgb":
                assert im_data.shape == (1, camera_cfg.height, camera_cfg.width, 3)
                assert (im_data / 255.0).mean() > 0.0
            elif im_type == "distance_to_camera":
                assert im_data.shape == (1, camera_cfg.height, camera_cfg.width, 1)
                assert im_data.mean() > 0.0
    del camera


def test_output_equal_to_usd_camera_intrinsics(setup_camera):
    """
    Test that the output of the ray caster camera and the usd camera are the same when both are
    initialized with the same intrinsic matrix.
    """
    sim, _, dt = setup_camera
    # create cameras
    offset_rot = (-0.1251, 0.3617, 0.8731, -0.3020)
    offset_pos = (2.5, 2.5, 4.0)
    intrinsics = [380.08, 0.0, 467.79, 0.0, 380.08, 262.05, 0.0, 0.0, 1.0]
    # get camera cfgs
    # TODO: add clipping range back, once correctly supported by tiled camera
    camera_tiled_cfg = TiledCameraCfg(
        prim_path="/World/Camera_tiled",
        offset=TiledCameraCfg.OffsetCfg(pos=offset_pos, rot=offset_rot, convention="ros"),
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsics,
            height=540,
            width=960,
        ),
        height=540,
        width=960,
        data_types=["depth"],
    )
    camera_usd_cfg = CameraCfg(
        prim_path="/World/Camera_usd",
        offset=CameraCfg.OffsetCfg(pos=offset_pos, rot=offset_rot, convention="ros"),
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsics,
            height=540,
            width=960,
        ),
        height=540,
        width=960,
        data_types=["distance_to_image_plane"],
    )

    # set aperture offsets to 0, as currently not supported for usd/ tiled camera
    camera_tiled_cfg.spawn.horizontal_aperture_offset = 0
    camera_tiled_cfg.spawn.vertical_aperture_offset = 0
    camera_usd_cfg.spawn.horizontal_aperture_offset = 0
    camera_usd_cfg.spawn.vertical_aperture_offset = 0
    # init cameras
    camera_tiled = TiledCamera(camera_tiled_cfg)
    camera_usd = Camera(camera_usd_cfg)

    # play sim
    sim.reset()
    sim.play()

    # perform steps
    for _ in range(5):
        sim.step()

    # update camera
    camera_usd.update(dt)
    camera_tiled.update(dt)

    # filter nan and inf from output
    cam_tiled_output = camera_tiled.data.output["depth"].clone()
    cam_usd_output = camera_usd.data.output["distance_to_image_plane"].clone()
    cam_tiled_output[torch.isnan(cam_tiled_output)] = 0
    cam_tiled_output[torch.isinf(cam_tiled_output)] = 0
    cam_usd_output[torch.isnan(cam_usd_output)] = 0
    cam_usd_output[torch.isinf(cam_usd_output)] = 0

    # check that both have the same intrinsic matrices
    torch.testing.assert_close(camera_tiled.data.intrinsic_matrices[0], camera_usd.data.intrinsic_matrices[0])

    # check the apertures
    torch.testing.assert_close(
        camera_usd._sensor_prims[0].GetHorizontalApertureAttr().Get(),
        camera_tiled._sensor_prims[0].GetHorizontalApertureAttr().Get(),
    )
    torch.testing.assert_close(
        camera_usd._sensor_prims[0].GetVerticalApertureAttr().Get(),
        camera_tiled._sensor_prims[0].GetVerticalApertureAttr().Get(),
    )

    # check image data
    torch.testing.assert_close(
        cam_tiled_output[..., 0],
        cam_usd_output[..., 0],
        atol=5e-5,
        rtol=5e-6,
    )

    del camera_tiled
    del camera_usd


def test_sensor_print(setup_camera):
    """Test sensor print is working correctly."""
    sim, camera_cfg, _ = setup_camera
    # Create sensor
    sensor = TiledCamera(cfg=camera_cfg)
    # Play sim
    sim.reset()
    # print info
    print(sensor)


def test_frame_offset_small_resolution(setup_camera):
    """Test frame offset issue with small resolution camera."""
    sim, camera_cfg, dt = setup_camera
    # Create sensor
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.height = 80
    camera_cfg.width = 80
    camera_cfg.offset.pos = (0.0, 0.0, 0.5)
    tiled_camera = TiledCamera(camera_cfg)
    # play sim
    sim.reset()
    # simulate some steps first to make sure objects are settled
    stage = stage_utils.get_current_stage()
    for i in range(10):
        prim = stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")
        UsdGeom.Gprim(prim).GetOrderedXformOps()[2].Set(Gf.Vec3d(1.0, 1.0, 1.0))
    for i in range(100):
        # step simulation
        sim.step()
        # update camera
        tiled_camera.update(dt)
    # collect image data
    image_before = tiled_camera.data.output["rgb"].clone() / 255.0

    # update scene
    for i in range(10):
        prim = stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")
        color = Gf.Vec3f(0, 0, 0)
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color])

    # update rendering
    sim.step()
    # update camera
    tiled_camera.update(dt)

    # make sure the image is different
    image_after = tiled_camera.data.output["rgb"].clone() / 255.0

    # check difference is above threshold
    assert torch.abs(image_after - image_before).mean() > 0.1  # images of same color should be below 0.01


def test_frame_offset_large_resolution(setup_camera):
    """Test frame offset issue with large resolution camera."""
    sim, camera_cfg, dt = setup_camera
    # Create sensor
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.height = 480
    camera_cfg.width = 480
    tiled_camera = TiledCamera(camera_cfg)

    # modify scene to be less stochastic
    stage = stage_utils.get_current_stage()
    for i in range(10):
        prim = stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")
        color = Gf.Vec3f(1, 1, 1)
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color])

    # play sim
    sim.reset()
    # simulate some steps first to make sure objects are settled
    for i in range(100):
        # step simulation
        sim.step()
        # update camera
        tiled_camera.update(dt)
    # collect image data
    image_before = tiled_camera.data.output["rgb"].clone() / 255.0

    # update scene
    for i in range(10):
        prim = stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")
        color = Gf.Vec3f(0, 0, 0)
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color])

    # update rendering
    sim.step()
    # update camera
    tiled_camera.update(dt)

    # make sure the image is different
    image_after = tiled_camera.data.output["rgb"].clone() / 255.0

    # check difference is above threshold
    assert torch.abs(image_after - image_before).mean() > 0.01  # images of same color should be below 0.001


def test_cpu_device_outputs_match_rendered_atlas():
    """CPU-simulation cameras return this frame's atlas tiles exactly, with contiguous RGB on the host."""
    stage_utils.create_new_stage()
    dt = 0.01
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device="cpu"))
    try:
        _populate_scene()
        stage_utils.update_stage()
        num_cameras = 5  # 3x2 atlas with a padded tile
        for i in range(num_cameras):
            prim_utils.create_prim(f"/World/Origin_{i}", "Xform", translation=(0.3 * i, 0.0, 0.0))
        camera_cfg = TiledCameraCfg(
            height=96,
            width=160,
            offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 4.0), rot=(0.0, 0.0, 1.0, 0.0), convention="ros"),
            prim_path="/World/Origin_.*/CameraSensor",
            update_period=0,
            data_types=["rgb", "distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 3.0)
            ),
        )
        camera = TiledCamera(camera_cfg)
        sim.reset()
        assert list(camera.data.output.keys()) == ["rgba", "rgb", "distance_to_camera"]
        for _ in range(5):
            sim.step()
        for _ in range(3):
            sim.step()
            camera.update(dt)
            rgba = camera.data.output["rgba"]
            rgb = camera.data.output["rgb"]
            distance = camera.data.output["distance_to_camera"]
            for tensor in (rgba, rgb, distance):
                assert tensor.device.type == "cpu"
            assert rgb.is_contiguous()
            assert rgb.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
            torch.testing.assert_close(rgb, rgba[..., :3], rtol=0, atol=0)
            assert rgb.float().mean() > 0.0
            # Same frame, read back through replicator's host path: tiles must match bit for bit.
            columns = camera._tiling_grid_shape()[0]
            atlas_rgba = camera._annotators["rgba"].get_data(device="cpu")
            atlas_distance = camera._annotators["distance_to_camera"].get_data(device="cpu")
            atlas_distance = atlas_distance.reshape(*atlas_distance.shape[:2], 1)
            for index in range(num_cameras):
                row, column = divmod(index, columns)
                rows = slice(row * camera_cfg.height, (row + 1) * camera_cfg.height)
                cols = slice(column * camera_cfg.width, (column + 1) * camera_cfg.width)
                np.testing.assert_array_equal(rgba[index].numpy(), atlas_rgba[rows, cols])
                expected = atlas_distance[rows, cols].copy()
                expected[expected > camera_cfg.spawn.clipping_range[1]] = np.inf
                np.testing.assert_array_equal(distance[index].numpy(), expected)
        del camera
    finally:
        rep.vp_manager.destroy_hydra_textures("Replicator")
        sim._timeline.stop()
        sim.clear_all_callbacks()
        sim.clear_instance()


"""
Helper functions.
"""


@staticmethod
def _populate_scene():
    """Add prims to the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.SphereLightCfg()
    cfg.func("/World/Light/GreySphere", cfg, translation=(4.5, 3.5, 10.0))
    cfg.func("/World/Light/WhiteSphere", cfg, translation=(-4.5, 3.5, 10.0))
    # Random objects
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    for i in range(10):
        # sample random position
        position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
        position *= np.asarray([1.5, 1.5, 0.5])
        # create prim
        prim_type = random.choice(["Cube", "Sphere", "Cylinder"])
        prim = prim_utils.create_prim(
            f"/World/Objects/Obj_{i:02d}",
            prim_type,
            translation=position,
            scale=(0.25, 0.25, 0.25),
            semantic_label=prim_type,
        )
        # cast to geom prim
        geom_prim = getattr(UsdGeom, prim_type)(prim)
        # set random color
        color = Gf.Vec3f(random.random(), random.random(), random.random())
        geom_prim.CreateDisplayColorAttr()
        geom_prim.GetDisplayColorAttr().Set([color])
        # add rigid properties
        SingleGeometryPrim(f"/World/Objects/Obj_{i:02d}", collision=True)
        SingleRigidPrim(f"/World/Objects/Obj_{i:02d}", mass=5.0)
