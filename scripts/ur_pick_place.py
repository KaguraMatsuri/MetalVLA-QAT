"""Physical UR5e/2F-85 task shared by collection, RL and autonomous evaluation.

Assets are loaded through the installed Menagerie API, without vendoring them.
No weld, teleport, collision disabling or expert takeover during model rollouts.
"""

import hashlib
import math
from pathlib import Path

import mujoco
import mujoco_menagerie as menagerie
import numpy as np


TOP = .72
HALF = .025
PERIOD = .1
SOURCE = (-.34, -.04, .0, .36)
DESTINATION = (.04, .34, .0, .36)
WORKSPACE_LOW = (-.39, -.10, TOP + .012)
WORKSPACE_HIGH = (.39, .42, TOP + .40)
CAMERAS = ("overhead", "ur_hand_wrist", "side")
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
SCHEMA = "ur5e_pick_place_pose_v1"
INSTRUCTION = "pick blue cube and place on red target"


def downward_rotation(yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.asarray(
        (
            (cosine, sine, 0.0),
            (sine, -cosine, 0.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )


def cells(bounds):
    x0, x1, y0, y1 = bounds
    xs, ys = np.linspace(x0, x1, 4), np.linspace(y0, y1, 4)
    return [(xs[x], xs[x+1], ys[y], ys[y+1]) for y in range(3) for x in range(3)]


def scenario(seed, region, ordinal, retry=0):
    """300 strata/cell: 5x5 centre-position strata times 12 continuous yaw bins.

    The two halves have equal area. Goal jitter is independently stratified.
    Retries use the same stratum, so failure filtering cannot omit hard bins.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, region, ordinal, retry]))
    permutation = np.random.default_rng(np.random.SeedSequence([seed, region])).permutation(300)
    slot = int(permutation[ordinal % 300])
    ix, iy, angle_bin = slot % 5, slot // 5 % 5, slot // 25
    def point(bounds, sx, sy):
        x0, x1, y0, y1 = cells(bounds)[region]
        return np.array([x0+(sx+rng.random())*(x1-x0)/5,
                         y0+(sy+rng.random())*(y1-y0)/5, TOP+HALF+.002])
    cube = point(SOURCE, ix, iy)
    goal = point(DESTINATION, (ix+2*iy) % 5, (iy+3*angle_bin) % 5)
    goal[2] = TOP + .002
    yaw = -math.pi+(angle_bin+rng.random())*2*math.pi/12
    return cube, goal, yaw, slot


def camera_quat(position, target):
    z = np.asarray(position)-np.asarray(target)
    z /= np.linalg.norm(z)
    x = np.cross((0., 0., 1.), z)
    if np.linalg.norm(x) < 1e-8:
        x = np.array([1., 0., 0.])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    result = np.empty(4)
    mujoco.mju_mat2Quat(result, np.column_stack((x, y, z)).ravel())
    return result


def build_model():
    scene = mujoco.MjSpec()
    scene.modelname = SCHEMA
    scene.compiler.degree = False
    scene.option.timestep = .002
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    scene.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    scene.option.impratio = 10
    scene.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                             size=(0, 0, .05), rgba=(.15, .17, .2, 1))
    scene.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                             pos=(0, -.03, TOP-.04), size=(.85, .48, .04),
                             rgba=(.45, .28, .14, 1), friction=(1, .005, .0001))
    for x in (-.55, .55):
        for y in (-.4, .45):
            scene.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(x, y, (TOP-.08)/2), size=(.035, .035, (TOP-.08)/2), rgba=(.25,.27,.3,1))
    scene.worldbody.add_light(pos=(0, -.2, 2.6), dir=(0, 0, -1))
    scene.worldbody.add_camera(name="overhead", pos=(0, .16, 1.95),
                              quat=(1,0,0,0), fovy=48)
    side_pos = (.0, .92, TOP+.085)
    scene.worldbody.add_camera(name="side", pos=side_pos,
        quat=camera_quat(side_pos, (0, .13, TOP+.085)), fovy=58)
    cube = scene.worldbody.add_body(name="cube", pos=(-.2,.15,TOP+HALF+.002))
    cube.add_freejoint(name="cube_freejoint")
    cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(HALF,)*3, mass=.08, rgba=(.1,.45,.95,1), friction=(1.2,.005,.0001))
    scene.worldbody.add_site(name="red_goal", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(.075, .002, 0.), rgba=(.95, .05, .05, .85), pos=(.2,.15,TOP+.002))
    arm = menagerie.get("universal_robots_ur5e").spec("ur5e")
    arm.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    arm.option.impratio = 10
    for key in list(arm.keys):
        arm.delete(key)
    hand = menagerie.get("robotiq_2f85").spec("2f85")
    hand.body("base").add_camera(name="wrist", pos=(.055,0,.08),
        quat=camera_quat((.055,0,.08), (0,0,.20)), fovy=80)
    arm.attach(hand, site=arm.site("attachment_site"), prefix="hand_")
    # Industrial position control includes gravity feedforward. MuJoCo's
    # official per-body gravcomp supplies it for the robot, not the object.
    # All robot/object collision geometries and object gravity remain active.
    for body in arm.bodies:
        if body.name != "world":
            body.gravcomp = 1.
    mount = scene.worldbody.add_frame(pos=(0, -.43, TOP))
    scene.attach(arm, frame=mount, prefix="ur_")
    return scene.compile()


def asset_provenance(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in ("universal_robots_ur5e", "robotiq_2f85"):
        asset = menagerie.get(name)
        license_text = (asset.path()/"LICENSE").read_text()
        (directory/f"{name}.LICENSE").write_text(license_text)
        result[name] = dict(license=asset.license, revision=asset.oid,
                           asset_sha256=asset.sha256,
                           source=f"https://github.com/google-deepmind/mujoco_menagerie/tree/main/{name}")
    return result


class URTask:
    # Reuse the existing MuJoCo damped Jacobian solver. It operates on indexed
    # joints, so six UR joints need no parallel IK implementation.
    def solve_ik(
            self, target_position: np.ndarray, target_yaw: float,
            seed_joints: np.ndarray | None = None,
            *, target_rotation: np.ndarray | None = None,
        ) -> np.ndarray:
            ik_data = mujoco.MjData(self.model)
            ik_data.qpos[:] = self.data.qpos
            if seed_joints is not None:
                ik_data.qpos[self.joint_qpos] = seed_joints
            mujoco.mj_forward(self.model, ik_data)
            lower = self.model.jnt_range[self.joint_ids, 0] + 0.01
            upper = self.model.jnt_range[self.joint_ids, 1] - 0.01
            full_rotation = target_rotation is not None
            target_rotation = (downward_rotation(target_yaw) if target_rotation is None
                               else np.asarray(target_rotation, dtype=np.float64))
            if (target_rotation.shape != (3, 3) or not np.isfinite(target_rotation).all()
                    or not np.allclose(target_rotation.T @ target_rotation, np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(target_rotation), 1., atol=1e-5)
                    or np.asarray(target_position).shape != (3,)
                    or not np.isfinite(target_position).all()):
                raise ValueError("IK requires a finite position and proper rotation matrix")
            position_jacobian = np.zeros((3, self.model.nv))
            rotation_jacobian = np.zeros((3, self.model.nv))
            for _ in range(250):
                rotation = ik_data.site_xmat[self.tcp_site].reshape(3, 3)
                position_error = target_position - ik_data.site_xpos[self.tcp_site]
                rotation_error = 0.5 * sum(
                    np.cross(rotation[:, axis], target_rotation[:, axis])
                    for axis in range(3)
                )
                if full_rotation:
                    # Official quaternion logarithm avoids the cross-product error's
                    # false zero at 180 degrees. Existing yaw-only calls stay unchanged.
                    desired_q, current_q, inverse_q, delta_q = (np.empty(4) for _ in range(4))
                    mujoco.mju_mat2Quat(desired_q, target_rotation.ravel())
                    mujoco.mju_mat2Quat(current_q, rotation.ravel())
                    mujoco.mju_negQuat(inverse_q, current_q)
                    mujoco.mju_mulQuat(delta_q, desired_q, inverse_q)
                    rotation_error = np.empty(3)
                    mujoco.mju_quat2Vel(rotation_error, delta_q, 1.)
                if (
                    np.linalg.norm(position_error) < 2e-4
                    and np.linalg.norm(rotation_error) < 2e-3
                ):
                    return ik_data.qpos[self.joint_qpos].copy()
                mujoco.mj_jacSite(
                    self.model, ik_data, position_jacobian, rotation_jacobian,
                    self.tcp_site,
                )
                jacobian = np.vstack(
                    (position_jacobian[:, self.joint_dof],
                     rotation_jacobian[:, self.joint_dof])
                )
                error = np.concatenate((position_error, rotation_error))
                delta = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + 1e-4 * np.eye(6), error
                )
                norm = np.linalg.norm(delta)
                if norm > 0.15:
                    delta *= 0.15 / norm
                ik_data.qpos[self.joint_qpos] = np.clip(
                    ik_data.qpos[self.joint_qpos] + delta, lower, upper
                )
                mujoco.mj_forward(self.model, ik_data)
            raise RuntimeError("IK did not converge")

    def __init__(self, width=320, height=180, render=True):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width) if render else None
        self.width, self.height = width, height
        self.joint_ids = np.array([self.model.joint("ur_"+name+"_joint").id for name in JOINT_NAMES])
        self.joint_qpos = self.model.jnt_qposadr[self.joint_ids]
        self.joint_dof = self.model.jnt_dofadr[self.joint_ids]
        self.arm_actuators = np.array([self.model.actuator("ur_"+name).id for name in JOINT_NAMES])
        self.gripper_actuator = self.model.actuator("ur_hand_fingers_actuator").id
        self.tcp_site = self.model.site("ur_hand_pinch").id
        self.cube_qpos = int(self.model.joint("cube_freejoint").qposadr[0])
        self.cube_dof = int(self.model.joint("cube_freejoint").dofadr[0])
        self.goal_site = self.model.site("red_goal").id
        self.pad_ids = [self.model.geom("ur_hand_"+p+"_pad1").id for p in ("left", "right")]
        self.pad_groups = [{self.model.geom("ur_hand_"+p+"_pad"+str(i)).id for i in (1,2)} for p in ("left", "right")]
        self.cube_geom = self.model.geom("cube_geom").id
        self.ctrl_min = self.model.actuator_ctrlrange[self.arm_actuators,0]
        self.ctrl_max = self.model.actuator_ctrlrange[self.arm_actuators,1]
        mujoco.mj_resetData(self.model, self.data)
        seed = np.array([-math.pi/2, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0.])
        self.data.qpos[self.joint_qpos] = seed
        mujoco.mj_forward(self.model, self.data)
        self.home = self.solve_ik(np.array([0.,.14,TOP+.32]), 0., seed,
                                 target_rotation=downward_rotation(0.))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def reset(self, seed, region, ordinal, retry=0):
        start, goal, yaw, slot = scenario(seed, region, ordinal, retry)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.joint_qpos] = self.home
        self.data.ctrl[self.arm_actuators] = self.home
        self.data.qpos[self.cube_qpos:self.cube_qpos+7] = (*start, math.cos(yaw/2), 0,0,math.sin(yaw/2))
        self.model.site_pos[self.goal_site] = goal
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=100)
        self.start = self.cube_position().copy()
        self.goal = goal.copy()
        self.previous = self.normalized_control()
        self.seed, self.region, self.ordinal, self.slot = seed, region, ordinal, slot
        self.retry = retry
        self.yaw = yaw
        self.trace = []
        self.path = 0.
        self.last_tcp = self.tcp_position().copy()
        self.ever_grasped = False
        self.lifted = False
        self.grasp_center_error = None
        self.elapsed = 0.
        self._control_step_remainder = 0.
        self.stable = 0.
        self.success = False
        self.last_closed = False
        self.last_target = np.r_[self.tcp_position(), 0.,0.]
        return start, goal, yaw

    def tcp_position(self):
        return self.data.site_xpos[self.tcp_site]

    def cube_position(self):
        return self.data.qpos[self.cube_qpos:self.cube_qpos+3]

    def normalized_control(self):
        joint = 2*(self.data.ctrl[self.arm_actuators]-self.ctrl_min)/(self.ctrl_max-self.ctrl_min)-1
        # Explicit fixed-zero slot preserves the model's 8-D interface, not a
        # fictitious seventh UR joint. Gripper remains the final channel.
        return np.r_[np.clip(joint,-1,1), 0., self.data.ctrl[self.gripper_actuator]/127.5-1].astype(np.float32)

    def state(self):
        joint = self.data.qpos[self.joint_qpos]
        grip = self.data.qpos[self.model.joint("ur_hand_left_driver_joint").qposadr[0]]/.8
        return np.r_[joint, 0., grip].astype(np.float32)

    def observe(self):
        images = []
        for name in CAMERAS:
            self.renderer.update_scene(self.data, camera=name)
            images.append(self.renderer.render().copy())
        return dict(images=np.stack(images), state=self.state(), previous=self.previous.copy(),
                    offset=(self.cube_position()-self.tcp_position()).astype(np.float32))

    def contacts(self):
        pads = set()
        for contact in self.data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.cube_geom in pair:
                for side, group in enumerate(self.pad_groups):
                    if pair & group:
                        pads.add(side)
        return len(pads) == 2

    def tick_metrics(self):
        tcp = self.tcp_position().copy()
        self.path += float(np.linalg.norm(tcp-self.last_tcp))
        self.last_tcp = tcp
        if self.contacts():
            self.ever_grasped = True
            error = float(np.linalg.norm(self.data.geom_xpos[self.pad_ids].mean(0)-self.cube_position()))
            self.grasp_center_error = error if self.grasp_center_error is None else min(self.grasp_center_error, error)
            if self.cube_position()[2] > TOP+HALF+.08:
                self.lifted = True
        placed = (self.lifted and np.linalg.norm(self.cube_position()[:2]-self.goal[:2]) < .020
                  and abs(self.cube_position()[2]-(TOP+HALF)) < .006
                  and np.linalg.norm(self.data.qvel[self.cube_dof:self.cube_dof+3]) < .025
                  and self.data.ctrl[self.gripper_actuator] < 32 and not self.contacts()
                  and np.linalg.norm(tcp-self.cube_position()) > .075)
        self.stable = self.stable + self.model.opt.timestep if placed else 0.
        self.success = self.stable >= .5

    def execute(self, xyz, yaw, closed, *, target_rotation=None, gripper_open=None, period=PERIOD):
        """Identical physical executor for expert, BC/RL and QAT rollouts.

        Cartesian targets are rate limited. Invalid IK holds current controls;
        it never substitutes a privileged cube/goal-dependent correction.
        """
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all() or not math.isfinite(yaw):
            raise ValueError("Nonfinite Cartesian command")
        if not math.isfinite(period) or period < self.model.opt.timestep:
            raise ValueError("Invalid control period")
        if gripper_open is not None and (not math.isfinite(gripper_open) or not 0 <= gripper_open <= 1):
            raise ValueError("Invalid gripper ratio")
        rotation = downward_rotation(yaw) if target_rotation is None else np.asarray(target_rotation, dtype=float)
        requested = [*xyz,float(yaw),int(closed)]
        xyz = np.clip(xyz, WORKSPACE_LOW, WORKSPACE_HIGH)
        delta = xyz-self.tcp_position()
        xyz = self.tcp_position()+delta*min(1., (.035*period/PERIOD)/max(np.linalg.norm(delta),1e-9))
        ik_ok = True
        try:
            joints = self.solve_ik(xyz, yaw, self.data.ctrl[self.arm_actuators],
                                   target_rotation=rotation)
            self.data.ctrl[self.arm_actuators] = np.clip(joints,self.ctrl_min,self.ctrl_max)
        except RuntimeError:
            ik_ok = False
        self.data.ctrl[self.gripper_actuator] = (255. if closed else 0.) if gripper_open is None else 255.*(1.-gripper_open)
        step_count = period/self.model.opt.timestep + self._control_step_remainder
        physics_steps = round(step_count)
        self._control_step_remainder = step_count-physics_steps
        for _ in range(physics_steps):
            mujoco.mj_step(self.model, self.data)
            self.tick_metrics()
        self.elapsed += physics_steps*self.model.opt.timestep
        self.previous = self.normalized_control()
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Nonfinite simulation state")
        self.trace.append(dict(time=round(self.elapsed,3),tcp=self.tcp_position().tolist(),
            cube=self.cube_position().tolist(), goal=self.goal.tolist(),
            raw_ctrl=self.data.ctrl.tolist(), executed_control=self.previous.tolist(),
            command=[*xyz,float(yaw),int(closed)], ik_ok=ik_ok,
            requested_pose=requested,
            bilateral_contact=self.contacts(), success=self.success))
        if target_rotation is not None or gripper_open is not None or period != PERIOD:
            self.trace[-1].update(requested_rotation=rotation.tolist(), gripper_open_ratio=gripper_open,
                                  control_period_s=period, physics_steps=physics_steps)
        self.last_target = np.r_[xyz,yaw,int(closed)]
        self.last_closed = bool(closed)
        return ik_ok

    def score(self, decisions=0):
        distance = float(np.linalg.norm(self.cube_position()[:2]-self.goal[:2]))
        cube_tcp = float(np.linalg.norm(self.cube_position()-self.tcp_position()))
        center = 0. if self.grasp_center_error is None else math.exp(-self.grasp_center_error/.015)
        placement = math.exp(-distance/.02)
        ideal = float(np.linalg.norm(self.start[:2]-self.goal[:2]))+.4
        efficiency = min(1.,ideal/max(ideal,self.path))
        speed = min(1.,5./max(5.,self.elapsed))
        # Only stable, released placement unlocks score >= 6. Failure shaping
        # is useful for RL but cannot earn more than any completed task.
        if self.success:
            reward = 6.+center+placement+efficiency+.7*speed+.3/(1+decisions/20)
        else:
            reward = .3*math.exp(-cube_tcp/.1)+.7*int(self.ever_grasped)+1.5*int(self.lifted)
            reward += .5*center+(.8*placement if self.lifted else 0.)
        return dict(success=bool(self.success), score=float(min(10,reward)),
                    grasp_center_error_m=self.grasp_center_error, placement_error_m=distance,
                    tcp_path_m=self.path, ideal_path_m=ideal, elapsed_sim_s=self.elapsed, decisions=decisions,
                    lifted=bool(self.lifted), bilateral_grasp=bool(self.ever_grasped), ever_grasped=bool(self.ever_grasped),
                    stable_release_s=self.stable)

    def expert(self):
        """Shortest feasible cube-symmetric yaw, direct raised transfer.

        All phases end on measured tracking/contact conditions, not wall time.
        Desired poses are recorded before the matching 100-ms executed action.
        """
        rotation = self.data.site_xmat[self.tcp_site].reshape(3,3)
        current_yaw = math.atan2(rotation[1,0],rotation[0,0])
        choices = [self.yaw+k*math.pi/2 for k in range(4)]
        choices.sort(key=lambda angle:abs(math.atan2(math.sin(angle-current_yaw),math.cos(angle-current_yaw))))
        yaw = None
        for candidate in choices:
            try:
                self.solve_ik(np.r_[self.start[:2],TOP+HALF+.002], candidate, self.home,
                              target_rotation=downward_rotation(candidate))
                self.solve_ik(np.r_[self.goal[:2],TOP+HALF+.002], candidate, self.home,
                              target_rotation=downward_rotation(candidate))
                yaw = candidate
                break
            except RuntimeError:
                continue
        if yaw is None:
            raise RuntimeError("No feasible cube-symmetric grasp yaw")
        data = {key:[] for key in ("images","state","previous","offset","desired","phase")}
        stages = [(*self.start[:2],TOP+.20,False,0), (*self.start[:2],TOP+HALF+.002,False,1),
                  (*self.start[:2],TOP+HALF+.002,True,2), (*self.start[:2],TOP+.20,True,3),
                  (*self.goal[:2],TOP+.20,True,4), (*self.goal[:2],TOP+HALF+.002,True,5),
                  (*self.goal[:2],TOP+HALF+.002,False,6), (*self.goal[:2],TOP+.20,False,7)]
        for x,y,z,closed,phase in stages:
            for tick in range(45):
                observation = self.observe() if self.renderer else dict(state=self.state(),previous=self.previous.copy(),
                    offset=(self.cube_position()-self.tcp_position()).astype(np.float32))
                for key,value in observation.items():
                    data[key].append(value)
                delta = np.array([x,y,z])-self.tcp_position()
                desired = self.tcp_position()+delta*min(1.,.032/max(np.linalg.norm(delta),1e-9))
                data["desired"].append(np.r_[desired,yaw,int(closed)].astype(np.float32))
                data["phase"].append(phase)
                self.execute(desired,yaw,closed)
                error = np.linalg.norm(np.array([x,y,z])-self.tcp_position())
                done = error < .008 and tick >= 2
                if phase == 2:
                    done = self.contacts() and tick >= 5
                if phase == 6:
                    done = tick >= 6 and not self.contacts()
                if phase == 7:
                    done = self.success
                if done:
                    break
            else:
                return data, self.score()
        return data, self.score()


def episode_metadata(task, result):
    return dict(schema=SCHEMA, seed=task.seed, region=task.region, ordinal=task.ordinal, retry=task.retry,
                stratum=task.slot, cube_start=task.start.tolist(), goal=task.goal.tolist(),
                cube_yaw=task.yaw, instruction=INSTRUCTION, cameras=list(CAMERAS),
                raw_image_size=[task.height,task.width], period_s=PERIOD,
                state_layout="six_ur_joints_fixed_zero_slot_measured_gripper",
                action_layout="desired_world_xyz_yaw_binary_gripper", **result)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()
