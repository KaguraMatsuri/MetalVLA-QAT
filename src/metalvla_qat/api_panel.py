"""Loopback-only API experiment panel; credentials live only in process memory."""

import base64
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import multiprocessing as mp
from pathlib import Path
import secrets
import sys
import threading
import time
from urllib.parse import urlsplit, parse_qs

from .api_policy import ApiConfig, ApiError, MAX_REQUESTS, REQUEST_TIMEOUT_S, PokePolicy, RemotePolicy, jpeg_frames

ROOT = Path(__file__).resolve().parents[2]
ACTIVE = {"initializing", "observing", "waiting_api", "executing", "stopping"}


def run_trial(config, connection, cancel):
    """A separate process lets stop interrupt even a blocked provider request."""
    sys.path.insert(0, str(ROOT / "scripts"))
    task = None
    policy = None
    counters = dict(requests=0, executed_steps=0)
    alignment = None
    try:
        from .mac.state import GPU_LOCK, exclusive_lock
        from ur_pick_place import URTask, CAMERAS, INSTRUCTION, PERIOD, WORKSPACE_LOW, WORKSPACE_HIGH
        from ur_pose_worker import precision_score

        with exclusive_lock(GPU_LOCK):
            task = URTask(render=True)
            task.reset(21270001, 0, 0)
            pose_executor = None
            if config.is_poke:
                import mujoco
                from .api_pose import POKE_CAMERA, POKE_PERIOD, PokeExecutor
                pose_executor = PokeExecutor(task, WORKSPACE_LOW, WORKSPACE_HIGH)
            policy = PokePolicy(config) if pose_executor is not None else RemotePolicy(config)

            def publish(phase, frames=None, **fields):
                connection.send(dict(phase=phase, **counters, **fields,
                                     **({"frames": frames} if frames is not None else {})))

            def observe():
                if pose_executor is not None:
                    mujoco.mj_forward(task.model, task.data)
                raw = task.observe()
                # Only proprioception/calibration leaves the simulator. Object
                # positions and privileged offset remain confined to scoring.
                cameras = []
                for name in CAMERAS:
                    index = task.model.camera(name).id
                    cameras.append(dict(name=name, position_world_m=task.data.cam_xpos[index].tolist(),
                        rotation_camera_to_world=task.data.cam_xmat[index].reshape(3, 3).tolist(),
                        vertical_fov_deg=float(task.model.cam_fovy[index])))
                rotation = task.data.site_xmat[task.tcp_site].reshape(3, 3)
                observation = dict(state=raw["state"].tolist(), previous_control=raw["previous"].tolist(),
                    tcp_world_m=task.tcp_position().tolist(), tcp_yaw_rad=math.atan2(rotation[1, 0], rotation[0, 0]),
                    cameras=cameras, image_width=task.width, image_height=task.height)
                measured = (dict(measured_state=pose_executor.state(), observation_sim_time_s=task.elapsed)
                            if pose_executor is not None else {})
                return jpeg_frames(raw["images"]), observation, measured

            frames, observation, measured = observe()
            publish("observing", frames, **measured, **(dict(protocol="pokebot", inference_camera=POKE_CAMERA,
                    action_frame="ur_base", control_period_s=POKE_PERIOD,
                    state_transport="inputs.state", gateway_state_unverified=config.gateway_state_unverified)
                    if pose_executor is not None else {}))
            for _ in range(MAX_REQUESTS):
                if cancel.is_set():
                    break
                counters["requests"] += 1
                publish("waiting_api", api_response=None)
                if pose_executor is not None:
                    actions = policy.predict(frames[CAMERAS.index(POKE_CAMERA)], measured["measured_state"], INSTRUCTION)
                    publish("waiting_api", received_actions=len(actions), request_id=policy.request_id,
                            api_response=policy.api_response)
                    if config.model not in ("poke-model", "vision-1"):
                        raise ApiError("unsupported_pose_model", request_id=policy.request_id)
                    actions = pose_executor.prepare(actions)
                else:
                    actions = policy.predict(frames, observation, INSTRUCTION, CAMERAS, WORKSPACE_LOW, WORKSPACE_HIGH)
                publish("executing", api_response=policy.api_response)
                last_frame = time.monotonic()
                chunk_started = time.monotonic()
                for action_index, action in enumerate(actions):
                    if cancel.is_set():
                        break
                    started = time.monotonic()
                    if pose_executor is not None:
                        pose_executor.execute(action)
                        closed = action[2] < .5
                        period = POKE_PERIOD
                    else:
                        x, y, z, yaw, closed = action
                        task.execute([x, y, z], yaw, closed)
                        period = PERIOD
                    task.trace[-1]["action_source"] = "remote_api"
                    counters["executed_steps"] += 1
                    if closed:
                        import numpy as np
                        distance = float(np.linalg.norm(task.data.geom_xpos[task.pad_ids].mean(0)-task.cube_position()))
                        alignment = distance if alignment is None else min(alignment, distance)
                    if time.monotonic()-last_frame >= .25 or task.success:
                        frames, _, measured = observe()
                        publish("executing", frames, elapsed_sim_s=task.elapsed, **measured)
                        last_frame = time.monotonic()
                    if task.success:
                        break
                    deadline = chunk_started+(action_index+1)*period if pose_executor is not None else started+period
                    cancel.wait(max(0., deadline-time.monotonic()))
                frames, observation, measured = observe()
                publish("observing", frames, elapsed_sim_s=task.elapsed, **measured)
                if task.success or cancel.is_set():
                    break
            result = precision_score(task, counters["requests"], 0, alignment)
            result.update(seed=21270001, region=0, ordinal=0, expert_takeover=False, viewer_opened=False)
            publish("stopped" if cancel.is_set() else "completed", result=result,
                    reason="success" if task.success else "stopped" if cancel.is_set() else "request_limit")
    except ApiError as exc:
        connection.send(dict(phase="failed", error=exc.public(),
                             api_response=policy.api_response if policy is not None else None, **counters))
    except Exception as exc:
        busy = isinstance(exc, RuntimeError) and exc.args and isinstance(exc.args[0], dict) and "active_job_lock" in exc.args[0]
        connection.send(dict(phase="failed", error={"code": "resource_busy" if busy else "simulation_error",
                                                   "type": type(exc).__name__},
                             api_response=policy.api_response if policy is not None else None, **counters))
    finally:
        if task is not None:
            task.close()
        connection.close()


class TrialSession:
    def __init__(self, worker=run_trial):
        self.worker = worker
        self.context = mp.get_context("spawn")
        self.lock = threading.RLock()
        self.process = None
        self.cancel = None
        self.state = dict(phase="idle", requests=0, executed_steps=0, max_requests=MAX_REQUESTS)
        self.frames = []
        self.frame_version = 0
        self.generation = 0

    def start(self, config):
        with self.lock:
            if self.process is not None and self.process.is_alive():
                raise ApiError("trial_active")
            if self.state["phase"] in ACTIVE:
                raise ApiError("trial_active")
            self.generation += 1
            generation = self.generation
            self.frames = []
            self.frame_version += 1
            self.state = dict(phase="initializing", requests=0, executed_steps=0, max_requests=MAX_REQUESTS)
            parent, child = self.context.Pipe(duplex=False)
            self.cancel = self.context.Event()
            process = self.context.Process(target=self.worker, args=(config, child, self.cancel), daemon=True)
            self.process = process
            try:
                process.start()
            except Exception:
                parent.close()
                child.close()
                self.state.update(phase="failed", error={"code": "worker_start_failed"})
                raise ApiError("worker_start_failed") from None
            child.close()
            threading.Thread(target=self._watch, args=(generation, process, parent), daemon=True).start()

    def _watch(self, generation, process, connection):
        phase_started = time.monotonic()
        try:
            while True:
                if connection.poll(.1):
                    try:
                        message = connection.recv()
                    except EOFError:
                        break
                    with self.lock:
                        if generation != self.generation:
                            break
                        if self.state["phase"] in ("stopping", "stopped"):
                            continue
                        if "frames" in message:
                            self.frames = message.pop("frames")
                            self.frame_version += 1
                        if message.get("phase") != self.state["phase"]:
                            phase_started = time.monotonic()
                        self.state.update(message)
                with self.lock:
                    if generation != self.generation:
                        break
                    phase = self.state["phase"]
                # Socket timeouts alone do not bound a peer trickling a response.
                limit = REQUEST_TIMEOUT_S + 2 if phase == "waiting_api" else 120 if phase == "initializing" else None
                if limit is not None and time.monotonic()-phase_started > limit:
                    process.terminate()
                    with self.lock:
                        if (generation == self.generation and process is self.process
                                and self.state["phase"] == phase):
                            self.state.update(phase="failed", error={"code": "api_timeout" if phase == "waiting_api" else "simulation_timeout"})
                    break
                if not process.is_alive() and not connection.poll():
                    break
        finally:
            connection.close()
            process.join(timeout=2)
            with self.lock:
                if generation == self.generation and self.state["phase"] in ACTIVE - {"stopping"}:
                    self.state.update(phase="failed", error={"code": "worker_exited"})

    def snapshot(self, since=-1):
        with self.lock:
            value = dict(self.state, frame_version=self.frame_version)
            if since != self.frame_version:
                value["frames"] = ["data:image/jpeg;base64,"+base64.b64encode(frame).decode("ascii") for frame in self.frames]
            return value

    def stop(self):
        with self.lock:
            process = self.process
            if process is None or self.state["phase"] not in ACTIVE - {"stopping"}:
                return
            generation = self.generation
            self.state["phase"] = "stopping"
            self.cancel.set()
        process.join(timeout=.5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        with self.lock:
            if (generation == self.generation and process is self.process
                    and self.state["phase"] == "stopping"):
                self.state["phase"] = "stopped"


class PanelServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address=("127.0.0.1", 0), default_base_url="", session=None):
        if address[0] != "127.0.0.1":
            raise ValueError({"loopback_required": True})
        self.session = session if session is not None else TrialSession()
        self.token = secrets.token_urlsafe(32)
        self.default_base_url = default_base_url
        super().__init__(address, PanelHandler)
        self.origin = "http://127.0.0.1:"+str(self.server_port)

    @property
    def url(self):
        return self.origin+"/#"+self.token

    def server_close(self):
        self.session.stop()
        super().server_close()


class PanelHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *args):
        pass

    def respond(self, status, value, content_type="application/json; charset=utf-8", nonce=None):
        body = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'; img-src data:; connect-src 'self'; "
                         "base-uri 'none'; frame-ancestors 'none'; form-action 'none'; "
                         + (f"script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'" if nonce else ""))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def authorize(self, mutation=False):
        if self.headers.get("Host") != self.server.origin.removeprefix("http://"):
            raise ApiError("invalid_host")
        origin = self.headers.get("Origin")
        if (origin is not None or mutation) and origin != self.server.origin:
            raise ApiError("invalid_origin")
        token = self.headers.get("X-Session-Token", "")
        if not token.isascii() or not hmac.compare_digest(token, self.server.token):
            raise ApiError("invalid_session")

    def do_GET(self):
        parts = urlsplit(self.path)
        if parts.path == "/":
            if self.headers.get("Host") != self.server.origin.removeprefix("http://"):
                self.respond(403, {"error": {"code": "invalid_host"}})
                return
            nonce = secrets.token_urlsafe(24)
            page = Path(__file__).with_name("api_panel.html").read_text().replace("__NONCE__", nonce)
            self.respond(200, page.encode(), "text/html; charset=utf-8", nonce)
            return
        try:
            self.authorize()
            if parts.path == "/state":
                try:
                    since = int(parse_qs(parts.query).get("since", ["-1"])[0])
                except ValueError:
                    raise ApiError("invalid_frame_version") from None
                self.respond(200, self.server.session.snapshot(since))
            elif parts.path == "/config":
                self.respond(200, {"base_url": self.server.default_base_url})
            else:
                self.respond(404, {"error": {"code": "not_found"}})
        except ApiError as exc:
            self.respond(403, {"error": exc.public()})

    def do_POST(self):
        try:
            self.authorize(mutation=True)
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ApiError("invalid_content_type")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ApiError("invalid_content_length") from None
            if not 0 < length <= 16384:
                raise ApiError("invalid_content_length")
            try:
                body = json.loads(self.rfile.read(length))
            except (ValueError, RecursionError):
                raise ApiError("invalid_json") from None
            if self.path == "/run":
                if not isinstance(body, dict) or set(body) != {"base_url", "model", "api_key"}:
                    raise ApiError("invalid_config")
                self.server.session.start(ApiConfig(**body))
            elif self.path == "/stop":
                if body != {}:
                    raise ApiError("invalid_config")
                self.server.session.stop()
            else:
                self.respond(404, {"error": {"code": "not_found"}})
                return
            self.respond(200, self.server.session.snapshot())
        except ApiError as exc:
            status = 409 if exc.code == "trial_active" else 403 if exc.code in ("invalid_host", "invalid_origin", "invalid_session") else 400
            self.respond(status, {"error": exc.public()})
