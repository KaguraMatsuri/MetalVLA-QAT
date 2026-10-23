"""Local-only API, browser-boundary and real MuJoCo smoke checks."""

from contextlib import contextmanager
import base64
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import random
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metalvla_qat.api_panel import PanelServer, TrialSession
from metalvla_qat.api_policy import ApiConfig, ApiError, MAX_ACTIONS, MAX_REQUESTS, RemotePolicy, action_chunk, completion_url

LOW, HIGH = (-.39, -.1, .732), (.39, .42, 1.12)
ACTION = [0., .14, 1.04, 0., False]
POSE = [0., .14, 1.04, 1., 0., 0., 0., 1.]


@contextmanager
def fake_api(mode="ok"):
    calls = []
    entered, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(dict(path=self.path, payload=payload, authorization=self.headers.get("Authorization")))
            entered.set()
            if mode == "block":
                release.wait(30)
            status = 503 if mode in ("overloaded", "overloaded_html", "long_error") else 401 if mode in ("unauthorized", "echo_id") or (mode == "fail_second" and len(calls) > 1) else 400 if mode == "request_id_error" else 307 if mode == "redirect" else 200
            if status == 200:
                if mode in ("poke_motion", "poke_two_chunks") and len(calls) > (2 if mode == "poke_two_chunks" else 1):
                    status = 401
                if mode in ("poke_motion", "poke_two_chunks"):
                    state = payload["inputs"]["state"]
                    rows = [[state[0]+.0005*(i+1), *state[1:7], .35] for i in range(40)]
                    result = {"output": rows}
                else:
                    result = {"output": [POSE]*40} if mode == "poke" else {"actions": [ACTION]}
                value = {"choices": [{"message": {"content": json.dumps(result)}}]}
            else:
                value = {"error": {"code": "system_cpu_overloaded", "message": "provider original detail: "+self.headers.get("Authorization")}} if mode == "overloaded" else {"error": self.headers.get("Authorization")}
                if mode == "long_error":
                    value = {"error": {"code": "system_cpu_overloaded", "message": "x"*9000+"original_tail"}}
            body = json.dumps(value).encode()
            if mode == "overloaded_html":
                body = ('<html>provider original detail '+self.headers.get("Authorization")+'<script>globalThis.apiBodyExecuted=true</script></html>').encode()
            self.send_response(status)
            self.send_header("X-Request-ID", calls[-1]["authorization"].removeprefix("Bearer ") if mode == "echo_id" else "a"*32)
            if status == 307:
                self.send_header("Location", "/stolen")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:"+str(server.server_port), calls, entered
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class FakeSession:
    def __init__(self):
        self.started = 0

    def start(self, config):
        self.started += 1

    def stop(self):
        pass

    def snapshot(self, since=-1):
        return dict(phase="idle", requests=0, frame_version=0)


class FixedStateGatewayConfig(ApiConfig):
    @property
    def gateway_state_unverified(self):
        return True


@contextmanager
def panel():
    server = PanelServer(default_base_url="https://example.test/v1", session=FakeSession())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def local_request(server, path, method="GET", body=None, token=True, origin=None, host=None, content_type="application/json"):
    headers = {"Content-Type": content_type}
    if token:
        headers["X-Session-Token"] = server.token
    if origin is not None:
        headers["Origin"] = origin
    if host is not None:
        headers["Host"] = host
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.read(), dict(response.getheaders())
    finally:
        connection.close()


class ApiTests(unittest.TestCase):
    def test_endpoint_normalization_is_idempotent(self):
        rng = random.Random(9)
        for _ in range(100):
            prefix = "/".join("p"+str(rng.randrange(100)) for _ in range(rng.randrange(3)))
            url = "https://example.test/"+prefix
            normalized = completion_url(url)
            self.assertEqual(completion_url(normalized), normalized)
            self.assertTrue(normalized.endswith("/chat/completions"))
        self.assertEqual(completion_url("https://example.test"), "https://example.test/v1/chat/completions")
        self.assertEqual(completion_url("https://example.test/v1/"), "https://example.test/v1/chat/completions")

    def test_rejects_unsafe_or_ambiguous_endpoint(self):
        for value in ("", "file:///secret", "http://example.test", "https://user:secret@example.test/v1",
                      "https://example.test/v1?api_key=secret", "https://example.test/#secret",
                      "https://example.test/\r\n", "https://example.test:bad/v1"):
            with self.subTest(value=value), self.assertRaises(ApiError):
                completion_url(value)

    def test_secret_not_in_repr_or_error(self):
        config = ApiConfig("https://example.test", "model", "test-secret-value")
        self.assertNotIn("test-secret-value", repr(config))
        with self.assertRaises(ApiError) as error:
            ApiConfig("https://example.test", "model", "test-secret\r\n")
        self.assertNotIn("test-secret", str(error.exception.public()))

    def test_action_roundtrip_and_bounds(self):
        rng = random.Random(19)
        for _ in range(100):
            actions = [[*(rng.uniform(a, b) for a, b in zip(LOW, HIGH)), rng.uniform(-math.pi, math.pi), bool(rng.randrange(2))]
                       for _ in range(rng.randrange(1, 10))]
            parsed = action_chunk(json.dumps({"actions": actions}), LOW, HIGH)
            self.assertEqual(parsed, list(map(tuple, actions)))
            self.assertTrue(all(all(a <= v <= b for a, v, b in zip(LOW, row[:3], HIGH)) for row in parsed))

    def test_invalid_action_never_partially_accepted(self):
        invalid = [True, float("nan"), float("inf"), 10**400, "0"]
        for number in invalid:
            row = list(ACTION)
            row[0] = number
            with self.subTest(number_type=type(number).__name__), self.assertRaises(ApiError):
                action_chunk({"actions": [ACTION, row]}, LOW, HIGH)
        for row in ([1., .14, 1.04, 0., False], [0., .14, 1.04, 9., False], [0., .14, 1.04, 0., 1]):
            with self.assertRaises(ApiError):
                action_chunk({"actions": [ACTION, row]}, LOW, HIGH)

    def test_action_count_and_fenced_json(self):
        self.assertEqual(len(action_chunk({"actions": [ACTION]*MAX_ACTIONS}, LOW, HIGH)), MAX_ACTIONS)
        for value in ({"actions": []}, {"actions": [ACTION]*(MAX_ACTIONS+1)}, {"actions": [ACTION], "code": "ignored"}, "not-json"):
            with self.assertRaises(ApiError):
                action_chunk(value, LOW, HIGH)
        text = "```json\n"+json.dumps({"actions": [ACTION]})+"\n```"
        self.assertEqual(action_chunk(text, LOW, HIGH), [tuple(ACTION)])

    def test_chat_completion_image_transport(self):
        with fake_api() as (url, calls, _):
            actions = RemotePolicy(ApiConfig(url, "vision-model", "test-secret")).predict(
                [b"jpeg"]*3, {"state": [0]*8}, "task", ("overhead", "wrist", "side"), LOW, HIGH)
        self.assertEqual(actions, [tuple(ACTION)])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["path"], "/v1/chat/completions")
        self.assertEqual(calls[0]["authorization"], "Bearer test-secret")
        content = calls[0]["payload"]["messages"][0]["content"]
        self.assertEqual(sum(item["type"] == "image_url" for item in content), 3)
        self.assertNotIn("test-secret", json.dumps(calls[0]["payload"]))

    def test_provider_errors_and_redirects_do_not_echo_secret_or_retry(self):
        for mode in ("unauthorized", "redirect"):
            with fake_api(mode) as (url, calls, _), self.assertRaises(ApiError) as error:
                RemotePolicy(ApiConfig(url, "vision", "test-secret")).predict([], {}, "task", (), LOW, HIGH)
            self.assertNotIn("test-secret", json.dumps(error.exception.public()))
            self.assertEqual(len(calls), 1)

    def test_request_ids_are_usable_without_echoing_credentials(self):
        with fake_api("request_id_error") as (url, calls, _), self.assertRaises(ApiError) as error:
            RemotePolicy(ApiConfig(url, "vision", "test-secret")).predict([], {}, "task", (), LOW, HIGH)
        self.assertEqual(error.exception.public(), dict(code="api_http_error", http_status=400, request_id="a"*32))
        self.assertEqual(len(calls), 1)
        secret = "abc123ef"*4
        with fake_api("echo_id") as (url, calls, _), self.assertRaises(ApiError) as error:
            RemotePolicy(ApiConfig(url, "vision", secret)).predict([], {}, "task", (), LOW, HIGH)
        self.assertNotIn(secret, json.dumps(error.exception.public()))

    def test_provider_machine_code_is_retained_without_error_message(self):
        with fake_api("overloaded") as (url, calls, _), self.assertRaises(ApiError) as error:
            RemotePolicy(ApiConfig(url, "vision", "test-secret")).predict([], {}, "task", (), LOW, HIGH)
        self.assertEqual(error.exception.public()["provider_code"], "system_cpu_overloaded")
        self.assertNotIn("test-secret", json.dumps(error.exception.public()))
        self.assertEqual(len(calls), 1)

    def test_actual_json_and_non_json_response_bodies_are_available(self):
        for mode in ("overloaded", "overloaded_html", "long_error"):
            with fake_api(mode) as (url, calls, _):
                policy = RemotePolicy(ApiConfig(url, "vision", "test-secret"))
                with self.assertRaises(ApiError):
                    policy.predict([], {}, "task", (), LOW, HIGH)
            response = policy.api_response
            self.assertEqual(response["http_status"], 503)
            self.assertNotIn("test-secret", json.dumps(response))
            self.assertNotIn("truncated", response)
            if mode == "overloaded":
                self.assertEqual(response["body"]["error"]["message"], "provider original detail: Bearer <redacted>")
                self.assertEqual(response["body"]["error"]["code"], "system_cpu_overloaded")
            elif mode == "long_error":
                self.assertTrue(response["body"]["error"]["message"].endswith("original_tail"))
            else:
                self.assertIn("<html>provider original detail", response["body"])
                self.assertIn("<script>", response["body"])
            self.assertEqual(len(calls), 1)

    def test_response_redaction_preserves_nonsecret_content_and_limits(self):
        from metalvla_qat.api_policy import public_api_response, MAX_RESPONSE_BYTES
        original = dict(error=dict(message="original 中文 <tag>", details=[1, None, True]), api_key="unknown-key", authorization="unknown-auth")
        response = public_api_response(400, json.dumps(original).encode(), "test-secret")
        self.assertEqual(response["body"]["error"], original["error"])
        self.assertNotIn("unknown-key", json.dumps(response))
        self.assertNotIn("unknown-auth", json.dumps(response))
        self.assertEqual(public_api_response(200, b'{"value":NaN}', "test-secret")["body"], '{"value":NaN}')
        response = public_api_response(502, b"x"*(MAX_RESPONSE_BYTES+1), "test-secret")
        self.assertTrue(response["truncated"])
        self.assertEqual(len(response["body"]), MAX_RESPONSE_BYTES)

    def test_panel_session_origin_and_host_boundaries(self):
        with panel() as server:
            self.assertEqual(local_request(server, "/state", token=False)[0], 403)
            self.assertEqual(local_request(server, "/state", host="other.test")[0], 403)
            self.assertEqual(local_request(server, "/stop", "POST", b"{}")[0], 403)
            self.assertEqual(local_request(server, "/stop", "POST", b"{}", origin="https://other.test")[0], 403)
            self.assertEqual(local_request(server, "/state")[0], 200)
            code, body, headers = local_request(server, "/")
            self.assertEqual(code, 200)
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
            self.assertNotIn(server.token.encode(), body)

    def test_panel_config_is_validated_and_key_never_returned(self):
        with panel() as server:
            body = json.dumps(dict(base_url="https://example.test/v1", model="vision", api_key="test-secret")).encode()
            code, response, _ = local_request(server, "/run", "POST", body, origin=server.origin)
            self.assertEqual(code, 200)
            self.assertNotIn(b"test-secret", response)
            self.assertEqual(server.session.started, 1)
            self.assertEqual(local_request(server, "/run", "POST", b"{}", origin=server.origin)[0], 400)
            self.assertEqual(server.session.started, 1)
            self.assertEqual(local_request(server, "/run", "POST", b"bad-json", origin=server.origin)[0], 400)

    def test_cancelled_worker_eof_does_not_become_failure(self):
        class ClosedPipe:
            def poll(self, timeout=0):
                return True

            def recv(self):
                raise EOFError()

            def close(self):
                pass

        session = TrialSession()
        session.state["phase"] = "stopping"
        session._watch(session.generation, SimpleNamespace(join=lambda timeout: None), ClosedPipe())
        self.assertEqual(session.snapshot()["phase"], "stopping")
        self.assertNotIn("error", session.snapshot())

    def test_repeated_stop_does_not_wait_for_the_same_process_twice(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        session = TrialSession()

        def join(timeout):
            calls.append(timeout)
            if len(calls) == 1:
                entered.set()
                release.wait(3)

        session.process = SimpleNamespace(join=join, is_alive=lambda: False)
        session.cancel = threading.Event()
        session.state["phase"] = "executing"
        thread = threading.Thread(target=session.stop)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            session.stop()
            self.assertEqual(len(calls), 1)
            self.assertEqual(session.snapshot()["phase"], "stopping")
        finally:
            release.set()
            thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(session.snapshot()["phase"], "stopped")

    def test_late_stop_cannot_overwrite_a_new_trial(self):
        session = TrialSession()
        new_process = MagicMock()
        session.context = SimpleNamespace(Pipe=lambda **_: (MagicMock(), MagicMock()),
            Event=threading.Event, Process=lambda **_: new_process)

        def join(timeout):
            # Another request completed this stop and started a new generation.
            session.state["phase"] = "stopped"
            with patch("metalvla_qat.api_panel.threading.Thread"):
                session.start(ApiConfig("http://127.0.0.1:1", "vision", "fixture"))

        session.process = SimpleNamespace(join=join, is_alive=lambda: False)
        session.cancel = threading.Event()
        session.state["phase"] = "executing"
        session.stop()
        self.assertIs(session.process, new_process)
        self.assertEqual(session.snapshot()["phase"], "initializing")
        self.assertFalse(session.cancel.is_set())

    def test_late_timeout_cannot_overwrite_stop_or_a_new_trial(self):
        for replace in (False, True):
            with self.subTest(replace=replace):
                session = TrialSession()
                session.state["phase"] = "waiting_api"

                def terminate():
                    session.state["phase"] = "stopping"
                    if replace:
                        session.generation += 1
                        session.process = MagicMock()
                        session.state["phase"] = "initializing"

                process = SimpleNamespace(terminate=terminate, join=lambda timeout: None)
                session.process = process
                connection = SimpleNamespace(poll=lambda timeout=0: False, close=lambda: None)
                with patch("metalvla_qat.api_panel.time.monotonic", side_effect=[0., 1000.]):
                    session._watch(session.generation, process, connection)
                self.assertEqual(session.snapshot()["phase"], "initializing" if replace else "stopping")
                self.assertNotIn("error", session.snapshot())


class PokeApiTests(unittest.TestCase):
    def image(self, mode="RGB"):
        stream = io.BytesIO()
        Image.new(mode, (32, 24)).save(stream, format="PNG")
        return stream.getvalue()

    def test_documented_single_image_state_and_output(self):
        from metalvla_qat.api_policy import PokePolicy
        state = [0., .14, 1.04, 1., 0., 0., 0., 1.]
        image = self.image()
        with fake_api("poke") as (url, calls, _):
            actions = PokePolicy(ApiConfig(url, "poke-model", "test-secret")).predict(image, state, "task")
        self.assertEqual(actions, [tuple(POSE)]*40)
        self.assertEqual(len(calls), 1)
        payload = calls[0]["payload"]
        self.assertEqual(set(payload), {"model", "messages", "inputs"})
        self.assertEqual(payload["inputs"], {"state": state})
        self.assertEqual(len(payload["messages"]), 1)
        message = payload["messages"][0]
        self.assertEqual(message["role"], "user")
        self.assertEqual([part["type"] for part in message["content"]], ["text", "image_url"])
        self.assertEqual(message["content"][0]["text"], "task")
        encoded = message["content"][1]["image_url"]["url"]
        self.assertTrue(encoded.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(encoded.split(",", 1)[1]), image)

    def test_output_normalization_preserves_pose_and_open_gripper(self):
        from metalvla_qat.api_policy import poke_action_chunk
        rng = random.Random(37)
        for _ in range(30):
            row = [rng.uniform(-1, 1) for _ in range(7)]+[rng.uniform(0, 1)]
            parsed = poke_action_chunk(json.dumps({"output": [row]*40}), 40)[0]
            self.assertEqual(parsed[:3], tuple(row[:3]))
            self.assertEqual(parsed[7], row[7])
            self.assertAlmostEqual(math.hypot(*parsed[3:7]), 1.)
            norm = math.hypot(*row[3:7])
            for a, b in zip(parsed[3:7], row[3:7]):
                self.assertAlmostEqual(a*norm, b)

    def test_rejects_whole_output_with_invalid_late_row(self):
        from metalvla_qat.api_policy import poke_action_chunk
        for number in (True, "0", float("nan"), float("inf"), 10**400):
            row = list(POSE)
            row[0] = number
            with self.subTest(kind=type(number).__name__), self.assertRaises(ApiError):
                poke_action_chunk({"output": [POSE]*39+[row]}, 40)
        invalid_rows = [POSE[:7], [*POSE[:3], 0., 0., 0., 0., 1.], [*POSE[:7], 1.02]]
        for row in invalid_rows:
            with self.assertRaises(ApiError):
                poke_action_chunk({"output": [POSE]*39+[row]}, 40)
        for value in ({"actions": [POSE]*40}, {"output": [POSE]*39}, {"output": []}, "not-json"):
            with self.assertRaises(ApiError):
                poke_action_chunk(value, 40)

    def test_bad_state_or_image_is_rejected_before_request(self):
        from metalvla_qat.api_policy import PokePolicy
        invalid = ([0]*8, POSE[:7], [True, *POSE[1:]], [float("nan"), *POSE[1:]])
        with fake_api("poke") as (url, calls, _):
            policy = PokePolicy(ApiConfig(url, "poke-model", "test-secret"))
            for state in invalid:
                with self.assertRaises(ApiError):
                    policy.predict(self.image(), state, "task")
            for image in (b"invalid", self.image("L")):
                with self.assertRaises(ApiError):
                    policy.predict(image, POSE, "task")
            self.assertEqual(calls, [])

    def test_previous_response_cannot_be_reused_for_a_rejected_new_request(self):
        from metalvla_qat.api_policy import PokePolicy
        with fake_api("poke") as (url, calls, _):
            policy = PokePolicy(ApiConfig(url, "poke-model", "test-secret"))
            policy.predict(self.image(), POSE, "task")
            self.assertEqual(policy.api_response["http_status"], 200)
            with self.assertRaises(ApiError):
                policy.predict(self.image(), [0]*8, "task")
            self.assertIsNone(policy.api_response)
            self.assertEqual(len(calls), 1)

    def test_provider_gripper_tolerance_and_model_horizons(self):
        from metalvla_qat.api_policy import poke_action_chunk
        for raw, expected in ((-.005, 0.), (1.005, 1.)):
            actions = poke_action_chunk({"output": [[*POSE[:7], raw]]*50}, 50)
            self.assertEqual(actions[0][7], expected)
        with self.assertRaises(ApiError):
            poke_action_chunk({"output": [POSE]*40}, 50)


class PokeExecutorTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from metalvla_qat.api_pose import PokeExecutor
        from ur_pick_place import URTask
        self.task = URTask(render=False)
        self.addCleanup(self.task.close)
        self.task.reset(21270001, 0, 0)
        self.executor = PokeExecutor(self.task, LOW, HIGH)

    def test_actual_base_frame_rotation_and_measured_cartesian_state(self):
        import numpy as np
        state = self.executor.state()
        world = self.task.tcp_position()
        np.testing.assert_allclose(state[:3], [-world[0], -world[1]-.43, world[2]-.72], atol=1e-12)
        self.assertAlmostEqual(state[7], 1.-float(self.task.state()[-1]))
        position, orientation, _ = self.executor.prepare([state])[0]
        np.testing.assert_allclose(position, world, atol=1e-12)
        np.testing.assert_allclose(orientation, self.task.data.site_xmat[self.task.tcp_site].reshape(3, 3), atol=1e-12)
        state[0] += .01
        position, _, _ = self.executor.prepare([state])[0]
        np.testing.assert_allclose(position, world+[-.01, 0., 0.], atol=1e-12)

    def test_out_of_bounds_late_target_cannot_partially_execute(self):
        state = self.executor.state()
        invalid = [*state]
        invalid[2] = -1.
        with self.assertRaises(ApiError):
            self.executor.prepare([state, invalid])
        self.assertEqual(self.task.trace, [])
        self.assertEqual(self.task.elapsed, 0.)

    def test_full_tilt_and_continuous_gripper_reach_shared_ik(self):
        import mujoco
        import numpy as np
        state = self.executor.state()
        # A tilted pose, not representable by the old fixed-downward yaw path.
        state[3:7] = [0., math.cos(.04), 0., math.sin(.04)]
        state[7] = .35
        action = self.executor.prepare([state])[0]
        self.assertTrue(self.executor.execute(action))
        self.assertAlmostEqual(float(self.task.data.ctrl[self.task.gripper_actuator]), 255.*.65)
        fk = mujoco.MjData(self.task.model)
        fk.qpos[:] = self.task.data.qpos
        fk.qpos[self.task.joint_qpos] = self.task.data.ctrl[self.task.arm_actuators]
        mujoco.mj_forward(self.task.model, fk)
        np.testing.assert_allclose(fk.site_xmat[self.task.tcp_site].reshape(3, 3), action[1], atol=.003)
        self.assertEqual(self.task.trace[-1]["gripper_open_ratio"], .35)

    def test_thirty_hz_uses_fractional_physics_steps_without_drift(self):
        from metalvla_qat.api_pose import POKE_PERIOD
        state = self.executor.state()
        state[7] = 1.
        action = self.executor.prepare([state])[0]
        before = self.task.data.time
        for _ in range(40):
            self.executor.execute(action)
        self.assertLessEqual(abs(self.task.elapsed-40*POKE_PERIOD), self.task.model.opt.timestep/2+1e-12)
        self.assertAlmostEqual(self.task.data.time-before, self.task.elapsed, places=10)
        self.assertEqual({r["physics_steps"] for r in self.task.trace}, {16, 17})


class MuJoCoTests(unittest.TestCase):
    def wait(self, session, predicate, limit=40):
        deadline = time.monotonic()+limit
        while time.monotonic() < deadline:
            state = session.snapshot()
            if predicate(state):
                return state
            time.sleep(.05)
        self.fail(str(session.snapshot()))

    def test_real_scene_failure_after_one_action(self):
        with fake_api("fail_second") as (url, calls, _):
            session = TrialSession()
            self.addCleanup(session.stop)
            session.start(ApiConfig(url, "vision", "test-secret"))
            state = self.wait(session, lambda s: s["phase"] == "failed")
            self.assertEqual(state["executed_steps"], 1)
            self.assertEqual(state["error"]["http_status"], 401)
            self.assertEqual(len(state["frames"]), 3)
            self.assertNotIn("test-secret", json.dumps(state))
            self.assertEqual(len(calls), 2)
            contract = json.loads(calls[0]["payload"]["messages"][0]["content"][0]["text"])
            self.assertNotIn("offset", contract["observation"])
            self.assertNotIn("cube", contract["observation"])
            self.assertNotIn("goal", contract["observation"])

    def test_poke_single_image_cartesian_state_drives_real_scene(self):
        with fake_api("poke_motion") as (url, calls, _):
            session = TrialSession()
            self.addCleanup(session.stop)
            session.start(ApiConfig(url, "poke-model", "test-secret"))
            state = self.wait(session, lambda s: s["phase"] == "failed")
            self.assertEqual(state["executed_steps"], 40,
                             {k: state.get(k) for k in ("phase", "requests", "error")})
            self.assertEqual(state["error"]["http_status"], 401)
            self.assertEqual(state["received_actions"], 40)
            self.assertEqual(state["inference_camera"], "ur_hand_wrist")
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(state["frames"]), 3)
            first, second = (c["payload"] for c in calls)
            self.assertEqual(len(first["messages"][0]["content"]), 2)
            # Real base-relative TCP y is about -0.57m, not a shoulder joint angle.
            self.assertAlmostEqual(first["inputs"]["state"][1], -.57, places=3)
            self.assertLess(second["inputs"]["state"][0]-first["inputs"]["state"][0], .03)
            self.assertGreater(second["inputs"]["state"][0]-first["inputs"]["state"][0], .01)
            self.assertNotEqual(first["messages"][0]["content"][1], second["messages"][0]["content"][1])
            self.assertNotIn("test-secret", json.dumps(state))

    def test_gateway_domain_flag_does_not_block_valid_local_actions(self):
        with fake_api("poke_motion") as (url, calls, _):
            session = TrialSession()
            self.addCleanup(session.stop)
            session.start(FixedStateGatewayConfig(url, "poke-model", "test-secret"))
            state = self.wait(session, lambda s: s["phase"] == "failed")
            self.assertEqual(state["executed_steps"], 40, state)
            self.assertEqual(state["error"]["code"], "api_http_error")
            self.assertEqual(state["error"]["http_status"], 401)
            self.assertTrue(state["gateway_state_unverified"])
            self.assertEqual(state["state_transport"], "inputs.state")
            self.assertEqual(state["request_id"], "a"*32)
            self.assertEqual(state["received_actions"], 40)
            self.assertEqual(len(calls), 2)

    def test_two_chunks_use_post_motion_state_and_the_same_frame_as_the_web(self):
        with fake_api("poke_two_chunks") as (url, calls, _):
            session = TrialSession()
            self.addCleanup(session.stop)
            session.start(ApiConfig(url, "poke-model", "test-secret"))
            state = self.wait(session, lambda s: s["phase"] == "failed")
            self.assertEqual(state["executed_steps"], 80)
            self.assertEqual(state["error"]["http_status"], 401)
            self.assertEqual(len(calls), 3)
            packets = [call["payload"] for call in calls]
            poses = [packet["inputs"]["state"] for packet in packets]
            images = [packet["messages"][0]["content"][1]["image_url"]["url"] for packet in packets]
            for before, after in zip(poses, poses[1:]):
                self.assertGreater(after[0]-before[0], .01)
                self.assertLess(after[0]-before[0], .03)
            self.assertEqual(len(set(images)), 3)
            self.assertEqual(poses[-1], state["measured_state"])
            self.assertEqual(images[-1], state["frames"][1])
            self.assertAlmostEqual(state["observation_sim_time_s"], state["elapsed_sim_s"])
            self.assertLessEqual(abs(state["observation_sim_time_s"]-80/30), .001+1e-10)

    def test_gateway_domain_flag_preserves_workspace_rejection(self):
        with fake_api("poke") as (url, calls, _):
            session = TrialSession()
            self.addCleanup(session.stop)
            session.start(FixedStateGatewayConfig(url, "poke-model", "test-secret"))
            state = self.wait(session, lambda s: s["phase"] == "failed")
            self.assertEqual(state["error"]["code"], "action_outside_workspace")
            self.assertEqual(state["executed_steps"], 0)
            self.assertEqual(len(calls), 1)
            response = state["api_response"]
            self.assertEqual(response["http_status"], 200)
            output = json.loads(response["body"]["choices"][0]["message"]["content"])["output"]
            self.assertEqual(len(output), 40)

    def test_request_budget_and_physical_result(self):
        with fake_api() as (url, calls, _):
            session = TrialSession()
            self.addCleanup(session.stop)
            session.start(ApiConfig(url, "vision", "test-secret"))
            state = self.wait(session, lambda s: s["phase"] in ("failed", "completed"))
            self.assertEqual(state["phase"], "completed", state)
            self.assertEqual(len(calls), MAX_REQUESTS)
            self.assertEqual(state["executed_steps"], MAX_REQUESTS)
            self.assertIs(state["result"]["success"], False)
            self.assertIs(state["result"]["expert_takeover"], False)

    def test_stop_interrupts_inflight_request_and_releases_worker(self):
        with fake_api("block") as (url, calls, entered):
            session = TrialSession()
            self.addCleanup(session.stop)
            session.start(ApiConfig(url, "vision", "test-secret"))
            self.assertTrue(entered.wait(30))
            with self.assertRaises(ApiError):
                session.start(ApiConfig(url, "vision", "test-secret"))
            started = time.monotonic()
            session.stop()
            self.assertLess(time.monotonic()-started, 3)
            self.assertFalse(session.process.is_alive())
            self.assertEqual(session.snapshot()["phase"], "stopped")
            self.assertNotIn("error", session.snapshot())
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
