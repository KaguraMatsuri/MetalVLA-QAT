"""Bounded Chat Completions transports for generic vision and PokeBot VLA."""

import base64
from dataclasses import dataclass, field
import io
import json
import math
import re
from urllib import error, parse, request

from PIL import Image

MAX_REQUESTS = 20
MAX_ACTIONS = 130
MAX_RESPONSE_BYTES = 1024 * 1024
REQUEST_TIMEOUT_S = 60
POKE_HORIZONS = {"poke-model": 40, "vision-1": 40, "vision-2": 40, "vision-3": 50, "vision-4": 40}


def _request_id(value, secret=None):
    return value if isinstance(value, str) and value != secret and re.fullmatch(r"[a-f0-9]{32}", value) else None


def _provider_code(value, secret=None):
    # Diagnostic codes only; messages and credential-shaped values stay private.
    return value if (isinstance(value, str) and value != secret
                     and not value.startswith(("pk_", "sk_"))
                     and re.fullmatch(r"[a-z][a-z_]{0,63}", value)) else None


def public_api_response(status, body, secret):
    """Keep the received body for the local page, with credentials removed."""
    truncated = len(body) > MAX_RESPONSE_BYTES
    text = body[:MAX_RESPONSE_BYTES].decode("utf-8", errors="replace")

    def clean_text(value):
        value = value.replace(secret, "<redacted>") if secret else value
        value = re.sub(r"\b(?:sk-|pk_)[A-Za-z0-9_-]+", "<redacted>", value)
        return re.sub(r"(?i)\bBearer[ \t]+[^\s\"'<>\\]+", "Bearer <redacted>", value)

    def clean_json(value):
        if isinstance(value, dict):
            sensitive = {"apikey", "authorization", "accesstoken", "token", "secret", "password", "key"}
            return {clean_text(k): "<redacted>" if re.sub(r"[_\s-]", "", k).lower() in sensitive
                    else clean_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean_json(v) for v in value]
        return clean_text(value) if isinstance(value, str) else value

    try:
        value = json.loads(text)
        # A response with NaN/Infinity must remain displayable by our strict
        # local JSON encoder. Preserve its received text in that case.
        json.dumps(value, allow_nan=False)
        value = clean_json(value)
    except (ValueError, RecursionError):
        value = clean_text(text)
    result = dict(http_status=status, body=value)
    if truncated:
        result["truncated"] = True
    return result


class ApiError(Exception):
    """Only machine codes cross the local UI/logging boundary."""

    def __init__(self, code, http_status=None, *, request_id=None, provider_code=None):
        self.code, self.http_status = code, http_status
        self.request_id = _request_id(request_id)
        self.provider_code = _provider_code(provider_code)
        super().__init__(code)

    def public(self):
        value = {"code": self.code}
        if self.http_status is not None:
            value["http_status"] = self.http_status
        if self.request_id is not None:
            value["request_id"] = self.request_id
        if self.provider_code is not None:
            value["provider_code"] = self.provider_code
        return value


def completion_url(value):
    if not isinstance(value, str) or not value.strip() or any(ord(c) <= 32 for c in value):
        raise ApiError("invalid_base_url")
    try:
        parts = parse.urlsplit(value.strip())
        _ = parts.port
        if (not parts.hostname or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment or parts.scheme not in ("http", "https")):
            raise ValueError()
        if parts.scheme != "https" and parts.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError()
    except ValueError:
        raise ApiError("invalid_base_url") from None
    path = parts.path.rstrip("/") or "/v1"
    if not path.endswith("/chat/completions"):
        path += "/chat/completions"
    return parse.urlunsplit((parts.scheme, parts.netloc, path, "", ""))


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    model: str
    api_key: str = field(repr=False)

    def __post_init__(self):
        object.__setattr__(self, "base_url", completion_url(self.base_url))
        for name, limit in (("model", 256), ("api_key", 8192)):
            value = getattr(self, name)
            if (not isinstance(value, str) or not value.strip() or len(value) > limit
                    or (name == "api_key" and not value.isascii())
                    or any(ord(c) < 32 or ord(c) == 127 for c in value)):
                raise ApiError("invalid_" + name)
            object.__setattr__(self, name, value.strip())

    @property
    def is_poke(self):
        return self.model in POKE_HORIZONS or parse.urlsplit(self.base_url).hostname == "platform.pokebot.tech"

    @property
    def gateway_state_unverified(self):
        # The supplied New API guide documents a fixed channel override and
        # discarded client inputs.state. This is diagnostic metadata, not a
        # hostname-based execution ban. The local executor validates the whole
        # action chunk against its actual pose and workspace contract.
        return parse.urlsplit(self.base_url).hostname == "platform.pokebot.tech"


def jpeg_frames(images):
    frames = []
    for image in images:
        stream = io.BytesIO()
        Image.fromarray(image).save(stream, format="JPEG", quality=85)
        frames.append(stream.getvalue())
    return frames


def action_chunk(value, low, high):
    """Reject the whole response before executing any of its actions."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1])
        try:
            value = json.loads(text)
        except (ValueError, RecursionError):
            raise ApiError("invalid_action_json") from None
    if not isinstance(value, dict) or set(value) != {"actions"}:
        raise ApiError("invalid_action_schema")
    actions = value["actions"]
    if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_ACTIONS:
        raise ApiError("invalid_action_count")
    result = []
    for action in actions:
        if not isinstance(action, list) or len(action) != 5 or not isinstance(action[4], bool):
            raise ApiError("invalid_action_schema")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in action[:4]):
            raise ApiError("invalid_action_number")
        try:
            numbers = [float(v) for v in action[:4]]
        except (ValueError, OverflowError):
            raise ApiError("invalid_action_number") from None
        if not all(math.isfinite(v) for v in numbers):
            raise ApiError("invalid_action_number")
        if any(not a <= v <= b for a, v, b in zip(low, numbers[:3], high, strict=True)):
            raise ApiError("action_outside_workspace")
        if abs(numbers[3]) > math.pi:
            raise ApiError("invalid_yaw")
        result.append((*numbers, action[4]))
    return result


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # urllib raises HTTPError, letting us display the original body while
        # still refusing to forward the Authorization header to another URL.
        return None


class _ChatPolicy:
    def __init__(self, config):
        self.config = config
        self.opener = request.build_opener(NoRedirect())
        self.request_id = None
        self.api_response = None

    def _complete(self, payload):
        self.request_id = None
        self.api_response = None
        req = request.Request(self.config.base_url, data=payload, method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.config.api_key})
        try:
            with self.opener.open(req, timeout=REQUEST_TIMEOUT_S) as response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
                request_id = response.headers.get("X-Request-ID")
                self.api_response = public_api_response(response.status, body, self.config.api_key)
            if len(body) > MAX_RESPONSE_BYTES:
                raise ApiError("api_response_too_large")
            parsed = json.loads(body)
        except error.HTTPError as exc:
            status, request_id = exc.code, exc.headers.get("X-Request-ID")
            provider_code = None
            try:
                body = exc.read(MAX_RESPONSE_BYTES + 1)
                self.api_response = public_api_response(status, body, self.config.api_key)
                details = json.loads(body)
                provider_code = _provider_code(details["error"].get("code"), self.config.api_key)
            except (ValueError, RecursionError, KeyError, TypeError, AttributeError, OSError):
                pass
            finally:
                exc.close()
            if self.api_response is None:
                self.api_response = dict(http_status=status, body=None, body_unavailable=True)
            code = "api_redirect_rejected" if 300 <= status < 400 else "api_http_error"
            raise ApiError(code, status, request_id=_request_id(request_id, self.config.api_key),
                           provider_code=provider_code) from None
        except (error.URLError, TimeoutError, OSError):
            raise ApiError("api_network_error") from None
        except (ValueError, RecursionError):
            raise ApiError("invalid_api_json") from None
        try:
            message = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ApiError("invalid_api_response") from None
        if not isinstance(message, str):
            raise ApiError("invalid_api_response")
        candidate = parsed.get("request_id") or request_id
        self.request_id = _request_id(candidate, self.config.api_key)
        return message


class RemotePolicy(_ChatPolicy):

    def predict(self, frames, observation, instruction, camera_names, low, high):
        self.api_response = None
        contract = dict(
            task=instruction, response_format="json_only", action_frame="world",
            action_units=["x_m", "y_m", "z_m", "yaw_rad", "gripper_closed_bool"],
            action_response={"actions": [[0.0, 0.14, 0.95, 0.0, False]]},
            max_actions=MAX_ACTIONS, control_period_s=.1, max_translation_per_step_m=.035,
            workspace_low=list(low), workspace_high=list(high),
            tool_orientation="downward", observation=observation,
        )
        content = [{"type": "text", "text": json.dumps(contract, allow_nan=False)}]
        for name, frame in zip(camera_names, frames, strict=True):
            content.extend([
                {"type": "text", "text": name},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(frame).decode("ascii")}},
            ])
        payload = json.dumps(dict(model=self.config.model,
                                  messages=[dict(role="user", content=content)],
                                  max_tokens=8192, stream=False), allow_nan=False).encode()
        return action_chunk(self._complete(payload), low, high)


def _pose_row(value, code):
    if not isinstance(value, (list, tuple)) or len(value) != 8:
        raise ApiError(code)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ApiError(code)
    try:
        row = [float(v) for v in value]
    except (ValueError, OverflowError):
        raise ApiError(code) from None
    if not all(math.isfinite(v) for v in row):
        raise ApiError(code)
    norm = math.hypot(*row[3:7])
    if not 0 < norm < math.inf:
        raise ApiError(code)
    return row, norm


def poke_action_chunk(value, horizon):
    """Absolute xyz metres, quaternion xyzw, gripper ratio (1 = open).

    Matches the provider client's normalization and 0.01 gripper tolerance;
    coordinate conversion and workspace checks belong to the local executor.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            raise ApiError("invalid_action_json") from None
    if not isinstance(value, dict) or not isinstance(value.get("output"), list):
        raise ApiError("invalid_action_schema")
    actions = value["output"]
    if not 1 <= len(actions) <= MAX_ACTIONS or (horizon is not None and len(actions) != horizon):
        raise ApiError("invalid_action_count")
    result = []
    for value in actions:
        row, norm = _pose_row(value, "invalid_pose_action")
        if not -.01 <= row[7] <= 1.01:
            raise ApiError("invalid_gripper")
        result.append((*row[:3], *(v/norm for v in row[3:7]), min(1., max(0., row[7]))))
    return result


class PokePolicy(_ChatPolicy):
    def predict(self, frame, state, instruction):
        self.api_response = None
        state, _ = _pose_row(state, "invalid_pose_state")
        limit = 4096 if self.config.model in ("vision-3", "vision-4") else 8192
        if (not isinstance(instruction, str) or len(instruction) > limit
                or (self.config.model in ("vision-3", "vision-4") and not instruction.strip())):
            raise ApiError("invalid_instruction")
        if self.config.model == "vision-3" and not 0 <= state[7] <= 1:
            raise ApiError("invalid_pose_state")
        try:
            with Image.open(io.BytesIO(frame)) as image:
                if (image.format not in ("JPEG", "PNG") or image.mode != "RGB"
                        or image.width*image.height > 16_000_000
                        or max(image.size)/min(image.size) > 20):
                    raise ApiError("invalid_image")
                media_type = "jpeg" if image.format == "JPEG" else "png"
                image.verify()
        except (OSError, TypeError, ValueError, Image.DecompressionBombError):
            raise ApiError("invalid_image") from None
        payload = json.dumps(dict(model=self.config.model, messages=[dict(role="user", content=[
            dict(type="text", text=instruction),
            dict(type="image_url", image_url=dict(url="data:image/"+media_type+";base64,"+base64.b64encode(frame).decode("ascii"))),
        ])], inputs=dict(state=state)), allow_nan=False).encode()
        if len(payload) > 8*1024*1024:
            raise ApiError("api_request_too_large")
        return poke_action_chunk(self._complete(payload), POKE_HORIZONS.get(self.config.model))
