# SPDX-License-Identifier: MIT
import base64
import email
import hashlib
import hmac
import json
import threading
import uuid
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from larex_actions.verifier import canonical_dispatch_request

PROCESSOR_ID = 'pero-ocr'
CAPABILITIES = {'incrementalPageResults': True, 'customFileResults': True, 'runReports': True}


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip('=')


def sign(payload: dict, path: str, secret: str) -> tuple[bytes, dict[str, str]]:
    body = json.dumps(payload).encode()
    timestamp = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
    nonce = uuid.uuid4().hex
    body_hash = _b64url(hashlib.sha256(body).digest())
    run_id = payload.get('runId') or payload['requestId']
    canonical = canonical_dispatch_request(
        method='POST',
        path_and_query=path,
        run_id=run_id,
        processor_id=PROCESSOR_ID,
        timestamp=timestamp,
        nonce=nonce,
        body_hash=body_hash,
    )
    signature = _b64url(hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).digest())
    return body, {
        'X-LAREX-Action-Auth': 'hmac-sha256;v=1',
        'X-LAREX-Action-Processor': PROCESSOR_ID,
        'X-LAREX-Action-Run-Id': run_id,
        'X-LAREX-Action-Timestamp': timestamp,
        'X-LAREX-Action-Nonce': nonce,
        'X-LAREX-Action-Body-SHA256': body_hash,
        'X-LAREX-Action-Signature': f'v1={signature}',
    }


class FakeLarex:
    def __init__(self, host: str = '127.0.0.1', origin: str = 'http://127.0.0.1') -> None:
        self.files: dict[str, bytes] = {}
        self.input: dict = {}
        self.results: list[tuple[dict, dict[str, bytes]]] = []
        self.heartbeats: list[dict] = []
        self.finished = threading.Event()
        larex = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def reply(self, content: bytes, content_type: str = 'application/json') -> None:
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def do_GET(self):
                if self.path == '/input':
                    self.reply(json.dumps(larex.input).encode())
                else:
                    self.reply(larex.files[self.path.removeprefix('/files/')], 'application/octet-stream')

            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                if self.path == '/heartbeat':
                    heartbeat = json.loads(body)
                    larex.heartbeats.append(heartbeat)
                    if heartbeat['status'] != 'running':
                        larex.finished.set()
                    self.reply(b'{"cancelRequested": false}')
                    return
                message = email.message_from_bytes(
                    b'Content-Type: ' + self.headers['Content-Type'].encode() + b'\r\n\r\n' + body
                )
                parts = {
                    part.get_param('name', header='content-disposition'): part.get_payload(decode=True)
                    for part in message.get_payload()
                }
                manifest = json.loads(parts.pop('manifest'))
                larex.results.append((manifest, parts))
                if manifest['status'] != 'running':
                    larex.finished.set()
                self.reply(b'{}')

        self.server = ThreadingHTTPServer((host, 0), Handler)
        self.origin = f'{origin}:{self.server.server_port}'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def prepare(self, pages: list[tuple[bytes, bytes | None]], **parameters) -> dict:
        run_id = f'run-{uuid.uuid4()}'
        items = []
        for number, (image, xml) in enumerate(pages, start=1):
            item = {'id': f'p{number}', 'name': f'page{number}', 'images': [], 'xml': []}
            for kind, content in (('images', image), ('xml', xml)):
                if content is not None:
                    name = f'{kind}{number}'
                    self.files[name] = content
                    item[kind].append({'id': name, 'fileName': name, 'downloadUrl': f'{self.origin}/files/{name}'})
            items.append(item)
        self.input = {
            'protocolVersion': 1,
            'runId': run_id,
            'processorKey': PROCESSOR_ID,
            'projectId': 'project-1',
            'parameters': parameters,
            'pages': items,
            'capabilities': CAPABILITIES,
        }
        return {
            'protocolVersion': 1,
            'runId': run_id,
            'processorId': PROCESSOR_ID,
            'workspaceId': 'workspace-1',
            'projectId': 'project-1',
            'pageIds': [item['id'] for item in items],
            'parameters': parameters,
            'secret': 'run-secret',
            'pullUrl': f'{self.origin}/input',
            'heartbeatUrl': f'{self.origin}/heartbeat',
            'resultUrl': f'{self.origin}/results',
            'capabilities': CAPABILITIES,
        }

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
