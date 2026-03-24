#!/usr/bin/env python3
"""
OBS screenshot client for the external OCR machine.

This first version keeps the API small and dependable:
- connect to OBS WebSocket
- request screenshots for a given source
- write image files to a cache directory
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture screenshots from OBS.")
    parser.add_argument("--host", default="127.0.0.1", help="OBS websocket host.")
    parser.add_argument("--port", type=int, default=4455, help="OBS websocket port.")
    parser.add_argument("--password", default="", help="OBS websocket password.")
    parser.add_argument("--source", required=True, help="OBS source name.")
    parser.add_argument("--out-dir", required=True, help="Image output directory.")
    parser.add_argument("--image-format", default="png", choices=["png", "jpg"], help="Image format.")
    parser.add_argument("--image-width", type=int, default=1920, help="Requested image width.")
    parser.add_argument("--count", type=int, default=1, help="Number of screenshots to capture.")
    parser.add_argument("--interval-ms", type=int, default=1500, help="Capture interval.")
    return parser.parse_args()


class ObsClient:
    def __init__(self, host: str, port: int, password: str) -> None:
        self._url = "ws://%s:%s" % (host, port)
        self._password = password
        self._message_id = 0
        self._socket = None

    async def __aenter__(self) -> "ObsClient":
        self._socket = await websockets.connect(self._url, max_size=32 * 1024 * 1024)
        await self._recv_json()
        await self._send_json({"op": 1, "d": {"rpcVersion": 1}})
        await self._recv_json()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._socket is not None:
            await self._socket.close()

    async def _send_json(self, payload: Dict[str, Any]) -> None:
        assert self._socket is not None
        await self._socket.send(json.dumps(payload))

    async def _recv_json(self) -> Dict[str, Any]:
        assert self._socket is not None
        return json.loads(await self._socket.recv())

    async def call(self, request_type: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        self._message_id += 1
        request_id = "req-%s" % self._message_id
        await self._send_json(
            {
                "op": 6,
                "d": {
                    "requestType": request_type,
                    "requestId": request_id,
                    "requestData": request_data,
                },
            }
        )
        while True:
            message = await self._recv_json()
            if message.get("op") != 7:
                continue
            data = message.get("d", {})
            if data.get("requestId") == request_id:
                return data


async def capture_once(client: ObsClient, source: str, image_format: str, image_width: int) -> bytes:
    response = await client.call(
        "SaveSourceScreenshot",
        {
            "sourceName": source,
            "imageFormat": image_format,
            "imageWidth": image_width,
            "imageCompressionQuality": 100,
            "imageFilePath": "",
        },
    )
    response_data = response.get("responseData", {})
    image_data = response_data.get("imageData")
    if not image_data:
        raise RuntimeError("OBS did not return screenshot data. Check source name and websocket support.")
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    return base64.b64decode(image_data)


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)


async def run_capture(args: argparse.Namespace) -> int:
    ensure_dir(args.out_dir)
    async with ObsClient(args.host, args.port, args.password) as client:
        for index in range(args.count):
            payload = await capture_once(client, args.source, args.image_format, args.image_width)
            filename = "capture_%s_%03d.%s" % (int(time.time()), index + 1, args.image_format)
            path = os.path.join(args.out_dir, filename)
            with open(path, "wb") as handle:
                handle.write(payload)
            print(path)
            if index + 1 < args.count:
                await asyncio.sleep(max(args.interval_ms, 0) / 1000.0)
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_capture(args))


if __name__ == "__main__":
    sys.exit(main())
