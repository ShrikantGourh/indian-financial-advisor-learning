#!/usr/bin/env python3
"""Serve the local LoRA pilot adapter for testing from build_log.html."""

import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME = "indian_financial_advisor:pilot-cpu"
MAX_BODY_BYTES = 64_000
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 128


def is_local_origin(origin: str) -> bool:
    if origin == "null":
        return True
    parsed = urlsplit(origin)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }


def load_model(adapter_path: Path):
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No trained adapter found at {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        local_files_only=True,
    ).eval()
    return model, tokenizer


def make_handler(model, tokenizer, generation_lock: threading.Lock):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            origin = self.headers.get("Origin")
            if origin and not is_local_origin(origin):
                status_code = 403
                body = json.dumps({"error": "Only local browser origins are allowed"}).encode("utf-8")

            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if origin and is_local_origin(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self._send_json(200, {})

        def do_GET(self) -> None:
            if self.path == "/api/tags":
                self._send_json(200, {"models": [{"name": MODEL_NAME, "model": MODEL_NAME}]})
            elif self.path == "/api/health":
                self._send_json(200, {"ok": True, "model": MODEL_NAME})
            else:
                self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self._send_json(404, {"error": "Not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "Invalid Content-Length"})
                return
            if content_length <= 0 or content_length > MAX_BODY_BYTES:
                self._send_json(413, {"error": f"Request body must be between 1 and {MAX_BODY_BYTES} bytes"})
                return

            try:
                request = json.loads(self.rfile.read(content_length))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                self._send_json(400, {"error": f"Invalid JSON request: {error}"})
                return

            messages = request.get("messages") if isinstance(request, dict) else None
            if not isinstance(messages, list) or not messages:
                self._send_json(400, {"error": "Expected a non-empty messages array"})
                return
            if any(
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                for message in messages
            ):
                self._send_json(400, {"error": "Messages must have supported roles and text content"})
                return
            if messages[-1]["role"] != "user":
                self._send_json(400, {"error": "The last message must be from the user"})
                return
            if sum(len(message["content"]) for message in messages) > MAX_BODY_BYTES:
                self._send_json(413, {"error": "Combined message text is too long"})
                return

            try:
                encoded = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                input_length = encoded["input_ids"].shape[1]
                if input_length > MAX_INPUT_TOKENS:
                    self._send_json(
                        413,
                        {"error": f"Prompt is {input_length} tokens; limit is {MAX_INPUT_TOKENS}. Shorten it and retry."},
                    )
                    return

                with generation_lock, torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                content = tokenizer.decode(
                    generated[0, input_length:],
                    skip_special_tokens=True,
                ).strip()
                if not content:
                    self._send_json(500, {"error": "The model returned an empty response"})
                    return
                self._send_json(
                    200,
                    {
                        "model": MODEL_NAME,
                        "message": {"role": "assistant", "content": content},
                        "done": True,
                    },
                )
            except Exception as error:
                logging.exception("Adapter inference failed")
                self._send_json(500, {"error": f"Adapter inference failed: {error}"})

        def log_message(self, format: str, *args) -> None:
            logging.info("%s - %s", self.address_string(), format % args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts/pilot-cpu",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11435)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model, tokenizer = load_model(args.adapter)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(model, tokenizer, threading.Lock()),
    )
    logging.info("Loaded %s from %s", MODEL_NAME, args.adapter)
    logging.info("Adapter test server listening at http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Stopping adapter test server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
