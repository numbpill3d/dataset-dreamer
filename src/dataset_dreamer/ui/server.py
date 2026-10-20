"""fastapi web server for dataset-dreamer ui."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dataset_dreamer.config import settings
from dataset_dreamer.core.pipeline import DatasetDreamer, create_default_config
from dataset_dreamer.signals import fetch_signals_sync, create_signal_metadata

app = FastAPI(title="dataset-dreamer", version="0.1.0")

# static files and templates
BASE_DIR = Path(__file__).parent.parent.parent
STATIC_DIR = BASE_DIR / "assets" / "static"
TEMPLATES_DIR = BASE_DIR / "assets" / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# global state
dreamer: DatasetDreamer | None = None
active_connections: list[WebSocket] = []


@app.on_event("startup")
async def startup():
    global dreamer
    dreamer = DatasetDreamer(create_default_config())
    # initial signal fetch
    await asyncio.to_thread(dreamer.fetch_signals_sync)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/signals")
async def get_signals():
    if dreamer is None:
        return {"error": "not initialized"}
    await asyncio.to_thread(dreamer.fetch_signals_sync)
    return create_signal_metadata(dreamer.signals)


@app.get("/api/data/preview")
async def data_preview(rows: int = 10):
    if dreamer is None or dreamer.data is None:
        return {"error": "no data loaded"}
    return {
        "columns": list(dreamer.data.columns),
        "dtypes": {c: str(dreamer.data[c].dtype) for c in dreamer.data.columns},
        "sample": dreamer.data.head(rows).to_dict(orient="records"),
        "shape": list(dreamer.data.shape),
    }


@app.post("/api/data/load")
async def load_data(request: Request):
    if dreamer is None:
        return {"error": "not initialized"}
    body = await request.json()
    path = body.get("path")
    if not path:
        return {"error": "path required"}
    try:
        dreamer.load_data(path)
        return {"success": True, "shape": list(dreamer.data.shape), "columns": list(dreamer.data.columns)}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/train")
async def train_model(request: Request):
    if dreamer is None:
        return {"error": "not initialized"}
    body = await request.json()
    try:
        # update config from request
        if "epochs" in body:
            dreamer.config.training.epochs = body["epochs"]
        if "model_type" in body:
            dreamer.config.training.model_type = body["model_type"]

        await asyncio.to_thread(dreamer.train)
        return {"success": True, "model_path": str(dreamer.config.generation.model_path)}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/generate")
async def generate_data(request: Request):
    if dreamer is None or dreamer.synthesizer is None:
        return {"error": "model not trained"}
    body = await request.json()
    try:
        n = body.get("rows", 1000)
        temperature = body.get("temperature", 1.0)
        overrides = body.get("signal_overrides", {})

        generated = await asyncio.to_thread(
            dreamer.generate,
            n=n,
            temperature=temperature,
            signal_overrides=overrides if overrides else None,
        )

        out_path = dreamer.save_output(generated)
        return {
            "success": True,
            "rows": len(generated),
            "output": str(out_path),
            "sample": generated.head(5).to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/model/list")
async def list_models():
    model_dir = Path(settings.MODEL_DIR)
    models = []
    if model_dir.exists():
        for p in model_dir.iterdir():
            if p.is_dir() and (p / "meta.json").exists():
                try:
                    import json
                    meta = json.loads((p / "meta.json").read_text())
                    models.append({
                        "name": p.name,
                        "type": meta.get("type"),
                        "path": str(p),
                    })
                except Exception:
                    pass
    return {"models": models}


@app.post("/api/model/load")
async def load_model(request: Request):
    global dreamer
    body = await request.json()
    path = body.get("path")
    if not path:
        return {"error": "path required"}
    try:
        dreamer.load_model(Path(path))
        return {"success": True}
    except Exception as e:
        return {"error": str(e)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # handle real-time updates
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif data.get("type") == "signals":
                await asyncio.to_thread(dreamer.fetch_signals_sync)
                await websocket.send_json(create_signal_metadata(dreamer.signals))
    except WebSocketDisconnect:
        active_connections.remove(websocket)


async def broadcast(message: dict):
    for conn in active_connections:
        try:
            await conn.send_json(message)
        except Exception:
            pass


def run(host: str = "127.0.0.1", port: int = 8080, reload: bool = False):
    uvicorn.run("dataset_dreamer.ui.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run()