import asyncio
import json
import tempfile
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl

import ingestion

app = FastAPI(title="BanFakeNews Ingestion API")

# Prototype CORS: wide open so the static Aurora frontend can call this
# from any origin (file://, localhost:5500, etc). Lock this down to your
# real frontend origin before this touches the internet.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    url: HttpUrl


class IngestResponse(BaseModel):
    job_id: str


# In-memory job store. Fine for a single-process prototype; swap for
# Redis (or similar) if you ever run this with more than one worker.
JOBS: dict[str, dict] = {}


async def _emit(queue: asyncio.Queue, step: str, status: str, extra: Optional[dict] = None):
    payload = {"step": step, "status": status}
    if extra:
        payload.update(extra)
    await queue.put(payload)


async def run_pipeline(job_id: str, url: str):
    job = JOBS[job_id]
    queue: asyncio.Queue = job["queue"]
    result: dict = {}
    current = "meta"
    try:
        current = "meta"
        await _emit(queue, "meta", "active")
        info = await asyncio.to_thread(ingestion.fetch_basic_info, url)
        result["title"] = info.get("title")
        result["description"] = (info.get("description") or "")[:280]
        await _emit(queue, "meta", "done")

        current = "channel"
        await _emit(queue, "channel", "active")
        result["channel"] = info.get("channel") or info.get("uploader")
        result["subscribers"] = info.get("channel_follower_count")
        await _emit(queue, "channel", "done")

        current = "engagement"
        await _emit(queue, "engagement", "active")
        result["views"] = info.get("view_count")
        result["likes"] = info.get("like_count")
        await _emit(queue, "engagement", "done")

        current = "comments"
        await _emit(queue, "comments", "active")
        comments = await asyncio.to_thread(ingestion.fetch_comments, url)
        result["comments_collected"] = len(comments)
        result["sample_comments"] = [
            (c.get("text") or "")[:120] for c in comments[:5]
        ]
        await _emit(queue, "comments", "done")

        current = "audio"
        await _emit(queue, "audio", "active")
        audio_path = await asyncio.to_thread(ingestion.download_audio, url, job["workdir"])
        result["audio_path"] = audio_path
        await _emit(queue, "audio", "done")

        current = "transcript"
        await _emit(queue, "transcript", "active")
        transcript = await asyncio.to_thread(ingestion.fetch_transcript, url, info)
        result["transcript_source"] = transcript["source"]  # "manual" | "none"
        result["transcript_text"] = transcript.get("text")
        result["transcript_note"] = transcript.get("note")
        await _emit(queue, "transcript", "done")

        current = "compile"
        await _emit(queue, "compile", "active")
        job["result"] = result
        job["status"] = "done"
        await _emit(queue, "compile", "done", extra={"result": result})

    except Exception as e:
        job["status"] = "error"
        await _emit(queue, current, "error", extra={"message": str(e)})
    finally:
        await queue.put(None)  # sentinel: closes the SSE stream


@app.post("/ingest", response_model=IngestResponse)
async def ingest(payload: IngestRequest):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "queue": asyncio.Queue(),
        "status": "running",
        "result": None,
        "workdir": tempfile.mkdtemp(prefix=f"ingest_{job_id}_"),
    }
    asyncio.create_task(run_pipeline(job_id, str(payload.url)))
    return IngestResponse(job_id=job_id)


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    async def event_generator():
        queue: asyncio.Queue = job["queue"]
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {"status": job["status"], "result": job["result"]}


@app.get("/health")
async def health():
    return {"ok": True}