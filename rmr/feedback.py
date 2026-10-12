"""Ratings in the loop: every voice turn gets an id, and a thumbs up / down on it is logged for training.

The server remembers the last ``MAX_TURNS`` turns (what it heard, the answer, the recipe, which responder answered)
in memory. ``POST /api/feedback {"turn", "rating": 1 | -1}`` appends the turn and its rating to
``<dir>/ratings.jsonl``; with ``"save_audio": true`` (the viewer's opt-in) the recording is kept too, as
``<dir>/audio/<turn>.wav``. Nothing is written for turns nobody rated.

With ``repo`` (a private Hugging Face dataset, env ``FEEDBACK_REPO`` on the live Space, whose disk does not
persist), a ``huggingface_hub.CommitScheduler`` pushes the folder every ``every`` minutes. These ratings feed the
reward model and preference tuning (scripts/dpo_from_ratings.py, docs/ROADMAP.md).
"""
import json
import os
import threading
import time
import uuid
from collections import OrderedDict

MAX_TURNS = 500


class Turns:
    """The last ``MAX_TURNS`` voice turns, by id."""

    def __init__(self, max_turns=MAX_TURNS):
        self.max_turns, self.turns, self.lock = max_turns, OrderedDict(), threading.Lock()

    def new(self, session, audio):
        tid = uuid.uuid4().hex[:12]
        with self.lock:
            self.turns[tid] = {"turn": tid, "session": session or "", "time": time.time(), "_audio": audio}
            while len(self.turns) > self.max_turns:
                self.turns.popitem(last=False)
        return tid

    def update(self, tid, **kw):
        with self.lock:
            if tid in self.turns:
                self.turns[tid].update(kw)

    def get(self, tid):
        with self.lock:
            return dict(self.turns[tid]) if tid in self.turns else None


class FeedbackLog:
    def __init__(self, folder, repo=None, every=5):
        self.folder, self.lock = folder, threading.Lock()
        os.makedirs(os.path.join(folder, "audio"), exist_ok=True)
        self.scheduler = None
        if repo:
            from huggingface_hub import CommitScheduler

            self.scheduler = CommitScheduler(repo_id=repo, repo_type="dataset", folder_path=folder, every=every,
                                             private=True, path_in_repo="live")

    def add(self, turn, rating, save_audio=False, note=""):
        audio = turn.pop("_audio", None)
        rec = {**turn, "rating": int(rating), "note": note[:500], "rated_at": time.time(), "audio": None}
        lock = self.scheduler.lock if self.scheduler else self.lock
        with lock:
            if save_audio and audio:
                rec["audio"] = f"audio/{turn['turn']}.wav"
                with open(os.path.join(self.folder, rec["audio"]), "wb") as f:
                    f.write(audio)
            with open(os.path.join(self.folder, "ratings.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec
