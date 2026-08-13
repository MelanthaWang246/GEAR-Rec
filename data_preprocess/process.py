#!/usr/bin/env python3
"""Prepare a filtered MOOCCubeX course-recommendation dataset.

This script mirrors the three stages used by the uploaded MovieLens code:

1. interactions:
   user.json -> item_map.txt, user_map.txt, train.txt, test.txt
2. item-prompts:
   course metadata + heterogeneous relations -> llm_input_item.json
3. user-prompts:
   train.txt + compact course profiles -> llm_input_user.json

The implementation intentionally uses only Python's standard library.  The
interaction stage uses SQLite so that the 770 MB user file does not have to be
loaded into RAM.

Expected input layout under --data-root:

    entities/course.json
    entities/user.json
    entities/concept.json
    entities/school.json
    entities/teacher.json
    relations/concept-course.txt
    relations/course-field.json
    relations/course-school.txt
    relations/course-teacher.txt

All JSON entity files are JSON Lines: one JSON object per line.

python .\process.py all `
  --data-root ".\data" `
  --output-dir ".\mooc" `
  --interaction-source enrollment `
  --min-user-interactions 60 `
  --max-user-interactions 300 `
  --sample-users 12000 `
  --min-user-core 60 `
  --min-item-core 10 `
  --train-ratio 0.8 `
  --seed 2020 `
  --prompt-profile rich `
  --max-concepts 30 `
  --second-order-k 10 `
  --max-history 150
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


DEFAULT_SEED = 2020


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Iterate either JSON Lines or a small top-level JSON array."""
    with path.open("r", encoding="utf-8") as f:
        first = ""
        while True:
            ch = f.read(1)
            if not ch:
                return
            if not ch.isspace():
                first = ch
                break
        f.seek(0)
        if first == "[":
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON array in {path}: {exc}") from exc
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON array in {path}")
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
            return

        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                yield obj


def clean_text(value: Any, max_chars: int | None = None) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars is not None and len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text



def normalize_course_id(value: Any) -> str:
    """Normalize MOOCCubeX course IDs to the ``C_<id>`` representation.

    ``entities/user.json`` stores ``course_order`` entries as bare numeric IDs,
    whereas ``entities/course.json`` and relation files use IDs such as
    ``C_682129``.  Failing to add the prefix silently removes enrollment
    interactions during course validation.
    """
    course_id = clean_text(value)
    if not course_id:
        return ""
    if course_id.startswith("C_"):
        return course_id
    return f"C_{course_id}"


def choose_user_ids(
    candidate_ids: Iterable[str],
    sample_size: int | None,
    seed: int,
) -> list[str]:
    """Choose a reproducible random subset and return it in sorted order."""
    candidates = sorted(set(candidate_ids))
    if sample_size is None or sample_size <= 0 or sample_size >= len(candidates):
        return candidates
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, sample_size))


def apply_iterative_core(
    conn: sqlite3.Connection,
    table: str,
    min_user_core: int,
    min_item_core: int,
    verbose: bool = True,
) -> list[dict[str, int]]:
    """Apply asymmetric user/item core filtering until the graph is stable."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError(f"Unsafe SQLite table name: {table!r}")
    history: list[dict[str, int]] = []
    iteration = 0
    while True:
        iteration += 1
        before = conn.total_changes
        conn.execute(
            f"DELETE FROM {table} WHERE course_id IN ("
            f"SELECT course_id FROM {table} GROUP BY course_id "
            "HAVING COUNT(*) < ?)",
            (min_item_core,),
        )
        deleted_item_interactions = conn.total_changes - before

        before = conn.total_changes
        conn.execute(
            f"DELETE FROM {table} WHERE user_id IN ("
            f"SELECT user_id FROM {table} GROUP BY user_id "
            "HAVING COUNT(*) < ?)",
            (min_user_core,),
        )
        deleted_user_interactions = conn.total_changes - before
        conn.commit()

        remaining = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        row = {
            "iteration": iteration,
            "deleted_item_interactions": deleted_item_interactions,
            "deleted_user_interactions": deleted_user_interactions,
            "remaining_interactions": remaining,
        }
        history.append(row)
        if verbose:
            print(
                f"  iteration {iteration}: removed "
                f"{deleted_item_interactions + deleted_user_interactions:,}, "
                f"remaining {remaining:,}"
            )
        if deleted_item_interactions == 0 and deleted_user_interactions == 0:
            break
    return history


def split_items(
    item_ids: list[int],
    train_ratio: float,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """Shuffle one user's items and leave at least one item in each split."""
    if len(item_ids) < 2:
        raise ValueError("Each retained user must have at least two interactions")
    shuffled = list(item_ids)
    rng.shuffle(shuffled)
    n_train = min(
        len(shuffled) - 1,
        max(1, math.ceil(len(shuffled) * train_ratio)),
    )
    return shuffled[:n_train], shuffled[n_train:]


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def degree_summary(
    conn: sqlite3.Connection,
    table: str,
    group_column: str,
) -> dict[str, float | int]:
    row = conn.execute(
        f"SELECT MIN(degree), MAX(degree), AVG(degree) FROM ("
        f"SELECT COUNT(*) AS degree FROM {table} GROUP BY {group_column})"
    ).fetchone()
    return {
        "min": int(row[0]) if row and row[0] is not None else 0,
        "max": int(row[1]) if row and row[1] is not None else 0,
        "mean": float(row[2]) if row and row[2] is not None else 0.0,
    }


def ordered_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def read_tsv_pairs(path: Path) -> Iterator[tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            left, right = parts[0].strip(), parts[1].strip()
            if left and right:
                yield left, right


def load_item_map(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    orig_to_mapped: dict[str, int] = {}
    mapped_to_orig: dict[int, str] = {}
    with require_file(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad mapping line at {path}:{line_no}")
            orig, mapped_text = parts
            mapped = int(mapped_text)
            orig_to_mapped[orig] = mapped
            mapped_to_orig[mapped] = orig
    return orig_to_mapped, mapped_to_orig


def collect_semantic_course_ids(data_root: Path) -> tuple[set[str], set[str]]:
    """Return (all course ids, course ids with at least one semantic/KG edge)."""
    course_file = require_file(data_root / "entities" / "course.json")
    all_courses: set[str] = set()
    semantic_courses: set[str] = set()

    for obj in iter_jsonl(course_file):
        course_id = clean_text(obj.get("id"))
        if not course_id:
            continue
        all_courses.add(course_id)
        if obj.get("field"):
            semantic_courses.add(course_id)

    optional_relation_files = [
        (data_root / "relations" / "concept-course.txt", "tsv_right"),
        (data_root / "relations" / "course-school.txt", "tsv_left"),
        (data_root / "relations" / "course-teacher.txt", "tsv_left"),
        (data_root / "relations" / "course-field.json", "json_course"),
    ]

    for path, kind in optional_relation_files:
        if not path.is_file():
            continue
        if kind == "tsv_right":
            for _, course_id in read_tsv_pairs(path):
                if course_id in all_courses:
                    semantic_courses.add(course_id)
        elif kind == "tsv_left":
            for course_id, _ in read_tsv_pairs(path):
                if course_id in all_courses:
                    semantic_courses.add(course_id)
        else:
            for obj in iter_jsonl(path):
                course_id = clean_text(obj.get("course_id"))
                if course_id in all_courses and obj.get("field"):
                    semantic_courses.add(course_id)

    return all_courses, semantic_courses



def build_video_course_maps(
    data_root: Path,
    valid_courses: set[str],
) -> dict[str, Any]:
    """Build direct and canonical mappings from watched videos to courses.

    MOOCCubeX may assign several ``video_id`` values to the same canonical
    content id (``ccid``), for example across different course offerings.  A
    direct ``video_id -> course`` lookup therefore misses historical or
    alternate video IDs that do not appear in the downloaded ``course.json``.

    Resolution policy:
      1. Prefer the exact video ID from ``course.json``.
      2. Otherwise map video_id -> ccid -> course only when that ccid belongs
         to exactly one eligible course.
      3. Leave ambiguous ccids unresolved rather than silently assigning the
         interaction to the wrong course.
    """
    course_file = require_file(data_root / "entities" / "course.json")
    direct: dict[str, str] = {}
    course_videos: list[tuple[str, str]] = []

    for course in iter_jsonl(course_file):
        course_id = clean_text(course.get("id"))
        if course_id not in valid_courses:
            continue
        resources = course.get("resource") or []
        if not isinstance(resources, list):
            continue
        for resource in resources:
            if not isinstance(resource, dict):
                continue
            video_id = clean_text(resource.get("resource_id"))
            resource_type = clean_text(resource.get("resource_type")).lower()
            if video_id and (resource_type == "video" or video_id.startswith("V_")):
                direct[video_id] = course_id
                course_videos.append((video_id, course_id))

    mapping_path = data_root / "relations" / "video_id-ccid.txt"
    video_to_ccid: dict[str, str] = {}
    if mapping_path.is_file():
        for video_id, ccid in read_tsv_pairs(mapping_path):
            video_to_ccid[video_id] = ccid

    ccid_to_courses: defaultdict[str, set[str]] = defaultdict(set)
    if video_to_ccid:
        for video_id, course_id in course_videos:
            ccid = video_to_ccid.get(video_id)
            if ccid:
                ccid_to_courses[ccid].add(course_id)

    unique_ccid_to_course = {
        ccid: next(iter(courses))
        for ccid, courses in ccid_to_courses.items()
        if len(courses) == 1
    }
    ambiguous_ccids = {
        ccid for ccid, courses in ccid_to_courses.items() if len(courses) > 1
    }

    return {
        "direct": direct,
        "video_to_ccid": video_to_ccid,
        "unique_ccid_to_course": unique_ccid_to_course,
        "ambiguous_ccids": ambiguous_ccids,
    }


def resolve_video_course(
    video_id: str,
    maps: dict[str, Any],
) -> tuple[str | None, str]:
    """Resolve one video ID and return ``(course_id, resolution_method)``."""
    direct = maps["direct"]
    if video_id in direct:
        return direct[video_id], "direct"

    video_to_ccid = maps["video_to_ccid"]
    ccid = video_to_ccid.get(video_id)
    if not ccid:
        return None, "unknown_video"

    unique_ccid_to_course = maps["unique_ccid_to_course"]
    if ccid in unique_ccid_to_course:
        return unique_ccid_to_course[ccid], "ccid"

    if ccid in maps["ambiguous_ccids"]:
        return None, "ambiguous_ccid"
    return None, "ccid_without_course"


def configure_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-524288")  # approximately 512 MiB when available


def flush_interaction_batch(
    conn: sqlite3.Connection,
    batch: list[tuple[str, str, int, int]],
    table: str = "raw_interactions",
) -> None:
    if not batch:
        return
    conn.executemany(
        f"INSERT OR IGNORE INTO {table}(user_id, course_id, user_seq, pos) "
        "VALUES (?, ?, ?, ?)",
        batch,
    )
    batch.clear()

def build_interactions(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_file = data_root / "entities" / "user.json"
    user_video_file = data_root / "relations" / "user-video.json"
    if args.interaction_source == "enrollment":
        require_file(user_file)
    else:
        require_file(user_video_file)

    all_courses, semantic_courses = collect_semantic_course_ids(data_root)
    valid_source_courses = semantic_courses if args.require_kg else all_courses
    if not valid_source_courses:
        raise RuntimeError("No valid courses found. Check course.json and relation files.")

    print(f"Courses in course.json: {len(all_courses):,}")
    print(f"Courses with semantic/KG information: {len(semantic_courses):,}")
    print(f"Courses eligible before filtering: {len(valid_source_courses):,}")

    db_path = Path(args.sqlite_path) if args.sqlite_path else output_dir / "interactions.sqlite"
    if db_path.exists() and not args.reuse_raw_sqlite:
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    configure_sqlite(conn)

    if not args.reuse_raw_sqlite:
        conn.execute("DROP TABLE IF EXISTS interactions")
        conn.execute("DROP TABLE IF EXISTS selected_users")
        conn.execute("DROP TABLE IF EXISTS raw_interactions")
        conn.execute(
            "CREATE TABLE raw_interactions ("
            "user_id TEXT NOT NULL, course_id TEXT NOT NULL, "
            "user_seq INTEGER NOT NULL, pos INTEGER NOT NULL, "
            "PRIMARY KEY(user_id, course_id)) WITHOUT ROWID"
        )

        batch: list[tuple[str, str, int, int]] = []
        read_users = 0
        kept_before_filter = 0

        if args.interaction_source == "enrollment":
            for user_seq, obj in enumerate(iter_jsonl(user_file)):
                user_id = clean_text(obj.get("id"))
                course_order = obj.get("course_order") or []
                if not user_id or not isinstance(course_order, list):
                    continue
                read_users += 1
                seen: set[str] = set()
                for pos, raw_course_id in enumerate(course_order):
                    course_id = normalize_course_id(raw_course_id)
                    if (
                        not course_id
                        or course_id in seen
                        or course_id not in valid_source_courses
                    ):
                        continue
                    seen.add(course_id)
                    batch.append((user_id, course_id, user_seq, pos))
                    kept_before_filter += 1
                    if len(batch) >= args.sqlite_batch_size:
                        flush_interaction_batch(conn, batch)
                if read_users % 100_000 == 0:
                    flush_interaction_batch(conn, batch)
                    conn.commit()
                    print(f"Read {read_users:,} users ...")
        else:
            video_maps = build_video_course_maps(data_root, valid_source_courses)
            print(
                "Direct video IDs mapped to eligible courses: "
                f"{len(video_maps['direct']):,}"
            )
            if video_maps["video_to_ccid"]:
                print(
                    "video_id-ccid mappings loaded: "
                    f"{len(video_maps['video_to_ccid']):,}"
                )
                print(
                    "Canonical ccids uniquely mapped to a course: "
                    f"{len(video_maps['unique_ccid_to_course']):,}"
                )
            else:
                eprint(
                    "WARNING: video_id-ccid.txt was not found; video mode "
                    "will use exact video IDs only."
                )

            total_video_events = 0
            resolution_counts: defaultdict[str, int] = defaultdict(int)
            users_with_mapped_video = 0
            unmatched_examples: list[str] = []

            for user_seq, obj in enumerate(iter_jsonl(user_video_file)):
                user_id = clean_text(obj.get("user_id"))
                seq = obj.get("seq") or []
                if not user_id or not isinstance(seq, list):
                    continue
                read_users += 1
                course_videos: defaultdict[str, set[str]] = defaultdict(set)
                first_pos: dict[str, int] = {}
                for pos, event in enumerate(seq):
                    if not isinstance(event, dict):
                        continue
                    video_id = clean_text(event.get("video_id"))
                    if not video_id:
                        continue
                    total_video_events += 1
                    course_id, method = resolve_video_course(video_id, video_maps)
                    resolution_counts[method] += 1
                    if not course_id:
                        if len(unmatched_examples) < 5:
                            unmatched_examples.append(video_id)
                        continue
                    course_videos[course_id].add(video_id)
                    first_pos.setdefault(course_id, pos)
                if course_videos:
                    users_with_mapped_video += 1
                for course_id, videos in course_videos.items():
                    if len(videos) < args.min_videos_per_course:
                        continue
                    batch.append((user_id, course_id, user_seq, first_pos[course_id]))
                    kept_before_filter += 1
                    if len(batch) >= args.sqlite_batch_size:
                        flush_interaction_batch(conn, batch)
                if read_users % 100_000 == 0:
                    flush_interaction_batch(conn, batch)
                    conn.commit()
                    print(f"Read {read_users:,} user-video records ...")

            mapped_events = resolution_counts["direct"] + resolution_counts["ccid"]
            match_rate = (
                100.0 * mapped_events / total_video_events
                if total_video_events else 0.0
            )
            print(f"Video events read: {total_video_events:,}")
            print(f"  exact video-id matches: {resolution_counts['direct']:,}")
            print(f"  ccid fallback matches: {resolution_counts['ccid']:,}")
            print(f"  ambiguous ccid events: {resolution_counts['ambiguous_ccid']:,}")
            print(f"Video-event-to-course match rate: {match_rate:.2f}%")
            print(f"Users with at least one mapped video: {users_with_mapped_video:,}")
            if unmatched_examples:
                print("Example unresolved video IDs: " + ", ".join(unmatched_examples))

        flush_interaction_batch(conn, batch)
        conn.commit()
        print(f"Users read: {read_users:,}")
        print(f"Eligible deduplicated interactions inserted: {kept_before_filter:,}")
        conn.execute(
            "CREATE INDEX idx_raw_course ON raw_interactions(course_id)"
        )
        conn.execute(
            "CREATE INDEX idx_raw_user_seq ON raw_interactions(user_seq, pos)"
        )
        conn.commit()
    elif not table_exists(conn, "raw_interactions"):
        raise RuntimeError(
            "--reuse-raw-sqlite requires a raw_interactions table created by "
            "this script. An old already-filtered interactions table is not safe to reuse."
        )

    raw_interactions = conn.execute(
        "SELECT COUNT(*) FROM raw_interactions"
    ).fetchone()[0]
    raw_users = conn.execute(
        "SELECT COUNT(DISTINCT user_id) FROM raw_interactions"
    ).fetchone()[0]
    raw_items = conn.execute(
        "SELECT COUNT(DISTINCT course_id) FROM raw_interactions"
    ).fetchone()[0]
    if raw_interactions == 0:
        raise RuntimeError("No valid interactions were extracted.")
    print(
        f"Valid raw graph: {raw_users:,} users, {raw_items:,} courses, "
        f"{raw_interactions:,} interactions"
    )

    max_degree = args.max_user_interactions
    if max_degree is None:
        candidate_query = (
            "SELECT user_id FROM raw_interactions GROUP BY user_id "
            "HAVING COUNT(*) >= ? ORDER BY user_id"
        )
        candidate_params: tuple[int, ...] = (args.min_user_interactions,)
    else:
        candidate_query = (
            "SELECT user_id FROM raw_interactions GROUP BY user_id "
            "HAVING COUNT(*) BETWEEN ? AND ? ORDER BY user_id"
        )
        candidate_params = (args.min_user_interactions, max_degree)

    candidate_ids = [
        row[0] for row in conn.execute(candidate_query, candidate_params)
    ]
    if not candidate_ids:
        raise RuntimeError(
            "No users satisfy the candidate activity range. "
            "Lower --min-user-interactions or increase --max-user-interactions."
        )
    selected_ids = choose_user_ids(candidate_ids, args.sample_users, args.seed)
    print(
        f"Candidate users in activity range: {len(candidate_ids):,}; "
        f"selected before core: {len(selected_ids):,}"
    )

    conn.execute("DROP TABLE IF EXISTS selected_users")
    conn.execute(
        "CREATE TABLE selected_users (user_id TEXT PRIMARY KEY) WITHOUT ROWID"
    )
    conn.executemany(
        "INSERT INTO selected_users(user_id) VALUES (?)",
        ((user_id,) for user_id in selected_ids),
    )

    conn.execute("DROP TABLE IF EXISTS interactions")
    conn.execute(
        "CREATE TABLE interactions ("
        "user_id TEXT NOT NULL, course_id TEXT NOT NULL, "
        "user_seq INTEGER NOT NULL, pos INTEGER NOT NULL, "
        "PRIMARY KEY(user_id, course_id)) WITHOUT ROWID"
    )
    conn.execute(
        "INSERT INTO interactions(user_id, course_id, user_seq, pos) "
        "SELECT r.user_id, r.course_id, r.user_seq, r.pos "
        "FROM raw_interactions r JOIN selected_users s ON r.user_id=s.user_id"
    )
    conn.execute("CREATE INDEX idx_interactions_course ON interactions(course_id)")
    conn.execute(
        "CREATE INDEX idx_interactions_user_seq ON interactions(user_seq, pos)"
    )
    conn.commit()

    before_core_interactions = conn.execute(
        "SELECT COUNT(*) FROM interactions"
    ).fetchone()[0]
    before_core_items = conn.execute(
        "SELECT COUNT(DISTINCT course_id) FROM interactions"
    ).fetchone()[0]
    print(
        f"Sampled graph before core: {len(selected_ids):,} users, "
        f"{before_core_items:,} courses, {before_core_interactions:,} interactions"
    )
    print(
        f"Applying iterative user-core={args.min_user_core}, "
        f"item-core={args.min_item_core} filtering ..."
    )
    core_history = apply_iterative_core(
        conn,
        table="interactions",
        min_user_core=args.min_user_core,
        min_item_core=args.min_item_core,
    )

    n_users = conn.execute(
        "SELECT COUNT(DISTINCT user_id) FROM interactions"
    ).fetchone()[0]
    n_items = conn.execute(
        "SELECT COUNT(DISTINCT course_id) FROM interactions"
    ).fetchone()[0]
    n_interactions = conn.execute(
        "SELECT COUNT(*) FROM interactions"
    ).fetchone()[0]
    if n_users == 0 or n_items == 0:
        raise RuntimeError("Core filtering removed all data. Relax the thresholds.")
    if args.min_user_core < 2:
        raise RuntimeError("At least two interactions per user are required for splitting.")

    print(
        f"After filtering: {n_users:,} users, {n_items:,} courses, "
        f"{n_interactions:,} interactions"
    )

    item_map: dict[str, int] = {}
    item_map_path = output_dir / "item_map.txt"
    with item_map_path.open("w", encoding="utf-8") as f:
        query = (
            "SELECT course_id, MIN(user_seq) AS first_user, MIN(pos) AS first_pos "
            "FROM interactions GROUP BY course_id "
            "ORDER BY first_user, first_pos, course_id"
        )
        for mapped_id, (course_id, _, _) in enumerate(conn.execute(query)):
            item_map[course_id] = mapped_id
            f.write(f"{course_id} {mapped_id}\n")

    conn.execute("DROP TABLE IF EXISTS user_mapping")
    conn.execute(
        "CREATE TABLE user_mapping ("
        "user_id TEXT PRIMARY KEY, mapped_id INTEGER UNIQUE NOT NULL"
        ") WITHOUT ROWID"
    )
    user_map_path = output_dir / "user_map.txt"
    with user_map_path.open("w", encoding="utf-8") as f:
        query = (
            "SELECT user_id, MIN(user_seq) AS first_seq FROM interactions "
            "GROUP BY user_id ORDER BY first_seq, user_id"
        )
        mapping_batch: list[tuple[str, int]] = []
        for mapped_id, (user_id, _) in enumerate(conn.execute(query)):
            f.write(f"{user_id} {mapped_id}\n")
            mapping_batch.append((user_id, mapped_id))
            if len(mapping_batch) >= args.sqlite_batch_size:
                conn.executemany(
                    "INSERT INTO user_mapping(user_id, mapped_id) VALUES (?, ?)",
                    mapping_batch,
                )
                mapping_batch.clear()
        if mapping_batch:
            conn.executemany(
                "INSERT INTO user_mapping(user_id, mapped_id) VALUES (?, ?)",
                mapping_batch,
            )
    conn.commit()

    train_path = output_dir / "train.txt"
    test_path = output_dir / "test.txt"
    rng = random.Random(args.seed)
    train_count = 0
    test_count = 0
    query = (
        "SELECT u.mapped_id, i.course_id FROM interactions i "
        "JOIN user_mapping u ON i.user_id=u.user_id "
        "ORDER BY u.mapped_id, i.pos, i.course_id"
    )

    def write_user(
        train_f: TextIO,
        test_f: TextIO,
        mapped_user: int,
        original_courses: list[str],
    ) -> tuple[int, int]:
        mapped_items = [item_map[c] for c in original_courses]
        train_items, test_items = split_items(mapped_items, args.train_ratio, rng)
        train_f.write(f"{mapped_user} " + " ".join(map(str, train_items)) + "\n")
        test_f.write(f"{mapped_user} " + " ".join(map(str, test_items)) + "\n")
        return len(train_items), len(test_items)

    with train_path.open("w", encoding="utf-8") as train_f, test_path.open(
        "w", encoding="utf-8"
    ) as test_f:
        current_user: int | None = None
        current_courses: list[str] = []
        for mapped_user, course_id in conn.execute(query):
            if current_user is None:
                current_user = mapped_user
            if mapped_user != current_user:
                tr, te = write_user(train_f, test_f, current_user, current_courses)
                train_count += tr
                test_count += te
                current_user = mapped_user
                current_courses = []
            current_courses.append(course_id)
        if current_user is not None:
            tr, te = write_user(train_f, test_f, current_user, current_courses)
            train_count += tr
            test_count += te

    user_degree = degree_summary(conn, "interactions", "user_id")
    item_degree = degree_summary(conn, "interactions", "course_id")
    density = n_interactions / (n_users * n_items)
    stats = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "interaction_source": args.interaction_source,
        "require_kg": args.require_kg,
        "candidate_user_interactions": {
            "min": args.min_user_interactions,
            "max": args.max_user_interactions,
        },
        "candidate_users": len(candidate_ids),
        "requested_sample_users": args.sample_users,
        "sampled_users_before_core": len(selected_ids),
        "min_user_core": args.min_user_core,
        "min_item_core": args.min_item_core,
        "raw_users": raw_users,
        "raw_items": raw_items,
        "raw_interactions": raw_interactions,
        "sampled_items_before_core": before_core_items,
        "sampled_interactions_before_core": before_core_interactions,
        "users": n_users,
        "items": n_items,
        "interactions": n_interactions,
        "train_interactions": train_count,
        "test_interactions": test_count,
        "density": density,
        "sparsity": 1.0 - density,
        "user_degree": user_degree,
        "item_degree": item_degree,
        "core_history": core_history,
        "min_videos_per_course": (
            args.min_videos_per_course if args.interaction_source == "video" else None
        ),
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    conn.close()
    print(f"Wrote mappings and splits to: {output_dir}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

def load_entity_names(path: Path, wanted_ids: set[str]) -> dict[str, str]:
    names: dict[str, str] = {}
    if not wanted_ids:
        return names
    for obj in iter_jsonl(require_file(path)):
        entity_id = clean_text(obj.get("id"))
        if entity_id not in wanted_ids:
            continue
        name = clean_text(obj.get("name")) or clean_text(obj.get("name_en")) or entity_id
        names[entity_id] = name
    return names


def build_course_profiles(
    data_root: Path,
    output_dir: Path,
    max_concepts: int,
    include_concept_context: bool,
) -> dict[int, dict[str, Any]]:
    orig_to_mapped, _ = load_item_map(output_dir / "item_map.txt")
    retained = set(orig_to_mapped)

    profiles_by_orig: dict[str, dict[str, Any]] = {}
    for obj in iter_jsonl(require_file(data_root / "entities" / "course.json")):
        course_id = clean_text(obj.get("id"))
        if course_id not in retained:
            continue
        raw_fields = obj.get("field") or []
        if not isinstance(raw_fields, list):
            raw_fields = [raw_fields]
        profiles_by_orig[course_id] = {
            "original_id": course_id,
            "mapped_id": orig_to_mapped[course_id],
            "name": clean_text(obj.get("name")) or course_id,
            "about": clean_text(obj.get("about"), 1200),
            "prerequisites": clean_text(obj.get("prerequisites"), 600),
            "fields": ordered_unique(raw_fields),
            "school_ids": [],
            "teacher_ids": [],
            "concept_ids": set(),
        }

    missing_courses = retained - set(profiles_by_orig)
    if missing_courses:
        raise RuntimeError(
            f"{len(missing_courses)} mapped courses are absent from course.json; "
            f"first examples: {sorted(missing_courses)[:5]}"
        )

    course_field_path = data_root / "relations" / "course-field.json"
    if course_field_path.is_file():
        for obj in iter_jsonl(course_field_path):
            course_id = clean_text(obj.get("course_id"))
            if course_id not in profiles_by_orig:
                continue
            values = obj.get("field") or []
            if not isinstance(values, list):
                values = [values]
            profiles_by_orig[course_id]["fields"] = ordered_unique(
                profiles_by_orig[course_id]["fields"] + list(values)
            )

    school_ids: set[str] = set()
    course_school_path = require_file(data_root / "relations" / "course-school.txt")
    for course_id, school_id in read_tsv_pairs(course_school_path):
        if course_id in profiles_by_orig:
            profiles_by_orig[course_id]["school_ids"].append(school_id)
            school_ids.add(school_id)

    teacher_ids: set[str] = set()
    course_teacher_path = require_file(data_root / "relations" / "course-teacher.txt")
    for course_id, teacher_id in read_tsv_pairs(course_teacher_path):
        if course_id in profiles_by_orig:
            profiles_by_orig[course_id]["teacher_ids"].append(teacher_id)
            teacher_ids.add(teacher_id)

    concept_course_path = require_file(data_root / "relations" / "concept-course.txt")
    concept_ids: set[str] = set()
    for concept_id, course_id in read_tsv_pairs(concept_course_path):
        if course_id in profiles_by_orig:
            profiles_by_orig[course_id]["concept_ids"].add(concept_id)
            concept_ids.add(concept_id)

    school_names = load_entity_names(
        data_root / "entities" / "school.json", school_ids
    )
    teacher_names = load_entity_names(
        data_root / "entities" / "teacher.json", teacher_ids
    )

    concept_names: dict[str, str] = {}
    concept_file = require_file(data_root / "entities" / "concept.json")
    for obj in iter_jsonl(concept_file):
        concept_id = clean_text(obj.get("id"))
        if concept_id in concept_ids:
            concept_names[concept_id] = clean_text(obj.get("name")) or concept_id

    concept_df: defaultdict[str, int] = defaultdict(int)
    for profile in profiles_by_orig.values():
        for concept_id in profile["concept_ids"]:
            concept_df[concept_id] += 1

    n_courses = len(profiles_by_orig)
    selected_concepts: set[str] = set()
    for profile in profiles_by_orig.values():
        concept_list = list(profile["concept_ids"])
        concept_list.sort(
            key=lambda cid: (
                -math.log((n_courses + 1) / (concept_df[cid] + 1)),
                concept_names.get(cid, cid),
            )
        )
        # The key above sorts ascending on the negative IDF, equivalent to
        # descending IDF. Rare, discriminative concepts are retained first.
        profile["key_concept_ids"] = concept_list[:max_concepts]
        selected_concepts.update(profile["key_concept_ids"])

    concept_context: dict[str, str] = {}
    if include_concept_context and selected_concepts:
        for obj in iter_jsonl(concept_file):
            concept_id = clean_text(obj.get("id"))
            if concept_id in selected_concepts:
                concept_context[concept_id] = clean_text(obj.get("context"), 160)

    profiles: dict[int, dict[str, Any]] = {}
    for course_id, profile in profiles_by_orig.items():
        mapped_id = int(profile["mapped_id"])
        key_ids: list[str] = profile.pop("key_concept_ids")
        profile.pop("concept_ids")
        profile["school_ids"] = ordered_unique(profile["school_ids"])
        profile["teacher_ids"] = ordered_unique(profile["teacher_ids"])
        profile["schools"] = [
            school_names.get(entity_id, entity_id) for entity_id in profile["school_ids"]
        ]
        profile["teachers"] = [
            teacher_names.get(entity_id, entity_id) for entity_id in profile["teacher_ids"]
        ]
        profile["key_concept_ids"] = key_ids
        profile["key_concepts"] = [concept_names.get(cid, cid) for cid in key_ids]
        if include_concept_context:
            profile["key_concept_contexts"] = [
                concept_context[cid] for cid in key_ids if concept_context.get(cid)
            ][:5]
        profiles[mapped_id] = profile

    profile_path = output_dir / "course_profiles.jsonl"
    with profile_path.open("w", encoding="utf-8") as f:
        for mapped_id in sorted(profiles):
            f.write(json.dumps(profiles[mapped_id], ensure_ascii=False) + "\n")

    # Human-readable analogue of ml1m_extended_movie.csv.  The JSONL file is
    # used by later stages; this CSV is convenient for inspection/ablation.
    csv_path = output_dir / "mooccubex_extended_course.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "original_id", "mapped_id", "name", "about", "prerequisites",
                "fields", "schools", "teachers", "key_concepts",
            ],
        )
        writer.writeheader()
        for mapped_id in sorted(profiles):
            profile = profiles[mapped_id]
            writer.writerow({
                "original_id": profile["original_id"],
                "mapped_id": mapped_id,
                "name": profile["name"],
                "about": profile["about"],
                "prerequisites": profile["prerequisites"],
                "fields": "|".join(profile["fields"]),
                "schools": "|".join(profile["schools"]),
                "teachers": "|".join(profile["teachers"]),
                "key_concepts": "|".join(profile["key_concepts"]),
            })
    print(f"Wrote {len(profiles)} course profiles to {profile_path}")
    print(f"Wrote merged course metadata to {csv_path}")
    return profiles


def sample_related(
    candidates: set[int],
    current: int,
    limit: int,
    seed: int,
) -> list[int]:
    values = sorted(candidates - {current})
    if len(values) <= limit:
        return values
    rng = random.Random(seed)
    return sorted(rng.sample(values, limit))


def join_nonempty(values: Iterable[str], fallback: str = "未知") -> str:
    cleaned = [clean_text(v) for v in values]
    cleaned = [v for v in cleaned if v]
    return "、".join(cleaned) if cleaned else fallback


def generate_item_prompts(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles = build_course_profiles(
        data_root=data_root,
        output_dir=output_dir,
        max_concepts=args.max_concepts,
        include_concept_context=args.include_concept_context,
    )

    by_field: defaultdict[str, set[int]] = defaultdict(set)
    by_school: defaultdict[str, set[int]] = defaultdict(set)
    by_teacher: defaultdict[str, set[int]] = defaultdict(set)
    by_concept: defaultdict[str, set[int]] = defaultdict(set)

    for mapped_id, profile in profiles.items():
        for field in profile["fields"]:
            by_field[field].add(mapped_id)
        for school_id in profile["school_ids"]:
            by_school[school_id].add(mapped_id)
        for teacher_id in profile["teacher_ids"]:
            by_teacher[teacher_id].add(mapped_id)
        for concept_id in profile["key_concept_ids"]:
            by_concept[concept_id].add(mapped_id)

    prompts: dict[int, str] = {}
    for mapped_id in sorted(profiles):
        profile = profiles[mapped_id]
        name = profile["name"]

        same_field: set[int] = set()
        for value in profile["fields"]:
            same_field.update(by_field[value])
        same_teacher: set[int] = set()
        for value in profile["teacher_ids"]:
            same_teacher.update(by_teacher[value])
        same_school: set[int] = set()
        for value in profile["school_ids"]:
            same_school.update(by_school[value])
        shared_concept: set[int] = set()
        for value in profile["key_concept_ids"]:
            shared_concept.update(by_concept[value])

        relation_groups = {
            "同领域课程": sample_related(
                same_field,
                mapped_id,
                args.second_order_k,
                args.seed + mapped_id * 11 + 1,
            ),
            "同教师课程": sample_related(
                same_teacher,
                mapped_id,
                args.second_order_k,
                args.seed + mapped_id * 11 + 2,
            ),
            "同学校课程": sample_related(
                same_school,
                mapped_id,
                args.second_order_k,
                args.seed + mapped_id * 11 + 3,
            ),
            "共享核心概念的课程": sample_related(
                shared_concept,
                mapped_id,
                args.second_order_k,
                args.seed + mapped_id * 11 + 4,
            ),
        }

        second_order_lines: list[str] = []
        related_by_type: dict[str, list[int]] = {}
        for label, related_ids in relation_groups.items():
            if not related_ids:
                continue
            related_by_type[label] = related_ids
            names = [profiles[item_id]["name"] for item_id in related_ids]
            second_order_lines.append(f"- {label}：{join_nonempty(names)}")
        profile["related_courses"] = related_by_type

        first_order_lines = [f"课程名称：{name}"]
        if args.prompt_profile == "rich":
            first_order_lines.append(f"课程简介：{profile['about'] or '未提供'}")
        first_order_lines.extend([
            f"所属领域：{join_nonempty(profile['fields'])}",
            f"开课学校：{join_nonempty(profile['schools'])}",
            f"授课教师：{join_nonempty(profile['teachers'])}",
        ])
        if args.prompt_profile == "rich":
            first_order_lines.append(
                f"先修知识：{profile['prerequisites'] or '未提供'}"
            )
        first_order_lines.append(
            f"代表性核心概念：{join_nonempty(profile['key_concepts'])}"
        )
        contexts = profile.get("key_concept_contexts") or []
        if contexts:
            first_order_lines.append(
                "部分概念上下文：" + "；".join(contexts)
            )

        second_order_text = (
            "\n".join(second_order_lines) if second_order_lines else "- 暂无可用的二阶关联课程"
        )
        if args.prompt_profile == "rich":
            output_focus = (
                "课程主题、所属领域、核心知识、先修基础、可能培养的能力，"
                "以及适合的学习者类型"
            )
        else:
            output_focus = (
                "课程主题、所属领域、核心知识、教师或学校特征，"
                "以及适合的学习者类型"
            )
        prompt = f"""假设你是一名在线课程推荐专家。现在给定课程《{name}》。

该课程在知识图谱中的一阶信息如下：
{chr(10).join('- ' + line for line in first_order_lines)}

从知识图谱采样得到的二阶关联信息如下：
{second_order_text}

请校正并补全上述知识，用一段连贯的中文描述概括这门课程。描述应包含{output_focus}。不得虚构具体教师经历、考核安排或未给出的课程事实，长度不超过300个汉字。"""
        prompts[mapped_id] = prompt

    # Rewrite profiles with the sampled second-order relations included.
    profile_path = output_dir / "course_profiles.jsonl"
    with profile_path.open("w", encoding="utf-8") as f:
        for mapped_id in sorted(profiles):
            f.write(json.dumps(profiles[mapped_id], ensure_ascii=False) + "\n")

    output_path = output_dir / "llm_input_item.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False)
    print(f"Wrote {len(prompts)} item prompts to {output_path}")


def load_profiles(path: Path) -> dict[int, dict[str, Any]]:
    profiles: dict[int, dict[str, Any]] = {}
    for obj in iter_jsonl(require_file(path)):
        mapped_id = int(obj["mapped_id"])
        profiles[mapped_id] = obj
    return profiles


def compact_user_course_text(
    profile: dict[str, Any],
    max_user_concepts: int,
    prompt_profile: str,
) -> str:
    concepts = profile.get("key_concepts") or []
    concepts = concepts[:max_user_concepts]
    fields = (profile.get("fields") or [])[:4]
    teachers = (profile.get("teachers") or [])[:3]
    schools = (profile.get("schools") or [])[:2]
    parts = [
        f"领域：{join_nonempty(fields)}",
        f"教师：{join_nonempty(teachers)}",
        f"学校：{join_nonempty(schools)}",
        f"核心概念：{join_nonempty(concepts)}",
    ]
    if prompt_profile == "rich":
        prerequisites = clean_text(profile.get("prerequisites"), 120) or "未提供"
        parts.append(f"先修知识：{prerequisites}")
    return f"课程《{profile.get('name', '未知课程')}》：{{" + "；".join(parts) + "}"


def write_json_entry(
    f: TextIO,
    key: int,
    value: str,
    first: bool,
) -> bool:
    if not first:
        f.write(",")
    f.write(json.dumps(str(key), ensure_ascii=False))
    f.write(":")
    f.write(json.dumps(value, ensure_ascii=False))
    return False


def generate_user_prompts(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    profiles = load_profiles(output_dir / "course_profiles.jsonl")
    train_path = require_file(output_dir / "train.txt")
    output_path = output_dir / "llm_input_user.json"

    processed = 0
    skipped_items = 0
    with train_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        dst.write("{")
        first = True
        for line_no, line in enumerate(src, 1):
            parts = line.split()
            if len(parts) < 2:
                continue
            user_id = int(parts[0])
            item_ids = [int(x) for x in parts[1:]]

            # Same principle as the uploaded MovieLens user notebook: use only
            # train.txt and cap long histories at 150 items.  A per-user RNG
            # makes the result reproducible and independent of file chunking.
            if len(item_ids) > args.max_history:
                rng = random.Random(args.seed + user_id)
                item_ids = rng.sample(item_ids, args.max_history)

            course_texts: list[str] = []
            for item_id in item_ids:
                profile = profiles.get(item_id)
                if profile is None:
                    skipped_items += 1
                    continue
                course_texts.append(
                    compact_user_course_text(
                        profile, args.max_user_concepts, args.prompt_profile
                    )
                )
            if not course_texts:
                continue

            history_text = "；\n".join(course_texts)
            if args.prompt_profile == "rich":
                record_fields = "课程名称、领域、教师、学校、核心概念和先修知识"
                preference_aspects = "学习领域、核心知识主题、课程难度或先修基础、教师/学校偏好"
            else:
                record_fields = "课程名称、领域、教师、学校和核心概念"
                preference_aspects = "学习领域、核心知识主题、教师/学校偏好"
            prompt = f"""假设你是一名在线教育课程推荐专家。下面给出某位学习者在训练集中的选课历史，每条记录包含{record_fields}：

{history_text}

请从{preference_aspects}等方面，总结该学习者的课程偏好和可能适合继续学习的课程类型。不要提及测试集或推荐具体课程编号。请输出一段连贯中文，不超过180个汉字。"""
            first = write_json_entry(dst, user_id, prompt, first)
            processed += 1
            if args.max_users is not None and processed >= args.max_users:
                break
            if processed % 100_000 == 0:
                print(f"Generated {processed:,} user prompts ...")
        dst.write("}")

    print(f"Wrote {processed:,} user prompts to {output_path}")
    if skipped_items:
        eprint(f"Warning: skipped {skipped_items:,} history items with no profile")


def run_all(args: argparse.Namespace) -> None:
    build_interactions(args)
    generate_item_prompts(args)
    generate_user_prompts(args)


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        required=True,
        help="MOOCCubeX root containing entities/ and relations/",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for maps, splits, profiles, and LLM input JSON files",
    )


def add_interaction_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=60,
        help="Candidate users must have at least this many valid distinct courses",
    )
    parser.add_argument(
        "--max-user-interactions",
        type=int,
        default=300,
        help="Candidate users may have at most this many valid distinct courses",
    )
    parser.add_argument(
        "--sample-users",
        type=int,
        default=12_000,
        help="Randomly select this many candidate users; 0 keeps all candidates",
    )
    parser.add_argument("--min-user-core", type=int, default=60)
    parser.add_argument("--min-item-core", type=int, default=10)
    parser.add_argument(
        "--min-core",
        type=int,
        default=None,
        help="Backward-compatible alias that overrides both core thresholds",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--interaction-source",
        choices=["enrollment", "video"],
        default="enrollment",
        help="Use course_order registrations or actual watched-video courses",
    )
    parser.add_argument(
        "--min-videos-per-course",
        type=int,
        default=1,
        help="For video mode, distinct watched videos required per course",
    )
    parser.add_argument(
        "--require-kg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only courses with at least one field/teacher/school/concept edge",
    )
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument(
        "--reuse-raw-sqlite",
        "--reuse-sqlite",
        dest="reuse_raw_sqlite",
        action="store_true",
        help="Reuse the immutable raw_interactions table created by this script",
    )
    parser.add_argument("--sqlite-batch-size", type=int, default=20_000)

def add_item_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-concepts", type=int, default=30)
    parser.add_argument("--second-order-k", type=int, default=10)
    parser.add_argument(
        "--prompt-profile",
        choices=["rich", "structure"],
        default="rich",
        help="rich adds course introduction/prerequisites; structure mirrors only KG attributes",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--include-concept-context",
        action="store_true",
        help="Add short contexts for up to five selected concepts",
    )


def add_user_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-history", type=int, default=150)
    parser.add_argument("--max-user-concepts", type=int, default=8)
    parser.add_argument(
        "--prompt-profile",
        choices=["rich", "structure"],
        default="rich",
    )
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_interactions = subparsers.add_parser("interactions")
    add_common_paths(p_interactions)
    add_interaction_args(p_interactions)
    p_interactions.set_defaults(func=build_interactions)

    p_items = subparsers.add_parser("item-prompts")
    add_common_paths(p_items)
    add_item_args(p_items)
    p_items.set_defaults(func=generate_item_prompts)

    p_users = subparsers.add_parser("user-prompts")
    p_users.add_argument("--output-dir", required=True)
    add_user_args(p_users)
    p_users.set_defaults(func=generate_user_prompts)

    p_all = subparsers.add_parser("all")
    add_common_paths(p_all)
    add_interaction_args(p_all)
    # Avoid duplicate --seed when composing arguments.
    p_all.add_argument("--max-concepts", type=int, default=30)
    p_all.add_argument("--second-order-k", type=int, default=10)
    p_all.add_argument("--include-concept-context", action="store_true")
    p_all.add_argument(
        "--prompt-profile", choices=["rich", "structure"], default="rich"
    )
    p_all.add_argument("--max-history", type=int, default=150)
    p_all.add_argument("--max-user-concepts", type=int, default=8)
    p_all.add_argument("--max-users", type=int, default=None)
    p_all.set_defaults(func=run_all)

    args = parser.parse_args()
    if hasattr(args, "train_ratio") and not 0 < args.train_ratio < 1:
        parser.error("--train-ratio must be between 0 and 1")
    if hasattr(args, "min_core") and args.min_core is not None:
        if args.min_core < 2:
            parser.error("--min-core must be at least 2")
        args.min_user_core = args.min_core
        args.min_item_core = args.min_core
    if hasattr(args, "min_user_core") and args.min_user_core < 2:
        parser.error("--min-user-core must be at least 2")
    if hasattr(args, "min_item_core") and args.min_item_core < 1:
        parser.error("--min-item-core must be at least 1")
    if hasattr(args, "min_user_interactions") and args.min_user_interactions < 2:
        parser.error("--min-user-interactions must be at least 2")
    if (
        hasattr(args, "max_user_interactions")
        and args.max_user_interactions is not None
        and args.max_user_interactions < args.min_user_interactions
    ):
        parser.error("--max-user-interactions must be >= --min-user-interactions")
    if hasattr(args, "sample_users") and args.sample_users < 0:
        parser.error("--sample-users must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError, sqlite3.Error) as exc:
        eprint(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
