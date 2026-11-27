"""Unattended multi-battle batch runner for Stage 2 BC/RL trajectory and clean-frame collection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10, help="Number of battles to run back-to-back")
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs" / time.strftime("batch-%Y%m%d-%H%M%S"))
    parser.add_argument("--record-bc", action="store_true", default=True, help="Record bc-trajectory.npz (default: on)")
    parser.add_argument("--no-record-bc", dest="record_bc", action="store_false")
    parser.add_argument("--save-clean-frames", action="store_true", help="Save clean unannotated frames for battle YOLO")
    parser.add_argument("--record-video", action="store_true", help="Keep annotated mp4 videos (disabled by default to save disk)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--adb", default=None)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cooldown", type=float, default=3.0, help="Seconds to wait between episodes")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_root / "batch_summary.json"
    results = []

    for idx in range(1, args.episodes + 1):
        ep_dir = args.out_root / f"ep-{idx:03d}"
        cmd = [
            sys.executable,
            str(ROOT / "run_stage1_fsm.py"),
            str(ROOT / "captures" / "yolo26s-self-loop"),
            str(ep_dir),
            "--full",
        ]
        if args.record_bc:
            cmd.append("--record-bc")
        if args.save_clean_frames:
            cmd.append("--save-clean-frames")
        if not args.record_video:
            cmd.append("--no-record-video")
        if args.config:
            cmd.extend(["--config", str(args.config)])
        if args.adb:
            cmd.extend(["--adb", args.adb])
        if args.serial:
            cmd.extend(["--serial", args.serial])
        if args.device:
            cmd.extend(["--device", args.device])

        print(f"\n[BATCH {idx}/{args.episodes}] Starting -> {ep_dir}", flush=True)
        t0 = time.monotonic()
        proc = subprocess.run(cmd, cwd=str(ROOT))
        dur = round(time.monotonic() - t0, 1)

        entry = {"episode": idx, "dir": str(ep_dir), "returncode": proc.returncode, "duration_s": dur}
        rj = ep_dir / "run.json"
        if rj.exists():
            data = json.loads(rj.read_text(encoding="utf-8"))
            entry["final_stars"] = data.get("final_stars")
            entry["final_destruction"] = data.get("final_destruction")
            entry["bc_steps"] = data.get("bc_steps_recorded")
            entry["side"] = data.get("plan", {}).get("side")
        results.append(entry)

        valid_dest = [r["final_destruction"] for r in results if isinstance(r.get("final_destruction"), int)]
        valid_stars = [r["final_stars"] for r in results if isinstance(r.get("final_stars"), int)]
        three_stars = sum(1 for s in valid_stars if s == 3)
        stats = {
            "completed": len(results),
            "target": args.episodes,
            "mean_destruction": round(sum(valid_dest) / len(valid_dest), 1) if valid_dest else None,
            "mean_stars": round(sum(valid_stars) / len(valid_stars), 2) if valid_stars else None,
            "three_star_count": three_stars,
            "episodes": results,
        }
        summary_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(
            f"[BATCH {idx}/{args.episodes}] rc={proc.returncode} dur={dur}s "
            f"stars={entry.get('final_stars')} dest={entry.get('final_destruction')}% "
            f"| running_avg={stats['mean_destruction']}% 3-stars={three_stars}/{len(valid_stars)}",
            flush=True,
        )
        if idx < args.episodes:
            time.sleep(args.cooldown)

    print(f"\nBatch complete! Summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
