"""Compare short- and long-gap optical flow on a recorded game viewport."""

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


def stabilize(previous, current):
    """Estimate camera translation from the four scenery patches outside the base."""
    height, width = current.shape
    regions = ((.04,.14,.18,.28), (.82,.14,.96,.28), (.03,.55,.15,.70), (.85,.55,.97,.70))
    shifts = []
    for left, top, right, bottom in regions:
        crop = np.s_[round(top*height):round(bottom*height), round(left*width):round(right*width)]
        a, b = (np.float32(frame[crop]) for frame in (previous, current))
        window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(a, b, window)
        if response >= .45 and max(map(abs, shift)) <= 20:
            shifts.append(shift)
    if len(shifts) < 2:
        return None
    x, y = np.median(shifts, axis=0)
    if np.max(np.abs(np.asarray(shifts) - (x,y))) > 2:
        return None
    return cv2.warpAffine(previous, np.float32([[1,0,x],[0,1,y]]), (width,height),
                          borderMode=cv2.BORDER_REPLICATE)


def motion_mask(flow, minimum, reticle):
    magnitude = np.linalg.norm(flow, axis=2)
    mask = (magnitude >= minimum) & reticle
    return mask, magnitude


def remove_specks(mask, minimum_area=16):
    count, component, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum_area
    return keep[component].astype(bool)


def overlay(image, mask, color):
    result = image.copy()
    result[mask] = (.35*result[mask] + .65*np.asarray(color)).astype(np.uint8)
    return result


def read_selected(path, seconds, near, far, width):
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError("Could not read recording FPS")
        final = [round(t*fps) for t in seconds]
        near_frames, far_frames = [max(1,round(t*fps)) for t in (near,far)]
        wanted = {index-gap for index in final for gap in (0,near_frames,far_frames)}
        if min(wanted)<0:
            raise ValueError("Time samples must be later than the longest gap")
        frames = {}
        for index in range(max(wanted)+1):
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Recording ended before frame {index}")
            if index in wanted:
                height = round(image.shape[0]*width/image.shape[1])
                frames[index] = cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA)
        return [(idx/fps, frames[idx], frames[idx-near_frames], frames[idx-far_frames])
                for idx in final], fps
    finally:
        capture.release()


def probe(path, output, seconds, near, far, width):
    cv2.setNumThreads(2)
    samples, fps = read_selected(path, seconds, near, far, width)
    output.mkdir()
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    panels = []
    rows = []
    for at, current, previous, older in samples:
        started = time.perf_counter()
        c, p, o = (cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in (current,previous,older))
        p, o = stabilize(p,c), stabilize(o,c)
        if p is None or o is None:
            rows.append({"time":at,"status":"camera_alignment_failed"})
            continue
        near_flow, far_flow = (dis.calc(c,im,None) for im in (p,o))
        height, actual_width = c.shape
        yy, xx = np.ogrid[:height,:actual_width]
        # Remove the game chrome and the outer scenery. Keep the battle grass.
        field = (yy >= height*.10) & (yy < height*.81) & (xx >= actual_width*.045) & (xx < actual_width*.96)
        near_mask, near_speed = motion_mask(near_flow,1.15,field)
        far_mask, far_speed = motion_mask(far_flow,2.3,field)
        direction = np.sum(near_flow*far_flow,axis=2)
        stable = remove_specks(near_mask & far_mask & (direction > .55*near_speed*far_speed))
        elapsed = (time.perf_counter()-started)*1000
        fields = (("Original",current), ("Short gap",overlay(current,remove_specks(near_mask),(255,200,30))),
                  ("Long gap",overlay(current,remove_specks(far_mask),(255,100,40))),
                  ("Direction agrees",overlay(current,stable,(60,255,70))))
        strips = []
        for name,im in fields:
            height = round(im.shape[0]*.5)
            tile = cv2.copyMakeBorder(cv2.resize(im,(width//2,height)),28,0,0,0,
                                      cv2.BORDER_CONSTANT,value=(12,12,12))
            cv2.putText(tile,f"{at:.1f}s  {name}",(10,19),cv2.FONT_HERSHEY_SIMPLEX,.56,
                        (240,240,240),1,cv2.LINE_AA)
            strips.append(tile)
        panels.append(np.hstack(strips))
        rows.append({"time":at,"status":"ok","processing_ms":round(elapsed,1),
                     "near_pixels":int(near_mask.sum()),"far_pixels":int(far_mask.sum()),
                     "consistent_pixels":int(stable.sum())})
    if panels:
        cv2.imwrite(str(output/"comparison.jpg"), np.vstack(panels),[cv2.IMWRITE_JPEG_QUALITY,92])
    valid = [row["processing_ms"] for row in rows if row["status"]=="ok"]
    report={"video_fps":fps,"near_gap_seconds":near,"far_gap_seconds":far,"width":width,
            "median_processing_ms":statistics.median(valid) if valid else None,"samples":rows}
    (output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--at",type=float,nargs="+",required=True)
    parser.add_argument("--near",type=float,default=.10)
    parser.add_argument("--far",type=float,default=.35)
    parser.add_argument("--width",type=int,default=960)
    args=parser.parse_args()
    if not (0 < args.near < args.far < min(args.at) and args.width >= 160):
        parser.error("Expected positive frame gaps shorter than the sample times")
    probe(args.video,args.output,args.at,args.near,args.far,args.width)


if __name__=="__main__":
    main()
