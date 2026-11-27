"""Track a tentative main-wave region from a recorded deployment and visible health bars."""

import argparse
import json
import statistics
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from battle_flow_probe import stabilize
from health_bar_tracks import HealthBarTracks
from deployed_wave_filter import DeployedWaveTracks, opening_prior, split_group_measurements


def health_bars(image, roi, *, yellow=False):
    left, top, right, bottom = roi
    region = image[top:bottom, left:right]
    hue, saturation, value = cv2.split(cv2.cvtColor(region, cv2.COLOR_BGR2HSV))
    if yellow:
        fill = (hue >= 19) & (hue <= 36) & (saturation >= 150) & (value >= 168)
    else:
        fill = (hue >= 44) & (hue <= 69) & (saturation >= 145) & (value >= 130)
    _, _, stats, _ = cv2.connectedComponentsWithStats(fill.astype(np.uint8))
    dark = (value < 88)
    bars = []
    for x, y, w, h, area in stats[1:]:
        if not (6 <= w <= 45 and 1 <= h <= 5 and w >= 2.5*h and area >= .55*w*h):
            continue
        if y < 2 or y+h+3 > region.shape[0]:
            continue
        outline = max(dark[y-2:y, x:x+w].mean(), dark[y+h:y+h+3, x:x+w].mean())
        if outline < (.35 if yellow else .24):
            continue
        if yellow and (w > 27 or h > 4):
            continue
        bars.append((int(left+x),int(top+y),int(w),int(h)))
    return bars


def has_level_badge(image, bar):
    x, y, _, _ = bar
    if x < 19 or y < 6:
        return False
    chip = image[y-6:y+7, x-19:x-1]
    hue, saturation, value = cv2.split(cv2.cvtColor(chip, cv2.COLOR_BGR2HSV))
    purple = (hue >= 125) & (hue <= 171) & (saturation >= 80) & (value >= 75)
    digits = (saturation < 95) & (value > 170)
    return np.count_nonzero(purple) >= 23 and np.count_nonzero(digits) >= 18


def deployment_group(image, observations):
    """Separate visible hero and ordinary unit indicators before estimating troop coverage."""
    heroes = [observation for observation in observations if has_level_badge(image, observation[1])]
    troops = []
    for observation in observations:
        point, box, _ = observation
        x,y,_,_ = box
        if any(abs(x-hero_box[0]) <= 28 and abs(y-hero_box[1]) <= 11
               for _,hero_box,_ in heroes):
            continue
        troops.append(point)
    if len(troops) < 2:
        return heroes, troops, None
    positions = np.asarray(troops)
    distances = np.linalg.norm(positions[:,None]-positions[None,:],axis=2)
    neighbors = distances < 65
    cluster = positions[neighbors[np.argmax(neighbors.sum(axis=1))]]
    if len(cluster) < 2:
        return heroes, troops, None
    center = np.mean(cluster, axis=0)
    radius = float(np.clip(np.max(np.linalg.norm(cluster-center,axis=1))+17, 20, 56))
    return heroes, troops, {"center":center.tolist(), "radius":radius, "indicators":len(cluster)}


def motion_evidence(previous, current, roi, dis):
    # Registration uses a small image; dense flow is calculated only in the search region.
    small = [cv2.resize(frame,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
             for frame in (previous,current)]
    before, after = [cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) for frame in small]
    aligned = stabilize(before,after)
    if aligned is None:
        return None
    left, top, right, bottom = (coordinate//2 for coordinate in roi)
    crop = np.s_[top:bottom,left:right]
    flow = dis.calc(np.ascontiguousarray(after[crop]), np.ascontiguousarray(aligned[crop]), None)
    mask = (np.linalg.norm(flow,axis=2) >= 1.05).astype(np.uint8)
    return mask, flow


def follow_hidden_wave(area, gap, step, motion, roi, heroes):
    """Predict a brief gap using local, coherent flow anchored to the last troop observation."""
    if area is None or motion is None or not 0 < gap <= .8:
        return None
    mask, flow = motion
    left, top, _, _ = roi
    center = np.asarray(area["center"])
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    px, py = 2*(xx + left//2), 2*(yy + top//2)
    radius = min(area["radius"] + 5, 51)
    support = ((px-center[0])**2 + (py-center[1])**2 <= radius**2) & mask.astype(bool)
    for point, _, _ in heroes:
        support &= (px-point[0])**2 + (py-point[1])**2 > 17**2
    vectors = -2 * flow[support]
    if len(vectors) < 32:
        return None
    median = np.median(vectors, axis=0)
    speed = np.linalg.norm(median)
    if not 2 <= speed <= 32:
        return None
    norms = np.linalg.norm(vectors, axis=1)
    coherent = (vectors @ median >= .75 * norms * speed) & (norms <= 2*speed+3)
    if coherent.mean() < .65:
        return None
    shift = np.median(vectors[coherent], axis=0) * step / .35
    return {**area, "center":(center+shift).tolist(), "indicators":0,
            "source":"motion_only", "age_seconds":round(gap,3),
            "coherent_pixels":int(coherent.sum())}


class TroopTrack:
    """Keep one short-lived position estimate, with separate observation and prediction timestamps."""

    def __init__(self):
        self.area = None
        self.seen = None
        self.updated = None
        self.pending = None
        self.velocity = np.zeros(2)

    def update(self, t, observation, motion, roi, heroes):
        if self.seen is not None and t-self.seen > 1.2:
            self.area = None
            self.velocity.fill(0)
        if self.area is None:
            if observation is None:
                self.pending = None
                return None
            measured = np.asarray(observation["center"])
            if self.pending is None or np.linalg.norm(measured-self.pending) > 32:
                self.pending = measured
                return None
            self.pending = None
            return self.accept(t, observation, measured)

        step = t-self.updated
        old = np.asarray(self.area["center"])
        expected = old+self.velocity*step
        if observation is not None:
            measured = np.asarray(observation["center"])
            if np.linalg.norm(measured-expected) <= 32+32*(t-self.seen):
                updated = .55*measured+.45*expected
                instant = (updated-old)/max(step,.10)
                speed = np.linalg.norm(instant)
                self.velocity = .65*self.velocity+.35*instant*min(1,85/max(speed,1))
                return self.accept(t, observation, updated)

        gap = t-self.seen
        moved = follow_hidden_wave(self.area,gap,step,motion,roi,heroes)
        if moved is not None:
            flow_position = np.asarray(moved["center"])
            if np.linalg.norm(flow_position-expected)<=24:
                updated = .60*flow_position+.40*expected
                self.velocity = .7*self.velocity+.3*(updated-old)/max(step,.10)
                return self.store(t, moved, updated)
        if motion is None or gap>.8:
            return None
        self.velocity *= .8
        return self.store(t, {**self.area,"source":"predicted","age_seconds":round(gap,3),
                              "indicators":0}, expected)

    def accept(self, t, observation, center):
        self.seen = t
        return self.store(t, {**observation,"source":"health_bars","age_seconds":0}, center)

    def store(self, t, area, center):
        self.updated = t
        self.area = {**area,"center":center.tolist()}
        return self.area


def nearby_motion(mask, point, roi):
    if mask is None:
        return False
    left, top, _, _ = roi
    x, y = round(point[0]/2-left//2), round(point[1]/2-top//2)
    if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
        return False
    region = mask[max(0,y-9):y+10,max(0,x-9):x+10]
    return bool(region.size and region.mean() >= .12)


def main_seed(plan, width, height):
    log = json.loads(plan.read_text(encoding="utf-8"))
    wave = next(stage for stage in log["stages"] if stage["name"] == "main_wave")
    pixels = next(command["points"] for command in wave["commands"] if command["unit"] == "miner_squad")
    return np.median(np.float32(pixels),axis=0)*[width/1800,height/1440]


def inspect(video, plan, output, start, until, hide_bars, use_bar_tracks, sample_rate, kalman=None):
    cv2.setNumThreads(2)
    capture=cv2.VideoCapture(str(video))
    source_fps=capture.get(cv2.CAP_PROP_FPS)
    width,height=(int(capture.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps<=0 or (width,height)!=(1200,960):
        capture.release()
        raise ValueError("Expected the calibrated 1200×960 battle recording")
    output.mkdir()
    stride=max(1,round(source_fps/sample_rate))
    period=stride/source_fps
    gap=round(source_fps*.35)
    frames=deque(maxlen=gap+1)
    writer=cv2.VideoWriter(str(output/"review.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),source_fps/stride,(width,height))
    if not writer.isOpened():
        capture.release()
        raise OSError("Could not open video writer")
    wave_tracks = None
    if kalman is not None:
        deployment_time, layout = kalman
        log = json.loads(plan.read_text(encoding="utf-8"))
        wave = next(stage for stage in log["stages"] if stage["name"] == "main_wave")
        if wave.get("status") != "sent" or any(
            wave.get("card_changes", {}).get(name, {}).get("appearance_delta", 0) < 30
            for name in ("miner_squad", "hog_squad")
        ):
            raise ValueError("Miner/Hog wave lacks post-deployment card evidence")
        center, velocity = opening_prior(wave["commands"],layout,width,height)
        wave_tracks = DeployedWaveTracks(deployment_time,center,velocity)
    else:
        center=main_seed(plan,width,height)
    seed=center.copy()
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    recent_bars=deque(maxlen=2)
    last_seen=None
    samples=[]
    troop_track = TroopTrack()
    bar_tracker = HealthBarTracks() if use_bar_tracks else None
    try:
        for index in range(round(until*source_fps)):
            ok,frame=capture.read()
            if not ok:
                break
            frames.append(frame)
            t=index/source_fps
            if t<start or index%stride or len(frames)<frames.maxlen:
                continue
            started=time.perf_counter()
            anchors = [center]
            if wave_tracks is not None:
                anchors.extend(wave_tracks.search_centers(t))
            radius = 130
            low_x, low_y = np.rint(np.min(anchors, axis=0)).astype(int)
            high_x, high_y = np.rint(np.max(anchors, axis=0)).astype(int)
            roi = (max(0, low_x-radius), max(90, low_y-radius),
                   min(width, high_x+radius), min(780, high_y+radius))
            motion=motion_evidence(frames[0],frame,roi,dis)
            observations=[]
            visible_bars=[]
            candidates=health_bars(frame,roi) if motion is not None else []
            for x,y,w,h in candidates:
                point=np.array([x+w/2,y+h+12])
                if all(np.linalg.norm(point-anchor)>radius for anchor in anchors):
                    continue
                visible_bars.append(point)
                moving=nearby_motion(motion[0],point,roi)
                persistent=len(recent_bars)==2 and all(
                    any(np.linalg.norm(point-old)<18 for old in older_bars)
                    for older_bars in recent_bars
                )
                if moving or persistent:
                    observations.append((point,(x,y,w,h),moving))
            recent_bars.append(visible_bars)
            if len(observations)>=2:
                points=np.asarray([point for point,_,_ in observations])
                distances=np.linalg.norm(points-center,axis=1)
                distances_between=np.linalg.norm(points[:,None]-points[None,:],axis=2)
                weights=1/(45+distances)
                support=(distances_between<70) @ weights
                cluster=distances_between[np.argmax(support)]<70
                if cluster.sum()>=2:
                    points,weights=points[cluster],weights[cluster]
                measured=np.average(points,axis=0,weights=weights)
                center=.65*center+.35*measured
                last_seen=t
                status="candidate_cluster"
            elif last_seen is not None and t-last_seen<=.8:
                status="short_hold"
            else:
                status="unknown"
            heroes, troops, measurement = deployment_group(frame, observations)
            masked = any(start <= t < end for start, end in hide_bars)
            bar_ids = []
            reacquired = []
            if bar_tracker is not None:
                troop_bars = []
                if not masked:
                    hero_boxes = [box for _,box,_ in heroes]
                    for x,y,w,h in candidates:
                        point = np.array([x+w/2,y+h+12])
                        if all(np.linalg.norm(point-anchor)>radius for anchor in anchors) or has_level_badge(frame,(x,y,w,h)):
                            continue
                        if any(abs(x-box[0]) <= 28 and abs(y-box[1]) <= 11 for box in hero_boxes):
                            continue
                        troop_bars.append((point,w,nearby_motion(motion[0],point,roi)))
                bar_ids, reacquired = bar_tracker.update(t,troop_bars,period)
                measurement = bar_tracker.group(bar_ids)
            if wave_tracks is None:
                protection = troop_track.update(t, None if masked else measurement, motion, roi, heroes)
                troop_areas = [protection] if protection is not None else []
            else:
                points = [bar.point for bar in bar_ids] if bar_tracker is not None else troops
                grouped = split_group_measurements(points) if not masked and t >= wave_tracks.deployment_time else []

                def local_motion(track, gap):
                    area = {"center": track.state[:2].tolist(), "radius": track.radius}
                    return follow_hidden_wave(area, gap, .35, motion, roi, heroes)

                troop_areas = wave_tracks.update(t, grouped, local_motion, None if masked else measurement)
                protection = next((area for area in troop_areas if area["id"] == 1), None)
            elapsed=(time.perf_counter()-started)*1000
            overlay=frame.copy()
            left,top,right,bottom=roi
            if motion is not None:
                shade=cv2.resize(motion[0],(right-left,bottom-top),interpolation=cv2.INTER_NEAREST).astype(bool)
                area=overlay[top:bottom,left:right]
                area[shade]=(.72*area[shade]+.28*np.array([248,140,45])).astype(np.uint8)
            for point,(x,y,w,h),_ in observations:
                cv2.rectangle(overlay,(x-2,y-2),(x+w+2,y+h+2),(70,255,110),1)
                cv2.circle(overlay,tuple(np.int32(point)),3,(70,255,110),-1)
            for point,(x,y,w,h),_ in heroes:
                cv2.rectangle(overlay,(x-18,y-5),(x+w+2,y+h+3),(220,90,255),2)
                cv2.drawMarker(overlay,tuple(np.int32(point)),(220,90,255),cv2.MARKER_DIAMOND,13,2)
            for protection_area in troop_areas:
                location=tuple(np.int32(protection_area["center"]))
                radius = round(protection_area["radius"])
                color = (35,220,255)
                if protection_area["source"] != "health_bars":
                    color = (25,147,255) if protection_area["source"] == "motion_only" else (180,180,180)
                    for angle in range(0, 360, 45):
                        cv2.ellipse(overlay,location,(radius,radius),0,angle,angle+26,color,2)
                else:
                    cv2.circle(overlay,location,radius,color,2)
                cv2.drawMarker(overlay,location,color,cv2.MARKER_CROSS,13,2)
                if "id" in protection_area:
                    cv2.putText(overlay,str(protection_area["id"]),(location[0]+8,location[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA)
            for bar in bar_ids:
                x, y = map(round, bar.point)
                color = (255,235,85) if bar.id not in reacquired else (65,170,255)
                cv2.circle(overlay,(x,y),3,color,-1)
                cv2.putText(overlay,str(bar.id),(x+5,y-3),cv2.FONT_HERSHEY_SIMPLEX,.46,color,1,cv2.LINE_AA)
            cv2.drawMarker(overlay,tuple(np.int32(seed)),(180,170,170),cv2.MARKER_CROSS,12,1)
            area_source = ", ".join(area["source"] for area in troop_areas) or "unknown"
            cv2.putText(overlay,f"{t:.1f}s  heroes={len(heroes)}  groups={len(troop_areas)}  {area_source}",(15,62),
                        cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2,cv2.LINE_AA)
            writer.write(overlay)
            samples.append({"time":round(t,3),"status":status,"bars":len(observations),
                            "center":center.tolist() if status!="unknown" else None,
                            "hero_candidates":[point.tolist() for point,_,_ in heroes],
                            "troop_candidates":[point.tolist() for point in troops],
                            "bar_tracks":[{"id":bar.id,"point":bar.point.tolist(),"observations":bar.count} for bar in bar_ids],
                            "bar_reacquired":reacquired,"troop_measurement":None if masked else measurement,
                            "masked":masked,"troop_area":protection,"troop_areas":troop_areas,"processing_ms":round(elapsed,2)})
    finally:
        capture.release()
        writer.release()
    if not samples:
        raise ValueError("No frames sampled")
    report={"start":start,"seed":seed.tolist(),"sample_rate":source_fps/stride,
            "median_processing_ms":statistics.median(row["processing_ms"] for row in samples),
            "samples":samples}
    (output/"tracking.json").write_text(json.dumps(report,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video",type=Path)
    parser.add_argument("plan",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--start",type=float,default=12)
    parser.add_argument("--until",type=float,default=37)
    parser.add_argument("--kalman",action="store_true",help="Use deployment-seeded Kalman tracking")
    parser.add_argument("--video-clock",type=Path)
    parser.add_argument("--deployment-time",type=float,help="Fallback deployment time in seconds for older recordings")
    parser.add_argument("--layout",type=Path,help="Optional defense positions for a weak initial direction")
    parser.add_argument("--bar-tracks",action="store_true",help="Associate individual health bars before grouping")
    parser.add_argument("--sample-rate",type=float,default=5)
    parser.add_argument("--hide-bars",type=float,nargs=2,action="append",default=[],metavar=("START","END"),
                        help="Suppress troop observations for a replay-only gap test")
    args=parser.parse_args()
    if not 1<=args.sample_rate<=15:
        parser.error("Expected a sample rate between 1 and 15 Hz")
    if not 1<=args.start<args.until:
        parser.error("Expected at least one second before start and until later than start")
    if any(not args.start <= start < end <= args.until for start,end in args.hide_bars):
        parser.error("Each hidden interval must fall within the sampled segment")
    kalman = None
    if args.kalman:
        if args.video_clock:
            plan = json.loads(args.plan.read_text())
            wave = next(stage for stage in plan["stages"] if stage["name"]=="main_wave")
            clock = json.loads(args.video_clock.read_text())
            deployment = wave["dispatch_started_monotonic_s"]-clock["recording_requested_monotonic_s"]
        elif args.deployment_time is not None:
            deployment = args.deployment_time
        else:
            parser.error("--kalman requires --video-clock or --deployment-time")
        if not 0 <= deployment < args.until:
            parser.error("Deployment time falls outside the recording")
        kalman = (deployment,json.loads(args.layout.read_text()) if args.layout else None)
    inspect(args.video,args.plan,args.output,args.start,args.until,args.hide_bars,args.bar_tracks,args.sample_rate,kalman)


if __name__=="__main__":
    main()
