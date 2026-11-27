"""Associate short health-bar observations across successive video frames."""

from dataclasses import dataclass

import numpy as np


@dataclass
class BarTrack:
    id: int
    point: np.ndarray
    velocity: np.ndarray
    width: int
    seen: float
    count: int = 1


class HealthBarTracks:
    def __init__(self, retain_seconds=.65, *, min_match_margin=None):
        self.retain_seconds = retain_seconds
        self.min_match_margin = min_match_margin
        self.tracks = {}
        self.next_id = 1
        self.group_ids = set()

    def update(self, timestamp, observations, period):
        self.tracks = {
            key: track for key, track in self.tracks.items()
            if timestamp-track.seen <= self.retain_seconds
        }
        costs = []
        for key, track in self.tracks.items():
            gap = timestamp-track.seen
            guess = track.point + track.velocity*min(gap,.45)
            for index, (point, width, _) in enumerate(observations):
                distance = np.linalg.norm(point-guess)
                movement = np.linalg.norm(point-track.point)
                limit = min(30, 10 + 38*gap)
                if distance <= limit and movement <= 10+75*gap:
                    cost = distance + .14*abs(track.width-width)
                    costs.append((cost,key,index))
        costs.sort()
        if self.min_match_margin is not None:
            costs = self.unambiguous_matches(costs)
        used_tracks = set()
        used_observations = set()
        reacquired = []
        for _, key, index in costs:
            if key in used_tracks or index in used_observations:
                continue
            track = self.tracks[key]
            point, width, _ = observations[index]
            gap = timestamp-track.seen
            velocity = (point-track.point)/max(gap,period)
            speed = np.linalg.norm(velocity)
            track.velocity = .65*track.velocity + .35*velocity*min(1,75/max(speed,1))
            track.point, track.width, track.seen = point,width,timestamp
            track.count += 1
            if gap > period*1.6 and track.count >= 3:
                reacquired.append(key)
            used_tracks.add(key)
            used_observations.add(index)
        for index,(point,width,moving) in enumerate(observations):
            if index in used_observations or not moving:
                continue
            key = self.next_id
            self.tracks[key] = BarTrack(key,point,np.zeros(2),width,timestamp)
            used_tracks.add(key)
            self.next_id += 1
        visible = [self.tracks[key] for key in sorted(used_tracks) if self.tracks[key].count >= 2]
        return visible, reacquired

    def unambiguous_matches(self, costs):
        by_track, by_observation = {}, {}
        for pair in costs:
            _, key, index = pair
            by_track.setdefault(key, []).append(pair)
            by_observation.setdefault(index, []).append(pair)
        return [
            pair for pair in costs
            if all(options[0] == pair and
                   (len(options) == 1 or options[1][0] - pair[0] >= self.min_match_margin)
                   for options in (by_track[pair[1]], by_observation[pair[2]]))
        ]

    def group(self, visible):
        if len(visible) < 2:
            return None
        points = np.asarray([track.point for track in visible])
        distances = np.linalg.norm(points[:,None]-points[None,:],axis=2)
        neighbors = distances <= 62
        groups = [np.flatnonzero(row) for row in neighbors]
        groups = [group for group in groups if len(group)>=2]
        if not groups:
            return None
        def score(group):
            keys = {visible[i].id for i in group}
            if self.group_ids:
                return (len(keys & self.group_ids), len(group))
            return (0, len(group))
        selected = max(groups,key=score)
        self.group_ids = {visible[i].id for i in selected}
        cluster = points[selected]
        center = np.mean(cluster,axis=0)
        radius = float(np.clip(np.max(np.linalg.norm(cluster-center,axis=1))+17,20,56))
        return {"center":center.tolist(),"radius":radius,"indicators":len(selected),
                "bar_track_ids":sorted(self.group_ids)}
