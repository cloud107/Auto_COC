"""Constant-velocity Kalman filter for a confirmed Miner/Hog deployment."""

import math

import numpy as np

POSITION = np.array([[1.,0,0,0],[0,1.,0,0]])
VELOCITY = np.array([[0,0,1.,0],[0,0,0,1.]])


class DeployedWaveFilter:
    """Forecast a troop group and distinguish confirmed observations from extrapolation."""

    def __init__(self, deployment_time, seed, initial_velocity, *, observation_gap=.35,
                 max_initial_reacquire_distance=None):
        self.deployment_time = deployment_time
        self.observation_gap = observation_gap
        self.max_initial_reacquire_distance = max_initial_reacquire_distance
        self.seed = np.asarray(seed,dtype=float)
        self.initial_velocity = np.asarray(initial_velocity,dtype=float)
        self.state = None
        self.covariance = None
        self.last_time = None
        self.position_seen = None
        self.pending = None
        self.radius = 30
        self.reacquired = False

    def predict(self, timestamp):
        if self.state is None:
            self.state = np.r_[self.seed,self.initial_velocity]
            self.covariance = np.diag([14.**2,14.**2,45.**2,45.**2])
            self.last_time = self.deployment_time
        self.reacquired = False
        dt = timestamp-self.last_time
        if dt < 0:
            raise ValueError("Tracking timestamps must be monotonic")
        transition = np.eye(4)
        transition[:2,2:] = np.eye(2)*dt
        self.state = transition @ self.state
        self.state[2:] *= math.exp(-.7*dt)
        process = np.diag([12.**2,12.**2,40.**2,40.**2])*dt
        self.covariance = transition @ self.covariance @ transition.T + process
        self.last_time = timestamp

    def correct(self, measurement, observation_matrix, sigma, gate):
        observation = np.asarray(measurement,dtype=float)
        innovation = observation - observation_matrix @ self.state
        noise = np.eye(2)*sigma**2
        projected = observation_matrix @ self.covariance @ observation_matrix.T + noise
        if innovation @ np.linalg.solve(projected,innovation) > gate:
            return False
        gain = np.linalg.solve(projected, observation_matrix @ self.covariance).T
        self.state += gain @ innovation
        remaining = np.eye(4)-gain @ observation_matrix
        self.covariance = remaining @ self.covariance @ remaining.T + gain @ noise @ gain.T
        return True

    def correct_position(self, timestamp, measured):
        age = timestamp-self.position_seen if self.position_seen is not None else timestamp-self.deployment_time
        point = np.array(measured["center"])
        # After a longer loss, require two nearby measurements before accepting a new anchor.
        if age > 1.35:
            distance = np.linalg.norm(point-self.state[:2])
            if (self.position_seen is None and self.max_initial_reacquire_distance is not None
                    and distance > self.max_initial_reacquire_distance):
                return False
            previous = self.pending
            self.pending = (timestamp,point)
            if previous is None or timestamp-previous[0]>self.observation_gap or np.linalg.norm(previous[1]-point)>35:
                return False
            if distance>95 and measured["indicators"]<3:
                return False
            self.state = np.r_[point,np.zeros(2)]
            self.covariance = np.diag([20.**2,20.**2,45.**2,45.**2])
            self.reacquired = True
        else:
            sigma = 12 if measured["indicators"] >= 3 else 19
            if np.linalg.norm(point-self.state[:2])>max(42,min(92,38+30*age)):
                return False
            if not self.correct(point,POSITION,sigma,9.21):
                return False
        self.pending = None
        self.position_seen = timestamp
        self.radius = .7*self.radius+.3*measured["radius"]
        return True

    def estimate(self, timestamp, observed, used_flow, indicators):
        age = timestamp-self.position_seen if self.position_seen is not None else timestamp-self.deployment_time
        uncertainty = float(2*np.sqrt(np.linalg.eigvalsh(self.covariance[:2,:2])[-1]))
        if age>1.35 or uncertainty>70:
            return None
        source = "health_bars" if observed else "motion_only" if used_flow else "predicted"
        return {"center":self.state[:2].tolist(),"velocity":self.state[2:].tolist(),
                "radius":self.radius,"source":source,"uncertainty_px":uncertainty,
                "age_seconds":round(age,3),"indicators":indicators if observed else 0,
                "reacquired":self.reacquired}


def opening_prior(commands, layout, width, height):
    """Seed the wave from the two squads; apply a small directional prior for Hogs."""
    squads = [c for c in commands if c["unit"] in {"miner_squad","hog_squad"}]
    points = [point for c in squads for point in c["points"]]
    if not points:
        raise ValueError("Missing deployed Miner/Hog coordinates")
    seed = np.median(points,axis=0)*[width/1800,height/1440]
    velocity = np.zeros(2)
    if layout:
        defenders = [d["bbox_xyxy"] for d in layout["detections"]
                     if d["class"] not in {"town_hall","clan_castle"}]
        if defenders:
            target = min((np.array([(a+c)/2,d])*[width/1800,height/1440]
                          for a,_,c,d in defenders),key=lambda p:np.linalg.norm(p-seed))
            direction = target-seed
            velocity = 10*direction/max(1,np.linalg.norm(direction))
    return seed,velocity


def split_group_measurements(troops):
    """Group nearby bars without using a bar in two measurements."""
    clusters = [np.asarray([point]) for point in troops]
    while len(clusters) > 1:
        separation, left, right = min(
            (np.linalg.norm(a[:, None] - b[None, :], axis=2).max(), i, j)
            for i, a in enumerate(clusters)
            for j, b in enumerate(clusters[i + 1:], i + 1)
        )
        if separation > 42:
            break
        clusters[left] = np.concatenate((clusters[left], clusters.pop(right)))
    measurements = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        center = cluster.mean(axis=0)
        radius = float(np.linalg.norm(cluster - center, axis=1).max() + 17)
        measurements.append({"center": center.tolist(), "radius": radius, "indicators": len(cluster)})
    return measurements


class DeployedWaveTracks:
    """Follow the deployment and admit persistent secondary groups near a known branch."""

    def __init__(self, deployment_time, seed, velocity, *, observation_gap=.31,
                 max_initial_reacquire_distance=None):
        self.deployment_time = deployment_time
        self.observation_gap = observation_gap
        self.max_initial_reacquire_distance = max_initial_reacquire_distance
        self.tracks = {1: self.new_track(deployment_time, seed, velocity)}
        self.next_id = 2
        self.pending = []

    def new_track(self, timestamp, seed, velocity):
        return DeployedWaveFilter(timestamp, seed, velocity,
                                  observation_gap=max(.35, self.observation_gap),
                                  max_initial_reacquire_distance=self.max_initial_reacquire_distance)

    def search_centers(self, timestamp):
        return [
            track.state[:2] for track in self.tracks.values()
            if track.state is not None and track.estimate(timestamp, False, False, 0) is not None
        ]

    def update(self, timestamp, measurements, local_motion, reacquisition_measurement=None,
               *, single_bar_positions=None):
        if timestamp < self.deployment_time:
            return []

        for track_id, track in list(self.tracks.items()):
            if track_id != 1 and timestamp - track.position_seen > 1.35:
                del self.tracks[track_id]
                continue
            track.predict(timestamp)

        observed = {}
        claimed = set()
        matches = sorted(
            (np.linalg.norm(track.state[:2] - measurement["center"]), track_id, i)
            for track_id, track in self.tracks.items()
            if timestamp - (track.position_seen if track.position_seen is not None else track.deployment_time) <= 1.35
            for i, measurement in enumerate(measurements)
        )
        for _, track_id, i in matches:
            if track_id not in observed and i not in claimed:
                if self.tracks[track_id].correct_position(timestamp, measurements[i]):
                    observed[track_id] = measurements[i]
                    claimed.add(i)

        primary = self.tracks[1]
        # Preserve the original wider reacquisition while no other branch has a lock.
        last_primary_observation = (primary.position_seen if primary.position_seen is not None
                                    else primary.deployment_time)
        if len(self.tracks) == 1 and timestamp - last_primary_observation > 1.35 and reacquisition_measurement:
            if primary.correct_position(timestamp, reacquisition_measurement):
                observed[1] = reacquisition_measurement
                claimed.update(i for i, measurement in enumerate(measurements)
                               if np.linalg.norm(np.subtract(measurement["center"],
                                                             reacquisition_measurement["center"])) < 40)

        remaining = [measurement for i, measurement in enumerate(measurements) if i not in claimed]
        self.confirm_branches(timestamp, remaining, observed)

        single_bar_positions = single_bar_positions or {}
        areas = []
        for track_id, track in self.tracks.items():
            measurement = observed.get(track_id)
            if measurement is None and any(
                np.linalg.norm(track.state[:2] - other["center"]) < 36 for other in observed.values()
            ):
                # Keep the internal track through a brief merge, but only draw one circle.
                continue
            used_flow = False
            used_bar = None
            if measurement is None and track.position_seen is not None:
                gap = timestamp - track.position_seen
                candidate = single_bar_positions.get(track_id)
                if candidate is not None and gap <= .8:
                    bar_id, center = candidate
                    if (np.linalg.norm(center - track.state[:2]) <= 25
                            and track.correct(center, POSITION, 22, 6)):
                        used_bar = bar_id
                if used_bar is None:
                    evidence = local_motion(track, gap)
                    if evidence is not None:
                        velocity = (np.asarray(evidence["center"]) - track.state[:2]) / .35
                        used_flow = track.correct(velocity, VELOCITY, 30, 6)
            estimate = track.estimate(timestamp, measurement is not None, used_flow,
                                      measurement["indicators"] if measurement else 0)
            if estimate is not None:
                if used_bar is not None:
                    estimate.update(source="single_bar", bar_track_id=used_bar)
                areas.append({"id": track_id, **estimate})
        separated = []
        by_evidence = sorted(areas, key=lambda area: (
            {"health_bars": 0, "single_bar": 1}.get(area["source"], 2), area["age_seconds"]
        ))
        for area in by_evidence:
            if all(np.linalg.norm(np.subtract(area["center"], kept["center"])) >= 36 for kept in separated):
                separated.append(area)
        return sorted(separated, key=lambda area: area["id"])

    def confirm_branches(self, timestamp, measurements, observed):
        previous = self.pending
        self.pending = []
        for measurement in measurements:
            center = np.asarray(measurement["center"])
            distances = [
                np.linalg.norm(center - track.state[:2])
                for track in self.tracks.values()
                if track.position_seen is not None and timestamp - track.position_seen <= .55
            ]
            if not distances or not 48 <= min(distances) <= 125:
                continue
            nearest = min(range(len(previous)), key=lambda i: np.linalg.norm(center - previous[i][1]),
                          default=None)
            count = 1
            if nearest is not None:
                last_time, last_center, hits = previous[nearest]
                if timestamp - last_time <= self.observation_gap and np.linalg.norm(center - last_center) <= 28:
                    previous.pop(nearest)
                    count = hits + 1
            if count < 3 or len(self.tracks) >= 4:
                self.pending.append((timestamp, center, count))
                continue

            track = self.new_track(timestamp, center, np.zeros(2))
            track.predict(timestamp)
            track.correct_position(timestamp, measurement)
            self.tracks[self.next_id] = track
            observed[self.next_id] = measurement
            self.next_id += 1
