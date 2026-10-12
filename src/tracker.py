"""
tracker.py - A very simple centroid tracker (Motion and Object Tracking).

Idea: an object does not move far between two consecutive frames. So every
new centroid is matched to the closest centroid from the previous frame and
inherits its ID. A centroid with no close match becomes a new object.
"""

import math
from collections import deque

from config import TRACK_MAX_DISTANCE, TRACK_MAX_MISSING, TRACK_TRAIL_LENGTH


class CentroidTracker:
    def __init__(self, max_distance=TRACK_MAX_DISTANCE, max_missing=TRACK_MAX_MISSING,
                 trail_length=TRACK_TRAIL_LENGTH):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.trail_length = trail_length
        self.next_id = 1
        self.positions = {}   # id -> last known (x, y)
        self.missing = {}     # id -> number of frames since it was last seen
        self.trails = {}      # id -> deque of recent (x, y) positions

    def update(self, centroids):
        """
        Give the tracker this frame's centroids [(x, y), ...].
        Returns a list of IDs in the same order as `centroids`.
        """
        ids = [None] * len(centroids)

        # 1. Measure the distance between every known object and every new centroid.
        pairs = []
        for object_id, (ox, oy) in self.positions.items():
            for index, (cx, cy) in enumerate(centroids):
                pairs.append((math.hypot(cx - ox, cy - oy), object_id, index))
        pairs.sort()  # closest pairs first

        # 2. Greedily match the closest pairs (each object / centroid is used once).
        used_objects, used_centroids = set(), set()
        for distance, object_id, index in pairs:
            if distance > self.max_distance:
                break
            if object_id in used_objects or index in used_centroids:
                continue
            ids[index] = object_id
            used_objects.add(object_id)
            used_centroids.add(index)

        # 3. Update matched objects, remember missing ones, register new ones.
        for index, object_id in enumerate(ids):
            if object_id is None:
                object_id = self.next_id
                self.next_id += 1
                ids[index] = object_id
                self.trails[object_id] = deque(maxlen=self.trail_length)
            self.positions[object_id] = centroids[index]
            self.missing[object_id] = 0
            self.trails[object_id].append(centroids[index])

        for object_id in list(self.positions):
            if object_id not in used_objects and object_id not in ids:
                self.missing[object_id] += 1
                if self.missing[object_id] > self.max_missing:   # forget old objects
                    del self.positions[object_id]
                    del self.missing[object_id]
                    del self.trails[object_id]

        return ids
