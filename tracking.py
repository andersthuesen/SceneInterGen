import sys
import torch
from collections import OrderedDict

from typing import List, Dict, TypeVar, Generic, Callable, OrderedDict
from common.data_types import TransposedTracks

TrackID = str
K = TypeVar("K")
T = TypeVar("T")

Tracks = OrderedDict[K, Dict[TrackID, T]]


class Tracker(Generic[K, T]):
    def __init__(self, cost_fn: Callable[[T, T], float]):
        self.cost_fn = cost_fn

    def track(self, untracked: OrderedDict[K, List[T]]) -> Tracks[K, T]:
        tracks = OrderedDict()
        latest_track = {}

        for frame_id, objs in untracked.items():
            track = {}

            # If we have no tracks yet, create one for each object.
            if len(latest_track) == 0:
                track = {str(i): obj for i, obj in enumerate(objs)}
            elif len(objs) > 0:
                # If we have tracks, we need to match the objects to the tracks.
                # We do this by computing the cost of matching each object to each track.
                costs = torch.zeros(len(objs), len(latest_track))
                latest_track_items = list(enumerate(latest_track.items()))
                for i, obj in enumerate(objs):
                    for j, (_, tracked_obj) in latest_track_items:
                        costs[i, j] = self.cost_fn(obj, tracked_obj)

                # Now we need to find the best matching between the objects and the tracks.
                # We do this by finding the minimum cost matching between the objects and the tracks.
                # We can do this using a naive greedy algorithm:
                while torch.min(costs) < torch.inf:
                    best_match = torch.argmin(costs)
                    i, j = best_match // len(latest_track), best_match % len(
                        latest_track
                    )

                    _, (obj_id, _) = latest_track_items[j]
                    if obj_id in track:
                        # Create new track
                        track[str(len(track))] = objs[i]
                    else:
                        track[obj_id] = objs[i]

                    # Remove the matched object by setting its cost to infinity.
                    costs[i, :] = torch.inf

                # Update the latest track with the new matches.
            latest_track.update(track)
            # Add the new track to the tracks.
            tracks[frame_id] = track

        return tracks

    @staticmethod
    def transpose_tracks(tracks: Tracks) -> TransposedTracks:
        transposed = {}
        for frame_idx, frame in tracks.items():
            for obj_id, obj in frame.items():
                if obj_id not in transposed:
                    transposed[obj_id] = {}
                transposed[obj_id][frame_idx] = obj

        return transposed


if __name__ == "__main__":

    def dist(a, b):
        return (a - b) ** 2

    untracked = OrderedDict([(0, [1]), (1, [2, 3]), (2, [2]), (3, [1, 4])])

    tracker = Tracker(cost_fn=dist)
    tracks = tracker.track(untracked)

    print("untracked", untracked)
    print("tracked", tracks)
