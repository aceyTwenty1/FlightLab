"""
Waypoint Guidance and Navigation
=================================

Provides waypoint-following guidance for autonomous flight.

The navigator generates heading, altitude, and airspeed commands
for the PID controller to track.

Features:
- Sequential waypoint tracking
- Cross-track error computation
- Turn anticipation
- Acceptance radius for waypoint switching
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Waypoint:
    x: float  # North (m)
    y: float  # East (m)
    altitude: float  # Altitude (m)
    airspeed: float = 20.0  # Target airspeed (m/s)
    name: str = ""

    def to_array(self):
        return np.array([self.x, self.y, self.altitude])


@dataclass
class GuidanceCommand:
    heading: float
    altitude: float
    airspeed: float
    cross_track_error: float = 0.0
    distance_to_waypoint: float = 0.0
    waypoint_index: int = 0


class WaypointNavigator:
    def __init__(
        self,
        waypoints: List[Waypoint],
        acceptance_radius: float = 15.0,
        lookahead_distance: float = 30.0,
    ):
        self.waypoints = waypoints
        self.acceptance_radius = acceptance_radius
        self.lookahead_distance = lookahead_distance
        self.current_index = 0

    @classmethod
    def from_tuples(cls, wp_list: List[Tuple[float, float, float]], **kwargs):
        waypoints = [Waypoint(x=w[0], y=w[1], altitude=w[2]) for w in wp_list]
        return cls(waypoints, **kwargs)

    def reset(self):
        self.current_index = 0

    @property
    def finished(self):
        return self.current_index >= len(self.waypoints)

    def update(self, position: np.ndarray, heading: float) -> GuidanceCommand:
        if self.finished:
            wp = self.waypoints[-1]
            return GuidanceCommand(
                heading=heading,
                altitude=wp.altitude,
                airspeed=wp.airspeed,
                waypoint_index=self.current_index,
            )

        wp = self.waypoints[self.current_index]
        dx = wp.x - position[0]
        dy = wp.y - position[1]
        dist = math.sqrt(dx**2 + dy**2)

        if dist < self.acceptance_radius:
            self.current_index += 1
            if self.finished:
                return GuidanceCommand(
                    heading=heading,
                    altitude=self.waypoints[-1].altitude,
                    airspeed=self.waypoints[-1].airspeed,
                    distance_to_waypoint=0.0,
                    waypoint_index=self.current_index,
                )
            wp = self.waypoints[self.current_index]
            dx = wp.x - position[0]
            dy = wp.y - position[1]
            dist = math.sqrt(dx**2 + dy**2)

        desired_heading = math.atan2(dy, dx)
        cte = -math.sin(heading) * (wp.x - position[0]) + math.cos(heading) * (wp.y - position[1])

        return GuidanceCommand(
            heading=desired_heading,
            altitude=wp.altitude,
            airspeed=wp.airspeed,
            cross_track_error=cte,
            distance_to_waypoint=dist,
            waypoint_index=self.current_index,
        )
