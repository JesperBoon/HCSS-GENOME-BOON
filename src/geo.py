"""Small geography helpers for the support-chain map: where a country sits and how to draw a directed arc."""

import numpy as np


def _ring_centroid(ring):
    """Area-weighted centroid and area of one polygon ring (planar approximation, fine for a map label)."""
    pts = np.asarray(ring, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2
    if abs(area) < 1e-12:
        return float(x.mean()), float(y.mean()), 0.0
    cx = ((x + x1) * cross).sum() / (6 * area)
    cy = ((y + y1) * cross).sum() / (6 * area)
    return float(cx), float(cy), abs(float(area))


def country_centroids(geojson: dict, key: str = "ISO_A3_EH") -> dict:
    """{iso3: (lat, lon)}: the centroid of each country's largest polygon (so France sits in Europe, not in
    the Atlantic between Paris and French Guiana)."""
    out = {}
    for feat in geojson["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        best = max((_ring_centroid(p[0]) for p in polys), key=lambda c: c[2])
        out[feat["properties"][key]] = (best[1], best[0])
    return out


def arc(lat1: float, lon1: float, lat2: float, lon2: float, n: int = 24, bend: float = 0.18):
    """A curved path from point 1 to point 2 (quadratic curve in lon/lat). The curve bows to the left of the
    direction of travel, so A -> B and B -> A are drawn as two different, opposite curves and direction can
    be read from the picture. Returns (lats, lons)."""
    t = np.linspace(0, 1, n)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    mid_lat, mid_lon = (lat1 + lat2) / 2 - bend * dlon, (lon1 + lon2) / 2 + bend * dlat
    lats = (1 - t) ** 2 * lat1 + 2 * (1 - t) * t * mid_lat + t ** 2 * lat2
    lons = (1 - t) ** 2 * lon1 + 2 * (1 - t) * t * mid_lon + t ** 2 * lon2
    return lats, lons
