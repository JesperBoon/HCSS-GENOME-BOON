"""Tests for src/geo.py."""

import numpy as np
import pytest

import geo


def feature(iso, geometry):
    return {"properties": {"ISO_A3_EH": iso}, "geometry": geometry}


SQUARE = [[(0, 0), (4, 0), (4, 2), (0, 2), (0, 0)]]                 # lon 0..4, lat 0..2


def test_centroid_of_a_rectangle_is_its_middle():
    c = geo.country_centroids({"features": [feature("AAA", {"type": "Polygon", "coordinates": SQUARE})]})
    assert c["AAA"] == pytest.approx((1.0, 2.0))                    # (lat, lon)


def test_a_country_with_islands_is_placed_on_its_largest_piece():
    big = [[(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)]]
    island = [[(100, 50), (101, 50), (101, 51), (100, 51), (100, 50)]]
    geom = {"type": "MultiPolygon", "coordinates": [island, big]}
    assert geo.country_centroids({"features": [feature("BBB", geom)]})["BBB"] == pytest.approx((15.0, 15.0))


def test_arc_starts_and_ends_at_the_given_points_and_has_n_points():
    lats, lons = geo.arc(10, 20, 30, 60, n=17)
    assert len(lats) == len(lons) == 17
    assert (lats[0], lons[0]) == pytest.approx((10, 20)) and (lats[-1], lons[-1]) == pytest.approx((30, 60))


def test_opposite_directions_bow_to_opposite_sides():
    la, lo = geo.arc(0, 0, 0, 40)
    lb, lob = geo.arc(0, 40, 0, 0)
    assert la[12] < 0 < lb[12] or la[12] > 0 > lb[12]               # one bows north, the other south
    assert np.sign(la[12]) == -np.sign(lb[12])


def test_a_straight_arc_when_bend_is_zero():
    lats, lons = geo.arc(0, 0, 10, 10, bend=0.0)
    assert np.allclose(lats, lons)
