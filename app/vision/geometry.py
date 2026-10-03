from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.vision.yolo_models import BoxDetection, SegDetection


@dataclass
class SpatialDecision:
    status: str
    confidence: float
    reason: str
    details: dict


def vehicle_footprint(vehicle: SegDetection, shape: tuple[int, int]) -> np.ndarray:
    """Visible road-contact area from the lower portion of the vehicle mask."""
    h, w = shape
    x1, y1, x2, y2 = vehicle.bbox
    cutoff = int(y1 + 0.62 * max(y2 - y1, 1))

    fp = vehicle.mask.copy().astype(np.uint8)
    fp[:max(0, cutoff), :] = 0

    kernel = np.ones((5, 5), np.uint8)
    fp = cv2.erode(fp, kernel, iterations=1)

    return fp.astype(bool)


def projected_vehicle_ground_mask(
    vehicle: SegDetection,
    shape: tuple[int, int],
    *,
    core: bool = False,
) -> np.ndarray:
    """
    Approximate the ground area hidden underneath the vehicle.

    A white parking line may disappear under the car because it is occluded,
    so direct pixel overlap between the car mask and line mask is not enough.
    """
    h, w = shape
    x1, y1, x2, y2 = vehicle.bbox

    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))

    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)

    if core:
        left = int(round(x1 + 0.20 * bw))
        right = int(round(x2 - 0.20 * bw))
        top = int(round(y1 + 0.54 * bh))
    else:
        left = int(round(x1 + 0.06 * bw))
        right = int(round(x2 - 0.06 * bw))
        top = int(round(y1 + 0.48 * bh))

    bottom = int(round(y2 + 0.04 * bh))

    left = max(0, min(w - 1, left))
    right = max(0, min(w - 1, right))
    top = max(0, min(h - 1, top))
    bottom = max(0, min(h - 1, bottom))

    mask = np.zeros((h, w), dtype=np.uint8)

    if right > left and bottom > top:
        cv2.rectangle(
            mask,
            (left, top),
            (right, bottom),
            1,
            thickness=-1,
        )

    return mask.astype(bool)


def mask_overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(int(a.sum()), 1)
    return float(np.logical_and(a, b).sum() / denom)


def mask_min_distance(a: np.ndarray, b: np.ndarray) -> float:
    if not a.any() or not b.any():
        return float("inf")

    inv = (~b).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    return float(dist[a].min()) if a.any() else float("inf")


def fit_extended_boundary_line(
    mask: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[float, float, float]] | None:
    """
    Fit a straight line through visible boundary pixels and extend it.

    Returns:
        line_mask
        line equation (a, b, c) where ax + by + c = 0
    """
    h, w = shape

    ys, xs = np.nonzero(mask)

    if len(xs) < 20:
        return None

    if len(xs) > 4000:
        idx = np.linspace(
            0,
            len(xs) - 1,
            4000,
        ).astype(int)

        xs = xs[idx]
        ys = ys[idx]

    pts = np.column_stack([xs, ys]).astype(np.float32)

    vx, vy, x0, y0 = cv2.fitLine(
        pts,
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    ).reshape(-1)

    norm = math.hypot(float(vx), float(vy))

    if norm < 1e-6:
        return None

    vx = float(vx) / norm
    vy = float(vy) / norm
    x0 = float(x0)
    y0 = float(y0)

    # ax + by + c = 0
    a = -vy
    b = vx
    c = vy * x0 - vx * y0

    span = int(round(2.0 * math.hypot(w, h)))

    p1 = (
        int(round(x0 - span * vx)),
        int(round(y0 - span * vy)),
    )

    p2 = (
        int(round(x0 + span * vx)),
        int(round(y0 + span * vy)),
    )

    line_mask = np.zeros((h, w), dtype=np.uint8)

    thickness = max(
        4,
        int(round(0.005 * math.hypot(w, h))),
    )

    cv2.line(
        line_mask,
        p1,
        p2,
        1,
        thickness=thickness,
    )

    return line_mask.astype(bool), (a, b, c)


def white_line_center_entry(
    line_mask: np.ndarray,
    vehicle: SegDetection,
    shape: tuple[int, int],
) -> tuple[bool, float]:
    """
    Detect a visible white line approaching the vehicle through its
    lower central region.

    This catches cases where the line disappears underneath the car
    due to occlusion and global line fitting is unreliable.
    """
    h, w = shape
    x1, y1, x2, y2 = vehicle.bbox

    bw = max(float(x2 - x1), 1.0)
    bh = max(float(y2 - y1), 1.0)

    # Central half of the vehicle width.
    left = int(max(0, x1 + 0.25 * bw))
    right = int(min(w - 1, x2 - 0.25 * bw))

    # Only inspect the lower/contact region of the vehicle.
    top = int(max(0, y2 - 0.18 * bh))
    bottom = int(min(h - 1, y2 + 0.06 * bh))

    zone = np.zeros((h, w), dtype=bool)

    if right > left and bottom > top:
        zone[top:bottom + 1, left:right + 1] = True

    distance = mask_min_distance(zone, line_mask)

    diag = math.hypot(w, h)

    # Allow for the white line disappearing slightly before the
    # visible car mask because the vehicle occludes it.
    threshold = max(10.0, 0.018 * diag)

    return distance <= threshold, distance


def point_line_distance(
    point: tuple[float, float],
    line: tuple[float, float, float],
) -> float:
    x, y = point
    a, b, c = line

    denom = max(math.hypot(a, b), 1e-6)

    return abs(a * x + b * y + c) / denom


def two_white_line_corridor(
    white_lines: list[SegDetection],
    vehicle: SegDetection,
    shape: tuple[int, int],
) -> tuple[bool, float, dict]:
    """
    Determine whether two white parking boundaries form a valid bay
    containing the vehicle.

    Positive only when:
    - two reasonably parallel white boundaries are available,
    - the vehicle ground center lies between them,
    - neither projected boundary crosses the central vehicle footprint.
    """
    if len(white_lines) < 2:
        return False, 0.0, {}

    x1, y1, x2, y2 = vehicle.bbox

    vehicle_w = max(float(x2 - x1), 1.0)
    vehicle_h = max(float(y2 - y1), 1.0)

    center = (
        (x1 + x2) / 2.0,
        y1 + 0.76 * vehicle_h,
    )

    core = projected_vehicle_ground_mask(
        vehicle,
        shape,
        core=True,
    )

    fitted = []

    for det in white_lines:
        result = fit_extended_boundary_line(
            det.mask,
            shape,
        )

        if result is None:
            continue

        line_mask, equation = result

        fitted.append(
            (
                det,
                line_mask,
                equation,
            )
        )

    if len(fitted) < 2:
        return False, 0.0, {}

    best = None

    for i in range(len(fitted)):
        for j in range(i + 1, len(fitted)):
            det1, mask1, eq1 = fitted[i]
            det2, mask2, eq2 = fitted[j]

            a1, b1, c1 = eq1
            a2, b2, c2 = eq2

            # Make both normals point in approximately the same direction.
            dot = a1 * a2 + b1 * b2

            if dot < 0:
                a2 = -a2
                b2 = -b2
                c2 = -c2
                dot = -dot

            # Require approximately parallel parking boundaries.
            if dot < 0.90:
                continue

            s1 = (
                a1 * center[0]
                + b1 * center[1]
                + c1
            )

            s2 = (
                a2 * center[0]
                + b2 * center[1]
                + c2
            )

            # Center must lie between the two parallel boundaries.
            between = s1 * s2 < 0

            if not between:
                continue

            separation = abs(s1 - s2)

            # Reject implausibly narrow or extremely wide "bays".
            if not (
                0.35 * vehicle_w
                <= separation
                <= 2.5 * vehicle_w
            ):
                continue

            crosses_1 = bool(
                np.logical_and(
                    mask1,
                    core,
                ).any()
            )

            crosses_2 = bool(
                np.logical_and(
                    mask2,
                    core,
                ).any()
            )

            # If either boundary passes through the vehicle core,
            # this is not evidence of correct parking.
            if crosses_1 or crosses_2:
                continue

            min_conf = min(
                det1.confidence,
                det2.confidence,
            )

            confidence = min(
                0.96,
                0.76 + 0.18 * min_conf,
            )

            details = {
                "class": "white_line_corridor",
                "between_lines": True,
                "line_parallel_score": round(dot, 3),
                "line_separation_px": round(
                    separation,
                    1,
                ),
                "crosses_line_1": crosses_1,
                "crosses_line_2": crosses_2,
            }

            if best is None or confidence > best[1]:
                best = (
                    True,
                    confidence,
                    details,
                )

    if best is not None:
        return best

    return False, 0.0, {}


def cat_eye_line_mask(
    cat_eyes: list[BoxDetection],
    shape: tuple[int, int],
    min_points: int = 3,
) -> np.ndarray | None:
    h, w = shape

    if len(cat_eyes) < min_points:
        return None

    pts = np.array(
        [
            [
                (b.bbox[0] + b.bbox[2]) / 2.0,
                (b.bbox[1] + b.bbox[3]) / 2.0,
            ]
            for b in cat_eyes
        ],
        dtype=np.float32,
    )

    if len(pts) < min_points:
        return None

    best: tuple[int, np.ndarray] | None = None

    threshold = max(
        8.0,
        0.012 * math.hypot(w, h),
    )

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            p1, p2 = pts[i], pts[j]

            v = p2 - p1
            norm = float(np.linalg.norm(v))

            if norm < 1e-6:
                continue

            dists = np.abs(
                np.cross(v, pts - p1) / norm
            )

            inliers = dists <= threshold
            count = int(inliers.sum())

            if best is None or count > best[0]:
                best = (count, pts[inliers])

    if best is None or best[0] < min_points:
        return None

    inlier_pts = best[1]

    vx, vy, x0, y0 = cv2.fitLine(
        inlier_pts,
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    ).flatten()

    if abs(vx) < 1e-6:
        p1 = (int(x0), 0)
        p2 = (int(x0), h - 1)
    else:
        y_left = int(
            round(
                y0
                + (0 - x0)
                * vy
                / vx
            )
        )

        y_right = int(
            round(
                y0
                + ((w - 1) - x0)
                * vy
                / vx
            )
        )

        p1 = (0, y_left)
        p2 = (w - 1, y_right)

    mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    thickness = max(
        8,
        int(
            round(
                0.012
                * math.hypot(w, h)
            )
        ),
    )

    cv2.line(
        mask,
        p1,
        p2,
        1,
        thickness=thickness,
    )

    return mask.astype(bool)


def ground_visibility(
    vehicle: SegDetection | None,
    shape: tuple[int, int],
) -> float:
    h, w = shape

    roi = np.zeros(
        (h, w),
        dtype=bool,
    )

    roi[int(h * 0.55):, :] = True

    if vehicle is not None:
        roi &= ~vehicle.mask

    return float(
        roi.sum()
        / max(
            int(h * w * 0.45),
            1,
        )
    )


def decide_q2(
    vehicle: SegDetection | None,
    boundaries: list[SegDetection],
    cat_eyes: list[BoxDetection],
    shape: tuple[int, int],
) -> SpatialDecision:

    if vehicle is None:
        return SpatialDecision(
            "unclear",
            0.35,
            "Target vehicle could not be localized reliably",
            {},
        )

    fp = vehicle_footprint(
        vehicle,
        shape,
    )

    projected_fp = projected_vehicle_ground_mask(
        vehicle,
        shape,
        core=False,
    )

    projected_core = projected_vehicle_ground_mask(
        vehicle,
        shape,
        core=True,
    )

    diag = math.hypot(
        shape[1],
        shape[0],
    )

    near_px = max(
        12.0,
        0.018 * diag,
    )

    x1, y1, x2, y2 = vehicle.bbox

    vehicle_w = max(
        float(x2 - x1),
        1.0,
    )

    vehicle_h = max(
        float(y2 - y1),
        1.0,
    )

    ground_center = (
        (x1 + x2) / 2.0,
        y1 + 0.78 * vehicle_h,
    )

    evidence = []

    positive_conf = 0.0
    negative_conf = 0.0

    negative_reason = None
    positive_reason = None

    for det in boundaries:

        overlap = mask_overlap_ratio(
            fp,
            det.mask,
        )

        distance = mask_min_distance(
            fp,
            det.mask,
        )

        item = {
            "class": det.class_name,
            "conf": round(
                det.confidence,
                3,
            ),
            "overlap": round(
                overlap,
                4,
            ),
            "distance_px": round(
                distance,
                1,
            ),
        }

        if det.class_name in {
            "forbidden_curb",
            "hatched_area",
        }:

            if (
                overlap >= 0.008
                or distance <= near_px
            ):
                conf = min(
                    0.99,
                    0.70
                    + 0.25
                    * det.confidence,
                )

                if conf > negative_conf:
                    negative_conf = conf
                    negative_reason = (
                        "Vehicle is on or adjacent to a prohibited "
                        f"{det.class_name.replace('_', ' ')}"
                    )

        elif det.class_name == "white_line":

            center_entry, center_entry_distance = white_line_center_entry(
                det.mask,
                vehicle,
                shape,
            )

            item["center_entry_distance_px"] = round(
                center_entry_distance,
                1,
            )
            item["center_entry"] = center_entry

            fitted = fit_extended_boundary_line(
                det.mask,
                shape,
            )

            projected_distance = mask_min_distance(
                projected_fp,
                det.mask,
            )

            item[
                "projected_distance_px"
            ] = round(
                projected_distance,
                1,
            )

            fitted_crosses_core = False
            center_distance = float("inf")

            if fitted is not None:

                fitted_mask, line_eq = fitted

                fitted_crosses_core = bool(
                    np.logical_and(
                        fitted_mask,
                        projected_core,
                    ).any()
                )

                center_distance = point_line_distance(
                    ground_center,
                    line_eq,
                )

                item[
                    "fitted_crosses_vehicle_core"
                ] = fitted_crosses_core

                item[
                    "fitted_center_distance_px"
                ] = round(
                    center_distance,
                    1,
                )

            support_is_near = (
                projected_distance
                <= near_px * 2.4
            )

            center_is_deep = (
                center_distance
                <= 0.28
                * min(
                    vehicle_w,
                    vehicle_h,
                )
            )

            # Strong direct evidence:
            # the visible white line approaches the vehicle through
            # the central lower/contact region.
            if center_entry:

                conf = min(
                    0.98,
                    0.86
                    + 0.12 * det.confidence,
                )

                if conf > negative_conf:
                    negative_conf = conf
                    negative_reason = (
                        "White parking boundary enters the central "
                        "vehicle footprint and continues underneath it"
                    )

            # Secondary evidence:
            # extrapolate the visible line underneath the vehicle.
            elif (
                fitted_crosses_core
                and support_is_near
                and center_is_deep
            ):

                conf = min(
                    0.98,
                    0.80
                    + 0.16
                    * det.confidence,
                )

                if conf > negative_conf:
                    negative_conf = conf
                    negative_reason = (
                        "White parking boundary passes through "
                        "the projected vehicle footprint"
                    )

            elif overlap >= 0.012:

                conf = min(
                    0.99,
                    0.68
                    + 3.5
                    * min(
                        overlap,
                        0.08,
                    )
                    + 0.12
                    * det.confidence,
                )

                if conf > negative_conf:
                    negative_conf = conf
                    negative_reason = (
                        "White parking boundary crosses "
                        "the visible vehicle footprint "
                        f"(overlap={overlap:.3f})"
                    )

            elif (
                projected_distance
                <= near_px * 1.8
            ):

                conf = min(
                    0.93,
                    0.58
                    + 0.25
                    * det.confidence,
                )

                if conf > positive_conf:
                    positive_conf = conf
                    positive_reason = (
                        "Vehicle is adjacent to a visible "
                        "white parking boundary without "
                        "evidence of crossing it"
                    )

        elif det.class_name == "allowed_curb":

            # -------------------------------------------------
            # Allowed curb / wheel-stop geometry
            # -------------------------------------------------
            # A correctly parked vehicle can have a visible
            # blue/black wheel stop in front of it with a gap
            # between the stop and the visible vehicle mask.
            #
            # Therefore simple mask distance is not enough.
            # Also check whether the curb:
            #   1. overlaps the vehicle horizontally,
            #   2. is below/in front of the vehicle in the image,
            #   3. is within a reasonable perspective distance.
            # -------------------------------------------------

            ys, xs = np.nonzero(det.mask)

            curb_in_front = False
            horizontal_overlap_ratio = 0.0
            vertical_gap_ratio = float("inf")

            if len(xs) > 0:
                curb_left = float(xs.min())
                curb_right = float(xs.max())
                curb_top = float(ys.min())
                curb_bottom = float(ys.max())

                overlap_width = max(
                    0.0,
                    min(float(x2), curb_right)
                    - max(float(x1), curb_left),
                )

                horizontal_overlap_ratio = (
                    overlap_width
                    / max(vehicle_w, 1.0)
                )

                vertical_gap = (
                    curb_top
                    - float(y2)
                )

                vertical_gap_ratio = (
                    vertical_gap
                    / max(vehicle_h, 1.0)
                )

                curb_center_y = (
                    curb_top
                    + curb_bottom
                ) / 2.0

                curb_in_front = (
                    horizontal_overlap_ratio >= 0.20
                    and
                    curb_center_y
                    >= float(y1) + 0.68 * vehicle_h
                    and
                    -0.05
                    <= vertical_gap_ratio
                    <= 0.85
                )

            item["curb_horizontal_overlap"] = round(
                horizontal_overlap_ratio,
                3,
            )

            item["curb_vertical_gap_ratio"] = (
                round(vertical_gap_ratio, 3)
                if math.isfinite(vertical_gap_ratio)
                else None
            )

            item["curb_in_front"] = curb_in_front

            # Strong evidence that the vehicle actually crosses
            # the curb/wheel stop.
            if overlap >= 0.05:

                conf = min(
                    0.96,
                    0.68
                    + 0.24 * det.confidence,
                )

                if conf > negative_conf:
                    negative_conf = conf
                    negative_reason = (
                        "Vehicle footprint crosses the allowed "
                        "curb or wheel stop"
                    )

            # New rule:
            # curb is visibly in front of and aligned with the
            # vehicle -> correctly positioned behind it.
            elif curb_in_front:

                conf = min(
                    0.97,
                    0.78
                    + 0.17 * det.confidence,
                )

                if conf > positive_conf:
                    positive_conf = conf
                    positive_reason = (
                        "Vehicle is correctly positioned behind "
                        "the visible allowed curb or wheel stop"
                    )

            # Fallback for side/nearby allowed curbs.
            elif distance <= near_px * 2.8:

                conf = min(
                    0.93,
                    0.62
                    + 0.24 * det.confidence,
                )

                if conf > positive_conf:
                    positive_conf = conf
                    positive_reason = (
                        "Vehicle is correctly positioned next to "
                        "the allowed curb or wheel stop"
                    )

        evidence.append(item)

    # -------------------------------------------------
    # Two-white-line parking bay
    # -------------------------------------------------
    # A vehicle does not need an allowed curb to be
    # correctly parked. If two white boundaries form a
    # bay and the vehicle lies between them without
    # crossing either boundary, Q2 can be positive.
    white_lines = [
        det
        for det in boundaries
        if det.class_name == "white_line"
    ]

    corridor_ok, corridor_conf, corridor_details = (
        two_white_line_corridor(
            white_lines,
            vehicle,
            shape,
        )
    )

    if corridor_details:
        evidence.append(corridor_details)

    # Do not let corridor evidence cancel an already
    # observed crossing/violation.
    if (
        corridor_ok
        and negative_conf < 0.60
        and corridor_conf > positive_conf
    ):
        positive_conf = corridor_conf
        positive_reason = (
            "Vehicle is positioned inside the parking bay "
            "between two visible white boundaries"
        )

    cat_mask = cat_eye_line_mask(
        cat_eyes,
        shape,
    )

    if cat_mask is not None:

        overlap = mask_overlap_ratio(
            fp,
            cat_mask,
        )

        distance = mask_min_distance(
            fp,
            cat_mask,
        )

        evidence.append(
            {
                "class": "cat_eye_boundary",
                "overlap": round(
                    overlap,
                    4,
                ),
                "distance_px": round(
                    distance,
                    1,
                ),
                "count": len(cat_eyes),
            }
        )

        mean_conf = float(
            np.mean(
                [
                    x.confidence
                    for x in cat_eyes
                ]
            )
        )

        if overlap >= 0.012:

            conf = min(
                0.95,
                0.64
                + 0.22
                * mean_conf,
            )

            if conf > negative_conf:
                negative_conf = conf
                negative_reason = (
                    "Vehicle footprint crosses the boundary "
                    "indicated by the cat-eye row"
                )

        elif distance <= near_px * 2.0:

            conf = min(
                0.92,
                0.60
                + 0.22
                * mean_conf,
            )

            if conf > positive_conf:
                positive_conf = conf
                positive_reason = (
                    "Vehicle is positioned next to the "
                    "boundary indicated by the cat-eye row"
                )

    if (
        negative_conf >= 0.60
        and positive_conf >= 0.60
        and abs(
            negative_conf
            - positive_conf
        ) < 0.15
    ):

        return SpatialDecision(
            "unclear",
            max(
                negative_conf,
                positive_conf,
            ),
            "Conflicting spatial evidence in this view",
            {"evidence": evidence},
        )

    if (
        negative_conf >= 0.60
        and negative_conf > positive_conf
    ):

        return SpatialDecision(
            "negative",
            negative_conf,
            negative_reason
            or "Incorrect spatial relationship",
            {"evidence": evidence},
        )

    if positive_conf >= 0.55:

        return SpatialDecision(
            "positive",
            positive_conf,
            positive_reason
            or "Correct spatial relationship",
            {"evidence": evidence},
        )

    return SpatialDecision(
        "unclear",
        0.45,
        "Boundary is visible, but this view does not show "
        "a decisive vehicle-to-boundary relationship",
        {"evidence": evidence},
    )
