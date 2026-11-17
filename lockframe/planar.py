"""Masked dense alignment and non-degenerate projective surface geometry."""
import cv2
import numpy as np

REGIONAL_NCC_THRESHOLD = .55
UNIT_CORNERS = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])


def project(points, matrix):
    points = np.asarray(points, np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return homogeneous[:, :2] / homogeneous[:, 2:3]


def as_homography(matrix):
    matrix = np.asarray(matrix, np.float64)
    return np.vstack([matrix, [0., 0., 1.]]) if matrix.shape == (2, 3) else matrix.copy()


def valid_surface(matrix, size, frame_shape=None):
    """Reject horizon crossings, reflections, collapses and extreme extrapolation."""
    matrix = as_homography(matrix)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-9:
        return False
    matrix /= matrix[2, 2]
    points = UNIT_CORNERS * size
    denominators = np.column_stack([points, np.ones(4)]) @ matrix[2]
    if np.any(denominators <= .05) or denominators.max() / denominators.min() > 8:
        return False
    corners = project(points, matrix)
    if not np.isfinite(corners).all():
        return False
    contour = corners.astype(np.float32)
    area = cv2.contourArea(contour, oriented=True)
    if not cv2.isContourConvex(contour) or not .08 * np.prod(size) <= area <= 12 * np.prod(size):
        return False
    edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    if edges.min() < 5 or edges.max() / edges.min() > 15:
        return False
    if frame_shape is not None:
        height, width = frame_shape
        if np.any(corners < 0) or np.any(corners > [width, height]):
            return False
    return True


def shifted(matrix, delta):
    translation = np.eye(3)
    translation[:2, 2] = delta
    return translation @ as_homography(matrix)


def template_pose(matrix, size, factor=1., angle=0.):
    """Keep all eight projective parameters while perturbing the local plane."""
    delta = cv2.getRotationMatrix2D(tuple(np.asarray(size, float) / 2), angle, factor)
    candidate = as_homography(matrix) @ as_homography(delta)
    if not valid_surface(candidate, size):
        return None
    corners = project(UNIT_CORNERS * size, candidate)
    low, high = np.floor(corners.min(axis=0)), np.ceil(corners.max(axis=0))
    return shifted(candidate, -low), low, (high - low).astype(int), candidate


def dense_refine(reference, gray, matrix):
    """Refine a measured pose using pixels throughout the selected mask."""
    mask = reference.mask.copy()
    mask[:2] = mask[-2:] = 0
    mask[:, :2] = mask[:, -2:] = 0
    if np.count_nonzero(mask) < 32:
        return None
    def normalized(image):
        values = image.astype(np.float32)
        inside = values[mask > 0]
        values = (values - float(inside.mean())) / max(float(inside.std()), 8.)
        return reference.masked_image(values)
    target = normalized(reference.image)
    image = normalized(reference.warp(gray, matrix))
    motion = cv2.MOTION_HOMOGRAPHY if reference.perspective else cv2.MOTION_AFFINE
    residual = np.eye(3 if reference.perspective else 2, 3, dtype=np.float32)
    try:
        score, residual = cv2.findTransformECC(target, image, residual, motion,
                                              (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-5), mask,
                                              1 if reference._shape_masked else 3)
    except cv2.error:
        return None
    if not np.isfinite(score) or score < .55 or not np.isfinite(residual).all():
        return None
    if not reference.perspective:
        u, singular, vt = np.linalg.svd(residual[:, :2])
        rotation = u @ vt
        if np.linalg.det(rotation) <= 0 or singular.min() < .5 or singular.max() / singular.min() > 1.16:
            return None
        linear = rotation * singular.mean()
        residual[:, 2] += (residual[:, :2] - linear) @ reference.center
        residual[:, :2] = linear
    points = UNIT_CORNERS * [reference.width, reference.height]
    displacement = np.linalg.norm(project(points, as_homography(residual)) - points, axis=1)
    if displacement.max() > max(8., min(reference.width, reference.height) * .25):
        return None
    result = as_homography(matrix) @ as_homography(residual)
    return result if reference.perspective else result[:2]
