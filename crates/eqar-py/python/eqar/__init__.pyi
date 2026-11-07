from typing import Any, Dict as _Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__version__: str
MAX_WIDTH: int
MIN_WIDTH: int
DICTIONARIES: Tuple[str, ...]

ACCEPT: int
R_NO_CANDIDATE: int
R_BRIDGE_FAILED: int
R_DEGENERATE_INIT: int
R_INIT_GEOMETRY: int
R_EDGE_FIT: int
R_DISPLACEMENT: int
R_DEGENERATE_CORNER: int
R_LOW_CONTRAST: int
R_HIGH_RESID: int
R_BORDER_NOT_BLACK: int
R_IDENTIFY_FAILED: int
R_ID_NOT_EXPECTED: int

FLAG_SCAN: int
FLAG_TRACK: int
FLAG_LOW_CONTRAST: int
FLAG_HIGH_RESID: int
FLAG_POLE_CAP: int
FLAG_SEAM: int
FLAG_SECOND_SCALE: int
FLAG_LEAN: int
FLAG_COASTED: int

PROV_LEGACY: int
PROV_LINEFIT: int
PROV_RELAXED: int

class Marker:
    """One detected marker (immutable). Array attributes return fresh copies."""

    @property
    def id(self) -> int: ...
    @property
    def corners(self) -> NDArray[np.float32]:
        """(4, 2) corner pixels, TL, TR, BR, BL of the marker's own frame."""
    @property
    def rays(self) -> NDArray[np.float32]:
        """(4, 3) unit ray of each corner."""
    @property
    def center(self) -> NDArray[np.float32]:
        """(2,) centre pixel."""
    @property
    def center_ray(self) -> NDArray[np.float32]:
        """(3,) unit ray of the centre."""
    @property
    def corner_valid(self) -> NDArray[np.bool_]: ...
    @property
    def corner_flags(self) -> NDArray[np.uint8]: ...
    @property
    def corner_sigma_scale(self) -> NDArray[np.float32]: ...
    @property
    def quality(self) -> float: ...
    @property
    def flags(self) -> int: ...
    @property
    def side_px(self) -> float: ...
    @property
    def edge_resid_px(self) -> float: ...
    @property
    def contrast_dn(self) -> float: ...
    @property
    def min_cross(self) -> float: ...
    @property
    def reject_code(self) -> int: ...
    @property
    def prov(self) -> int: ...
    @property
    def n_bit_errors(self) -> int: ...
    @property
    def border_fail(self) -> int: ...
    @property
    def n_erased(self) -> int: ...
    def to_dict(self) -> _Dict[str, Any]: ...

class Prior:
    """A known marker and where it is expected, for ``Detector.verify``."""

    def __init__(self, id: int, rays: ArrayLike, sigma_deg: float = 1.0) -> None: ...
    @staticmethod
    def from_marker(marker: Marker, sigma_deg: float = 1.0) -> Prior: ...
    @property
    def id(self) -> int: ...
    @property
    def rays(self) -> NDArray[np.float32]: ...
    @property
    def sigma_deg(self) -> float: ...

class Detector:
    """A detector bound to one image size (``width == 2 * height``) and parameters."""

    def __init__(
        self,
        width: int,
        height: int,
        dictionary: str = "4x4_50",
        *,
        quality: str = "default",
        max_markers: int = 32,
        time_budget_ms: Optional[float] = 2.0,
        use_second_scale: bool = True,
        reject_high_resid: bool = False,
        expect_id: Optional[int] = None,
        gamma_lut: Optional[ArrayLike] = None,
        in_use_ids: Optional[Sequence[int]] = None,
        coast_max: int = 0,
    ) -> None: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def dictionary(self) -> str: ...
    @property
    def max_markers(self) -> int: ...
    def detect(self, img: NDArray[np.uint8], *, format: str = "bgr") -> List[Marker]: ...
    def track(
        self,
        img: NDArray[np.uint8],
        markers: Sequence[Marker],
        frame_gap: int = 1,
        *,
        format: str = "bgr",
    ) -> List[Marker]: ...
    def verify(
        self, img: NDArray[np.uint8], priors: Sequence[Prior], *, format: str = "bgr"
    ) -> List[Marker]: ...
    def reset(self) -> None: ...
    def proposals(self, img: NDArray[np.uint8], *, format: str = "bgr") -> NDArray[np.float32]:
        """(n, 4) rows ``[rx, ry, rz, angular_radius]`` (diagnostic)."""
    @property
    def last_timings(self) -> _Dict[str, float]:
        """ms per stage: propose, bridge, decode, refine, gate, total."""
    @property
    def last_stats(self) -> _Dict[str, Any]: ...

def detect(
    img: NDArray[np.uint8],
    dictionary: str = "4x4_50",
    *,
    format: str = "bgr",
    quality: str = "default",
    max_markers: int = 32,
    time_budget_ms: Optional[float] = None,
    use_second_scale: bool = True,
    reject_high_resid: bool = False,
    expect_id: Optional[int] = None,
    gamma_lut: Optional[ArrayLike] = None,
    in_use_ids: Optional[Sequence[int]] = None,
) -> List[Marker]: ...
def generate_marker(id: int, dictionary: str = "4x4_50", size: int = 400) -> NDArray[np.uint8]: ...
