"""Synthetic defect generators. Every generator paints into masks from the shared MaskBank."""

GENERATORS = ("copypaste", "rgcb", "defectfill", "magic")


def build_generator(name: str, cfg: dict, category: str):
    cfg = dict(cfg or {})
    if name == "copypaste":
        from .copypaste import CopyPasteGenerator

        return CopyPasteGenerator(**cfg)
    if name == "rgcb":
        from .rgcb import RGCBGenerator

        return RGCBGenerator(**cfg)
    if name == "defectfill":
        from .sd_generators import DefectFillGenerator

        return DefectFillGenerator(category=category, **cfg)
    if name == "magic":
        from .sd_generators import MagicGenerator

        return MagicGenerator(category=category, **cfg)
    raise ValueError(f"Unknown generator {name!r}; choose from {GENERATORS}")
