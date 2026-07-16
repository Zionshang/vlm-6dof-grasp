"""FastSAM 分割器组件(复用 perception.segmenters.FastSAMSegmenter)。"""
from registry import register


@register("segmenter", "fastsam")
def build_fastsam_segmenter(ctx=None, cfg=None, hw=None, manager=None, **kw):
    from perception.segmenters import FastSAMSegmenter
    import paths
    ROOT = paths.PROJECT_ROOT
    scfg = cfg or {}
    return FastSAMSegmenter(
        str(ROOT / scfg.get("weights", "fastsam/weight/FastSAM-s.pt")),
        use_sam=scfg.get("use_sam", True),
    )
