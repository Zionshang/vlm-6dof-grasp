"""VLM candidate selector component."""
from perception_errors import PerceptionEmptyError
from registry import register
from saver import save_2d_grasp

from .image_candidates import prepare_image_candidates


class VLMSelector:
    def __init__(self, model_name, prompts_dir, top_k=8):
        from vlm.src.apps.grasp_selection import GraspSelectionApp

        self.app = GraspSelectionApp(
            model_name=model_name, prompts_dir=prompts_dir,
        )
        self.top_k = int(top_k)

    def select(self, color, grasps, intrinsic, output_dir, mask=None):
        del mask
        images, candidates = prepare_image_candidates(
            color, grasps, intrinsic, output_dir, self.top_k,
        )
        if not candidates:
            raise PerceptionEmptyError("二维几何筛选无候选")
        response = self.app.run(save_2d_grasp(output_dir, images))
        selected = int(response.get("selected_id", 0))
        return candidates[selected] if 0 <= selected < len(candidates) else candidates[0]


@register("candidate_selector", "vlm")
def build_vlm_selector(cfg=None, hw=None, ctx=None, dependencies=None):
    import paths

    cfg = cfg or {}
    return VLMSelector(
        cfg.get("model", "qwen3-vl:8b-instruct-q4_K_M"),
        str(paths.PROJECT_ROOT / "third_party/vlm/prompts"),
        cfg.get("top_k", 8),
    )
