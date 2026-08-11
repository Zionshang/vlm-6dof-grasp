"""Local web dashboard component."""
from registry import register


@register("dashboard", "web")
def build_web_dashboard(cfg=None, hw=None, ctx=None, dependencies=None):
    from web_dashboard import WebDashboard
    import paths

    cfg = cfg or {}
    output_dir = paths.PROJECT_ROOT / cfg.get("output_dir", "output/piper_run_test")
    return WebDashboard(
        output_dir, host=cfg.get("host", "127.0.0.1"),
        port=cfg.get("port", 8765),
        open_browser=cfg.get("open_browser", True),
        max_points=cfg.get("max_points", 12000),
    )
