import yaml
from pathlib import Path
from typing import Dict, Any

class PromptManager:
    def __init__(self, prompt_dir: str):
        self.prompt_dir = Path(prompt_dir)
        self.templates: Dict[str, Dict[str, Any]] = {}
        self._load_prompts()

    def _load_prompts(self):
        if not self.prompt_dir.exists():
            return
            
        for file_path in self.prompt_dir.glob("*.yaml"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if data:
                        self.templates.update(data)
            except Exception as e:
                print(f"Error loading prompt file {file_path}: {e}")

    def get_template(self, template_name: str) -> Any:
        """Get raw template (str or dict) by name (e.g. 'detection.v1')"""
        parts = template_name.split('.')
        current = self.templates
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise ValueError(f"Template '{template_name}' not found.")
        
        return current

    def format_prompt(self, template_name: str, **kwargs) -> Any:
        template = self.get_template(template_name)
        try:
            if isinstance(template, str):
                return template.format(**kwargs)
            elif isinstance(template, dict):
                # Recursively format strings in dictionary
                formatted = {}
                for k, v in template.items():
                    if isinstance(v, str):
                        formatted[k] = v.format(**kwargs)
                    else:
                        formatted[k] = v
                return formatted
            else:
                return template
        except KeyError as e:
            raise ValueError(f"Missing variable for template '{template_name}': {e}")
