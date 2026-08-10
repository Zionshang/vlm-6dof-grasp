from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import ollama

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: Union[str, Dict[str, str]], images: Optional[List[Union[str, bytes]]] = None, **kwargs) -> str:
        pass

class OllamaClient(LLMClient):
    def __init__(self, model_name: str, host: str = None,
                 num_ctx: int = 4096, keep_alive: Union[str, int] = 0):
        self.model_name = model_name
        self.host = host
        self.num_ctx = int(num_ctx)
        self.keep_alive = keep_alive
        self.client = ollama.Client(host=host) if host else ollama.Client()

    def check(self):
        """Fail fast when the configured Ollama service/model is unavailable."""
        response = self.client.list()
        models = (response.get('models', []) if hasattr(response, 'get')
                  else getattr(response, 'models', []))
        names = {
            (m.get('model') or m.get('name')) if isinstance(m, dict)
            else getattr(m, 'model', None) or getattr(m, 'name', None)
            for m in models
        }
        if self.model_name not in names:
            raise RuntimeError(
                f"Ollama model '{self.model_name}' is unavailable at "
                f"{self.host or 'http://127.0.0.1:11434'}; available={sorted(n for n in names if n)}"
            )

    def generate(self, prompt: Union[str, Dict[str, str]], images: Optional[List[Union[str, bytes]]] = None, **kwargs) -> str:
        options = dict(kwargs.pop('options', {}) or {})
        options.setdefault('num_ctx', self.num_ctx)
        kwargs['options'] = options
        if 'keep_alive' not in kwargs:
            # The grasp pipeline shares one GPU with several PyTorch models.
            # Do not leave Ollama's KV cache/model resident after detection.
            kwargs['keep_alive'] = self.keep_alive

        messages = []
        
        if isinstance(prompt, str):
            # Legacy/Simple mode
            msg = {'role': 'user', 'content': prompt}
            if images:
                msg['images'] = images
            messages.append(msg)
        elif isinstance(prompt, dict):
            # Structural Prompt Mode (System + User)
            if 'system' in prompt:
                messages.append({'role': 'system', 'content': prompt['system']})
            
            if 'user' in prompt:
                user_msg = {'role': 'user', 'content': prompt['user']}
                if images:
                    user_msg['images'] = images
                messages.append(user_msg)
            else:
                # Fallback if no user key, try to use values as content
                pass
        
        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return response.get('message', {}).get('content', '')

    def warmup(self):
        """Sends a dummy request to load the model into memory."""
        try:
            print(f"[Ollama] Warming up model {self.model_name}...")
            # Use a very simple prompt with no images just to load weights
            self.generate("Hello", keep_alive=self.keep_alive)
            print("[Ollama] Model warmed up.")
        except Exception as e:
            print(f"[Ollama] Warmup failed (Non-fatal): {e}")

    def unload(self):
        """Unloads the model from memory to free VRAM."""
        try:
            # Directly use ollama lib for control commands to force unload
            self.client.generate(model=self.model_name, prompt="", keep_alive=0)
            print(f"[Ollama] Unloaded model {self.model_name}")
        except Exception as e:
            print(f"[Ollama] Failed to unload {self.model_name}: {e}")
