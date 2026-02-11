from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import ollama

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: Union[str, Dict[str, str]], images: Optional[List[Union[str, bytes]]] = None, **kwargs) -> str:
        pass

class OllamaClient(LLMClient):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: Union[str, Dict[str, str]], images: Optional[List[Union[str, bytes]]] = None, **kwargs) -> str:
        # Default keep_alive to 10 min if not specified
        if 'keep_alive' not in kwargs:
            kwargs['keep_alive'] = '10m'

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
        
        response = ollama.chat(
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
            self.generate("Hello", keep_alive='10m')
            print("[Ollama] Model warmed up.")
        except Exception as e:
            print(f"[Ollama] Warmup failed (Non-fatal): {e}")

    def unload(self):
        """Unloads the model from memory to free VRAM."""
        try:
            # Directly use ollama lib for control commands to force unload
            ollama.generate(model=self.model_name, prompt="", keep_alive=0)
            print(f"[Ollama] Unloaded model {self.model_name}")
        except Exception as e:
            print(f"[Ollama] Failed to unload {self.model_name}: {e}")
