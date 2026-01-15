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
