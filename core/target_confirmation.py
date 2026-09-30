"""Minimal confidence-only temporal target confirmation."""


class TargetConfirmer:
    def __init__(self, high_confidence=.9, high_frames=5,
                 normal_confidence=.8, normal_frames=10):
        self.high_confidence = float(high_confidence)
        self.high_frames = int(high_frames)
        self.normal_confidence = float(normal_confidence)
        self.normal_frames = int(normal_frames)
        self.reset()

    def reset(self):
        self.high_count = self.normal_count = 0
        self.label = None

    def update(self, instances):
        if not instances:
            self.reset()
            return None
        target = max(instances, key=lambda item: item.score)
        if self.label != target.label:
            self.reset()
            self.label = target.label
        if target.score >= self.high_confidence:
            self.high_count += 1
            self.normal_count += 1
        elif target.score >= self.normal_confidence:
            self.high_count = 0
            self.normal_count += 1
        else:
            self.reset()
            return None
        if self.high_count >= self.high_frames:
            return target, f"{self.high_count}帧≥{self.high_confidence:.2f}"
        if self.normal_count >= self.normal_frames:
            return target, f"{self.normal_count}帧≥{self.normal_confidence:.2f}"
        return None

    def update_all(self, instances):
        """Confirm a scene, then return every instance meeting its threshold."""
        normal = [item for item in instances
                  if item.score >= self.normal_confidence]
        high = [item for item in normal
                if item.score >= self.high_confidence]
        if not normal:
            self.reset()
            return None
        self.normal_count += 1
        all_high = len(high) == len(normal)
        self.high_count = self.high_count + 1 if all_high else 0
        if self.high_count >= self.high_frames:
            return high, f"{self.high_count}帧≥{self.high_confidence:.2f}"
        if self.normal_count >= self.normal_frames:
            return normal, f"{self.normal_count}帧≥{self.normal_confidence:.2f}"
        return None
