"""Observable AOTI compiler configuration for profile and option tests."""

from contextlib import contextmanager


class FakeConfig:
    """Observable compiler configuration, not a numerical compiler substitute."""

    def __init__(self):
        self.emulate_divison_rounding = False
        self.fallback_by_default = False
        self.selective_decompose = False
        self.post_grad_custom_pre_pass = None
        self.patch_calls = 0

    @contextmanager
    def patch(self, settings):
        self.patch_calls += 1
        previous = {key: getattr(self, key) for key in settings}
        try:
            for key, value in settings.items():
                setattr(self, key, value)
            yield
        finally:
            for key, value in previous.items():
                setattr(self, key, value)
