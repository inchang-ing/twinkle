# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for check_unsafe: package builtins are allowed in safe mode.

Regression for the safe-mode part of
https://github.com/modelscope/twinkle/issues/298 — with
``TWINKLE_TRUST_REMOTE_CODE=0``, Twinkle's own built-in loss / template classes
were rejected because ``any_callable`` treated every ``Callable``/``type`` as
unsafe. Built-in symbols are importable on the worker and must be allowed; only
user-authored callables should require remote-code trust.
"""
import os

import pytest

from twinkle.utils.unsafe import any_callable, check_unsafe, is_package_builtin


class _UserLoss:
    """A user-defined class (module is this test file, not twinkle.*)."""

    def __call__(self):
        return 0


def _user_func():
    return 0


@pytest.fixture(autouse=True)
def safe_mode(monkeypatch):
    """Force safe mode (TWINKLE_TRUST_REMOTE_CODE=0) for every test."""
    monkeypatch.setenv('TWINKLE_TRUST_REMOTE_CODE', '0')


class TestPackageBuiltinAllowed:
    def test_builtin_loss_class_is_package_builtin(self):
        from twinkle.loss import CrossEntropyLoss

        assert is_package_builtin(CrossEntropyLoss)

    def test_builtin_loss_class_passes_check_unsafe(self):
        # The core regression: a built-in loss class must NOT raise in safe mode.
        from twinkle.loss import CrossEntropyLoss

        check_unsafe(CrossEntropyLoss)

    def test_builtin_loss_instance_passes_check_unsafe(self):
        from twinkle.loss import CrossEntropyLoss

        check_unsafe(CrossEntropyLoss())

    def test_builtin_loss_as_kwarg_passes(self):
        from twinkle.loss import CrossEntropyLoss

        check_unsafe(loss=CrossEntropyLoss)

    def test_builtin_loss_in_container_passes(self):
        from twinkle.loss import CrossEntropyLoss, MSELoss

        check_unsafe([CrossEntropyLoss, MSELoss])
        check_unsafe({'primary': CrossEntropyLoss})


class TestUserCallableBlocked:
    def test_user_lambda_is_not_package_builtin(self):
        fn = lambda: 0  # noqa: E731
        assert not is_package_builtin(fn)

    def test_user_class_is_not_package_builtin(self):
        assert not is_package_builtin(_UserLoss)

    def test_user_function_raises_in_safe_mode(self):
        with pytest.raises(ValueError, match='does not support Callable'):
            check_unsafe(_user_func)

    def test_user_class_raises_in_safe_mode(self):
        with pytest.raises(ValueError, match='does not support Callable'):
            check_unsafe(_UserLoss)

    def test_user_lambda_raises_in_safe_mode(self):
        with pytest.raises(ValueError):
            check_unsafe(lambda: 0)

    def test_user_callable_in_container_raises(self):
        with pytest.raises(ValueError):
            check_unsafe({'fn': _user_func})

    def test_user_callable_kwarg_raises(self):
        with pytest.raises(ValueError):
            check_unsafe(fn=_user_func)


class TestNonCallableAlwaysAllowed:
    def test_plain_data_passes(self):
        check_unsafe('cross_entropy', 3, [1, 2, 3], {'a': 'b'})
        assert not any_callable(('cross_entropy', 3))


class TestTrustedMode:
    def test_user_callable_allowed_when_trusted(self, monkeypatch):
        monkeypatch.setenv('TWINKLE_TRUST_REMOTE_CODE', '1')
        check_unsafe(_user_func, _UserLoss, lambda: 0)
