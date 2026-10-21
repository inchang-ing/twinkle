# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from collections.abc import Mapping
from typing import Callable

# Modules owned by the Twinkle distribution. Symbols imported from these are
# part of the installed package (resolvable on the worker via a plain import),
# so they do not need ``TWINKLE_TRUST_REMOTE_CODE`` to cross the Ray boundary.
_PACKAGE_MODULE_PREFIXES = ('twinkle.', 'twinkle_client.', 'twinkle_agentic.')


def is_package_builtin(obj) -> bool:
    """Return True for symbols defined by the Twinkle distribution itself.

    Built-in loss classes, templates and other first-party types live in
    ``twinkle.*`` (or the sibling ``twinkle_client``/``twinkle_agentic``
    packages). They are stable, importable symbols — the receiving worker can
    rebuild them from its own copy of the package — so they are safe to pass
    across the Ray boundary even in safe mode. User-authored callables (lambdas,
    closures, locally defined classes) have a ``__module__`` outside these
    packages and must still require remote-code trust.
    """
    module = getattr(obj, '__module__', '')
    return isinstance(module, str) and module.startswith(_PACKAGE_MODULE_PREFIXES)


def any_callable(args):
    if isinstance(args, Mapping):
        return any(any_callable(arg) for arg in args.values())
    elif isinstance(args, (tuple, list, set)):
        return any(any_callable(arg) for arg in args)
    else:
        # Package-owned callables/types are safe to ship (see is_package_builtin);
        # only external/user symbols count as unsafe.
        return isinstance(args, (Callable, type)) and not is_package_builtin(args)


def check_unsafe(*args, **kwargs):
    if not trust_remote_code():
        if any_callable(args) or any_callable(kwargs):
            raise ValueError('Twinkle does not support Callable or Type inputs in safe mode.')


def trust_remote_code():
    return os.environ.get('TWINKLE_TRUST_REMOTE_CODE', '1') != '0'
