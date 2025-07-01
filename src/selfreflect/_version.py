#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("selfreflect")
except PackageNotFoundError:
    __version__ = "1.0"

del version, PackageNotFoundError
