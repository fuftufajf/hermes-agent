"""Copied partial context elisions cannot become effectful tool inputs."""

import pytest

from agent.compression_marker import _COMPRESSION_MARKER_PREFIX
from agent.tool_dispatch_helpers import _context_pruned_argument_paths


@pytest.mark.parametrize("suffix", [" 1,800", " 1,800 of", " 1,800 of 2,000 chars omitted"])
@pytest.mark.parametrize("tool", ["write_file", "patch", "terminal", "execute_code", "plugin_unknown_effect"])
def test_partial_rendered_marker_is_blocked(tool, suffix):
    args = {"payload": [{"body": "prefix " + _COMPRESSION_MARKER_PREFIX + suffix}]}
    assert _context_pruned_argument_paths(tool, args) == ["$.payload[0].body"]


def test_marker_documentation_and_read_only_inspection_remain_allowed():
    assert _context_pruned_argument_paths("write_file", {"content": _COMPRESSION_MARKER_PREFIX + " {omitted:,} of {total:,}"}) == []
    assert _context_pruned_argument_paths("read_file", {"path": _COMPRESSION_MARKER_PREFIX + " 1,800"}) == []
