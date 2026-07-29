#!/usr/bin/env bash
# Deprecated compatibility wrapper. New code should source scripts/cann_env.sh.

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cann_env.sh
source "$_SCRIPT_DIR/cann_env.sh"
unset _SCRIPT_DIR
