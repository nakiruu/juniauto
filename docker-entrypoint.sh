#!/bin/sh
# JuniAuto container entrypoint.
#
# Bind-mounted volumes (./logs, ./data/cache) arrive with host-side ownership,
# which the container's non-root `juni` service user cannot write to. Runs a
# one-shot chown as root, then drops privileges via `runuser` (util-linux,
# always available in Ubuntu). Safe to run every start — noop if perms are
# already correct.

set -eu

if [ "$(id -u)" = "0" ]; then
    # Only try to chown paths that exist (bind mounts may point at nothing).
    for dir in /app/logs /app/cache; do
        [ -d "$dir" ] && chown -R juni:juni "$dir" 2>/dev/null || true
    done
    exec runuser -u juni -- "$@"
fi

# Already unprivileged (rare — e.g. --user override in compose). Just exec.
exec "$@"
