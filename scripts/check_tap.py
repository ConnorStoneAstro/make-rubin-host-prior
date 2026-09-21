#!/usr/bin/env python
"""Say whether this machine can reach the DP2 TAP service, and why not.

    python scripts/check_tap.py

Checks, in order: that a token was found and where it came from, that Gafaelfawr
recognises it, that it carries the scope TAP needs, and that a trivial query
comes back.  Prints no secret: the token's type prefix, never its value.
"""

from __future__ import annotations

import logging
import sys

from rubin_host_prior.rubin.extract import (
    TAP_SCOPE,
    check_tap_scope,
    discover_tap_url,
    find_token,
    run_adql,
    tap_client,
    token_info,
)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        token, source = find_token()
    except RuntimeError as exc:
        print(f"FAIL  no token\n      {exc}")
        return 1
    print(f"ok    token from {source} (looks like {token.split('-', 1)[0]}-...)")

    url = discover_tap_url("dp2")
    print(f"ok    TAP endpoint {url}")

    try:
        info = token_info(token)
    except RuntimeError as exc:
        print(f"FAIL  {exc}")
        return 1
    if not info:
        print("warn  could not reach Gafaelfawr to check the token; carrying on")
    else:
        print(f"ok    token belongs to {info.get('username', '?')}, "
              f"scopes {info.get('scopes')}")
        if info.get("expires"):
            print(f"      expires {info['expires']}")

    try:
        check_tap_scope(token)
    except RuntimeError as exc:
        print(f"FAIL  {exc}")
        return 1
    print(f"ok    token carries {TAP_SCOPE}")

    try:
        table = run_adql(tap_client(), "SELECT TOP 1 objectId FROM dp2.Object")
    except Exception as exc:
        print(f"FAIL  query rejected: {exc!r}")
        return 1
    print(f"ok    query returned {len(table)} row; TAP works from here")
    return 0


if __name__ == "__main__":
    sys.exit(main())
