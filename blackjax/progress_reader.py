# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone reader for file-based progress output.

Usage: python -m blackjax.progress_reader /tmp/bjx_progress.txt
"""
import argparse
import time


def read_progress(path):
    """Read '<step> <total>' from ``path``.

    Returns
    -------
    ``(step, total)`` tuple of ints, or ``None`` if the file does not exist
    or does not yet contain a parseable ``"<step> <total>"`` payload (e.g. it
    was read mid-write).
    """
    try:
        with open(path) as fh:
            parts = fh.read().split()
        return int(parts[0]), int(parts[1])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path")
    p.add_argument("--interval", type=float, default=0.2)
    args = p.parse_args(argv)

    from tqdm.auto import tqdm

    bar = None
    last = -1
    missing_after_start = 0
    while True:
        result = read_progress(args.path)
        if result is not None:
            step, total = result
            if bar is None:
                bar = tqdm(total=total, desc="BlackJAX", unit="step")
            if step != last:
                bar.n = step + 1
                bar.refresh()
                last = step
        elif bar is not None:
            # File deleted after we saw progress -> run finished.
            missing_after_start += 1
            if missing_after_start > 3:
                break
        time.sleep(args.interval)
    if bar is not None:
        bar.close()


if __name__ == "__main__":
    main()
