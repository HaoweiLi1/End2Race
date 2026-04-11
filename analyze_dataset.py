#!/usr/bin/env python3
"""Analyze Dataset_Austin/: output a CSV matrix of episode statuses.
Rows = lattice parameter sets, Columns = runtime scenarios, Cells = status.

Status codes:
  o   = success overtake
  f   = success follow
  olq = overtake low-quality
  flq = follow low-quality
  c   = collision
  (empty) = no data
"""

import csv
import os
import re
import sys
from collections import defaultdict


def parse_filename(fname):
    """Extract runtime params from filename like f_ol1_e0_o15_s0.8.csv"""
    m = re.match(r'^([of])_ol(\d+)_e(\d+)_o(\d+)_s([\d.]+)\.\w+$', fname)
    if not m:
        return None
    return {
        'behavior': m.group(1),
        'opp_raceline': int(m.group(2)),
        'ego_idx': int(m.group(3)),
        'opp_idx': int(m.group(4)),
        'opp_speed_scale': float(m.group(5)),
    }


def scan_dataset(root):
    """Scan all lattice param dirs and collect episode statuses.

    Returns:
        episodes: dict[runtime_key] -> dict[lattice_param] -> status
        lattice_params: sorted list of all lattice param dirs
    """
    episodes = defaultdict(dict)
    lattice_params = []

    for entry in sorted(os.listdir(root)):
        entry_path = os.path.join(root, entry)
        if not os.path.isdir(entry_path) or not entry.startswith('cw'):
            continue
        lattice_params.append(entry)

        for subdir, status_map in [
            ('success',     {'o': 'o',  'f': 'f'}),
            ('low_quality', {'o': 'olq', 'f': 'flq'}),
            ('collision',   {'o': 'c',  'f': 'c'}),
        ]:
            dirpath = os.path.join(entry_path, subdir)
            if not os.path.isdir(dirpath):
                continue
            seen = set()
            for fname in os.listdir(dirpath):
                info = parse_filename(fname)
                if info is None:
                    continue
                runtime_key = (info['opp_raceline'], info['ego_idx'],
                               info['opp_idx'], info['opp_speed_scale'])
                if runtime_key in seen:
                    continue
                seen.add(runtime_key)
                status = status_map[info['behavior']]
                episodes[runtime_key][entry] = status

    return episodes, sorted(lattice_params)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'Dataset_Austin'

    print(f"Scanning {root}/ ...")
    episodes, lattice_params = scan_dataset(root)

    runtime_keys = sorted(episodes.keys())
    col_names = [f"ol{ol}_e{eidx}_o{oidx}_s{spd}" for ol, eidx, oidx, spd in runtime_keys]

    print(f"  {len(lattice_params)} lattice param sets x {len(runtime_keys)} runtime scenarios")

    out_path = os.path.join(root, 'status_matrix.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lattice_params'] + col_names)
        for lp in lattice_params:
            row = [lp]
            for key in runtime_keys:
                row.append(episodes[key].get(lp, ''))
            writer.writerow(row)

    print(f"  Saved to {out_path}")


if __name__ == '__main__':
    main()
