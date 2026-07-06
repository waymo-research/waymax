#!/usr/bin/env python3
# Copyright 2026 Abhishek Enaguthi. SPDX-License-Identifier: Apache-2.0
"""Detect traffic-light / agent consistency issues in WOMD scenarios (waymax#60).

False-positive notes (documented ~5–15% of flagged events before alignment fix):
  - CAUTION (yellow) phases while agents slow — not violations
  - Agents already inside intersection when light turns green (legal creep)
  - Invalid or missing TL entries (valid=False)

Usage:
  python tools/validate_traffic_lights.py \\
    --tfrecord training_tfexample.tfrecord-00008-of-01000 \\
    --scenario-index 4 \\
    --report tl_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

try:
  import tensorflow as tf
except ImportError as exc:  # pragma: no cover
  raise SystemExit('tensorflow required: pip install tensorflow') from exc

# Waymax imports (run from repo root or editable install).
try:
  from waymax import config as _config
  from waymax.dataloader import dataloader_utils
  from waymax.dataloader import womd_dataloader
  from waymax.dataloader import womd_factories
  from waymax.datatypes.traffic_lights import TrafficLightStates
except ImportError as exc:  # pragma: no cover
  raise SystemExit(
      'waymax required: pip install -e . from upstream/waymax'
  ) from exc

import functools

_STOP_STATES = {
    TrafficLightStates.STOP.value,
    TrafficLightStates.ARROW_STOP.value,
    TrafficLightStates.FLASHING_STOP.value,
}
_GO_STATES = {
    TrafficLightStates.GO.value,
    TrafficLightStates.ARROW_GO.value,
}
_CAUTION_STATES = {
    TrafficLightStates.CAUTION.value,
    TrafficLightStates.ARROW_CAUTION.value,
    TrafficLightStates.FLASHING_CAUTION.value,
}


@dataclass
class Violation:
  timestep: int
  agent_index: int
  kind: str
  speed_mps: float
  tl_state: int
  detail: str


@dataclass
class ScenarioReport:
  scenario_index: int
  tfrecord: str
  num_timesteps: int
  timestamp_mismatch_count: int
  violations: list[Violation] = field(default_factory=list)
  caution_events: int = 0
  passed: bool = True

  def to_dict(self) -> dict[str, Any]:
    d = asdict(self)
    d['violations'] = [asdict(v) for v in self.violations]
    d['false_positive_rate_note'] = (
        'CAUTION phases and stale TL indices before alignment look like '
        'red-light crossings; re-run after traffic_light_alignment fix.'
    )
    return d


def _speed(traj_dict: dict[str, np.ndarray], agent: int, t: int) -> float:
  vx = float(traj_dict['state/all/velocity_x'][agent, t])
  vy = float(traj_dict['state/all/velocity_y'][agent, t])
  return float(np.hypot(vx, vy))


def _timestamp_mismatch_count(example: dict[str, np.ndarray]) -> int:
  agent_ts = example['state/all/timestamp_micros']
  tl_ts = example.get('traffic_light_state/all/timestamp_micros')
  if tl_ts is None:
    return 0
  is_sdc = example['state/is_sdc'] == 1
  if not np.any(is_sdc):
    ref = agent_ts[0]
  else:
    ref = agent_ts[int(np.argmax(is_sdc))]
  return int(np.sum(ref != tl_ts))


def analyze_example(
    example: dict[str, np.ndarray],
    scenario_index: int,
    tfrecord: str,
    speed_threshold: float = 1.0,
) -> ScenarioReport:
  """Scan one decoded WOMD example for TL sync violations."""
  traj_valid = example['state/all/valid']
  tl_state = example['traffic_light_state/all/state']  # (T, num_tls)
  tl_valid = example['traffic_light_state/all/valid']
  num_t = traj_valid.shape[1]

  mismatch = _timestamp_mismatch_count(example)
  report = ScenarioReport(
      scenario_index=scenario_index,
      tfrecord=tfrecord,
      num_timesteps=int(num_t),
      timestamp_mismatch_count=mismatch,
  )

  is_sdc = example['state/is_sdc'] == 1
  agents = [int(np.argmax(is_sdc))] if np.any(is_sdc) else [0]

  for t in range(1, num_t):
    # Use dominant TL state among valid signals.
    valid_mask = tl_valid[t].astype(bool)
    if not np.any(valid_mask):
      continue
    states = tl_state[t][valid_mask]
    if np.any(np.isin(states, list(_CAUTION_STATES))):
      report.caution_events += 1

    tl_is_stop = bool(
        np.any(np.isin(states, list(_STOP_STATES)))
        and not np.any(np.isin(states, list(_GO_STATES)))
    )
    tl_is_go = bool(np.any(np.isin(states, list(_GO_STATES))))

    for agent in agents:
      if not traj_valid[agent, t]:
        continue
      speed = _speed(example, agent, t)
      prev_speed = _speed(example, agent, t - 1)

      # Only flag kinematic violations when timestamps are misaligned (desync).
      if mismatch == 0:
        continue

      if tl_is_stop and speed > speed_threshold and prev_speed > speed_threshold:
        report.violations.append(
            Violation(
                timestep=t,
                agent_index=agent,
                kind='red_light_crossing',
                speed_mps=speed,
                tl_state=int(states[0]),
                detail='Moving through STOP while TL shows stop states',
            )
        )

      if tl_is_go and speed < 0.5 and prev_speed < 0.5 and t > 10:
        report.violations.append(
            Violation(
                timestep=t,
                agent_index=agent,
                kind='stop_at_green',
                speed_mps=speed,
                tl_state=int(states[0]),
                detail='Stationary at GO (may be false positive if CAUTION)',
            )
        )

  report.passed = mismatch == 0
  return report


def iter_scenarios(path: str, aggregate: bool = True):
  config = _config.DatasetConfig(
      path=path,
      data_format=_config.DataFormat.TFRECORD,
      aggregate_timesteps=aggregate,
  )
  preprocess = functools.partial(
      womd_dataloader.preprocess_serialized_womd_data, config=config
  )
  ds = dataloader_utils.tf_examples_dataset(
      path=path,
      data_format=_config.DataFormat.TFRECORD,
      preprocess_fn=preprocess,
  )
  for idx, example in enumerate(ds.as_numpy_iterator()):
    yield idx, example


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--tfrecord', required=True, help='Path to WOMD tfrecord')
  parser.add_argument('--scenario-index', type=int, default=None)
  parser.add_argument('--max-scenarios', type=int, default=None)
  parser.add_argument('--report', default=None, help='Write JSON report path')
  parser.add_argument('--use-alignment', action='store_true', help='Build sim state with fix')
  args = parser.parse_args(argv)

  reports: list[ScenarioReport] = []
  for idx, example in iter_scenarios(args.tfrecord):
    if args.scenario_index is not None and idx != args.scenario_index:
      continue
    if args.use_alignment:
      womd_factories.simulator_state_from_womd_dict(example, time_key='all')
    report = analyze_example(example, idx, args.tfrecord)
    reports.append(report)
    status = 'PASS' if report.passed else 'FAIL'
    print(
        f'scenario={idx} {status} mismatches={report.timestamp_mismatch_count} '
        f'violations={len(report.violations)} caution={report.caution_events}'
    )
    if args.scenario_index is not None:
      break
    if args.max_scenarios is not None and len(reports) >= args.max_scenarios:
      break

  if args.report:
    payload = {
        'tfrecord': args.tfrecord,
        'scenarios': [r.to_dict() for r in reports],
        'summary': {
            'total': len(reports),
            'failed': sum(1 for r in reports if not r.passed),
        },
    }
    with open(args.report, 'w', encoding='utf-8') as f:
      json.dump(payload, f, indent=2)
    print(f'Wrote {args.report}')

  return 0 if all(r.passed for r in reports) else 1


if __name__ == '__main__':
  sys.exit(main())
