"""Deterministic Fibonacci interaction state detection.

The detector deliberately uses only fixed Fibonacci zones and dwell/confirmation
rules. It does not estimate volatility, swings, or change points.
"""

from enum import Enum

import numpy as np


class InteractionState(str, Enum):
    IDLE = "idle"
    APPROACHING = "approaching"
    CONSOLIDATING = "consolidating"
    FINISHED = "finished"


class EventType(str, Enum):
    HOVER = "hover"
    BREAKOUT = "breakout"
    PULLBACK = "pullback"
    TIMEOUT = "timeout"


def _validated_inputs(
    prices,
    fibonacci_levels,
    tolerance,
    min_dwell_steps,
    confirmation_steps,
    max_dwell_steps,
):
    prices = np.asarray(prices, dtype=float).reshape(-1)
    levels = np.unique(np.asarray(fibonacci_levels, dtype=float).reshape(-1))

    if prices.size == 0 or not np.all(np.isfinite(prices)):
        raise ValueError("prices must contain at least one finite value")
    if levels.size == 0 or not np.all(np.isfinite(levels)):
        raise ValueError("fibonacci_levels must contain at least one finite value")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    for name, value in (
        ("min_dwell_steps", min_dwell_steps),
        ("confirmation_steps", confirmation_steps),
        ("max_dwell_steps", max_dwell_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if max_dwell_steps < min_dwell_steps:
        raise ValueError("max_dwell_steps must be at least min_dwell_steps")
    return prices, levels


def _zone_level(price, levels, tolerance):
    """Return the nearest zone containing price, preferring the lower tie."""
    distances = np.abs(levels - price)
    matches = np.flatnonzero(distances <= tolerance)
    if matches.size == 0:
        return None
    nearest_distance = np.min(distances[matches])
    nearest = matches[np.isclose(distances[matches], nearest_distance)]
    return float(levels[nearest[0]])


def _side(price, level, tolerance):
    if price < level - tolerance:
        return "below"
    if price > level + tolerance:
        return "above"
    return "inside"


def detect_fibonacci_interactions(
    prices,
    fibonacci_levels,
    tolerance=0.02,
    min_dwell_steps=3,
    confirmation_steps=10,
    max_dwell_steps=20,
    return_state_trace=False,
):
    """Detect Fibonacci hover, breakout, pullback, and timeout outputs.

    HOVER is emitted once when APPROACHING becomes CONSOLIDATING. A pullback
    may finish directly from APPROACHING, without a HOVER, when price exits
    toward its approach origin and stays there for the confirmation window.
    Breakouts are only possible after HOVER.

    When ``return_state_trace`` is true, return ``(outputs, state_trace)``.
    """
    prices, levels = _validated_inputs(
        prices,
        fibonacci_levels,
        tolerance,
        min_dwell_steps,
        confirmation_steps,
        max_dwell_steps,
    )
    states = [InteractionState.IDLE for _ in range(prices.size)]
    outputs = []
    state = InteractionState.IDLE
    active = None
    i = 1  # An entry needs a previous, outside observation to define approach.

    while i < prices.size:
        if state is InteractionState.IDLE:
            level = _zone_level(prices[i], levels, tolerance)
            if level is None:
                i += 1
                continue

            previous_side = _side(prices[i - 1], level, tolerance)
            if previous_side == "inside":
                i += 1
                continue

            active = {
                "fib_level": level,
                "entry_index": i,
                "approach_side": previous_side,
                "approach_direction": "up" if previous_side == "below" else "down",
                "consolidation_index": None,
                "hover_index": None,
                "dwell_time": 1,
            }
            state = InteractionState.APPROACHING
            states[i] = state
            if min_dwell_steps == 1:
                state = InteractionState.CONSOLIDATING
                active["consolidation_index"] = i
                active["hover_index"] = i
                states[i] = state
                outputs.append(
                    {
                        **active,
                        "event_type": EventType.HOVER,
                        "is_terminal": False,
                        "output_index": i,
                        "pullback_stage": None,
                        "exit_direction": None,
                        "exit_index": None,
                        "confirmation_end_index": None,
                        "exit_price": np.nan,
                        "max_price_after": np.nan,
                        "min_price_after": np.nan,
                    }
                )
            i += 1
            continue

        level = active["fib_level"]
        current_side = _side(prices[i], level, tolerance)

        if state is InteractionState.APPROACHING:
            if current_side != "inside":
                # Before consolidation, only a confirmed return toward the
                # approach origin is an event. An opposite-side exit is not a
                # breakout and is discarded.
                if current_side != active["approach_side"]:
                    active = None
                    state = InteractionState.IDLE
                    continue

                exit_index = i
                confirmation_end = exit_index + confirmation_steps - 1
                if confirmation_end >= prices.size:
                    break
                confirmation = prices[exit_index : confirmation_end + 1]
                confirmation_sides = [
                    _side(price, level, tolerance) for price in confirmation
                ]
                confirmed = all(
                    side == active["approach_side"] for side in confirmation_sides
                )
                if confirmed:
                    states[exit_index : confirmation_end + 1] = [
                        InteractionState.APPROACHING
                    ] * confirmation_steps
                    states[confirmation_end] = InteractionState.FINISHED
                    outputs.append(
                        {
                            **active,
                            "event_type": EventType.PULLBACK,
                            "is_terminal": True,
                            "output_index": confirmation_end,
                            "pullback_stage": "approaching",
                            "exit_direction": (
                                "up" if current_side == "above" else "down"
                            ),
                            "exit_index": exit_index,
                            "confirmation_end_index": confirmation_end,
                            "exit_price": float(prices[exit_index]),
                            "max_price_after": float(np.max(confirmation)),
                            "min_price_after": float(np.min(confirmation)),
                        }
                    )
                    active = None
                    state = InteractionState.IDLE
                    i = confirmation_end + 1
                    continue

                first_invalid_offset = next(
                    offset
                    for offset, side in enumerate(confirmation_sides[1:], start=1)
                    if side != active["approach_side"]
                )
                invalid_index = exit_index + first_invalid_offset
                states[exit_index:invalid_index] = [
                    InteractionState.APPROACHING
                ] * (invalid_index - exit_index)
                active = None
                state = InteractionState.IDLE
                i = invalid_index
                continue

            active["dwell_time"] += 1
            states[i] = state
            if active["dwell_time"] >= min_dwell_steps:
                state = InteractionState.CONSOLIDATING
                active["consolidation_index"] = i
                active["hover_index"] = i
                states[i] = state
                outputs.append(
                    {
                        **active,
                        "event_type": EventType.HOVER,
                        "is_terminal": False,
                        "output_index": i,
                        "pullback_stage": None,
                        "exit_direction": None,
                        "exit_index": None,
                        "confirmation_end_index": None,
                        "exit_price": np.nan,
                        "max_price_after": np.nan,
                        "min_price_after": np.nan,
                    }
                )
            i += 1
            continue

        # max_dwell_steps limits total time in CONSOLIDATING, not only the
        # observations that happen to remain inside the zone.
        timeout_index = active["consolidation_index"] + max_dwell_steps
        if i >= timeout_index:
            states[timeout_index] = InteractionState.FINISHED
            outputs.append(
                {
                    **active,
                    "event_type": EventType.TIMEOUT,
                    "is_terminal": True,
                    "output_index": timeout_index,
                    "pullback_stage": None,
                    "exit_direction": None,
                    "exit_index": timeout_index,
                    "confirmation_end_index": None,
                    "exit_price": float(prices[timeout_index]),
                    "max_price_after": np.nan,
                    "min_price_after": np.nan,
                }
            )
            active = None
            state = InteractionState.IDLE
            i = timeout_index + 1
            continue

        if current_side == "inside":
            active["dwell_time"] += 1
            states[i] = InteractionState.CONSOLIDATING
            i += 1
            continue

        exit_index = i
        confirmation_end = exit_index + confirmation_steps - 1
        if timeout_index < prices.size and confirmation_end >= timeout_index:
            states[exit_index:timeout_index] = [
                InteractionState.CONSOLIDATING
            ] * (timeout_index - exit_index)
            states[timeout_index] = InteractionState.FINISHED
            outputs.append(
                {
                    **active,
                    "event_type": EventType.TIMEOUT,
                    "is_terminal": True,
                    "output_index": timeout_index,
                    "pullback_stage": None,
                    "exit_direction": None,
                    "exit_index": timeout_index,
                    "confirmation_end_index": None,
                    "exit_price": float(prices[timeout_index]),
                    "max_price_after": np.nan,
                    "min_price_after": np.nan,
                }
            )
            active = None
            state = InteractionState.IDLE
            i = timeout_index + 1
            continue
        if confirmation_end >= prices.size:
            break

        confirmation = prices[exit_index : confirmation_end + 1]
        confirmation_sides = [
            _side(price, level, tolerance) for price in confirmation
        ]
        expected_side = current_side
        confirmed = all(side == expected_side for side in confirmation_sides)
        if confirmed:
            states[exit_index : confirmation_end + 1] = [
                InteractionState.CONSOLIDATING
            ] * confirmation_steps
            event_type = (
                EventType.PULLBACK
                if expected_side == active["approach_side"]
                else EventType.BREAKOUT
            )
            outputs.append(
                {
                    **active,
                    "event_type": event_type,
                    "is_terminal": True,
                    "output_index": confirmation_end,
                    "pullback_stage": (
                        "consolidating" if event_type is EventType.PULLBACK else None
                    ),
                    "exit_direction": "up" if expected_side == "above" else "down",
                    "exit_index": exit_index,
                    "confirmation_end_index": confirmation_end,
                    "exit_price": float(prices[exit_index]),
                    "max_price_after": float(np.max(confirmation)),
                    "min_price_after": float(np.min(confirmation)),
                }
            )
            states[confirmation_end] = InteractionState.FINISHED
            active = None
            state = InteractionState.IDLE
            i = confirmation_end + 1
            continue

        # Resume the same consolidated interaction at the first observation
        # that invalidated this confirmation attempt, whether it re-entered
        # the zone or jumped directly to the opposite side.
        first_invalid_offset = next(
            offset
            for offset, side in enumerate(confirmation_sides[1:], start=1)
            if side != expected_side
        )
        invalid_index = exit_index + first_invalid_offset
        states[exit_index:invalid_index] = [
            InteractionState.CONSOLIDATING
        ] * (invalid_index - exit_index)
        state = InteractionState.CONSOLIDATING
        i = invalid_index

    if return_state_trace:
        return outputs, states
    return outputs
