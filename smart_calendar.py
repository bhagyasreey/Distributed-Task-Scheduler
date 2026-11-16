"""
AI-Powered Smart Calendar Algorithm
===================================
An adaptive, online-learning scheduling engine that:
1. Estimates task duration using Online Recursive Least Squares (RLS) +
   category-level Bayesian/EMA bias calibration (correcting planning fallacy).
2. Learns personal circadian & weekly mood/energy patterns using Gaussian
   kernel-smoothed hourly updates across weekday/weekend regimes + live momentum.
3. Schedules and reschedules unfinished tasks by matching task cognitive load
   against live and predicted mood/energy curves, deadline feasibility, and
   context-switching costs.

Pure Python 3 standard library implementation (zero external dependencies)
with full JSON state persistence.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: str
    title: str
    category: str = "general"          # e.g., "deep_work", "creative", "admin", "learning"
    complexity: int = 3                # 1 (trivial) to 5 (very complex)
    cognitive_load: float = 0.5        # 0.0 (low) to 1.0 (high)
    estimated_minutes: float = 45.0    # User's initial estimate
    priority: int = 3                  # 1 (low) to 5 (critical)
    deadline: Optional[datetime] = None
    status: str = "pending"            # "pending", "scheduled", "completed", "unfinished"
    elapsed_minutes: float = 0.0       # Time already spent if partially done
    actual_minutes: Optional[float] = None
    reschedule_count: int = 0
    splittable: bool = False
    min_block_minutes: int = 15

    def to_dict(self) -> dict:
        d = asdict(self)
        d["deadline"] = self.deadline.isoformat() if self.deadline else None
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        d = dict(data)
        if d.get("deadline"):
            d["deadline"] = datetime.fromisoformat(d["deadline"])
        return cls(**d)


@dataclass
class TimeSlot:
    start: datetime
    end: datetime

    @property
    def duration_minutes(self) -> float:
        return max(0.0, (self.end - self.start).total_seconds() / 60.0)


@dataclass
class ScheduledBlock:
    task_id: str
    title: str
    category: str
    start: datetime
    end: datetime
    predicted_minutes: float
    uncertainty_minutes: float
    predicted_energy: float
    predicted_mood: float
    affinity_score: float
    reason: str


# ---------------------------------------------------------------------------
# AI Component 1: Online Task Duration Estimator (RLS + Category Calibration)
# ---------------------------------------------------------------------------

class OnlineDurationEstimator:
    """
    Predicts true task duration (in minutes) and uncertainty using:
    1. Exponentially-weighted Recursive Least Squares (RLS) over task & context
       features, adapting continuously as the user's pace changes.
    2. Per-category Exponential Moving Average (EMA) ratio calibration
       (actual / estimated) to rapidly correct systematic planning fallacy.
    """

    NUM_FEATURES = 6  # [bias, est_min, complexity, cog_load, fatigue, reschedules]

    def __init__(
        self,
        forgetting_factor: float = 0.95,
        l2_reg: float = 1.0,
        ema_alpha: float = 0.25,
    ) -> None:
        self.lam = forgetting_factor
        self.ema_alpha = ema_alpha
        # Initialize weights to trust user estimate initially: [0.0, 1.0, 0.0, ...]
        self.weights: List[float] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        inv_reg = 1.0 / l2_reg
        self.P: List[List[float]] = [
            [inv_reg if i == j else 0.0 for j in range(self.NUM_FEATURES)]
            for i in range(self.NUM_FEATURES)
        ]
        self.category_multiplier: Dict[str, float] = {}
        self.category_counts: Dict[str, int] = {}
        self.total_updates: int = 0

    def _extract_features(
        self, task: Task, predicted_energy: float
    ) -> List[float]:
        fatigue = max(0.0, min(1.0, 1.0 - predicted_energy))
        return [
            1.0,
            max(5.0, task.estimated_minutes),
            float(task.complexity),
            float(task.cognitive_load),
            fatigue * 30.0,
            float(min(task.reschedule_count, 5)),
        ]

    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def _mat_vec(self, M: List[List[float]], v: List[float]) -> List[float]:
        return [self._dot(row, v) for row in M]

    def predict(
        self, task: Task, predicted_energy: float = 0.7
    ) -> Tuple[float, float]:
        """
        Returns (predicted_remaining_minutes, uncertainty_std_minutes).
        """
        x = self._extract_features(task, predicted_energy)
        rls_pred = self._dot(self.weights, x)

        cat_mult = self.category_multiplier.get(task.category, 1.0)
        cat_count = self.category_counts.get(task.category, 0)
        cat_pred = task.estimated_minutes * cat_mult

        rls_weight = min(0.75, self.total_updates / 20.0) if cat_count > 0 else 0.5
        if self.total_updates < 3:
            rls_weight = 0.2

        blended = rls_weight * rls_pred + (1.0 - rls_weight) * cat_pred
        total_pred = max(5.0, min(blended, max(task.estimated_minutes * 3.0, 240.0)))
        remaining = max(5.0, total_pred - task.elapsed_minutes)

        px = self._mat_vec(self.P, x)
        var = max(1.0, self._dot(x, px))
        uncertainty = min(remaining * 0.6, math.sqrt(var) * 2.5)
        return round(remaining, 1), round(uncertainty, 1)

    def update(
        self, task: Task, actual_total_minutes: float, slot_energy: float = 0.7
    ) -> None:
        """Online update after a task completes."""
        actual = max(1.0, actual_total_minutes)
        x = self._extract_features(task, slot_energy)

        px = self._mat_vec(self.P, x)
        denom = self.lam + self._dot(x, px)
        k = [v / denom for v in px]
        err = actual - self._dot(self.weights, x)

        self.weights = [w + ki * err for w, ki in zip(self.weights, k)]

        n = self.NUM_FEATURES
        new_P = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                new_P[i][j] = (self.P[i][j] - k[i] * px[j]) / self.lam
        self.P = new_P

        ratio = max(0.3, min(3.0, actual / max(5.0, task.estimated_minutes)))
        prev_mult = self.category_multiplier.get(task.category, 1.0)
        count = self.category_counts.get(task.category, 0)
        alpha = self.ema_alpha if count > 2 else 0.5
        self.category_multiplier[task.category] = (1.0 - alpha) * prev_mult + alpha * ratio
        self.category_counts[task.category] = count + 1
        self.total_updates += 1

    def to_dict(self) -> dict:
        return {
            "lam": self.lam,
            "ema_alpha": self.ema_alpha,
            "weights": self.weights,
            "P": self.P,
            "category_multiplier": self.category_multiplier,
            "category_counts": self.category_counts,
            "total_updates": self.total_updates,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OnlineDurationEstimator":
        obj = cls(
            forgetting_factor=data["lam"],
            ema_alpha=data["ema_alpha"],
        )
        obj.weights = data["weights"]
        obj.P = data["P"]
        obj.category_multiplier = data["category_multiplier"]
        obj.category_counts = data["category_counts"]
        obj.total_updates = data["total_updates"]
        return obj


# ---------------------------------------------------------------------------
# AI Component 2: Circadian & Weekly Mood/Energy Pattern Learner
# ---------------------------------------------------------------------------

class CircadianEnergyMoodModel:
    """
    Learns the user's personal energy and mood curves across hours of the day
    and weekday vs. weekend regimes.
    Combines Gaussian kernel-smoothed hourly buckets with exponential-decay
    momentum from live user check-ins.
    """

    def __init__(self, learning_rate: float = 0.25, kernel_sigma: float = 1.5) -> None:
        self.lr = learning_rate
        self.sigma = kernel_sigma
        default_energy = [
            0.25, 0.20, 0.18, 0.18, 0.20, 0.30,
            0.45, 0.60, 0.72, 0.78, 0.80, 0.75,
            0.65, 0.55, 0.52, 0.58, 0.65, 0.68,
            0.65, 0.60, 0.52, 0.45, 0.38, 0.30,
        ]
        default_mood = [0.55] * 24
        self.energy_grid: List[List[float]] = [list(default_energy), list(default_energy)]
        self.mood_grid: List[List[float]] = [list(default_mood), list(default_mood)]
        self.sample_weights: List[List[float]] = [[0.0] * 24, [0.0] * 24]

        self.last_checkin_time: Optional[datetime] = None
        self.last_energy: Optional[float] = None
        self.last_mood: Optional[float] = None

    @staticmethod
    def _regime(ts: datetime) -> int:
        return 1 if ts.weekday() >= 5 else 0

    def record_checkin(
        self, timestamp: datetime, energy: float, mood: float
    ) -> None:
        """Record an observed (energy, mood) in [0.0, 1.0] and update curves."""
        e = max(0.0, min(1.0, energy))
        m = max(0.0, min(1.0, mood))
        regime = self._regime(timestamp)
        hour_f = timestamp.hour + timestamp.minute / 60.0

        for h in range(24):
            dist = min(abs(h - hour_f), 24.0 - abs(h - hour_f))
            w = math.exp(-0.5 * (dist / self.sigma) ** 2)
            if w < 0.05:
                continue
            eff_lr = self.lr * w
            self.energy_grid[regime][h] = (
                (1.0 - eff_lr) * self.energy_grid[regime][h] + eff_lr * e
            )
            self.mood_grid[regime][h] = (
                (1.0 - eff_lr) * self.mood_grid[regime][h] + eff_lr * m
            )
            self.sample_weights[regime][h] += w

        self.last_checkin_time = timestamp
        self.last_energy = e
        self.last_mood = m

    def predict(
        self, timestamp: datetime, use_live_momentum: bool = True
    ) -> Tuple[float, float]:
        """Predict (energy, mood) at a given timestamp using linear interpolation."""
        regime = self._regime(timestamp)
        h_low = timestamp.hour
        h_high = (h_low + 1) % 24
        frac = timestamp.minute / 60.0

        base_e = (
            (1.0 - frac) * self.energy_grid[regime][h_low]
            + frac * self.energy_grid[regime][h_high]
        )
        base_m = (
            (1.0 - frac) * self.mood_grid[regime][h_low]
            + frac * self.mood_grid[regime][h_high]
        )

        if (
            use_live_momentum
            and self.last_checkin_time is not None
            and self.last_energy is not None
            and self.last_mood is not None
        ):
            delta_hours = (timestamp - self.last_checkin_time).total_seconds() / 3600.0
            if 0.0 <= delta_hours <= 6.0:
                decay = math.exp(-delta_hours / 2.5)
                base_e = decay * self.last_energy + (1.0 - decay) * base_e
                base_m = decay * self.last_mood + (1.0 - decay) * base_m

        return round(base_e, 3), round(base_m, 3)

    def to_dict(self) -> dict:
        return {
            "lr": self.lr,
            "sigma": self.sigma,
            "energy_grid": self.energy_grid,
            "mood_grid": self.mood_grid,
            "sample_weights": self.sample_weights,
            "last_checkin_time": (
                self.last_checkin_time.isoformat() if self.last_checkin_time else None
            ),
            "last_energy": self.last_energy,
            "last_mood": self.last_mood,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CircadianEnergyMoodModel":
        obj = cls(learning_rate=data["lr"], kernel_sigma=data["sigma"])
        obj.energy_grid = data["energy_grid"]
        obj.mood_grid = data["mood_grid"]
        obj.sample_weights = data["sample_weights"]
        if data.get("last_checkin_time"):
            obj.last_checkin_time = datetime.fromisoformat(data["last_checkin_time"])
        obj.last_energy = data.get("last_energy")
        obj.last_mood = data.get("last_mood")
        return obj


# ---------------------------------------------------------------------------
# AI Component 3: Adaptive Smart Calendar & Rescheduler
# ---------------------------------------------------------------------------

class SmartCalendarEngine:
    """
    Combines duration estimation, mood/energy forecasting, and constraint-aware
    scheduling to build and adaptively reschedule a user's calendar.
    """

    def __init__(
        self,
        work_start_hour: int = 9,
        work_end_hour: int = 21,
        buffer_minutes: int = 5,
    ) -> None:
        self.work_start_hour = work_start_hour
        self.work_end_hour = work_end_hour
        self.buffer_minutes = buffer_minutes
        self.duration_model = OnlineDurationEstimator()
        self.energy_mood_model = CircadianEnergyMoodModel()
        self.tasks: Dict[str, Task] = {}

    def add_task(self, task: Task) -> None:
        self.tasks[task.id] = task

    def complete_task(
        self,
        task_id: str,
        actual_minutes: float,
        completed_at: datetime,
        observed_energy: Optional[float] = None,
        observed_mood: Optional[float] = None,
    ) -> None:
        """Record completion, update duration estimator & mood/energy curves."""
        task = self.tasks[task_id]
        task.status = "completed"
        task.actual_minutes = actual_minutes

        if observed_energy is not None and observed_mood is not None:
            self.energy_mood_model.record_checkin(
                completed_at, observed_energy, observed_mood
            )
            slot_e = observed_energy
        else:
            slot_e, _ = self.energy_mood_model.predict(completed_at)

        total_actual = task.elapsed_minutes + actual_minutes
        self.duration_model.update(task, total_actual, slot_energy=slot_e)

    def mark_unfinished(
        self,
        task_id: str,
        elapsed_minutes: float,
        at_time: datetime,
        current_energy: Optional[float] = None,
        current_mood: Optional[float] = None,
    ) -> None:
        """Mark a task unfinished and log user's current energy/mood."""
        task = self.tasks[task_id]
        task.status = "unfinished"
        task.elapsed_minutes += max(0.0, elapsed_minutes)
        task.reschedule_count += 1
        if current_energy is not None and current_mood is not None:
            self.energy_mood_model.record_checkin(at_time, current_energy, current_mood)

    def compute_free_slots(
        self,
        start_time: datetime,
        horizon_days: int,
        busy_blocks: Optional[List[TimeSlot]] = None,
    ) -> List[TimeSlot]:
        """Generate free time slots across working hours, subtracting busy blocks."""
        busy = sorted(busy_blocks or [], key=lambda b: b.start)
        free_slots: List[TimeSlot] = []

        for day_offset in range(horizon_days + 1):
            day_date = (start_time + timedelta(days=day_offset)).date()
            win_start = datetime.combine(day_date, datetime.min.time()).replace(
                hour=self.work_start_hour
            )
            win_end = datetime.combine(day_date, datetime.min.time()).replace(
                hour=self.work_end_hour
            )
            if day_offset == 0:
                win_start = max(win_start, start_time)
            if win_start >= win_end:
                continue

            cursor = win_start
            for b in busy:
                if b.end <= cursor:
                    continue
                if b.start >= win_end:
                    break
                if b.start > cursor:
                    free_slots.append(TimeSlot(cursor, b.start))
                cursor = max(cursor, b.end)
            if cursor < win_end:
                free_slots.append(TimeSlot(cursor, win_end))

        return [s for s in free_slots if s.duration_minutes >= 15.0]

    def _score_task_slot(
        self,
        task: Task,
        slot_start: datetime,
        pred_minutes: float,
        energy: float,
        mood: float,
        prev_category: Optional[str],
        reference_now: datetime,
    ) -> Tuple[float, str]:
        """Score how well a task fits a candidate start time (higher = better)."""
        state_index = 0.65 * energy + 0.35 * mood
        if task.cognitive_load >= 0.65:
            alignment = 2.0 * state_index - 1.2 * abs(task.cognitive_load - state_index)
        elif task.cognitive_load <= 0.35:
            alignment = 1.0 - 0.6 * state_index
        else:
            alignment = 1.0 - abs(task.cognitive_load - state_index)

        priority_score = 0.35 * task.priority + 0.20 * min(task.reschedule_count, 4)

        # Deadline feasibility & safety margin (never reward last-minute procrastination)
        deadline_score = 0.0
        task_end = slot_start + timedelta(minutes=pred_minutes)
        if task.deadline:
            hours_margin = (task.deadline - task_end).total_seconds() / 3600.0
            if hours_margin < 0:
                deadline_score = -5.0  # Misses deadline
            elif hours_margin >= 2.0:
                deadline_score = 0.5   # Comfortable safety buffer

        # Prefer earlier slots when energy/mood alignment is comparable
        hours_away = max(0.0, (slot_start - reference_now).total_seconds() / 3600.0)
        earliness_penalty = 0.025 * hours_away

        context_bonus = 0.25 if prev_category == task.category else 0.0

        total = (
            2.2 * alignment
            + priority_score
            + deadline_score
            + context_bonus
            - earliness_penalty
        )

        reason = (
            f"energy={energy:.2f}, mood={mood:.2f}, load={task.cognitive_load:.2f}, "
            f"est={pred_minutes:.0f}m"
        )
        return round(total, 3), reason

    def schedule_and_reschedule(
        self,
        now: datetime,
        busy_blocks: Optional[List[TimeSlot]] = None,
        horizon_days: int = 3,
        current_energy: Optional[float] = None,
        current_mood: Optional[float] = None,
    ) -> Tuple[List[ScheduledBlock], List[Task]]:
        """
        Schedules all pending and unfinished tasks across free slots.
        Returns (scheduled_blocks, unscheduled_tasks).
        """
        if current_energy is not None and current_mood is not None:
            self.energy_mood_model.record_checkin(now, current_energy, current_mood)

        candidates = [
            t for t in self.tasks.values() if t.status in ("pending", "unfinished")
        ]
        candidates.sort(
            key=lambda t: (
                t.deadline is None,
                t.deadline or datetime.max,
                -(t.priority + 0.5 * t.reschedule_count),
            )
        )

        free_slots = self.compute_free_slots(now, horizon_days, busy_blocks)
        scheduled: List[ScheduledBlock] = []
        unscheduled: List[Task] = []
        prev_category_by_day: Dict[object, str] = {}

        for task in candidates:
            best_idx: Optional[int] = None
            best_score = -1e9
            best_meta: Optional[Tuple[datetime, float, float, float, float, str]] = None

            candidate_windows: List[Tuple[int, datetime, TimeSlot]] = []
            for idx, slot in enumerate(free_slots):
                candidate_windows.append((idx, slot.start, slot))
                if slot.duration_minutes >= 120:
                    step = slot.start + timedelta(minutes=60)
                    while (slot.end - step).total_seconds() / 60.0 >= 30:
                        candidate_windows.append((idx, step, slot))
                        step += timedelta(minutes=60)

            for idx, cand_start, slot in candidate_windows:
                energy, mood = self.energy_mood_model.predict(cand_start)
                pred_min, unc_min = self.duration_model.predict(task, energy)
                avail = (slot.end - cand_start).total_seconds() / 60.0
                if avail < pred_min:
                    continue

                score, reason = self._score_task_slot(
                    task=task,
                    slot_start=cand_start,
                    pred_minutes=pred_min,
                    energy=energy,
                    mood=mood,
                    prev_category=prev_category_by_day.get(cand_start.date()),
                    reference_now=now,
                )
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_meta = (
                        cand_start,
                        pred_min,
                        unc_min,
                        energy,
                        mood,
                        reason,
                    )

            if best_idx is not None and best_meta is not None:
                cand_start, pred_min, unc_min, energy, mood, reason = best_meta
                chosen_slot = free_slots.pop(best_idx)
                block_end = cand_start + timedelta(minutes=pred_min)
                buffered_end = min(
                    chosen_slot.end,
                    block_end + timedelta(minutes=self.buffer_minutes),
                )

                if (cand_start - chosen_slot.start).total_seconds() / 60.0 >= 15:
                    free_slots.append(TimeSlot(chosen_slot.start, cand_start))
                if (chosen_slot.end - buffered_end).total_seconds() / 60.0 >= 15:
                    free_slots.append(TimeSlot(buffered_end, chosen_slot.end))
                free_slots.sort(key=lambda s: s.start)

                prev_category_by_day[cand_start.date()] = task.category
                scheduled.append(
                    ScheduledBlock(
                        task_id=task.id,
                        title=task.title,
                        category=task.category,
                        start=cand_start,
                        end=block_end,
                        predicted_minutes=pred_min,
                        uncertainty_minutes=unc_min,
                        predicted_energy=energy,
                        predicted_mood=mood,
                        affinity_score=best_score,
                        reason=reason,
                    )
                )
            else:
                unscheduled.append(task)

        scheduled.sort(key=lambda b: b.start)
        return scheduled, unscheduled

    def save_state(self, filepath: str) -> None:
        state = {
            "work_start_hour": self.work_start_hour,
            "work_end_hour": self.work_end_hour,
            "buffer_minutes": self.buffer_minutes,
            "duration_model": self.duration_model.to_dict(),
            "energy_mood_model": self.energy_mood_model.to_dict(),
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, filepath: str) -> "SmartCalendarEngine":
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        engine = cls(
            work_start_hour=state["work_start_hour"],
            work_end_hour=state["work_end_hour"],
            buffer_minutes=state["buffer_minutes"],
        )
        engine.duration_model = OnlineDurationEstimator.from_dict(state["duration_model"])
        engine.energy_mood_model = CircadianEnergyMoodModel.from_dict(
            state["energy_mood_model"]
        )
        engine.tasks = {
            tid: Task.from_dict(td) for tid, td in state["tasks"].items()
        }
        return engine


# ---------------------------------------------------------------------------
# Verification & Multi-Day Adaptation Simulation
# ---------------------------------------------------------------------------

def run_verification_and_simulation() -> None:
    rng = random.Random(42)
    engine = SmartCalendarEngine(work_start_hour=9, work_end_hour=20)
    base_day = datetime(2026, 11, 17, 9, 0)

    def true_user_energy_mood(ts: datetime) -> Tuple[float, float]:
        h = ts.hour + ts.minute / 60.0
        energy = (
            0.55
            + 0.30 * math.exp(-((h - 10.0) / 2.0) ** 2)
            - 0.28 * math.exp(-((h - 14.5) / 1.5) ** 2)
            + 0.18 * math.exp(-((h - 18.0) / 1.8) ** 2)
        )
        mood = (
            0.60
            + 0.20 * math.exp(-((h - 10.5) / 2.5) ** 2)
            - 0.15 * math.exp(-((h - 14.5) / 1.8) ** 2)
        )
        return max(0.1, min(1.0, energy)), max(0.1, min(1.0, mood))

    early_errors: List[float] = []
    late_errors: List[float] = []

    for day in range(14):
        for i in range(4):
            hour = [9, 11, 14, 17][i]
            ts = base_day + timedelta(days=day, hours=hour - 9)
            e, m = true_user_energy_mood(ts)
            engine.energy_mood_model.record_checkin(ts, e, m)

            cat = "deep_work" if i in (0, 1) else "admin"
            est = 60.0 if cat == "deep_work" else 30.0
            cog = 0.85 if cat == "deep_work" else 0.25
            task = Task(
                id=f"sim_{day}_{i}",
                title=f"{cat} task {day}-{i}",
                category=cat,
                complexity=4 if cat == "deep_work" else 2,
                cognitive_load=cog,
                estimated_minutes=est,
            )
            engine.add_task(task)
            pred_min, _ = engine.duration_model.predict(task, e)

            mult = 1.40 if cat == "deep_work" else 0.85
            actual = est * mult * (1.0 + 0.15 * (1.0 - e)) + rng.uniform(-3.0, 3.0)
            err = abs(pred_min - actual)
            if day < 3:
                early_errors.append(err)
            elif day >= 10:
                late_errors.append(err)

            engine.complete_task(task.id, actual, ts, e, m)

    early_mae = sum(early_errors) / len(early_errors)
    late_mae = sum(late_errors) / len(late_errors)
    assert late_mae < early_mae, f"Expected adaptation: {late_mae:.2f} < {early_mae:.2f}"

    now = base_day + timedelta(days=14, hours=5)  # 14:00 (2:00 PM slump)
    t_deep = Task(
        id="unfinished_arch",
        title="Finish Distributed Cache Design",
        category="deep_work",
        complexity=5,
        cognitive_load=0.90,
        estimated_minutes=60.0,
        priority=4,
        deadline=now + timedelta(days=2),
    )
    t_admin = Task(
        id="quick_inbox",
        title="Clear Expense Receipts",
        category="admin",
        complexity=1,
        cognitive_load=0.20,
        estimated_minutes=30.0,
        priority=3,
    )
    engine.add_task(t_deep)
    engine.add_task(t_admin)
    engine.mark_unfinished(
        "unfinished_arch",
        elapsed_minutes=20.0,
        at_time=now,
        current_energy=0.25,
        current_mood=0.30,
    )

    blocks, unscheduled = engine.schedule_and_reschedule(
        now=now,
        horizon_days=2,
        current_energy=0.25,
        current_mood=0.30,
    )
    assert not unscheduled, "All tasks should be scheduled"
    by_id = {b.task_id: b for b in blocks}
    assert by_id["quick_inbox"].start < by_id["unfinished_arch"].start
    assert by_id["unfinished_arch"].predicted_energy > 0.55

    print(f"Duration MAE improved: Day 1-3 = {early_mae:.1f}m -> Day 11-14 = {late_mae:.1f}m")
    print(f"Learned deep_work multiplier: {engine.duration_model.category_multiplier['deep_work']:.2f}x")
    print(f"Learned admin multiplier:     {engine.duration_model.category_multiplier['admin']:.2f}x")
    print("\nAdaptive Reschedule Output:")
    for b in blocks:
        print(
            f"  [{b.start.strftime('%a %H:%M')} - {b.end.strftime('%H:%M')}] "
            f"{b.title} (~{b.predicted_minutes:.0f}m ±{b.uncertainty_minutes:.0f}m) | {b.reason}"
        )


if __name__ == "__main__":
    run_verification_and_simulation()