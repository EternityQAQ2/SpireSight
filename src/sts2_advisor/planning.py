from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .domain import Advice, Candidate, DecisionContext, Evidence


PLANNABLE_DECISIONS = {"combat_next_action", "shop", "fake_merchant", "rewards"}


@dataclass(frozen=True)
class PlannedStep:
    action_type: str
    label: str
    params: dict


@dataclass
class ActivePlan:
    decision_type: str
    pending: list[PlannedStep]
    previous_context: DecisionContext
    reason: str
    confidence: float
    evidence: list[Evidence]
    evidence_ids: list[str]
    warnings: list[str]

    @classmethod
    def from_advice(
        cls, advice: Advice, context: DecisionContext, evidence: list[Evidence],
    ) -> ActivePlan | None:
        if context.decision_type not in PLANNABLE_DECISIONS or len(advice.plan_candidate_ids) < 2:
            return None
        candidates = {item.candidate_id: item for item in context.candidates}
        steps = [step_from_candidate(candidates[item]) for item in advice.plan_candidate_ids]
        return cls(
            context.decision_type,
            steps,
            context,
            advice.reason,
            advice.confidence,
            list(evidence),
            list(advice.evidence_ids),
            list(advice.warnings),
        )

    def advance(self, context: DecisionContext) -> Advice | None:
        """Consume the previous step and rebind the remaining plan to current indices."""
        if context.decision_type != self.decision_type or len(self.pending) < 2:
            return None

        resolved: list[Candidate] | None = None
        for completed_count in range(len(self.pending) - 1, 0, -1):
            completed = self.pending[:completed_count]
            if not transition_sequence_is_safe(self.previous_context, completed, context):
                continue
            rebound: list[Candidate] = []
            available = list(context.candidates)
            for step in self.pending[completed_count:]:
                candidate = resolve_step(step, available)
                if candidate is None:
                    break
                rebound.append(candidate)
                available.remove(candidate)
            else:
                resolved = rebound
                break
        if not resolved:
            return None

        self.pending = [step_from_candidate(item) for item in resolved]
        self.previous_context = context
        return Advice(
            decision_id=context.decision_id,
            candidate_id=resolved[0].candidate_id,
            reason="继续执行上一轮已验证的短计划：" + self.reason,
            confidence=self.confidence,
            evidence_ids=self.evidence_ids,
            warnings=self.warnings,
            plan_candidate_ids=[item.candidate_id for item in resolved],
        )


def step_from_candidate(candidate: Candidate) -> PlannedStep:
    return PlannedStep(candidate.action_type, candidate.label, dict(candidate.params))


def resolve_step(step: PlannedStep, candidates: list[Candidate]) -> Candidate | None:
    matches = [
        item
        for item in candidates
        if item.legal
        and item.action_type == step.action_type
        and item.label == step.label
        and stable_candidate_params(item) == stable_step_params(step)
    ]
    if not matches:
        return None
    if step.action_type == "play_card":
        return max(matches, key=lambda item: int(item.params.get("card_index", -1)))
    return matches[0]


def stable_candidate_params(candidate: Candidate) -> dict:
    return {
        key: value
        for key, value in candidate.params.items()
        if key not in {"index", "card_index", "slot"}
    }


def stable_step_params(step: PlannedStep) -> dict:
    return {
        key: value
        for key, value in step.params.items()
        if key not in {"index", "card_index", "slot"}
    }


def transition_is_safe(
    previous: DecisionContext, completed: PlannedStep, current: DecisionContext,
) -> bool:
    return transition_sequence_is_safe(previous, [completed], current)


def transition_sequence_is_safe(
    previous: DecisionContext, completed: list[PlannedStep], current: DecisionContext,
) -> bool:
    if previous.decision_type != current.decision_type:
        return False
    if previous.decision_type == "combat_next_action":
        return combat_transition_is_safe(previous, completed, current)
    if previous.decision_type in {"shop", "fake_merchant"}:
        if not completed_items_were_removed(previous, completed, current):
            return False
        return (
            all(step.action_type == "shop_purchase" for step in completed)
            and shop_gold_change_is_expected(previous, completed, current)
        )
    if previous.decision_type == "rewards":
        return completed_items_were_removed(previous, completed, current)
    return False


def matches_step(candidate: Candidate, step: PlannedStep) -> bool:
    return (
        candidate.action_type == step.action_type
        and candidate.label == step.label
        and stable_candidate_params(candidate) == stable_step_params(step)
    )


def completed_items_were_removed(
    previous: DecisionContext, completed: list[PlannedStep], current: DecisionContext,
) -> bool:
    signatures = Counter(step_signature(step) for step in completed)
    for signature, count in signatures.items():
        before = sum(candidate_signature(item) == signature for item in previous.candidates)
        after = sum(candidate_signature(item) == signature for item in current.candidates)
        if before < count or after != before - count:
            return False
    return True


def candidate_signature(candidate: Candidate) -> tuple:
    return candidate.action_type, candidate.label, tuple(sorted(stable_candidate_params(candidate).items()))


def step_signature(step: PlannedStep) -> tuple:
    return step.action_type, step.label, tuple(sorted(stable_step_params(step).items()))


def shop_gold_change_is_expected(
    previous: DecisionContext, completed: list[PlannedStep], current: DecisionContext,
) -> bool:
    container = previous.situation.get(previous.decision_type) or {}
    shop = container.get("shop", {}) if previous.decision_type == "fake_merchant" else container
    price = 0
    for step in completed:
        item_index = step.params.get("index")
        item = next((value for value in shop.get("items", []) if value.get("index") == item_index), None)
        if item is None or not isinstance(item.get("price"), (int, float)):
            return False
        price += item["price"]
    old_gold = previous.player.get("gold")
    new_gold = current.player.get("gold")
    return (
        isinstance(old_gold, (int, float))
        and isinstance(new_gold, (int, float))
        and old_gold - new_gold == price
    )


def combat_transition_is_safe(
    previous: DecisionContext, completed: list[PlannedStep], current: DecisionContext,
) -> bool:
    if not completed or any(step.action_type != "play_card" for step in completed):
        return False
    old_battle = previous.situation.get("battle") or {}
    new_battle = current.situation.get("battle") or {}
    if (
        old_battle.get("round") != new_battle.get("round")
        or new_battle.get("turn") != "player"
        or not new_battle.get("is_play_phase", False)
    ):
        return False

    old_hand = previous.player.get("hand", [])
    expected = Counter(card_signature(card) for card in old_hand)
    for step in completed:
        card_index = step.params.get("card_index")
        played = next((card for card in old_hand if card.get("index") == card_index), None)
        if played is None or expected[card_signature(played)] <= 0:
            return False
        expected[card_signature(played)] -= 1
        if expected[card_signature(played)] == 0:
            del expected[card_signature(played)]
    actual = Counter(card_signature(card) for card in current.player.get("hand", []))
    return actual == expected


def card_signature(card: dict) -> tuple:
    return (
        card.get("id"),
        card.get("name"),
        card.get("cost"),
        card.get("description"),
        bool(card.get("is_upgraded")),
    )
