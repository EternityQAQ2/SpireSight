"""Supported card-effect grammar and resource costs; pure functions."""
from __future__ import annotations

import re


def energy_cost(card: dict, energy: int) -> int | None:
    cost = str(card.get("cost", ""))
    return energy if cost.upper() == "X" else int(cost) if cost.isdigit() else None


def energy_gain(card: dict) -> int:
    # Description contains icons rather than numbers on the current mod.
    if card.get("id") == "BLOODLETTING":
        return 3 if card.get("is_upgraded") else 2
    return 0


def known_effect(card: dict) -> str | None:
    """Whitelist effects AND their description grammar; enchanted cards stop here."""
    text = re.sub(r"\s+", "", str(card.get("description", "")))
    patterns = {
        "strike": r"造成\d+点伤害。",
        "perfected_strike": r"造成\d+点伤害。你每有一张名字中含有“打击”的牌，伤害\+2。",
        "defend": r"获得\d+点格挡。",
        "bash": r"造成\d+点伤害。给予\d+层易伤。",
        "taunt": r"获得\d+点格挡。给予\d+层易伤。",
        "iron_wave": r"获得\d+点格挡。造成\d+点伤害。",
        "body_slam": r"造成你当前格挡值的伤害。(?:（造成\d+点伤害）)?",
        "inflame": r"获得\d+点力量。",
        "setup_strike": r"造成\d+点伤害。在本回合内获得\d+点力量。",
        "twin_strike": r"造成\d+点伤害两次。",
        "rage": r"打出此牌后，你在这个回合内每打出一张攻击牌，获得\d+点格挡。",
        "shrug": r"获得\d+点格挡。抽\d+张牌。",
        "pommel": r"造成\d+点伤害。抽\d+张牌。",
    }
    ident = str(card.get("id", "")).upper()
    effect = ("strike" if ident.startswith("STRIKE_") else
              "defend" if ident.startswith("DEFEND_") else
              {"BASH": "bash", "TAUNT": "taunt", "IRON_WAVE": "iron_wave",
               "BODY_SLAM": "body_slam", "INFLAME": "inflame", "BLUDGEON": "strike",
               "PERFECTED_STRIKE": "perfected_strike",
               "SETUP_STRIKE": "setup_strike", "TWIN_STRIKE": "twin_strike",
               "RAGE": "rage", "SHRUG_IT_OFF": "shrug", "POMMEL_STRIKE": "pommel"}.get(ident))
    if effect and re.fullmatch(patterns[effect], text):
        return effect
    return None
