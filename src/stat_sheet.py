"""Standard counts from explicitly reviewed volleyball events."""
import pandas as pd

EVENTS = ["Attack attempt", "Kill", "Attack error", "Serve", "Ace", "Service error", "Assist", "Dig", "Solo block", "Block assist"]


def build_stat_sheet(events, players):
    checked = events[events["Reviewed"].eq(True)]
    rows = []
    for player in players:
        counts = checked.loc[checked.Player == player, "Event"].value_counts()
        count = lambda name: int(counts.get(name, 0))
        kills, errors = count("Kill"), count("Attack error")
        attempts = count("Attack attempt") + kills + errors
        rows.append({
            "Player": player, "Kills": kills, "Attack errors": errors,
            "Attack attempts": attempts,
            "Hitting percentage": round((kills-errors)/attempts, 3) if attempts else None,
            "Serve attempts": count("Serve")+count("Ace")+count("Service error"),
            "Aces": count("Ace"), "Service errors": count("Service error"),
            "Assists": count("Assist"), "Digs": count("Dig"),
            "Solo blocks": count("Solo block"), "Block assists": count("Block assist"),
        })
    return pd.DataFrame(rows)
