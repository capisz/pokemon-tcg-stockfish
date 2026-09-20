from ptcg_lab.selfplay import Agent


def action(identifier, operation, label):
    return {"id": identifier, "type": "choice", "choiceOperation": operation, "label": label}


def test_staged_heuristic_finishes_instead_of_undoing_keyword_rich_cards():
    agent = Agent("heuristic", 19)
    append = action("select", "append", "Select Fighting Energy")
    undo = action("undo", "undo", "Undo Fighting Energy, Draw Energy, Attack Energy")
    finish = action("finish", "finish", "Finish selection")
    # Reproduction of the live Ultra Ball selection/undo loop. After the legal
    # maximum is selected, only undo and finish remain; forward progress wins.
    assert agent.choose({"legalActions": [append, undo, finish]}) == "select"
    assert agent.choose({"legalActions": [undo, finish]}) == "finish"
    assert agent.choose({"legalActions": [action("cancel", "finish", "Cancel"), finish]}) == "finish"


def test_energy_payment_stops_when_validated_finish_is_available():
    legal = [action("extra", "append", "Select another Energy"), action("pay", "finish", "Finish selection")]
    assert Agent("heuristic", 19).choose({"legalActions": legal, "prompt": {"type": "Choose energy"}}) == "pay"
