# tests/test_ml_rules.py

from helpdesk_ai.backend.app.ml.train_classifier import baseline_predict
from helpdesk_ai.backend.app.ml.predictor import compute_priority, route_to_queue


def test_baseline_predict_account():
    text = "I cannot login to my student account"
    assert baseline_predict(text) == "Account"


def test_baseline_predict_financy():
    text = "I have a question about my tuition payment"
    assert baseline_predict(text) == "Financy"


def test_baseline_predict_administration():
    text = "Question about admission application status"
    assert baseline_predict(text) == "Administration"


def test_baseline_predict_general():
    text = "When does the semester start?"
    assert baseline_predict(text) == "General"


def test_compute_priority_high_on_urgent_deadline():
    text = "URGENT: exam tomorrow and I cannot login to LMS"
    prio = compute_priority(text, "Account")
    assert prio == "High"


def test_compute_priority_medium_for_normal_issue():
    text = "I cannot access my email account"
    prio = compute_priority(text, "Account")
    assert prio == "Medium"


def test_compute_priority_low_for_general_question():
    text = "What are the library opening hours?"
    prio = compute_priority(text, "General")
    assert prio == "Low"


def test_route_to_queue_it():
    assert route_to_queue("Technical") == "IT"
    assert route_to_queue("Account") == "IT"


def test_route_to_queue_finance():
    assert route_to_queue("Financy") == "Finance"
    assert route_to_queue("Finance") == "Finance"


def test_route_to_queue_admissions_and_default():
    assert route_to_queue("Administration") == "Admissions"
    assert route_to_queue("UnknownCategory") == "General"
    assert route_to_queue(None) == "General"
