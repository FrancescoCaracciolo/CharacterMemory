"""Provider-independent, bounded decision questions and validated answers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal
import json
import math


class DecisionError(ValueError):
    """Invalid request/response or unavailable decision provider."""


@dataclass(frozen=True)
class BooleanQuestion:
    instructions: str
    criteria: dict[str, str] | None = None


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str
    criteria: dict[str, str]


@dataclass(frozen=True)
class ScoreQuestion:
    instructions: str
    criteria: list[str]


Question = BooleanQuestion | ChoiceQuestion | ScoreQuestion
ProbabilityOrigin = Literal['native', 'estimated', 'unavailable']


@dataclass(frozen=True)
class DecisionRequest:
    state: Any
    questions: dict[str, Question]

    def payload(self) -> dict:
        if not isinstance(self.state, (str, dict, list)):
            raise DecisionError('State must be text, an object, or an array')
        if not isinstance(self.questions, dict) or not self.questions or any(not isinstance(k, str) or not k for k in self.questions):
            raise DecisionError('Questions require non-empty names')
        questions = {}
        for key, q in self.questions.items():
            if not isinstance(q, (BooleanQuestion, ChoiceQuestion, ScoreQuestion)):
                raise DecisionError('Unsupported question type')
            if not isinstance(q.instructions, str) or not q.instructions:
                raise DecisionError('Instructions must be non-empty text')
            kind = 'choice' if isinstance(q, ChoiceQuestion) else 'score' if isinstance(q, ScoreQuestion) else 'noul'
            if isinstance(q, ChoiceQuestion) and (not isinstance(q.criteria, dict) or not q.criteria or any(not isinstance(k, str) or not k or not isinstance(v, str) for k, v in q.criteria.items())):
                raise DecisionError('Choices require named labels')
            if isinstance(q, ScoreQuestion) and (not isinstance(q.criteria, list) or not q.criteria or any(not isinstance(v, str) for v in q.criteria)):
                raise DecisionError('Scores require an ordered rubric')
            if isinstance(q, BooleanQuestion) and q.criteria is not None and (not isinstance(q.criteria, dict) or not set(q.criteria) <= {'true', 'false'}):
                raise DecisionError('Boolean criteria must use true/false keys')
            questions[key] = dict(type=kind, instructions=q.instructions)
            if q.criteria is not None:
                questions[key]['criteria'] = q.criteria
        result = dict(state=self.state, questions=questions)
        try:
            json.dumps(result, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise DecisionError('State and questions must be JSON compatible') from exc
        return result


@dataclass(frozen=True)
class BooleanAnswer:
    value: bool
    probability: float | None = None
    confidence: float | None = None
    origin: ProbabilityOrigin = 'unavailable'


@dataclass(frozen=True)
class ChoiceAnswer:
    value: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    origin: ProbabilityOrigin = 'unavailable'


@dataclass(frozen=True)
class ScoreAnswer:
    value: float
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    origin: ProbabilityOrigin = 'unavailable'


@dataclass(frozen=True)
class DecisionResponse:
    answers: dict[str, BooleanAnswer | ChoiceAnswer | ScoreAnswer]
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None


def number(value, low=0.0, high=1.0):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
        raise DecisionError('Invalid numeric answer')
    return float(value)


def parse_response(request: DecisionRequest, raw: dict, origin: ProbabilityOrigin = 'native') -> DecisionResponse:
    """Strict boundary validation shared by HTTP and structured-LLM adapters."""
    request.payload()
    try:
        if not isinstance(raw['model'], str) or not raw['model']:
            raise DecisionError('Missing resolved model')
        if set(raw['answers']) != set(request.questions):
            raise DecisionError('Response question IDs do not match request')
        answers = {}
        for key, q in request.questions.items():
            a = raw['answers'][key]
            expected = 'choice' if isinstance(q, ChoiceQuestion) else 'score' if isinstance(q, ScoreQuestion) else 'noul'
            if a['type'] != expected:
                raise DecisionError('Answer type does not match question')
            confidence = number(a['confidence']) if 'confidence' in a else None
            if isinstance(q, BooleanQuestion):
                p = number(a['noul'])
                answers[key] = BooleanAnswer(p >= .5, p, confidence, origin)
                continue
            labels = set(q.criteria) if isinstance(q, ChoiceQuestion) else {str(i) for i in range(len(q.criteria))}
            probs = a.get('probabilities')
            if probs is not None:
                if set(probs) != labels:
                    raise DecisionError('Probability labels do not match criteria')
                probs = {k: number(v) for k, v in probs.items()}
                if abs(sum(probs.values()) - 1) > .001:
                    raise DecisionError('Probabilities must sum to one')
            elif origin == 'native':
                raise DecisionError('Native answer missing probabilities')
            if isinstance(q, ChoiceQuestion):
                value = a['choice']
                if value not in labels or (probs and probs[value] < max(probs.values()) - 1e-8):
                    raise DecisionError('Invalid selected choice')
                answers[key] = ChoiceAnswer(value, probs, confidence, origin if probs else 'unavailable')
            else:
                value = number(a['score'], 0, len(q.criteria) - 1)
                if probs and abs(value - sum(int(k) * p for k, p in probs.items())) > .01:
                    raise DecisionError('Score does not match distribution')
                answers[key] = ScoreAnswer(value, probs, confidence, origin if probs else 'unavailable')
        return DecisionResponse(answers, raw['model'], raw.get('usage', {}), raw.get('id'))
    except (KeyError, TypeError, AttributeError) as exc:
        raise DecisionError('Malformed decision response') from exc


class DecisionClient(ABC):
    @abstractmethod
    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Evaluate named questions without changing application state."""

    def decide_many(self, requests: list[DecisionRequest]) -> list[DecisionResponse]:
        return [self.decide(request) for request in requests]
