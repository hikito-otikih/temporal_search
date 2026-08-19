from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


NonEmptyText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
]
Boundary = Literal['start', 'end', 'state', 'transition', 'interval', 'unknown']
TemporalRelationType = Literal[
    'sequence_start',
    'after',
    'before',
    'during',
    'simultaneous',
    'independent',
    'unknown',
]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class RewriteRequest(StrictSchema):
    common_query: str | None = Field(default=None, max_length=8000)
    query: list[NonEmptyText] = Field(min_length=1, max_length=32)

    @field_validator('common_query', mode='before')
    @classmethod
    def normalize_empty_common_query(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode='after')
    def limit_total_query_length(self):
        if sum(len(item) for item in self.query) > 24000:
            raise ValueError('query content must not exceed 24000 characters')
        return self


class VideoContext(StrictSchema):
    scene: NonEmptyText
    main_entities: list[NonEmptyText] = Field(min_length=1, max_length=32)


class TemporalRelation(StrictSchema):
    relation: TemporalRelationType
    reference_event_id: int | None

    @model_validator(mode='after')
    def validate_reference_presence(self):
        relation_requires_reference = self.relation in {
            'after',
            'before',
            'during',
            'simultaneous',
        }
        if relation_requires_reference and self.reference_event_id is None:
            raise ValueError(
                f'{self.relation} requires a reference_event_id'
            )
        if not relation_requires_reference and self.reference_event_id is not None:
            raise ValueError(
                f'{self.relation} requires reference_event_id to be null'
            )
        return self


class RewrittenEvent(StrictSchema):
    event_id: int = Field(ge=0)
    original_query: NonEmptyText
    target_moment_vi: NonEmptyText
    retrieval_queries_vi: list[NonEmptyText] = Field(min_length=2, max_length=2)
    retrieval_queries_en: list[NonEmptyText] = Field(min_length=2, max_length=2)
    # Self-report, parallel to retrieval_queries_en (same length, same
    # order): the model's own judgment of whether each entry is genuinely
    # English, re-checked as a distinct field rather than assumed from
    # having followed the earlier "write this in English" instruction.
    # Deliberately not trusted alone - _validate_standalone_context in
    # service.py still runs py3langid independently on the same text; a
    # model that mislabels its own output (or rubber-stamps "en" without
    # re-checking) is still caught by that server-side check.
    retrieval_queries_en_language: list[Literal['en', 'not_en']] = Field(
        min_length=2, max_length=2
    )
    subject: NonEmptyText
    action: NonEmptyText
    visible_state: NonEmptyText
    anchor_query: NonEmptyText
    pre_state: NonEmptyText
    post_state: NonEmptyText
    boundary: Boundary
    temporal_relation: TemporalRelation
    required_entities: list[NonEmptyText] = Field(max_length=32)
    soft_context: list[NonEmptyText] = Field(max_length=32)
    excluded_context: list[NonEmptyText] = Field(max_length=32)
    inferred_information: list[NonEmptyText] = Field(max_length=32)
    ambiguities: list[NonEmptyText] = Field(max_length=32)


class RewriteResponse(StrictSchema):
    video_context: VideoContext
    events: list[RewrittenEvent] = Field(min_length=1, max_length=32)

    @model_validator(mode='after')
    def validate_event_ids_and_references(self):
        event_ids = [event.event_id for event in self.events]
        expected_ids = list(range(len(self.events)))
        if event_ids != expected_ids:
            raise ValueError(
                f'event_id values must be contiguous and ordered: {expected_ids}'
            )

        if self.events[0].temporal_relation.relation != 'sequence_start':
            raise ValueError('event 0 must use sequence_start')
        if any(
            event.temporal_relation.relation == 'sequence_start'
            for event in self.events[1:]
        ):
            raise ValueError('only event 0 may use sequence_start')

        known_ids = set(event_ids)
        for event in self.events:
            relation = event.temporal_relation
            reference_id = relation.reference_event_id
            if relation.relation == 'sequence_start' and reference_id is not None:
                raise ValueError(
                    'sequence_start must have reference_event_id set to null'
                )
            if reference_id is not None:
                if reference_id not in known_ids:
                    raise ValueError(
                        f'event {event.event_id} references unknown event '
                        f'{reference_id}'
                    )
                if reference_id == event.event_id:
                    raise ValueError(
                        f'event {event.event_id} cannot reference itself'
                    )
        return self
