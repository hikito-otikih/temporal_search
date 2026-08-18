"""Direct unit tests for the standalone-context repair/validation internals
(rewrite.service._repair_standalone_context and friends).

These bypass the HTTP retry loop entirely and construct RewriteResponse
objects by hand, so each scenario can be pinned exactly - reproducing the
specific field-level asymmetry that let repair raise an uncaught exception
requires precise control over which fields already satisfy the context
check and which don't, control that's fragile to get right indirectly
through Vietnamese-tokenization-dependent LLM-mock fixtures.
"""

import unittest

from pydantic import ValidationError

from rewrite.exceptions import SemanticValidationError
from rewrite.schemas import RewriteResponse
from rewrite.service import (
    _repair_standalone_context,
    _repair_standalone_context_or_fail,
    _validate_standalone_context,
)


def _event(**overrides) -> dict:
    base = {
        'event_id': 0,
        'original_query': 'Con rồng cử động đầu tại khu chợ hoa.',
        'target_moment_vi': 'Con rồng cử động đầu tại khu chợ hoa lần đầu tiên.',
        'retrieval_queries_vi': [
            'Con rồng cử động đầu tại khu chợ hoa.',
            'Khoảnh khắc đầu tiên rồng cử động ở chợ hoa.',
        ],
        'retrieval_queries_en': [
            'The dragon moves its head for the first time at the flower market.',
            'The moment the dragon first moves its head at the flower market.',
        ],
        'subject': 'con rồng',
        'action': 'cử động đầu',
        'visible_state': 'đầu rồng đang cử động',
        'anchor_query': 'Con rồng cử động đầu tại khu chợ hoa lần đầu tiên.',
        'pre_state': 'Trước đó đầu rồng chưa cử động.',
        'post_state': 'Sau đó đầu rồng đã cử động xong.',
        'boundary': 'start',
        'temporal_relation': {'relation': 'sequence_start', 'reference_event_id': None},
        'required_entities': ['con rồng'],
        'soft_context': ['chợ hoa'],
        'excluded_context': [],
        'inferred_information': [],
        'ambiguities': [],
    }
    base.update(overrides)
    return base


def _response(**event_overrides) -> RewriteResponse:
    payload = {
        'video_context': {'scene': 'chợ hoa ngày Tết', 'main_entities': ['một con rồng']},
        'events': [_event(**event_overrides)],
    }
    return RewriteResponse.model_validate(payload)


class RepairStandaloneContextTests(unittest.TestCase):
    def test_repair_does_not_raise_when_only_anchor_query_needs_context(self) -> None:
        # context_terms extracted from "chợ hoa Tết" = {chợ, hoa, tết}
        # (minimum_matches = 2). target_moment_vi/retrieval_queries_vi
        # already contain "chợ"/"hoa" (both also present in original_query,
        # so neither counts as *inherited* - satisfied without needing
        # repair). anchor_query has neither "chợ" nor "hoa" and needs
        # repair, which prepends the full "chợ hoa Tết" prefix - introducing
        # "tết" specifically, which is NOT in original_query and so IS a
        # genuinely new inherited term, but one repair's own narrow
        # standalone_terms check (before the fix, built only from
        # target_moment_vi + retrieval_queries_vi) could never see, since
        # neither of those fields ever contains "tết" at all. Before the
        # fix this raised SemanticValidationError uncaught; after the fix
        # the symmetric check sees it too and sets inferred_information
        # itself, so the final re-validation passes.
        common_query = 'chợ hoa Tết'
        analysis = _response(
            anchor_query='Con rồng cử động đầu.',
            inferred_information=[],
        )

        repaired = _repair_standalone_context(analysis, common_query)

        self.assertIn('tết', repaired.events[0].anchor_query.casefold())
        self.assertTrue(repaired.events[0].inferred_information)
        # Must not raise - this call is the fixed function's own
        # post-repair check, reproduced here for an explicit assertion.
        _validate_standalone_context(repaired, common_query)

    def test_repair_reraises_max_length_overflow_via_full_revalidation(self) -> None:
        # anchor_query sits right at the 4000-char schema limit already;
        # repair's context_prefix prepend pushes it over. Attribute
        # assignment during repair doesn't re-check max_length on its own
        # (validate_assignment isn't enabled) - only the full
        # RewriteResponse.model_validate() round-trip at the end of
        # _repair_standalone_context catches this, converted by the wrapper
        # into a normal OllamaServiceError instead of shipping an
        # over-length field with the response.
        from rewrite.exceptions import OllamaServiceError

        long_anchor = 'a' * 3995  # short of 4000, but "chợ hoa. " prepended pushes it over
        common_query = 'chợ hoa'
        analysis = _response(anchor_query=long_anchor, inferred_information=[])

        with self.assertRaises(OllamaServiceError):
            _repair_standalone_context_or_fail(analysis, common_query)


class EnglishEchoValidationTests(unittest.TestCase):
    def test_english_query_identical_to_original_query_is_rejected(self) -> None:
        common_query = 'chợ hoa Tết'
        analysis = _response(
            original_query='the dragon moves its head',
            retrieval_queries_en=[
                'the dragon moves its head',  # byte-identical to original_query
                'A second, genuinely translated query.',
            ],
        )
        with self.assertRaises(SemanticValidationError):
            _validate_standalone_context(analysis, common_query)

    def test_english_query_left_in_vietnamese_is_rejected(self) -> None:
        # Regression test for a real reported bug: only a byte-identical
        # echo of original_query was rejected, so an LLM that paraphrased
        # (rather than translated) into Vietnamese - or just left the field
        # in Vietnamese outright - passed validation and returned HTTP 200.
        # This query is not identical to original_query, so the echo check
        # alone would miss it; the Vietnamese-signature check catches it on
        # script alone (chợ/hoa/lần/đầu/tiên all carry Vietnamese diacritics
        # outside plain English text).
        common_query = 'chợ hoa Tết'
        analysis = _response(
            retrieval_queries_en=[
                'Con rồng cử động đầu tại khu chợ hoa lần đầu tiên.',
                'A second, genuinely translated query.',
            ],
        )
        with self.assertRaises(SemanticValidationError):
            _validate_standalone_context(analysis, common_query)

    def test_genuinely_translated_english_query_passes(self) -> None:
        # original_query left at _event()'s default, which already contains
        # "chợ hoa" - matching target_moment_vi/anchor_query/retrieval_
        # queries_vi's default text, so nothing is genuinely "inherited"
        # and the unrelated inferred_information check stays out of the way
        # of what this test is actually about: the EN-echo check.
        common_query = 'chợ hoa Tết'
        analysis = _response(
            retrieval_queries_en=[
                'The dragon at the flower market moves its head for the first time.',
                'The moment the dragon first moves its head at the Tet flower market.',
            ],
        )
        _validate_standalone_context(analysis, common_query)


if __name__ == '__main__':
    unittest.main()
