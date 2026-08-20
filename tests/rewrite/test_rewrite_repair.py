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
    _is_english,
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
        'retrieval_queries_en_language': ['en', 'en'],
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
        # original_query left at _event()'s Vietnamese default (not
        # overridden to English text) - the echo check only applies when
        # the source actually needed translating; see the test below for
        # the opposite, English-original case.
        common_query = 'chợ hoa Tết'
        analysis = _response(
            retrieval_queries_en=[
                'Con rồng cử động đầu tại khu chợ hoa.',  # byte-identical to original_query
                'A second, genuinely translated query.',
            ],
        )
        with self.assertRaises(SemanticValidationError):
            _validate_standalone_context(analysis, common_query)

    def test_identical_copy_of_an_already_english_original_query_is_accepted(self) -> None:
        # Regression test for a real reported bug: the echo check used to
        # be unconditional, rejecting ANY retrieval_queries_en entry that
        # byte-matched original_query - but when original_query is itself
        # already English (a mixed-language or already-English source
        # query), an unchanged copy IS the correct "translation"; there is
        # nothing to translate. The old check misreported this valid, exact
        # copy as if it were an untranslated echo of Vietnamese source text.
        common_query = None
        analysis = _response(
            original_query='the dragon moves its head',
            target_moment_vi='the dragon moves its head',
            anchor_query='the dragon moves its head',
            retrieval_queries_vi=[
                'the dragon moves its head',
                'the moment the dragon moves its head',
            ],
            retrieval_queries_en=[
                'the dragon moves its head',  # byte-identical to original_query - now valid
                'the moment the dragon moves its head',
            ],
        )
        _validate_standalone_context(analysis, common_query)  # must not raise

    def test_english_query_left_in_vietnamese_is_rejected(self) -> None:
        # Regression test for a real reported bug: only a byte-identical
        # echo of original_query was rejected, so an LLM that paraphrased
        # (rather than translated) into Vietnamese - or just left the field
        # in Vietnamese outright - passed validation and returned HTTP 200.
        # This query is not identical to original_query, so the echo check
        # alone would miss it.
        #
        # retrieval_queries_en_language[0] is set to "not_en" - self-report
        # is the authoritative gate (see _validate_standalone_context's own
        # comment), so this simulates a model that correctly notices its
        # own untranslated output, not bad text with no signal of it.
        common_query = 'chợ hoa Tết'
        analysis = _response(
            retrieval_queries_en=[
                'Con rồng cử động đầu tại khu chợ hoa lần đầu tiên.',
                'A second, genuinely translated query.',
            ],
            retrieval_queries_en_language=['not_en', 'en'],
        )
        with self.assertRaises(SemanticValidationError):
            _validate_standalone_context(analysis, common_query)

    def test_langid_disagreement_alone_does_not_reject_when_self_report_says_en(self) -> None:
        # Regression test for a real reported bug: a live 18-video YouCook2
        # benchmark found py3langid alone (even with LANGID_ENGLISH_MARGIN's
        # rank-based tolerance) still occasionally misclassifies genuinely
        # correct English - and because the misclassification is often
        # systematic for a given phrasing, retrying just reproduces
        # similarly-phrased text that fails the same way, exhausting
        # MAX_VALIDATION_ATTEMPTS and hard-failing the whole video's
        # rewrite call even though the output was correct. Every observed
        # failure had self_report correctly saying "en" while only
        # py3langid disagreed - blocking on that disagreement was the bug,
        # not a feature. This uses the same ASCII-Vietnamese sentence the
        # old test suite used to assert *rejection* on py3langid's verdict
        # alone; the point now is the opposite - self-report overrides it.
        common_query = 'chợ hoa Tết'
        analysis = _response(
            retrieval_queries_en=[
                'Con rong cu dong dau tai khu cho hoa.',
                'A second, genuinely translated query.',
            ],
            retrieval_queries_en_language=['en', 'en'],
        )
        _validate_standalone_context(analysis, common_query)  # must not raise

    def test_self_reported_not_en_is_rejected_even_when_langid_is_fooled(self) -> None:
        # Regression test for exactly why two independent signals matter:
        # this ASCII-Vietnamese sentence ("Chiec thuyen luot nhe tren song
        # luc hoang hon.") is a MEASURED py3langid false negative - it gets
        # classified as English (see rewrite py3langid benchmark). If the
        # model's own self-report correctly flags it as not_en, that must
        # still be enough to reject it - the self-report is not redundant
        # with py3langid, it catches a real, measured gap in it.
        common_query = 'chợ hoa Tết'
        analysis = _response(
            retrieval_queries_en=[
                'Chiec thuyen luot nhe tren song luc hoang hon.',
                'A second, genuinely translated query.',
            ],
            retrieval_queries_en_language=['not_en', 'en'],
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


class IsEnglishDetectionTests(unittest.TestCase):
    """Direct unit tests for _is_english()'s own detection accuracy -
    separate from EnglishEchoValidationTests, which exercises the full
    _validate_standalone_context flow where self-report (not _is_english
    alone) determines rejection. These preserve the regression coverage for
    what _is_english() itself can and can't detect, independent of how its
    verdict is used."""

    def test_diacritic_vietnamese_is_detected(self) -> None:
        # The old character-set regex only matched a narrow Latin Extended
        # Additional range plus đ/ơ/ư/ă, missing plain single-diacritic
        # Vietnamese like this entirely.
        self.assertFalse(_is_english('Múa lân màu vàng.'))

    def test_ascii_vietnamese_is_usually_detected(self) -> None:
        # Not guaranteed (statistical, not script-based) - but this exact
        # sentence is a measured, reliable catch.
        self.assertFalse(_is_english('Con rong cu dong dau tai khu cho hoa.'))

    def test_preserved_vietnamese_proper_noun_does_not_fail_english(self) -> None:
        self.assertTrue(
            _is_english('A person walking through the old town of Hội An in the evening.')
        )

    def test_previously_false_positive_english_sentences_now_pass(self) -> None:
        # Regression test for a real reported bug: LANGID_ENGLISH_MARGIN's
        # own comment has the full story - classify()'s top-1-only verdict
        # misclassified these two live-observed, genuinely correct English
        # sentences as non-English on every one of 3 separate live
        # generations, which (before self-report became the authoritative
        # gate) hard-failed the whole video's rewrite call. In both cases
        # English was a close runner-up (measured via langid.rank()), not a
        # confidently-rejected outsider - exactly what rank()+margin fixes.
        self.assertTrue(_is_english('cutting cheese into cubes during salad preparation'))
        self.assertTrue(
            _is_english('sprinkling salt on garlic during Caesar salad preparation')
        )
        self.assertTrue(_is_english('The chef chops vegetables quickly.'))


if __name__ == '__main__':
    unittest.main()
