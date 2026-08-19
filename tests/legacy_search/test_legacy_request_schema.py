"""Regression test for a real reported bug: objectFilterMode=true with no
object_name_list silently became an accidental "does this video have
object-detection metadata at all" filter instead of a real content filter -
check_object_satisfaction() builds required_object_names from
object_name_list, and an empty set is trivially a subset of anything, so
satisfiedObjects only ever went False when the detection JSON itself was
missing/unparseable. TemporalSearchRequest now rejects that combination
outright at the request boundary instead of accepting it and silently doing
something the caller almost certainly didn't intend.
"""

import unittest

from pydantic import ValidationError

from legacy_search.schemas import TemporalSearchRequest


class TemporalSearchRequestObjectFilterValidationTests(unittest.TestCase):
    def test_object_filter_mode_without_labels_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            TemporalSearchRequest(query=["cat"], objectFilterMode=True)

    def test_object_filter_mode_with_empty_label_list_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            TemporalSearchRequest(query=["cat"], objectFilterMode=True, object_name_list=[])

    def test_object_filter_mode_with_labels_is_accepted(self) -> None:
        request = TemporalSearchRequest(
            query=["cat"], objectFilterMode=True, object_name_list=["cat", "dog"]
        )
        self.assertEqual(request.object_name_list, ["cat", "dog"])

    def test_object_filter_mode_off_does_not_require_labels(self) -> None:
        request = TemporalSearchRequest(query=["cat"])
        self.assertFalse(request.objectFilterMode)
        self.assertIsNone(request.object_name_list)


if __name__ == "__main__":
    unittest.main()
