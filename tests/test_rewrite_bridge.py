import unittest
from unittest.mock import AsyncMock, patch

from adaptive_search.rewrite_bridge import (
    BOUNDARY_MAPPING,
    build_session_plan,
    rewrite_queries_and_build_plan,
    variant_texts_for_events,
)
from rewrite_schema import RewriteResponse


def rewritten_event(index, query, boundary="start"):
    return {
        "event_id": index,
        "original_query": query,
        "target_moment_vi": f"Khoảnh khắc con lân màu vàng đen trắng ở sự kiện {index}.",
        "anchor_query": f"Con lân màu vàng đen trắng trong màn múa lân ở sự kiện {index}.",
        "retrieval_queries_vi": [
            f"Con lân màu vàng đen trắng thực hiện hành động ở sự kiện {index}.",
            f"Khoảnh khắc lân màu vàng đen trắng ở sự kiện {index}.",
        ],
        "retrieval_queries_en": [
            f"The yellow, black, and white lion performs in event {index}.",
            f"The lion-dance moment in event {index}.",
        ],
        "subject": "con lân",
        "action": "thực hiện hành động",
        "visible_state": "hành động của con lân có thể nhìn thấy",
        "pre_state": f"Ngay trước sự kiện {index}, con lân chưa thực hiện hành động.",
        "post_state": f"Ngay sau sự kiện {index}, con lân vừa hoàn thành hành động.",
        "boundary": boundary,
        "temporal_relation": {
            "relation": "sequence_start" if index == 0 else "independent",
            "reference_event_id": None,
        },
        "required_entities": ["con lân"],
        "soft_context": ["màn múa lân"],
        "excluded_context": [],
        "inferred_information": ["Từ common_query: lân màu vàng đen trắng."],
        "ambiguities": [],
    }


def analysis(queries, boundaries=None):
    boundaries = boundaries or ["start"] * len(queries)
    return RewriteResponse.model_validate(
        {
            "video_context": {
                "scene": "múa lân",
                "main_entities": ["một con lân màu vàng, đen và trắng"],
            },
            "events": [
                rewritten_event(index, query, boundaries[index])
                for index, query in enumerate(queries)
            ],
        }
    )


class RewriteBridgeTests(unittest.TestCase):
    def test_event_mapping_preserves_rewrite_fields(self):
        plan = build_session_plan(
            analysis(["Khoảnh khắc đầu tiên con lân xoay vòng."]),
            common_query="Video múa lân.",
        )

        self.assertEqual(plan.common_query, "Video múa lân.")
        (event,) = plan.events
        self.assertEqual(event.event_id, "evt0")
        self.assertEqual(event.original_query, "Khoảnh khắc đầu tiên con lân xoay vòng.")
        self.assertIn("Con lân màu vàng đen trắng", event.anchor_query)
        self.assertIn("chưa thực hiện", event.pre_state)
        self.assertIn("hoàn thành", event.post_state)
        self.assertEqual(event.boundary_type, "onset")

    def test_boundary_literals_map_to_domain_profiles(self):
        self.assertEqual(BOUNDARY_MAPPING["start"], "onset")
        self.assertEqual(BOUNDARY_MAPPING["end"], "offset")
        self.assertEqual(BOUNDARY_MAPPING["transition"], "transition")
        self.assertEqual(BOUNDARY_MAPPING["state"], "state")
        self.assertEqual(BOUNDARY_MAPPING["interval"], "state")
        self.assertEqual(BOUNDARY_MAPPING["unknown"], "unknown")

        plan = build_session_plan(
            analysis(
                ["Sự kiện bắt đầu.", "Sự kiện kết thúc.", "Trạng thái kéo dài."],
                boundaries=["start", "end", "state"],
            )
        )
        self.assertEqual(
            [event.boundary_type for event in plan.events],
            ["onset", "offset", "state"],
        )

    def test_variants_are_ordered_and_deduplicated(self):
        plan = build_session_plan(analysis(["Sự kiện một."]))
        variants = plan.variant_texts_for("evt0")

        self.assertEqual(len(variants), 4)
        self.assertEqual(len(variants), len(set(variants)))

    def test_variant_fallback_to_anchor_query(self):
        event = plan_events = build_session_plan(analysis(["Sự kiện một."])).events
        texts = variant_texts_for_events(
            {},
            event,
            ["evt0"],
        )
        self.assertEqual(texts["evt0"], (event[0].anchor_query,))

    def test_variant_selection_rejects_unknown_events(self):
        events = build_session_plan(analysis(["Sự kiện một."])).events
        with self.assertRaises(ValueError):
            variant_texts_for_events({}, events, ["nope"])

    def test_wrapper_builds_plan_from_rewrite_call(self):
        import asyncio

        expected = analysis(["Sự kiện một."])
        rewrite_mock = AsyncMock(return_value=expected)

        with patch(
            "adaptive_search.rewrite_bridge.rewrite_queries", rewrite_mock
        ):
            plan = asyncio.run(
                rewrite_queries_and_build_plan(
                    queries=["Sự kiện một."],
                    common_query="Video múa lân.",
                )
            )

        rewrite_mock.assert_awaited_once_with(
            queries=["Sự kiện một."],
            common_query="Video múa lân.",
        )
        self.assertEqual(plan.events[0].event_id, "evt0")
        self.assertEqual(
            plan.retrieval_variants["evt0"][0], expected.events[0].retrieval_queries_vi[0]
        )


if __name__ == "__main__":
    unittest.main()
