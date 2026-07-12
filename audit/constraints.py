"""The four data-quality constraints audited in E1 (ka1t #5, sV8H #1).

The first three are the construction constraints stated verbatim in the paper
(§4.1). The fourth (speaker consistency) is the annotation-integrity check the
rebuttal plan adds. This module is the single source of truth: the LLM
pre-screener, the annotation sheets, and the aggregator all import KEYS/CONSTRAINTS
from here so the rubric shown to the authors is identical to the one the model saw.
"""

# key -> (short label, paper/plan definition, the yes/no question the annotator answers)
CONSTRAINTS = {
    "spatial_separation": (
        "Spatial Separation",
        "The scenario must involve one speaker experiencing an event while conversing "
        "with another speaker about it. This ensures there are no contradictions arising "
        "from both speakers being in the same spatial context.",
        "Is exactly one speaker experiencing the narrated event while physically apart "
        "from the other, with no contradiction that would place both in the same place?",
    ),
    "temporal_implicitness": (
        "Temporal Implicitness",
        "The (timely) response must avoid direct references to the elapsed time. This "
        "prevents dull time-acknowledging responses and lexical overlap with the "
        "ground-truth time interval that would create a shortcut.",
        "Does the timely_response avoid naming/echoing the elapsed time interval "
        "(no explicit duration that overlaps the ground-truth time_elapsed)?",
    ),
    "mutual_exclusivity": (
        "Mutual Exclusivity",
        "The time-conditioned response must become untimely under contrary temporal "
        "conditions: a delayed (timely) response should be incoherent with no interval, "
        "and an instant (untimely) response should be incoherent when an interval exists. "
        "This prevents time-agnostic responses that stay coherent regardless of timing.",
        "Would the timely_response become incoherent if delivered instantly, AND the "
        "untimely_response become incoherent after the elapsed time — i.e. are the two "
        "genuinely time-exclusive rather than time-agnostic?",
    ),
    "speaker_consistency": (
        "Speaker Consistency",
        "The speaker labels are internally consistent: turns alternate coherently, and "
        "target_speaker is the event-experiencing speaker who produces the delayed "
        "response, with no role swap mid-dialogue.",
        "Are the speaker labels consistent and is target_speaker correctly the "
        "event-experiencing speaker who gives the timely (delayed) response?",
    ),
    "duration_validity": (
        "Duration Validity",
        "The ground-truth elapsed time is a realistic duration for the narrated event: "
        "long enough for the event to plausibly complete, and not implausibly long or "
        "short for what the narrative describes.",
        "Is the ground-truth time_elapsed a realistic amount of time for the narrated "
        "event to take?",
    ),
}

KEYS = list(CONSTRAINTS)


def render_sample(ex: dict) -> str:
    """Human/LLM-readable rendering of one sample for judging."""
    lines = [
        f"Narrative (seed event): {ex['narrative']}",
        f"Ground-truth elapsed time before the timely response: {ex['time_elapsed']}",
        f"Target speaker (should be the event experiencer): {ex['target_speaker']}",
        "Dialogue context:",
    ]
    for spk, utt in zip(ex["speaker_list"], ex["context"]):
        lines.append(f"    {spk}: {utt}")
    lines.append(f"timely_response (delivered after the elapsed time): {ex['timely_response']}")
    lines.append(f"untimely_response (delivered instantly): {ex['untimely_response']}")
    return "\n".join(lines)
