
VALID_ACTIONS = ("ANSWER", "SEARCH", "ABSTAIN")
INVALID_ACTION = "INVALID"

# An unrecognized decision is executed as ABSTAIN: the controller never
# answers on the strength of an output it could not parse. The parsed
# action stays INVALID and is scored as an incorrect decision.
INVALID_EXECUTION_FALLBACK = "ABSTAIN"

STRIP_CHARS = " \t\r\n.,;:!?\"'`*()[]{}<>-_"


def parse_action(text):
    """Return ANSWER, SEARCH, ABSTAIN, or INVALID for a raw model output."""
    normalized = text.strip(STRIP_CHARS).upper()

    if normalized in VALID_ACTIONS:
        return normalized

    return INVALID_ACTION


def execution_action(parsed_action):
    if parsed_action == INVALID_ACTION:
        return INVALID_EXECUTION_FALLBACK

    return parsed_action
