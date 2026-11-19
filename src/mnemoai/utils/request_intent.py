"""Conservative, local recognition of messages that do not request tool work."""

import re

_SOCIAL = re.compile(
    r"^(?:hi|hello|hey|thanks|thank you|bye|goodbye|ciao|salve|grazie|buongiorno|"
    r"buonasera|arrivederci)[\s!.?]*$", re.IGNORECASE,
)


def obvious_conversation(prompt: str) -> bool:
    """Only empty input and unambiguous greetings/acknowledgments, never task prefixes."""
    return not prompt.strip() or bool(_SOCIAL.fullmatch(prompt.strip()))
