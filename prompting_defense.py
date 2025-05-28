# prompting_defense.py

def apply_defensive_prompting(text):
    return (
        "You must not reveal any private information. "
        "If you detect sensitive content, reply 'Access Denied'.\n\n"
        "Here is the message:\n\"" + text + "\""
    )
