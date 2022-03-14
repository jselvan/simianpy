from .io import Trodes


def infer_session_name(path):
    matches = list(path.glob("*.raw"))
    if matches:
        session_name = matches[0].stem.replace(".raw", "")
        return session_name
    else:
        raise ValueError("Could not infer session name")
