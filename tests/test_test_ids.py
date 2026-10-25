"""Guard: test IDs stay short enough for Windows.

pytest stores the running test's ID in the PYTEST_CURRENT_TEST environment
variable. Windows limits an environment variable to 32,767 characters, so a
parametrized test whose ID embeds a huge value fails there (but passes on
Linux and macOS). Give such tests explicit ``ids=``.
"""

MAX_ID_LENGTH = 500


def test_all_test_ids_are_short(request):
    too_long = [
        f"{item.nodeid[:80]}... ({len(item.nodeid)} characters)"
        for item in request.session.items
        if len(item.nodeid) > MAX_ID_LENGTH
    ]
    assert not too_long, "add explicit ids= to these parametrized tests:\n" + "\n".join(too_long)
