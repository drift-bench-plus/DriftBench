

# ---------------------------------------------- harness fidelity (2026-08-24)
def test_tool_name_on_the_action_line_is_reassembled():
    """Some models write `Action: <tool>` + a block of ARGUMENTS instead of
    `Action: Operation` + `tool(args)`. Both express the same call. The old parser
    kept the block verbatim, dropped the function name, and the model looped until
    the turn cap (glm-4.7: 63.5 calls/episode, 52% turn-capped, Success 24% with
    correct intent). Reassembly is a fidelity fix, not a mechanism change."""
    from intent_graph.runtime.agent_api import parse_action, ActionKind
    a = parse_action('Action: get_reservation_details\n```\nreservation_id="OBUT9V"\n```')
    assert a.kind is ActionKind.ACT
    assert a.command == 'get_reservation_details(reservation_id="OBUT9V")'


def test_standard_wire_format_is_untouched_by_reassembly():
    from intent_graph.runtime.agent_api import parse_action, ActionKind
    a = parse_action('Action: Operation\n```\nget_user_details(user_id="x_1")\n```')
    assert a.kind is ActionKind.ACT
    assert a.command == 'get_user_details(user_id="x_1")'


def test_answer_and_clarify_still_win_over_named_operation():
    from intent_graph.runtime.agent_api import parse_action, ActionKind
    assert parse_action("Action: Answer\nFinal Answer: done").kind is ActionKind.PROPOSE
    assert parse_action("Action: Clarify\nContent: which one?").kind is ActionKind.ASK


def test_multiline_arguments_are_comma_joined():
    """glm-4.7 writes one argument per line with no commas. Joining verbatim made
    invalid call syntax; one measured episode retried the same broken call 91 times."""
    from intent_graph.runtime.agent_api import parse_action, ActionKind
    a = parse_action('Action: search_direct_flight\n```\norigin="IAH"\n'
                     'destination="JFK"\ndate="2024-05-20"\n```')
    assert a.kind is ActionKind.ACT
    assert a.command == ('search_direct_flight(origin="IAH", destination="JFK", '
                         'date="2024-05-20")')


def test_wire_keywords_are_never_treated_as_tool_names():
    """REGRESSION 2026-08-25: the tool-name-on-the-Action-line repair matched the wire
    format's own keyword, so `Action: Operation` + a non-callable body (WebShop-style
    `search[...]`, bare SQL) was wrapped into `Operation(search[...])` and could only
    fail -- 602 such errors in one glm-4.7 retail cell."""
    from intent_graph.runtime.agent_api import parse_action, ActionKind
    a = parse_action("Action: Operation\n```\nsearch[red running shoes]\n```")
    assert a.kind is ActionKind.ACT
    assert a.command == "search[red running shoes]"
    b = parse_action("Action: Operation\n```\nSELECT * FROM orders\n```")
    assert b.command == "SELECT * FROM orders"
