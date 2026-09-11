"""Small shared bits for the SQL adapters."""

# Identifiers that would need quoting if ever emitted bare.  We always backtick-quote
# identifiers, so this exists to document why, and as a hook if a dialect check is added.
MYSQL_RESERVED = frozenset({
    "select", "from", "where", "group", "order", "by", "table", "index", "key",
    "rank", "row", "insert", "update", "delete", "values", "set", "and", "or",
})
