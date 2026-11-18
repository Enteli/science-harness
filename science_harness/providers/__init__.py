"""Provider-facing accounting helpers, independent of benchmark scoring."""


def usage_by_role(responses, roles):
    result = {}
    for role in sorted(set(roles)):
        replies = [r for r, k in zip(responses, roles) if k == role]
        events = []
        for reply in replies:
            usage = reply.get("usage")
            if isinstance(usage, list):
                events.extend(e for e in usage if isinstance(e, dict))
            elif isinstance(usage, dict) and usage:
                events.append({"inputTokens": usage.get("input_tokens", usage.get("prompt_tokens")),
                               "outputTokens": usage.get("output_tokens", usage.get("completion_tokens"))})
        row = {"attempts": roles.count(role), "reported_responses": len(events)}
        for field, source in (("input_tokens", "inputTokens"), ("output_tokens", "outputTokens")):
            complete = len(events) == len(replies) == roles.count(role) and all(
                type(e.get(source)) is int and e[source] >= 0 for e in events)
            row[field] = sum(e[source] for e in events) if complete else None
        result[role] = row
    return result
