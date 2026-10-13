"""Line counts and slices with bounded memory, even for a single enormous line."""


def count_lines(stream):
    total, last = 0, ""
    while chunk := stream.read(64 * 1024):
        total += chunk.count("\n")
        last = chunk[-1]
    return total + bool(last and last != "\n")


def iter_lines(stream, limit):
    """Yield (bounded prefix, clipped) while consuming the remainder in small pieces."""
    limit = max(1, limit)
    while True:
        prefix = stream.readline(limit + 1)
        if not prefix:
            return
        clipped = len(prefix) > limit
        tail = prefix
        while tail and not tail.endswith("\n"):
            tail = stream.readline(64 * 1024)
        yield prefix[:limit] if clipped else prefix, clipped


def selected_lines(path, encoding, start, end, limit):
    with open(path, encoding=encoding) as stream:
        for number, item in enumerate(iter_lines(stream, limit), 1):
            if number > end:
                break
            if number >= start:
                yield item
