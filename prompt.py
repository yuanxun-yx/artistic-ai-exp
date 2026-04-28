def compute_length_bounds(max_new_tokens: int) -> tuple[int, int]:
    low = int(max_new_tokens * 0.45)
    high = int(max_new_tokens * 0.65)
    return low, high
