"""按演员与经纪人的渡河规则，穷举可达状态；不调用或评估语言模型。

规则出处：https://arxiv.org/html/2506.06941v3，附录 A.2.3。
只核对给定人数与船容量；状态空间随人数增长。
"""

import argparse
import json
from collections import deque
from functools import cache
from itertools import combinations


def solve(pairs, capacity):
    """状态记录左岸人员和船的位置；左右岸与船上都必须满足安全条件。"""
    if pairs < 1 or not 1 <= capacity <= 2 * pairs:
        raise ValueError("pairs must be positive; capacity must be in [1, 2*pairs]")
    everyone = (1 << (2 * pairs)) - 1
    actors_mask = (1 << pairs) - 1

    @cache
    def safe(people):
        actors = people & actors_mask
        agents = people >> pairs
        return agents == 0 or (actors & ~agents) == 0

    start = (everyone, True)
    goal = (0, False)
    queue = deque([start])
    previous = {start: None}
    arrived_by = {}
    expanded = 0
    while queue:
        state = queue.popleft()
        expanded += 1
        if state == goal:
            moves = []
            while previous[state] is not None:
                old_state = previous[state]
                passengers = arrived_by[state]
                names = [
                    f"a{i + 1}" if i < pairs else f"A{i - pairs + 1}"
                    for i in range(2 * pairs) if passengers & (1 << i)
                ]
                moves.append({"from": "left" if old_state[1] else "right",
                              "passengers": names})
                state = old_state
            moves.reverse()
            return {"pairs": pairs, "capacity": capacity, "solvable": True,
                    "minimum_crossings": len(moves), "expanded_states": expanded,
                    "search_exhausted": False, "moves": moves}

        left, boat_left = state
        available = left if boat_left else everyone ^ left
        people = [i for i in range(2 * pairs) if available & (1 << i)]
        for size in range(1, min(capacity, len(people)) + 1):
            for group in combinations(people, size):
                passengers = sum(1 << i for i in group)
                new_left = left ^ passengers
                if not (safe(passengers) and safe(new_left) and safe(everyone ^ new_left)):
                    continue
                next_state = (new_left, not boat_left)
                if next_state in previous:
                    continue
                previous[next_state] = state
                arrived_by[next_state] = passengers
                queue.append(next_state)

    return {"pairs": pairs, "capacity": capacity, "solvable": False,
            "minimum_crossings": None, "expanded_states": expanded,
            "search_exhausted": True, "moves": None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=int, help="number of actor-agent pairs")
    parser.add_argument("capacity", type=int, help="maximum passengers per crossing")
    args = parser.parse_args()
    if args.pairs < 1 or not 1 <= args.capacity <= 2 * args.pairs:
        parser.error("pairs must be positive; capacity must be in [1, 2*pairs]")
    print(json.dumps(solve(args.pairs, args.capacity), ensure_ascii=False, indent=2))
