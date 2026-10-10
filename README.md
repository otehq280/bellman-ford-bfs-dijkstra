# Cheapest Flights Within K Stops: solve LC 787 without the off-by-one trap, and know when to reach for Bellman-Ford, BFS, or Dijkstra

You read the problem, you think "shortest path", you write Dijkstra, and it comes back wrong. Not wrong in a subtle way either. It returns 400 when the expected answer is 700.

That's the usual first encounter with Cheapest Flights Within K Stops (LeetCode 787, Medium). The problem looks like a weighted shortest-path question, and then a single constraint — at most `k` stops — quietly invalidates the algorithm you were about to write. Here's what's actually going on, three working approaches, and the specific mistakes that cost people their submissions.

## What the problem is really asking

You get `n` cities, a list of directed flights where `flights[i] = [from, to, price]`, plus `src`, `dst` and `k`. Return the cheapest price from `src` to `dst` using at most `k` stops, or `-1` if no such route exists.

Constraints worth keeping in mind: `n <= 100`, `1 <= price <= 10^4`, no duplicate flights between the same pair of cities, and `src != dst`.

The trap is hiding in the word "stops". A stop is an intermediate city you pass through, so a route with `k` stops uses `k + 1` flights. If `k = 0`, you're allowed exactly one direct flight and nothing else.

> At most `k` stops = at most `k + 1` edges. Every third failed submission on this problem comes from reading that as `k` edges.

The three examples make the constraint bite:

| Input | k | Answer | Why |
| --- | --- | --- | --- |
| `n=4`, `flights=[[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]]`, `src=0, dst=3` | 1 | 700 | `0→1→3` costs 700. The route `0→1→2→3` costs only 400 but uses 2 stops |
| `n=3`, `flights=[[0,1,100],[1,2,100],[0,2,500]]`, `src=0, dst=2` | 1 | 200 | `0→1→2` beats the 500 direct flight |
| Same graph, same source and destination | 0 | 500 | With zero stops allowed, only the direct flight qualifies |

That second and third row are the same graph with one number changed, and the answer jumps from 200 to 500. Nothing about the graph moved.

## Why plain Dijkstra gets this wrong

Standard Dijkstra leans on one assumption: when you pop a node, its distance is final, because any additional edge only adds cost (weights here are positive). Track hop counts on top of that and the assumption breaks — the cheapest way to reach an intermediate city might burn too many of your stops, while a more expensive arrival at the same city leaves budget for the cheap leg at the end.

On the first example, Dijkstra without hop tracking finds `0→1→2→3` at 400 and returns it. Correct algorithm, wrong problem. The 400 route needs 2 stops and you only get 1.

There's a second failure mode on the *modified* version. If you track stops but still settle each city once, or dedupe purely on cost, you throw away the higher-cost arrival with fewer stops used — which is exactly the state the constrained optimum depends on. That's why every correct Dijkstra variant here has to allow revisiting a city with a different stop count.

## Approach 1: layered Bellman-Ford (write this one)

Bellman-Ford relaxes every edge, repeatedly. Run it in rounds and something useful falls out: after round `i`, every node holds the cheapest cost achievable using at most `i` edges. You need at most `k + 1` edges, so you run `k + 1` rounds and read off `dist[dst]`.

python
from typing import List

def findCheapestPrice(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    INF = float("inf")
    dist = [INF] * n
    dist[src] = 0

    for _ in range(k + 1):          # k stops = k + 1 edges
        prev = dist[:]              # read the previous layer, write the new one
        for u, v, price in flights:
            if prev[u] == INF:
                continue            # unreachable: don't add to INF
            if prev[u] + price < dist[v]:
                dist[v] = prev[u] + price

    return -1 if dist[dst] == INF else dist[dst]


Tracing example 2 (`n=3`, `flights=[[0,1,100],[1,2,100],[0,2,500]]`, `src=0`, `dst=2`, `k=1`) shows how the rounds enforce the budget:

| Round | Edges used to reach 0 | Reach 1 | Reach 2 |
| --- | --- | --- | --- |
| Start | 0 | ∞ | ∞ |
| 1 (≤1 edge, 0 stops) | 0 | 100 | 500 |
| 2 (≤2 edges, 1 stop) | 0 | 100 | 200 |

Round 2 finds `0→1→2` at 200 because round 1 had already established `dist[1] = 100`. Complexity is `O(K · E)` time and `O(V)` space, where `E` is the number of flights. No heap, no adjacency list, no visited set.

### The one line that matters: `prev = dist[:]`

Skip that copy and relax the array in place, and the result depends on the order of the `flights` array. Why? Because a single round can chain two hops together.

Take the `k = 0` case from example 3. Correct answer: 500, direct flight only. Now relax in place without copying. The loop reads edge `0→1` and sets `dist[1] = 100`. Later in the *same* round it reads edge `1→2`, sees `dist[1] = 100`, and sets `dist[2] = 200`. One round, two edges consumed, and your answer for zero allowed stops is 200. Wrong, and it will pass small tests by accident often enough to be maddening.

The copy makes each round extend every path by exactly one edge. It's the whole algorithm.

## Approach 2: level-by-level BFS over stops

Same idea, different container. Keep a frontier of cities reached, expand it one level per stop, and hold a separate array for the next level so a city discovered this round can't chain into another hop this round.

python
def findCheapestPrice(n, flights, src, dst, k):
    INF = float("inf")
    adj = [[] for _ in range(n)]
    for u, v, p in flights:
        adj[u].append((v, p))

    dist = [INF] * n
    dist[src] = 0
    frontier = [src]

    for _ in range(k + 1):
        nxt_dist = dist[:]
        nxt_frontier = []
        for city in frontier:
            for v, p in adj[city]:
                candidate = dist[city] + p
                if candidate < nxt_dist[v]:
                    nxt_dist[v] = candidate
                    nxt_frontier.append(v)
        dist, frontier = nxt_dist, nxt_frontier

    return -1 if dist[dst] == INF else dist[dst]


Cost is the same `O(K · E)`. On LeetCode you'll see people argue about whether this counts as "BFS" or as Bellman-Ford with an adjacency list — it's the latter dressed up as BFS, and it works for the same reason: reads come from the committed layer, writes go to the next one.

## Approach 3: Dijkstra with stops as part of the state

Valid, more code, and the version people argue about in the comments.

python
import heapq

def findCheapestPrice(n, flights, src, dst, k):
    INF = float("inf")
    adj = [[] for _ in range(n)]
    for u, v, p in flights:
        adj[u].append((v, p))

    fewest_stops = [INF] * n
    heap = [(0, src, 0)]            # (cost so far, city, stops used)

    while heap:
        cost, city, stops = heapq.heappop(heap)
        if city == dst:
            return cost
        if stops > k or stops >= fewest_stops[city]:
            continue
        fewest_stops[city] = stops
        for v, p in adj[city]:
            heapq.heappush(heap, (cost + p, v, stops + 1))

    return -1


The heap is keyed on cost, but the state isn't a city — it's a city plus how many stops you used getting there. A city can be popped more than once, at different stop counts, and that's required, not a bug.

Worst-case time is `O(E · K · log(E · K))` and space is `O(V · K)` for the heap entries. You'll see `O(E log V)` claimed for this solution on LeetCode, and it isn't true — nodes get re-expanded once per stop budget, which is the entire point.

A memoized DFS is also fine: state is `(city, stops_remaining)`, giving `O(V · K)` states, and each state scans its outgoing edges for `O(K · E)` total time. The memo key has to include the remaining stops. Memoizing on city alone produces a fast, confidently wrong answer.

## Choosing between them

| Approach | Time | Space | When it's the right call |
| --- | --- | --- | --- |
| Layered Bellman-Ford | `O(K · E)` | `O(V)` | Default choice. Shortest code, clearest correctness story, no heap |
| Level BFS | `O(K · E)` | `O(V)` | When you already have an adjacency list built |
| Modified Dijkstra | `O(E · K · log(E · K))` | `O(V · K)` | If you want to show you understand why the stop dimension must be in the state |
| Memoized DFS | `O(K · E)` | `O(V · K)` | When recursion matches how you think about the constraint |

For an interview, Bellman-Ford is the one to reach for. It's fewer moving parts, and the correctness explanation is one sentence: after round `i`, every distance reflects paths of at most `i` edges.

Interviewers tend to probe in a predictable direction. "Why not Dijkstra?" is the opening question — the honest answer is that the hop cap breaks the settle-once guarantee, and a shorter-hop route to an intermediate city may be worth more money. "What if weights were negative?" Bellman-Ford already handles that, which is a side benefit. "What's the complexity?" Answer `O(K · E)` and be specific that `K` is the stop limit, not the node count.

## The bug checklist

Before you submit, run through these:

1. **Off-by-one on the loop.** `range(k + 1)`, not `range(k)`. Zero stops still means one flight.
2. **Relaxing into the array you're reading.** Copy first. This is the single most common failure.
3. **`INF` arithmetic.** In Java or C++, `Integer.MAX_VALUE + price` overflows into a negative number that will happily pass every comparison. Guard with `if (dist[from] == INF) continue;`.
4. **Deduping by cost alone.** In the Dijkstra variant, skipping a city you've already settled discards the higher-cost, lower-stop state you need.
5. **Assuming a DAG.** The first example contains the cycle `0→1→2→0`. Don't try to topologically sort it.
6. **Returning early.** Fine in the heap version, because the first time you pop `dst` you have the cheapest valid arrival. Not fine in Bellman-Ford — let all `k + 1` rounds finish.

That list is short because the problem is short. What makes LC 787 worth the time is that it forces a specific recognition: weighted shortest path, except the state isn't a node, it's a node plus a budget.

## Where this problem sits in the pattern map

Graph problems with a budget dimension are a small family, and they all share the same move: add the constraint to the state. Minimum cost to reach a node with at most `K` edges. Cheapest path with at most `K` refueling stops. Shortest path through at most `K` intermediate hops. Once you've seen the shape, "modify the state, not the algorithm" is the reflex.

The AlgoMonster pattern framework handles this with a decision flowchart that runs roughly: is it a graph? Is it a shortest-path problem? Is the graph weighted? Weighted shortest path routes you to Dijkstra. LC 787 is the case where the flowchart's default answer needs one extra column in the state, and knowing why the flowchart lands there is more useful than memorizing the diagram.

Worth knowing what the platform actually contains before you compare it to anything: 48 patterns, 325 lessons and problems, 678 illustrations, and 3,000+ editorials. Explanations are text and diagrams, not video. There's a Speedrun mode for timed review, an AI assistant for hints and concept explanations, and company question banks refreshed monthly. Solution code is available in Python, Java, JavaScript, C++, Go and Haskell. The gaps reviewers mention consistently: problem volume is far smaller than LeetCode's, system design coverage is introductory, and there's no live coaching or mentorship.

### Current plans

| Plan | What you get | Price | Billing |
| --- | --- | --- | --- |
| Free | Limited access to a subset of lessons and problems | $0 | No payment |
| Pro — Yearly | All patterns, lessons and problems; unlimited AI assistant; unlimited Speedrun; company question banks updated monthly; 325 lessons and problems, 678 illustrations, 3,000+ editorials; 3-day free trial | $99/year (works out to $8.25/month) | Billed annually, cancel or upgrade anytime |
| Pro — Lifetime | Everything in the annual plan, plus lifetime updates and an internal referral to a top tech company for qualified candidates | $189 | One-time payment |

There's no monthly subscription and no team tier. Whether the annual or lifetime plan makes more sense comes down to how many times you expect to do this. If you're inside a single interview cycle that ends in three months, the yearly plan covers it. If you go through prep every couple of years, the lifetime plan pays for itself across two cycles.

If you're working through graph problems right now and want the pattern explanations alongside your LeetCode practice, 👉 [👉 see the current plans and free tier](https://algo.monster/subscribe?fpr=li69).

One honest caveat: if you already recognize weighted-shortest-path problems on sight and just need reps, more problems is what you need, and the free tier plus LeetCode will carry you further than a subscription. The platform's value shows up earlier, when "why is my Dijkstra wrong" is still a live question — 👉 [👉 start with the free lessons before paying](https://algo.monster?fpr=li69).

## The short version

Cheapest Flights Within K Stops is a shortest-path problem with a hop budget bolted on. Plain Dijkstra ignores the budget and returns routes that use too many stops. Bellman-Ford handles it naturally: `k + 1` rounds, copy the distance array each round, read from the copy and write to the live array. `O(K · E)` time, `O(V)` space, about ten lines. If you want the heap version, the stop count goes into the state and you accept `O(E · K · log(E · K))` as the real complexity.

The two things to take away are smaller than the algorithm. `k` stops means `k + 1` flights. And the copy isn't a detail — it's the reason the answer is correct.

If you want the pattern drills for the rest of the graph family, including the constrained-shortest-path variants that show up right after this one, 👉 [👉 unlock the full pattern library and Speedrun mode](https://algo.monster/subscribe?fpr=li69).
