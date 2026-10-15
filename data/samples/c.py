import heapq


def dijkstra(graph, source):
    dist = {node: float("inf") for node in graph}
    dist[source] = 0
    queue = [(0, source)]
    while queue:
        d, node = heapq.heappop(queue)
        if d > dist[node]:
            continue
        for neighbour, weight in graph[node].items():
            candidate = d + weight
            if candidate < dist[neighbour]:
                dist[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    return dist


graph = {"a": {"b": 1, "c": 4}, "b": {"c": 2}, "c": {}}
print(dijkstra(graph, "a"))
