import gc
import random
import time
from collections import deque

import matplotlib.pyplot as plt

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # zad2: stala kolejnosc gora, prawo, dol, lewo


# zad1: plansza = macierz 0/1 (0 wolne, 1 sciana); start (0,0) i cel (n-1,n-1) zawsze wolne
def random_maze(n, density, rng):
    grid = [[int(rng.random() < density) for _ in range(n)] for _ in range(n)]
    grid[0][0] = grid[n - 1][n - 1] = 0
    return grid


def is_walkable(grid, r, c):
    return 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 0


# zad2: funkcja generujaca potomkow (sasiednie wolne pola)
def neighbors(grid, pos, dirs=DIRS):
    r, c = pos
    return [(r + dr, c + dc) for dr, dc in dirs if is_walkable(grid, r + dr, c + dc)]


# zad4 i zad5: sciezka odtworzona po wskaznikach na rodzica + statystyki
def make_result(t0, found, goal, parent, open_list, closed, max_open):
    elapsed = (time.perf_counter() - t0) * 1e6
    path, node = [], goal if found else None
    while node is not None:  # od celu do startu potem odwracamy
        path.append(node)
        node = parent[node]
    path.reverse()
    stats = {
        "open": len(open_list),
        "closed": len(closed),
        "time_us": elapsed,
        "length": max(len(path) - 1, 0),
        "max_open": max_open,
    }
    return path, stats


# zad3: bfs - open jest kolejka fifo
def bfs(grid, start, goal, dirs=DIRS):
    t0 = time.perf_counter()
    closed = set()  # Closed: odwiedzone stany
    parent = {start: None}  # pusty wskaznik rodzica dla startu
    open_list = deque([start])
    in_open = {start}
    max_open = 1

    while open_list:
        s = open_list.popleft()
        in_open.remove(s)
        if s == goal:
            return make_result(t0, True, goal, parent, open_list, closed, max_open)
        for t in neighbors(grid, s, dirs):
            if t not in closed and t not in in_open:
                parent[t] = s
                open_list.append(t)
                in_open.add(t)
        closed.add(s)
        max_open = max(max_open, len(open_list))
    return make_result(t0, False, goal, parent, open_list, closed, max_open)


# zad3: dfs - open jest stosem lifo
def dfs(grid, start, goal, dirs=DIRS):
    t0 = time.perf_counter()
    closed = set()
    parent = {start: None}
    open_list = [start]
    in_open = {start}
    max_open = 1

    while open_list:
        s = open_list.pop()
        in_open.remove(s)
        if s == goal:
            return make_result(t0, True, goal, parent, open_list, closed, max_open)
        for t in neighbors(grid, s, dirs):
            if t not in closed and t not in in_open:
                parent[t] = s
                open_list.append(t)
                in_open.add(t)
        closed.add(s)
        max_open = max(max_open, len(open_list))
    return make_result(t0, False, goal, parent, open_list, closed, max_open)


def print_results(name, path, stats):
    found = f"sciezka ({stats['length']} krokow): {path}"
    print(f"{name}: {found if path else 'brak sciezki'}")
    print(
        f"  Open na koniec={stats['open']}, Closed={stats['closed']}, "
        f"max Open={stats['max_open']}, czas={stats['time_us']:.1f} us"
    )


# zad6: dla kazdego rozmiaru 20 rozwiazywalnych labiryntow, bfs i dfs na tych samych
# instancjach, stale ziarno, wyniki usrednione
def run_series(sizes, density=0.2, iters=20, seed=2026):
    algorithms = {"BFS": bfs, "DFS": dfs}
    rng = random.Random(seed)
    avg = {name: {} for name in algorithms}
    gc.disable()  # garbage collector zaklocalby pomiar czasu
    for n in sizes:
        sums = {name: {} for name in algorithms}
        valid = 0
        while valid < iters:
            grid = random_maze(n, density, rng)
            goal = (n - 1, n - 1)
            results = {n: alg(grid, (0, 0), goal) for n, alg in algorithms.items()}
            if not next(iter(results.values()))[0]:
                continue  # if brak rozwiazania to losujemy nowy labirynt
            for name, (_, stats) in results.items():
                for k, x in stats.items():
                    sums[name][k] = sums[name].get(k, 0) + x
            valid += 1
        for name in algorithms:
            for k, x in sums[name].items():
                avg[name].setdefault(k, []).append(x / iters)
            s = {k: x[-1] for k, x in avg[name].items()}
            print(
                f"  N={n:<4} {name}  czas={s['time_us']:>9.0f} us  Closed={s['closed']:>8.0f}"
                f"  sciezka={s['length']:>6.0f}  max Open={s['max_open']:>5.0f}"
            )
        gc.collect()
    gc.enable()
    return avg


# zad7: 4 wykresy porownawcze (os y logarytmiczna, bo wartosci rosna o kilka rzedow)
def plot(sizes, avg, filename="wykresy_rozmiar.png"):
    metrics = [
        ("time_us", "czas wykonania [us]"),
        ("closed", "liczba stanow w Closed"),
        ("length", "dlugosc sciezki [kroki]"),
        ("max_open", "maksymalny rozmiar Open"),
    ]
    _, axs = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, title) in zip(axs.flat, metrics):
        for name, stats in avg.items():
            ax.plot(sizes, stats[key], "o-", label=name)
        ax.set(title=title, xlabel="rozmiar N (plansza NxN, 20% scian)", yscale="log")
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    example = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0]]  # z instrukcji
    print("potomkowie (0,0) w przykladzie z instrukcji:", neighbors(example, (0, 0)))
    mazes = {
        "przyklad z instrukcji": example,
        "losowy 10x10": random_maze(10, 0.2, random.Random(42)),
        "bez rozwiazania": [[0, 1], [1, 0]],
    }
    for title, grid in mazes.items():
        print(f"\n--- {title} ---")
        goal = (len(grid) - 1, len(grid) - 1)
        print_results("BFS", *bfs(grid, (0, 0), goal))
        print_results("DFS", *dfs(grid, (0, 0), goal))

    sizes = [10, 25, 50, 100, 200, 300, 400, 500]
    print("\neksperyment: rosnacy rozmiar N, 20% scian (srednie z 20 labiryntow)")
    plot(sizes, run_series(sizes))
    print("\nzapisano: wykresy_rozmiar.png")


# zad8: wnioski
"""
1. Dlugosc sciezki: bfs zawsze daje najkrotsza (dla 500x500 ok. 1000 krokow), dfs nie
   gwarantuje optymalnosci - jego sciezki sa kilka razy dluzsze (ok. 8750 krokow).
2. Czas i Closed: dfs znajduje cel duzo szybciej i odwiedza mniej pol (500x500: ok. 12 tys.
   stanow i 15 ms wobec 200 tys. i 260 ms dla bfs), bo bfs rozchodzi sie "fala" po calej
   planszy, a dfs idzie od razu w glab.
3. Pamiec: na siatce to dfs ma wiekszy max Open (500x500: ok. 6200 wobec 470), bo na stosie
   zostaja nierozwiniete sasiedztwa calej dlugiej galezi; front bfs rosnie liniowo z N.
4. Bfs - gdy potrzebna jest najkrotsza sciezka; dfs - gdy wystarczy dowolna i liczy sie czas.
5. Kolejnosc sasiadow: dfs rozwija najpierw ostatnio dodanego potomka (stos). Dla kolejnosci
   gora,prawo,dol,lewo probuje najpierw lewo, potem dol - przy lewej krawedzi schodzi wiec
   w dol i dalej w prawo, czyli w strone celu w prawym dolnym rogu. Kolejnosc, w ktorej
   priorytet maja kierunki od celu (gora/lewo), daje dfs-owi duzo dluzsze sciezki i wiecej
   stanow w Closed. Na dlugosc sciezki bfs kolejnosc nie wplywa (zawsze najkrotsza).
"""
