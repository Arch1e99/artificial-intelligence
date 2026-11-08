import gc
import random
import time
from collections import deque

import matplotlib.pyplot as plt

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # zad2: stala kolejnosc - gora, prawo, dol, lewo


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
    while node is not None:  # od celu do startu, potem odwracamy
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


# zad3: bfs - Open jest kolejka FIFO
def bfs(grid, start, goal, dirs=DIRS):
    t0 = time.perf_counter()
    closed = set()  # Closed: odwiedzone stany
    parent = {start: None}  # pusty wskaznik rodzica dla startu
    open_list = deque([start])  # Open: kolejka
    in_open = {start}  # kopia Open jako zbior, test "t in Open" w O(1)
    max_open = 1

    while open_list:
        s = open_list.popleft()  # FIFO: pobieramy z poczatku kolejki
        in_open.remove(s)
        if s == goal:  # test znalezienia rozwiazania
            return make_result(t0, True, goal, parent, open_list, closed, max_open)
        for t in neighbors(grid, s, dirs):
            if t not in closed and t not in in_open:
                parent[t] = s
                open_list.append(t)
                in_open.add(t)
        closed.add(s)
        max_open = max(max_open, len(open_list))
    return make_result(t0, False, goal, parent, open_list, closed, max_open)


# zad3: dfs - Open jest stosem LIFO
def dfs(grid, start, goal, dirs=DIRS):
    t0 = time.perf_counter()
    closed = set()  # Closed: odwiedzone stany
    parent = {start: None}  # pusty wskaznik rodzica dla startu
    open_list = [start]  # Open: stos
    in_open = {start}  # kopia Open jako zbior, test "t in Open" w O(1)
    max_open = 1

    while open_list:
        s = open_list.pop()  # LIFO: pobieramy z wierzcholka stosu
        in_open.remove(s)
        if s == goal:  # test znalezienia rozwiazania
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


# zad6: dla kazdej wartosci parametru 20 rozwiazywalnych labiryntow, wszystkie
# algorytmy na tych samych instancjach, stale ziarno, wyniki usrednione
def run_series(values, make_maze, algorithms, iters=20, seed=2026):
    rng = random.Random(seed)
    avg = {name: {} for name in algorithms}
    gc.disable()  # garbage collector zaklocalby pomiar czasu
    for v in values:
        sums = {name: {} for name in algorithms}
        valid = 0
        while valid < iters:
            grid = make_maze(v, rng)
            goal = (len(grid) - 1, len(grid) - 1)
            results = {n: alg(grid, (0, 0), goal) for n, alg in algorithms.items()}
            if not next(iter(results.values()))[0]:
                continue  # brak rozwiazania -> losujemy nowy labirynt
            for name, (_, stats) in results.items():
                for k, x in stats.items():
                    sums[name][k] = sums[name].get(k, 0) + x
            valid += 1
        for name in algorithms:
            for k, x in sums[name].items():
                avg[name].setdefault(k, []).append(x / iters)
            s = {k: x[-1] for k, x in avg[name].items()}
            print(
                f"  {v:<5} {name:<24} czas={s['time_us']:>9.0f} us  Closed={s['closed']:>8.0f}"
                f"  sciezka={s['length']:>6.0f}  max Open={s['max_open']:>5.0f}"
            )
        gc.collect()
    gc.enable()
    return avg


# zad7: 4 wykresy porownawcze
def plot(values, avg, xlabel, filename, logy=False):
    metrics = [
        ("time_us", "czas wykonania [us]"),
        ("closed", "liczba stanow w Closed"),
        ("length", "dlugosc sciezki [kroki]"),
        ("max_open", "maksymalny rozmiar Open"),
    ]
    _, axs = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, title) in zip(axs.flat, metrics):
        for name, stats in avg.items():
            ax.plot(values, stats[key], "o-", label=name)
        ax.set(title=title, xlabel=xlabel, yscale="log" if logy else "linear")
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

    bfs_dfs = {"BFS": bfs, "DFS": dfs}
    sizes = [10, 25, 50, 100, 200, 300, 400, 500]
    print("\neksperyment 1: rosnacy rozmiar N, 20% scian (srednie z 20 labiryntow)")
    avg = run_series(sizes, lambda n, rng: random_maze(n, 0.2, rng), bfs_dfs)
    plot(sizes, avg, "rozmiar N (plansza NxN, 20% scian)", "wykresy_rozmiar.png", logy=True)

    densities = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    print("\neksperyment 2: plansza 50x50, rosnaca gestosc scian")
    avg = run_series(densities, lambda d, rng: random_maze(50, d, rng), bfs_dfs)
    plot(densities, avg, "gestosc scian (plansza 50x50)", "wykresy_gestosc.png")

    # wplyw kolejnosci sasiadow na dfs
    orders = {
        "BFS": bfs,
        "DFS gora,prawo,dol,lewo": dfs,
        "DFS dol,prawo,gora,lewo": lambda g, s, e: dfs(g, s, e, [(1, 0), (0, 1), (-1, 0), (0, -1)]),
        "DFS prawo,dol,lewo,gora": lambda g, s, e: dfs(g, s, e, [(0, 1), (1, 0), (0, -1), (-1, 0)]),
    }
    sizes = [10, 25, 50, 100, 200]
    print("\neksperyment 3: wplyw kolejnosci sasiadow na DFS")
    avg = run_series(sizes, lambda n, rng: random_maze(n, 0.2, rng), orders)
    plot(sizes, avg, "rozmiar N (plansza NxN, 20% scian)", "wykresy_kolejnosc.png", logy=True)
    print("\nzapisano: wykresy_rozmiar.png, wykresy_gestosc.png, wykresy_kolejnosc.png")


# zad8: wnioski
"""
1. Dlugosc sciezki: bfs zawsze daje najkrotsza (dla 500x500 ok. 1000 krokow), dfs nie
   gwarantuje optymalnosci - jego sciezki sa kilka razy dluzsze (ok. 8750 krokow).
2. Czas i Closed: dfs znajduje cel duzo szybciej i odwiedza mniej pol (500x500: ok. 12 tys.
   stanow i 15 ms wobec 200 tys. i 260 ms dla bfs), bo bfs rozchodzi sie "fala" po calej
   planszy. Wraz z gestoscia scian bfs odwiedza mniej pol, dfs wiecej (slepe zaulki).
3. Pamiec: na siatce to dfs ma wiekszy max Open (500x500: ok. 6200 wobec 470), bo na stosie
   zostaja nierozwiniete sasiedztwa calej dlugiej galezi; front bfs rosnie liniowo z N.
4. Bfs - gdy potrzebna jest najkrotsza sciezka; dfs - gdy wystarczy dowolna i liczy sie czas.
5. Kolejnosc sasiadow: dfs rozwija najpierw ostatnio dodanego potomka. Dla gora,prawo,dol,lewo
   idzie w dol/prawo, czyli w strone celu; kolejnosci preferujace gore/lewo daja 4-6x dluzsze
   sciezki i wiecej Closed. Na dlugosc sciezki bfs kolejnosc nie wplywa.
"""
