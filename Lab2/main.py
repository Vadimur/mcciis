import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
import random, time, threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ----------------------- Constants & Utilities -----------------------

DIRECTIONS = ['N', 'E', 'S', 'W']
DIR_DELTA = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}

MAX_ENERGY = 100.0
ENERGY_FROM_HERBIVORE = 40.0
ENERGY_FROM_PLANT = ENERGY_FROM_HERBIVORE / 2.0
METABOLISM_CARN = 1.0
METABOLISM_HERB = 2.0
MUTATION_STD = 0.08

RANDOM_SEED = None  # set to int for reproducible runs

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ----------------------- Data Classes -----------------------

@dataclass
class Agent:
    type: str  # 'herbivore' or 'carnivore'
    energy: float
    age: int
    generation: int
    x: int
    y: int
    direction: str  # 'N','E','S','W'
    weights: list  # length 12*4
    biases: list   # length 4

    def compute_outputs(self, inputs):
        # linear outputs; no activation required for winner-takes-all
        outputs = []
        for j in range(4):
            s = self.biases[j]
            base = j*12
            for i in range(12):
                s += inputs[i] * self.weights[base + i]
            outputs.append(s)
        return outputs

    def choose_action(self, inputs, front_has_food):
        """
        inputs: list of 12 normalized sensor values
        front_has_food: whether there's a plant in the cell in front
        Returns (action_index, outputs)
        """
        outputs = self.compute_outputs(inputs)

        # small stochastic noise so agent can escape fixed cycles (helps exploration)
        noise_std = 0.02
        outputs = [o + random.gauss(0, noise_std) for o in outputs]

        # if herbivore and there's food in front -> bias 'eat' action
        if front_has_food:
            outputs[3] += 1.5  # strong bonus; tune if needed

        # fallback: if no sensory signal at all, prefer to move
        if sum(inputs) <= 1e-6:
            # ensure move (index 2) is likely
            outputs[2] += 0.5

        idx = max(range(len(outputs)), key=lambda i: outputs[i])
        return idx, outputs

@dataclass
class Plant:
    x: int
    y: int

# ----------------------- World Model -----------------------

class World:
    def __init__(self, N=30, initial_plants=200, initial_herb=100, initial_carn=50, seed=None, max_iterations=1000, enforce_limits=True):
        if seed is not None:
            random.seed(seed)
        elif RANDOM_SEED is not None:
            random.seed(RANDOM_SEED)

        self.N = N
        self.max_agents = N * N
        self.plants = {}  # (x,y)->Plant
        self.agents = []  # list of Agent
        self.iteration = 0
        self.max_iterations = max_iterations
        self.enforce_limits = enforce_limits

        # stats
        self.history_counts = []  # each entry: {'plants':..,'herbivores':..,'carnivores':..}
        self.max_age_seen = {'herbivore':0, 'carnivore':0, 'plant':0}
        self.reproduction_counts = {'herbivore':0, 'carnivore':0}
        self.reproduction_counts_total = {'herbivore': 0, 'carnivore': 0}  # total across all time
        self.reproduction_history = []  # number of reproductions per iteration

        self.reset(initial_plants, initial_herb, initial_carn)

    def reset(self, initial_plants, initial_herb, initial_carn):
        self.plants.clear(); self.agents.clear(); self.iteration = 0
        self.history_counts.clear()
        self.max_age_seen = {'herbivore':0, 'carnivore':0, 'plant':0}
        self.reproduction_counts = {'herbivore':0, 'carnivore':0}

        # spawn plants uniquely
        all_cells = [(x,y) for x in range(self.N) for y in range(self.N)]
        random.shuffle(all_cells)
        for i in range(min(initial_plants, len(all_cells))):
            x,y = all_cells.pop()
            self.plants[(x,y)] = Plant(x,y)

        # enforce initial constraints for herbivores/carnivores 25%-50% of max_agents
        if self.enforce_limits:
            min_allowed = int(0.25 * self.max_agents)
            max_allowed = int(0.5 * self.max_agents)
            initial_herb = clamp(initial_herb, min_allowed, max_allowed)
            initial_carn = clamp(initial_carn, min_allowed, max_allowed)

        for i in range(initial_herb):
            x,y = random.randrange(self.N), random.randrange(self.N)
            self.agents.append(self._make_random_agent('herbivore', x, y))

        for i in range(initial_carn):
            x,y = random.randrange(self.N), random.randrange(self.N)
            self.agents.append(self._make_random_agent('carnivore', x, y))

        self._record_stats()

    def _make_random_agent(self, atype, x, y):
        weights = [random.uniform(0,1) for _ in range(12*4)]
        biases = [random.uniform(0,1) for _ in range(4)]
        biases[3] += 0.2
        energy = random.uniform(0.5*MAX_ENERGY, 0.8*MAX_ENERGY)
        direction = random.choice(DIRECTIONS)
        return Agent(type=atype, energy=energy, age=0, generation=1, x=x, y=y,
                     direction=direction, weights=weights, biases=biases)

    def toroidal(self, x, y):
        return (x % self.N, y % self.N)

    def step(self):
        print('--------------------------------------------------------------------------------------------------------------------')
        if self.iteration >= self.max_iterations:
            return False  # signal stopping condition

        self.iteration += 1
        random.shuffle(self.agents)  # randomize action order
        new_agents = []

        for agent in list(self.agents):
            # perceive
            inputs = self._compute_inputs_for(agent)

            dx, dy = DIR_DELTA[agent.direction]
            fx, fy = self.toroidal(agent.x + dx, agent.y + dy)

            if agent.type == 'herbivore':
                front_has_food = (fx, fy) in self.plants
            else:
                front_has_food = any(a.x == fx and a.y == fy
                                     and a.type == 'herbivore'
                                     for a in self.agents)

            action_idx, outputs = agent.choose_action(inputs, front_has_food)
            # actions: 0-left,1-right,2-move,3-eat
            if action_idx == 0:
                agent.direction = self._turn(agent.direction, left=True)
            elif action_idx == 1:
                agent.direction = self._turn(agent.direction, left=False)
            elif action_idx == 2:
                dx,dy = DIR_DELTA[agent.direction]
                nx,ny = self.toroidal(agent.x + dx, agent.y + dy)
                agent.x, agent.y = nx, ny
            elif action_idx == 3:
                self._attempt_eat(agent)

            # metabolism & aging
            if agent.type == 'herbivore':
                agent.energy -= METABOLISM_HERB
            else:
                agent.energy -= METABOLISM_CARN
            agent.age += 1

            # death by energy
            if agent.energy <= 0:
                try:
                    self.agents.remove(agent)
                except ValueError:
                    pass
                continue

            # reproduction check
            print(f'{agent.type} {agent.age} {agent.energy:0.2f} {agent.generation} {agent.weights} {agent.biases}')
            if agent.energy >= 0.8 * MAX_ENERGY:
                cnt_type = sum(1 for a in self.agents if a.type == agent.type)
                if cnt_type < int(0.5 * self.max_agents):
                    child = self._reproduce(agent)
                    new_agents.append(child)
                    self.reproduction_counts[agent.type] += 1
                    agent.energy /= 2.0  # split energy
            # track max age
            self.max_age_seen[agent.type] = max(self.max_age_seen.get(agent.type, 0), agent.age)

        self.agents.extend(new_agents)
        self._record_stats()

        # return True to indicate continue; False to indicate stop
        return self.iteration < self.max_iterations

    def _compute_inputs_for(self, agent):
        front_positions = self._positions_in_front(agent, distance=2)
        left_positions = self._positions_to_side(agent, side='left', distance=2)
        right_positions = self._positions_to_side(agent, side='right', distance=2)
        near_positions = self._positions_proximity(agent)

        def count_types(pos_list, typeset):
            c = 0
            for (x,y) in pos_list:
                x,y = self.toroidal(x,y)
                if 'plant' in typeset:
                    if (x,y) in self.plants:
                        c += 1
                for a in self.agents:
                    if a.x == x and a.y == y and a.type in typeset:
                        c += 1
            return c

        i0 = count_types(front_positions, {'herbivore'})
        i1 = count_types(front_positions, {'carnivore'})
        i2 = count_types(front_positions, {'plant'})
        i3 = count_types(left_positions, {'herbivore'})
        i4 = count_types(left_positions, {'carnivore'})
        i5 = count_types(left_positions, {'plant'})
        i6 = count_types(right_positions, {'herbivore'})
        i7 = count_types(right_positions, {'carnivore'})
        i8 = count_types(right_positions, {'plant'})
        i9 = count_types(near_positions, {'herbivore'})
        i10 = count_types(near_positions, {'carnivore'})
        i11 = count_types(near_positions, {'plant'})

        inputs = [i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11]
        # normalize by number of examined cells to keep values in manageable range
        sizes = [len(front_positions)]*3 + [len(left_positions)]*3 + [len(right_positions)]*3 + [len(near_positions)]*3
        inputs = [inputs[i] / max(1.0, sizes[i]) for i in range(12)]
        return inputs

    def _positions_in_front(self, agent, distance=2):
        dx,dy = DIR_DELTA[agent.direction]
        pos = []
        for d in range(1, distance+1):
            cx = agent.x + dx*d; cy = agent.y + dy*d
            pos.append((cx,cy))
            # lateral offsets
            if agent.direction in ('N','S'):
                pos.append((cx-1, cy)); pos.append((cx+1, cy))
            else:
                pos.append((cx, cy-1)); pos.append((cx, cy+1))
        return pos

    def _positions_to_side(self, agent, side='left', distance=2):
        dir_idx = DIRECTIONS.index(agent.direction)
        if side == 'left':
            side_dir = DIRECTIONS[(dir_idx - 1) % 4]
        else:
            side_dir = DIRECTIONS[(dir_idx + 1) % 4]
        pos = []
        for d in range(1, distance+1):
            sx = agent.x + DIR_DELTA[side_dir][0]*d
            sy = agent.y + DIR_DELTA[side_dir][1]*d
            pos.append((sx, sy))
            pos.append((sx + DIR_DELTA[agent.direction][0], sy + DIR_DELTA[agent.direction][1]))
        return pos

    def _positions_proximity(self, agent):
        pos = [(agent.x, agent.y)]
        for (dx,dy) in DIR_DELTA.values():
            pos.append((agent.x+dx, agent.y+dy))
        return pos

    def _turn(self, direction, left=True):
        idx = DIRECTIONS.index(direction)
        return DIRECTIONS[(idx - 1) % 4] if left else DIRECTIONS[(idx + 1) % 4]

    def _attempt_eat(self, agent):
        dx,dy = DIR_DELTA[agent.direction]
        fx,fy = self.toroidal(agent.x + dx, agent.y + dy)

        if agent.type == 'herbivore':
            print(f'plants: {self.plants.keys()}')
            if (fx,fy) in self.plants:
                del self.plants[(fx,fy)]
                agent.energy = min(MAX_ENERGY, agent.energy + ENERGY_FROM_PLANT)
                self._spawn_plant(exclude={(fx,fy)})
                print('EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEAT')
                return True
            return False
        else:
            prey = None
            for a in self.agents:
                if a.x == fx and a.y == fy and a.type == 'herbivore':
                    prey = a; break
            if prey is not None:
                try:
                    self.agents.remove(prey)
                except ValueError:
                    pass
                agent.energy = min(MAX_ENERGY, agent.energy + ENERGY_FROM_HERBIVORE)
                return True
            return False

    def _spawn_plant(self, exclude=None):
        if exclude is None: exclude = set()
        possible = [(x,y) for x in range(self.N) for y in range(self.N) if (x,y) not in self.plants and (x,y) not in exclude]
        if not possible: return
        x,y = random.choice(possible)
        self.plants[(x,y)] = Plant(x,y)

    def _reproduce(self, parent):
        child_weights = [w + random.gauss(0, MUTATION_STD) for w in parent.weights]
        child_biases = [b + random.gauss(0, MUTATION_STD) for b in parent.biases]
        dx = random.choice([-1,0,1]); dy = random.choice([-1,0,1])
        cx,cy = self.toroidal(parent.x + dx, parent.y + dy)
        child = Agent(type=parent.type, energy=parent.energy/2.0, age=0, generation=parent.generation+1,
                      x=cx, y=cy, direction=random.choice(DIRECTIONS), weights=child_weights, biases=child_biases)
        return child

    def _record_stats(self):
        counts = {'plants': len(self.plants),
                  'herbivores': sum(1 for a in self.agents if a.type == 'herbivore'),
                  'carnivores': sum(1 for a in self.agents if a.type == 'carnivore')}
        self.history_counts.append(counts)

        # append per-iteration reproduction counts
        self.reproduction_history.append(self.reproduction_counts.copy())
        # also update total reproduction counts
        for t in ['herbivore', 'carnivore']:
            self.reproduction_counts_total[t] = max(self.reproduction_counts_total[t], self.reproduction_counts[t])
        self.reproduction_counts = {'herbivore': 0, 'carnivore': 0}


# ----------------------- GUI Application -----------------------

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Artificial Life — Food Chain Simulator")
        self.world = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # Parameters
        self.N = tk.IntVar(value=25)
        self.initial_plants = tk.IntVar(value=120)
        self.initial_herb = tk.IntVar(value=80)
        self.initial_carn = tk.IntVar(value=40)
        self.speed = tk.DoubleVar(value=10.0)  # steps per second
        self.max_iterations = tk.IntVar(value=100)
        self.use_limits = tk.BooleanVar(value=True)

        self.cell_size = 18
        self._build_ui()

    def _build_ui(self):
        ctrl = ttk.Frame(self.root); ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        ttk.Label(ctrl, text="Grid N:").grid(row=0,column=0)
        ttk.Entry(ctrl, textvariable=self.N, width=5).grid(row=0,column=1)
        ttk.Label(ctrl, text="Plants:").grid(row=0,column=2)
        ttk.Entry(ctrl, textvariable=self.initial_plants, width=6).grid(row=0,column=3)
        ttk.Label(ctrl, text="Herbivores:").grid(row=0,column=4)
        ttk.Entry(ctrl, textvariable=self.initial_herb, width=6).grid(row=0,column=5)
        ttk.Label(ctrl, text="Carnivores:").grid(row=0,column=6)
        ttk.Entry(ctrl, textvariable=self.initial_carn, width=6).grid(row=0,column=7)
        ttk.Label(ctrl, text="Max iterations:").grid(row=0,column=10)
        ttk.Entry(ctrl, textvariable=self.max_iterations, width=6).grid(row=0,column=11)
        ttk.Checkbutton(ctrl, text="Enable 25–50% population limit",
                        variable=self.use_limits).grid(row=1, column=0, columnspan=4, sticky="w")

        ttk.Button(ctrl, text="Init World", command=self.init_world).grid(row=0,column=12,padx=4)
        self.start_btn = ttk.Button(ctrl, text="Start", command=self.toggle_run); self.start_btn.grid(row=0,column=13,padx=2)
        ttk.Button(ctrl, text="Step", command=self.step_once).grid(row=0,column=14,padx=2)
        ttk.Button(ctrl, text="Reset", command=self.reset).grid(row=0,column=15,padx=2)

        # Main frame with canvas (left) and chart (right)
        main = ttk.Frame(self.root); main.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        left = ttk.Frame(main); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        # Canvas for world
        self.canvas = tk.Canvas(left, bg='white', width=600, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._on_canvas_resize())

        # Stats labels
        stats = ttk.Frame(left); stats.pack(side=tk.TOP, fill=tk.X)
        self.iter_label = ttk.Label(stats, text="Iteration: 0"); self.iter_label.pack(side=tk.LEFT, padx=6)
        self.counts_label = ttk.Label(stats, text="P:0 H:0 C:0"); self.counts_label.pack(side=tk.LEFT, padx=6)

        # Matplotlib figure for embedded chart
        self.fig, self.ax = plt.subplots(figsize=(4,4))
        self.ax.set_title("Population counts over time")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Count")
        self.line_plants, = self.ax.plot([], [], label='plants', color='green')
        self.line_herb, = self.ax.plot([], [], label='herbivores', color='blue')
        self.line_carn, = self.ax.plot([], [], label='carnivores', color='red')
        self.ax.legend()

        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig.tight_layout()

    def init_world(self):
        N = max(5, self.N.get())
        W = max(200, self.canvas.winfo_width() or 600)
        H = max(200, self.canvas.winfo_height() or 600)
        self.cell_size = max(4, min(30, min(W,H)//N))
        max_iter = max(1, self.max_iterations.get())
        self.world = World(
            N=N,
            initial_plants=self.initial_plants.get(),
            initial_herb=self.initial_herb.get(),
            initial_carn=self.initial_carn.get(),
            seed=RANDOM_SEED,
            max_iterations=max_iter,
            enforce_limits=self.use_limits.get()
        )
        self._draw_world(); self._update_stats_labels(); self._update_chart(initial=True)

    def toggle_run(self):
        if self.world is None:
            messagebox.showinfo("Info", "Initialize world first."); return
        if not self.running:
            self.running = True; self.start_btn.config(text="Pause")
            self._start_thread()
        else:
            self.running = False; self.start_btn.config(text="Start")

    def _start_thread(self):
        def loop():
            while self.running:
                start = time.time()
                with self.lock:
                    cont = self.world.step()
                # schedule UI updates on main thread
                self.root.after(0, self._draw_world)
                self.root.after(0, self._update_stats_labels)
                # update chart less frequently to save CPU
                self.root.after(0, self._update_chart)
                if not cont:
                    # reached max iterations -> stop
                    self.running = False
                    self.root.after(0, lambda: self.start_btn.config(text="Start"))
                    msg = f"Reached max iterations: {self.world.iteration}\n\n"
                    msg += "Maximum age reached:\n"
                    msg += f"Herbivores: {self.world.max_age_seen['herbivore']}\n"
                    msg += f"Carnivores: {self.world.max_age_seen['carnivore']}\n\n"
                    msg += "Total reproductions:\n"
                    msg += f"Herbivores: {sum(r['herbivore'] for r in self.world.reproduction_history)}\n"
                    msg += f"Carnivores: {sum(r['carnivore'] for r in self.world.reproduction_history)}"

                    self.root.after(0, lambda: messagebox.showinfo("Simulation complete", msg))
                    break
                elapsed = time.time() - start
                delay = max(0, (1.0 / max(0.1, self.speed.get())) - elapsed)
                time.sleep(delay)
        self.thread = threading.Thread(target=loop, daemon=True); self.thread.start()

    def step_once(self):
        if self.world is None:
            messagebox.showinfo("Info", "Initialize world first."); return
        with self.lock:
            cont = self.world.step()
        self._draw_world(); self._update_stats_labels(); self._update_chart()
        if not cont:
            messagebox.showinfo("Info", f"Reached max iterations: {self.world.iteration}")

    def reset(self):
        self.running = False
        self.start_btn.config(text="Start")
        if self.world:
            self.world.reproduction_history.clear()
            self.world.reproduction_counts_total = {'herbivore': 0, 'carnivore': 0}
            self.world.reproduction_counts = {'herbivore': 0, 'carnivore': 0}
        self.init_world()

    def _on_canvas_resize(self):
        self._draw_world()

    def _draw_world(self):
        if self.world is None: return
        self.canvas.delete("all")
        N = self.world.N; cs = self.cell_size
        W = cs * N; H = cs * N
        self.canvas.config(scrollregion=(0,0,W,H))
        # draw grid
        for x in range(N):
            for y in range(N):
                x1 = x*cs; y1 = y*cs; x2 = x1+cs; y2 = y1+cs
                self.canvas.create_rectangle(x1,y1,x2,y2, outline='#eee', fill='white')
        # draw plants
        for (x,y), p in self.world.plants.items():
            x1 = x*cs + cs*0.12; y1 = y*cs + cs*0.12; x2 = x1 + cs*0.76; y2 = y1 + cs*0.76
            self.canvas.create_oval(x1,y1,x2,y2, fill='green', outline='')
        # draw agents
        for a in self.world.agents:
            x1 = a.x*cs + cs*0.18; y1 = a.y*cs + cs*0.18; x2 = x1 + cs*0.64; y2 = y1 + cs*0.64
            color = 'red' if a.type == 'carnivore' else 'blue'  # herbivores blue, carnivores red
            self.canvas.create_rectangle(x1,y1,x2,y2, fill=color, outline='black')
            # direction arrow
            mx = (x1 + x2)/2; my = (y1 + y2)/2
            if a.direction == 'N':
                self.canvas.create_line(mx, my, mx, my - cs*0.28, arrow=tk.LAST, width=1.5)
            elif a.direction == 'S':
                self.canvas.create_line(mx, my, mx, my + cs*0.28, arrow=tk.LAST, width=1.5)
            elif a.direction == 'E':
                self.canvas.create_line(mx, my, mx + cs*0.28, my, arrow=tk.LAST, width=1.5)
            elif a.direction == 'W':
                self.canvas.create_line(mx, my, mx - cs*0.28, my, arrow=tk.LAST, width=1.5)

    def _update_stats_labels(self):
        if self.world is None: return
        self.iter_label.config(text=f"Iteration: {self.world.iteration}")
        last = self.world.history_counts[-1] if self.world.history_counts else {'plants':0,'herbivores':0,'carnivores':0}
        self.counts_label.config(text=f"P:{last['plants']}  H:{last['herbivores']}  C:{last['carnivores']}")

    def _update_chart(self, initial=False):
        if self.world is None: return

        if self.world.reproduction_history:
            reps = list(range(len(self.world.reproduction_history)))
            herb_reps = [r['herbivore'] for r in self.world.reproduction_history]
            carn_reps = [r['carnivore'] for r in self.world.reproduction_history]
            # you can add new lines in chart
            if not hasattr(self, 'line_herb_rep'):
                self.line_herb_rep, = self.ax.plot(reps, herb_reps, label='herbivore reproductions', linestyle='--',
                                                   color='blue')
                self.line_carn_rep, = self.ax.plot(reps, carn_reps, label='carnivore reproductions', linestyle='--',
                                                   color='red')
                self.ax.legend()
            else:
                self.line_herb_rep.set_data(reps, herb_reps)
                self.line_carn_rep.set_data(reps, carn_reps)

        xs = list(range(len(self.world.history_counts)))
        plants = [h['plants'] for h in self.world.history_counts]
        herbs = [h['herbivores'] for h in self.world.history_counts]
        carns = [h['carnivores'] for h in self.world.history_counts]

        self.line_plants.set_data(xs, plants)
        self.line_herb.set_data(xs, herbs)
        self.line_carn.set_data(xs, carns)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        try:
            self.canvas_fig.draw()
        except Exception:
            pass

# ----------------------- Main -----------------------

def main():
    root = tk.Tk()
    app = App(root)
    root.geometry("1100x750")
    root.mainloop()

if __name__ == '__main__':
    main()