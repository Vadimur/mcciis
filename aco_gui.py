import tkinter as tk
from tkinter import ttk, messagebox
import random
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from aco_core import ACO
import time

class ACOApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ant Colony Optimization")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        # Default params
        self.params = {
            "num_cities": tk.IntVar(value=20),
            "num_ants": tk.IntVar(value=20),
            "iterations": tk.IntVar(value=100),
            "alpha": tk.DoubleVar(value=1.0),
            "beta": tk.DoubleVar(value=3.0),
            "rho": tk.DoubleVar(value=0.5),
            "q": tk.DoubleVar(value=100.0),
            "seed": tk.IntVar(value=1)
        }
        self.coords = []
        self.aco = None

        self._build_ui()

    def _build_ui(self):
        frame = ttk.Frame(self)
        frame.pack(side="left", fill="y", padx=8, pady=8)

        ttk.Label(frame, text="Algorithm settings").pack()

        for name, var in self.params.items():
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=name.replace("_", " ").title()+":", width=14).pack(side="left")
            ttk.Entry(row, textvariable=var, width=8).pack(side="left")

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=6)
        ttk.Button(btn_frame, text="Generate Cities", command=self.generate_cities).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Start ACO", command=self.start_aco).pack(side="left", padx=4)

        # Info labels
        self.info_text = tk.StringVar(value="Ready")
        ttk.Label(frame, textvariable=self.info_text, wraplength=240).pack(pady=6)

        # Plot area
        self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.axis("off")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

    def generate_cities(self):
        n = self.params["num_cities"].get()
        seed = self.params["seed"].get()
        random.seed(seed)
        np.random.seed(seed)
        self.coords = [(random.random() * 10, random.random() * 10) for _ in range(n)]
        self._draw_initial()

    def _draw_initial(self):
        self.ax.clear()

        # Remove axes and grid
        self.ax.grid(False)

        xs = [p[0] for p in self.coords]
        ys = [p[1] for p in self.coords]

        # Draw cities
        self.ax.scatter(xs, ys)

        # Make city numbers bold
        for i, (x, y) in enumerate(self.coords):
            self.ax.text(x, y, str(i+1), horizontalalignment='left', verticalalignment='bottom')

        self.ax.set_title("Cities")
        self.canvas.draw()

    def start_aco(self):
        if not self.coords:
            messagebox.showerror("Error", "Generate or load cities first")
            return
        ants = self.params["num_ants"].get()
        iterations = self.params["iterations"].get()
        alpha = self.params["alpha"].get()
        beta = self.params["beta"].get()
        rho = self.params["rho"].get()
        q = self.params["q"].get()
        seed = self.params["seed"].get()

        self.aco = ACO(self.coords, ants=ants, alpha=alpha, beta=beta, rho=rho, q=q, iterations=iterations, seed=seed)
        # run with callbacks that update plot occasionally
        def callback(gen, best_route, best_distance, pheromone):
            if gen % max(1, iterations//20) == 0 or gen == iterations:
                self._draw_solution(best_route, best_distance, pheromone)
                self.info_text.set(f"Generation {gen} — best distance {best_distance:.4f}")
                self.update_idletasks()

        start_time = time.time()
        res = self.aco.run(callback=callback)
        execution_time = round(time.time() - start_time, 3)
        path_str = " → ".join(map(str, res.best_route))
        self._draw_solution(res.best_route, res.best_distance, self.aco.pheromone)
        txt = (f"Finished.\n"
               f"Execution time: {execution_time} s\n"
               f"Best distance: {res.best_distance:.4f}\n"
               f"Ants: {res.ants}\n"
               f"Cities: {res.cities}\n"
               f"First found generation: {res.generation_found}\n"
               f"Shortest route: {path_str}")
        messagebox.showinfo("Result", txt)
        self.info_text.set(txt)

    def _draw_solution(self, tour, dist, pheromone):
        self.ax.clear()

        # Remove axes and background grid
        self.ax.grid(False)

        xs = [p[0] for p in self.coords]
        ys = [p[1] for p in self.coords]

        # Draw cities
        self.ax.scatter(xs, ys)

        # Make city numbers BOLD
        for i, (x, y) in enumerate(self.coords):
            self.ax.text(x, y, str(i), fontweight='bold')

        # Draw ALL pheromone edges (gray + thickness depends on pheromone)
        max_ph = pheromone.max()
        for i in range(len(self.coords)):
            for j in range(i + 1, len(self.coords)):
                width = pheromone[i][j] / max_ph * 2.5  # relative thickness
                self.ax.plot(
                    [self.coords[i][0], self.coords[j][0]],
                    [self.coords[i][1], self.coords[j][1]],
                    linewidth=width,
                    color="gray",
                    alpha=0.4
                )

        # Draw BEST ROUTE (red)
        if tour:
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i + 1) % len(tour)]
                x1, y1 = self.coords[a]
                x2, y2 = self.coords[b]
                self.ax.plot([x1, x2], [y1, y2],
                             linewidth=2.8,
                             color="red")

        self.ax.set_title(f"Best distance: {dist:.4f}")
        self.canvas.draw()

    def on_close(self):
        self.destroy()

if __name__ == "__main__":
    import sys
    app = ACOApp()
    app.mainloop()
