import pandas as pd
import matplotlib.pyplot as plt

# 1) Tabla 1.5: resumen de ganancias
summary = pd.read_csv("summary.csv")
print("=== Tabla 1.5: Resumen de ganancias ===")
print(summary)
print()

# 2) Tabla 1.4: costos por ruta Greedy vs SA
routes_g = pd.read_csv("routes_greedy.csv")
routes_sa = pd.read_csv("routes_sa.csv")

# Merge por route_id (puede que el número de rutas difiera,
# se usa outer join para ver todas)
tabla_14 = pd.merge(routes_g, routes_sa, on="route_id",
                    how="outer", suffixes=("_greedy", "_sa"))

print("=== Tabla 1.4: Costos por ruta Greedy vs SA ===")
print(tabla_14)
print()

# 3) Tabla 1.7: log de iteraciones del SA (muchas iteraciones)
log_sa = pd.read_csv("sa_iterations.csv")

print("=== Tabla 1.7: Log de iteraciones (primeras 30) ===")
print(log_sa.head(30))  # puedes cambiar 30 por lo que quieras
print()

# 4) Gráfico 2.1: convergencia del SA (best_profit vs iter)
plt.figure()
plt.plot(log_sa["iter"], log_sa["best_profit"])
plt.xlabel("Iteración")
plt.ylabel("Mejor profit global")
plt.title("Convergencia del Simulated Annealing")
plt.grid(True)
plt.tight_layout()
plt.show()
