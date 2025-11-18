#include <bits/stdc++.h>
using namespace std;

#include <random>
#include <chrono>

mt19937_64 rng;

void initRNG(unsigned long long seed) {
    rng.seed(seed);
}


struct Node {
    int id;
    double x, y;
    char quality;   // 'A','B','C' o '-' para planta
    double amount;  // litros
    bool isPlant;
};

struct Instance {
    int numTrucks;
    vector<double> truckCap;

    int numQualities;
    vector<double> quotas;
    vector<double> revenues; // por calidad (A,B,C)

    vector<Node> nodes;
    int plantIndex; // índice en nodes

    vector<vector<double>> dist; // matriz de distancias
};

struct TruckRoute {
    int truckIdx;
    vector<int> nodes; // secuencia de índices de nodes (incluyendo planta al inicio y fin)
};

struct CustomerPos {
    int routeIndex;
    int pos; // índice dentro de route.nodes
};

// -------------------- Utilidades --------------------
double routeLoad(const Instance &inst, const TruckRoute &route) {
    double load = 0.0;
    // clientes están entre 1 y size-2 (0 y size-1 son planta)
    for (size_t i = 1; i + 1 < route.nodes.size(); ++i) {
        int idx = route.nodes[i];
        const Node &nd = inst.nodes[idx];
        if (nd.isPlant) continue;
        load += nd.amount;
    }
    return load;
}

bool generateNeighborSwap(const Instance &inst,
                          const vector<TruckRoute> &current,
                          vector<TruckRoute> &neighbor) {
    neighbor = current;

    struct Pos { int r; int i; };
    vector<Pos> positions;

    // 1) Construimos la lista de TODAS las posiciones de clientes
    for (int r = 0; r < (int)neighbor.size(); ++r) {
        auto &rt = neighbor[r].nodes;
        if (rt.size() <= 2) continue; // ruta sin clientes
        for (int i = 1; i + 1 < (int)rt.size(); ++i) {
            positions.push_back({r, i});
        }
    }

    if (positions.size() < 2) return false;

    // Distribución uniforme entera sobre los índices de positions
    uniform_int_distribution<int> posDist(0, (int)positions.size() - 1);

    int tries = 100;
    while (tries--) {
        // 2) Elegimos dos posiciones de clientes completamente al azar
        Pos p1 = positions[posDist(rng)];
        Pos p2 = positions[posDist(rng)];

        if (p1.r == p2.r && p1.i == p2.i) continue; // misma posición, no sirve

        neighbor = current;

        // 3) Hacemos el swap de los índices de nodo
        int &a = neighbor[p1.r].nodes[p1.i];
        int &b = neighbor[p2.r].nodes[p2.i];
        std::swap(a, b);

        // 4) Revisamos capacidad de ambas rutas afectadas
        auto checkCap = [&](int r) {
            double load = 0.0;
            for (int i = 1; i + 1 < (int)neighbor[r].nodes.size(); ++i) {
                int idx = neighbor[r].nodes[i];
                load += inst.nodes[idx].amount;
            }
            return load <= inst.truckCap[neighbor[r].truckIdx] + 1e-9;
        };

        if (checkCap(p1.r) && checkCap(p2.r)) {
            return true; // vecino válido generado con SWAP aleatorio
        }
    }

    return false;
}




int qualityIndex(char c) {
    if (c == 'A') return 0;
    if (c == 'B') return 1;
    if (c == 'C') return 2;
    return -1;
}

char qualityChar(int idx) {
    static const string Q = "ABC";
    if (idx >= 0 && idx < (int)Q.size()) return Q[idx];
    return '?';
}

// -------------------- Lectura de instancia --------------------

Instance readInstance(istream &in) {
    Instance inst;

    // Número de camiones
    in >> inst.numTrucks;
    inst.truckCap.resize(inst.numTrucks);
    for (int i = 0; i < inst.numTrucks; ++i) {
        in >> inst.truckCap[i];
    }

    // Número de calidades
    in >> inst.numQualities;
    inst.quotas.resize(inst.numQualities);
    for (int i = 0; i < inst.numQualities; ++i) {
        in >> inst.quotas[i];
    }

    inst.revenues.resize(inst.numQualities);
    for (int i = 0; i < inst.numQualities; ++i) {
        in >> inst.revenues[i];
    }

    int numNodes;
    in >> numNodes;

    inst.nodes.clear();
    inst.nodes.reserve(numNodes);

    inst.plantIndex = -1;

    for (int k = 0; k < numNodes; ++k) {
        Node node;
        char typeChar;
        in >> node.id >> node.x >> node.y >> typeChar >> node.amount;
        node.quality = typeChar;
        node.isPlant = (typeChar == '-');

        if (node.isPlant) {
            inst.plantIndex = (int)inst.nodes.size();
        }
        inst.nodes.push_back(node);
    }

    if (inst.plantIndex == -1) {
        cerr << "ERROR: No se encontró la planta en el archivo de instancia.\n";
        exit(1);
    }

    // Construir matriz de distancias euclidianas
    int N = (int)inst.nodes.size();
    inst.dist.assign(N, vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double dx = inst.nodes[i].x - inst.nodes[j].x;
            double dy = inst.nodes[i].y - inst.nodes[j].y;
            inst.dist[i][j] = sqrt(dx * dx + dy * dy);
        }
    }

    return inst;
}

// -------------------- Greedy inicial --------------------

vector<TruckRoute> greedyInitialSolution(const Instance &inst) {
    int N = (int)inst.nodes.size();
    vector<bool> visited(N, false);
    // La planta no se marca como visitada porque se puede usar muchas veces
    for (int i = 0; i < N; ++i) {
        if (inst.nodes[i].isPlant) visited[i] = true; // nunca "visitamos" la planta como granja
    }

    vector<TruckRoute> solution;

    int remainingFarms = 0;
    for (int i = 0; i < N; ++i) {
        if (!inst.nodes[i].isPlant) remainingFarms++;
    }

    for (int k = 0; k < inst.numTrucks && remainingFarms > 0; ++k) {
        TruckRoute route;
        route.truckIdx = k;

        int current = inst.plantIndex;
        double remainingCap = inst.truckCap[k];

        route.nodes.push_back(inst.plantIndex); // empieza en planta

        while (true) {
            int best = -1;
            double bestDist = 1e18;

            for (int j = 0; j < N; ++j) {
                if (inst.nodes[j].isPlant) continue;
                if (visited[j]) continue;
                if (inst.nodes[j].amount > remainingCap) continue;

                double d = inst.dist[current][j];
                if (d < bestDist) {
                    bestDist = d;
                    best = j;
                }
            }

            if (best == -1) break; // no cabe ninguna granja más en este camión

            // Asignar la granja 'best' a esta ruta
            route.nodes.push_back(best);
            visited[best] = true;
            remainingCap -= inst.nodes[best].amount;
            remainingFarms--;
            current = best;
        }

        // Volver a la planta
        route.nodes.push_back(inst.plantIndex);
        if (route.nodes.size() > 2) {
            solution.push_back(route);
        }
    }

    if (remainingFarms > 0) {
        cerr << "ADVERTENCIA: no se pudieron asignar todas las granjas con los camiones disponibles.\n";
    }

    return solution;
}

// -------------------- Evaluación de solución --------------------

struct EvalResult {
    double profit;                 // revenue - cost
    double cost;
    double revenue;
    vector<double> totalMilkPerQuality; // litros finales por calidad (después de blending)
    bool quotasSatisfied;          // true sii se cumplen todas las cuotas
};


EvalResult evaluateSolution(const Instance &inst,
                            const vector<TruckRoute> &sol) {
    EvalResult res;
    res.cost = 0.0;
    res.revenue = 0.0;
    res.totalMilkPerQuality.assign(inst.numQualities, 0.0);

    // --------------------------------------
    // 1. Costo de transporte y blending por ruta
    // --------------------------------------
    for (const auto &route : sol) {
        if (route.nodes.size() < 2) continue;

        // Costo de viaje de esta ruta
        for (size_t i = 0; i + 1 < route.nodes.size(); ++i) {
            int u = route.nodes[i];
            int v = route.nodes[i + 1];
            res.cost += inst.dist[u][v];
        }

        // Mezcla de calidades en este camión
        int worstQuality = -1; // 0=A,1=B,2=C
        double volume = 0.0;

        for (size_t i = 0; i < route.nodes.size(); ++i) {
            int idx = route.nodes[i];
            const Node &nd = inst.nodes[idx];
            if (nd.isPlant) continue;

            int q = qualityIndex(nd.quality);
            if (q == -1) continue;

            if (q > worstQuality) worstQuality = q;
            volume += nd.amount;
        }

        // Toda la leche de esta ruta se suma a la peor calidad encontrada
        if (volume > 0.0 && worstQuality != -1 &&
            worstQuality < (int)res.totalMilkPerQuality.size()) {
            res.totalMilkPerQuality[worstQuality] += volume;
        }
    }

    // --------------------------------------
    // 2. Ingreso total
    // --------------------------------------
    for (int t = 0; t < inst.numQualities; ++t) {
        res.revenue += res.totalMilkPerQuality[t] * inst.revenues[t];
    }

    // --------------------------------------
    // 3. Ganancia
    // --------------------------------------
    res.profit = res.revenue - res.cost;

    // --------------------------------------
    // 4. Chequeo de cuotas (solo informativo)
    // --------------------------------------
    res.quotasSatisfied = true;
    for (int t = 0; t < inst.numQualities; ++t) {
        if (res.totalMilkPerQuality[t] < inst.quotas[t]) {
            res.quotasSatisfied = false;
            break;
        }
    }

    return res;
}


// -------------------- Impresión de solución --------------------

void printSolution(const Instance &inst,
                   const vector<TruckRoute> &sol,
                   const EvalResult &ev, 
                   unsigned long long seed) {
    cout << "Seed: " << seed << "\n";

    // Ganancia final | costo total | ganancia total
    // Aquí interpretamos como: profit cost revenue (como en el ejemplo del enunciado)
    cout  << "Profit: " << ev.profit << "\n" << "Costo Total: " << ev.cost << "\n" << "Ingresos por Leche: " << ev.revenue << "\n";
    cout << "-----------------------------------------------------------------------------------------\n";

    // Para cada ruta: imprimir ruta, costo y litros+tipo
    for (const auto &route : sol) {
        // Construir string de ruta
        // Ojo: usamos los ID originales del archivo
        string path;
        for (size_t i = 0; i < route.nodes.size(); ++i) {
            int idx = route.nodes[i];
            int id = inst.nodes[idx].id;
            if (i > 0) path += "-";
            path += to_string(id);
        }

        
        // Costo de la ruta
        double routeCost = 0.0;
        for (size_t i = 0; i + 1 < route.nodes.size(); ++i) {
            int u = route.nodes[i];
            int v = route.nodes[i + 1];
            routeCost += inst.dist[u][v];
        }

        // Mezcla de la ruta (misma lógica de evaluación)
        int worstQuality = -1;
        double volume = 0.0;
        for (size_t i = 0; i < route.nodes.size(); ++i) {
            int idx = route.nodes[i];
            const Node &nd = inst.nodes[idx];
            if (nd.isPlant) continue;
            int q = qualityIndex(nd.quality);
            if (q == -1) continue;
            if (q > worstQuality) worstQuality = q;
            volume += nd.amount;
        }

        char qChar = (worstQuality == -1 ? '-' : qualityChar(worstQuality));
        cout << path << " (Camino de Granjas)\n" << routeCost  << " (Costo de Camino)\n";
        if (volume > 0.0 && qChar != '-') {
            cout << volume << " Volumen de Leche de calidad " << qChar << "\n";
            cout << "-----------------------------------------------------------------------------------------\n";
        } else {
            cout << "0-\n";
        }
    }

    // Info adicional opcional (puedes quitarlo si tu profesor quiere salida estricta):
    cerr << "Leche final en planta por calidad:\n";
    for (int t = 0; t < inst.numQualities; ++t) {
        cerr << "Calidad " << qualityChar(t)
             << ": " << ev.totalMilkPerQuality[t] << "\n";
    }
}

vector<TruckRoute> simulatedAnnealing(const Instance &inst,
                                      const vector<TruckRoute> &initial,
                                      EvalResult &bestEval) {
    vector<TruckRoute> current = initial;
    EvalResult currentEval = evaluateSolution(inst, current);

    vector<TruckRoute> best = current;
    bestEval = currentEval;

    double T = 1000.0;       // temperatura inicial
    double alpha = 0.999;    // factor de enfriamiento
    int maxIter = 50000;     // número de iteraciones

    // Distribución uniforme real en [0,1)
    uniform_real_distribution<double> ur(0.0, 1.0);

    for (int iter = 0; iter < maxIter && T > 1e-4; ++iter) {
        vector<TruckRoute> neighbor;
        if (!generateNeighborSwap(inst, current, neighbor)) {
            T *= alpha;
            continue;
        }

        EvalResult neighborEval = evaluateSolution(inst, neighbor);

        double delta = neighborEval.profit - currentEval.profit;

        if (delta >= 0) {
            // 1) Si la nueva solución es mejor, siempre se acepta
            current = std::move(neighbor);
            currentEval = neighborEval;
        } else {
            // 2) Si es peor, se acepta con probabilidad exp(delta / T)
            double p = exp(delta / T);   // delta < 0, así que 0 < p < 1
            double r = ur(rng);          // r ~ Uniforme(0,1)

            if (r < p) {
                // ACEPTAMOS una solución peor
                current = std::move(neighbor);
                currentEval = neighborEval;
            }
        }

        // 3) Actualizar mejor global
        if (currentEval.profit > bestEval.profit) {
            best = current;
            bestEval = currentEval;
        }

        // 4) Enfriamiento de la temperatura
        T *= alpha;
    }

    return best;
}



int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Seed a partir del reloj
    unsigned long long seed =
        chrono::high_resolution_clock::now().time_since_epoch().count();
    initRNG(seed);

    ifstream in("a48.txt");
    if (!in) {
        cerr << "ERROR: no se pudo abrir instancia.txt\n";
        return 1;
    }

    Instance inst = readInstance(in);

    vector<TruckRoute> initialSol = greedyInitialSolution(inst);

    EvalResult bestEval;
    vector<TruckRoute> bestSol = simulatedAnnealing(inst, initialSol, bestEval);

    printSolution(inst, bestSol, bestEval, seed);
}

