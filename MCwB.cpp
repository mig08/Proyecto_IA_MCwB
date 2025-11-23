#include <bits/stdc++.h>
using namespace std;

#include <random>
#include <chrono>

mt19937_64 rng;

void initRNG(unsigned long long seed) {
    rng.seed(seed);
}

// ================== ESTRUCTURAS ==================

struct Node {
    int id;
    double x, y;
    char quality;   // A, B, C o - para planta
    double litros;  // litros de leche
    bool isPlant;
};

struct Instance {
    int numTrucks;
    vector<double> truckCap;

    int numQualities;
    vector<double> quotas;        // cuotas minimas por calidad
    vector<double> calidad_inst;  // ingresos por litro por calidad

    vector<Node> nodes;
    int plantIndex;

    vector<vector<double>> dist;
};

struct TruckRoute {
    int camion_id;         // indice del camion
    vector<int> nodes;     // indices de Instance::nodes
};

struct CustomerPos {
    int routeIndex;
    int pos; // indice dentro de route.nodes
};

// ================== UTILIDADES ==================

double routeLoad(const Instance &inst, const TruckRoute &route) {
    double load = 0.0;
    for (size_t i = 1; i + 1 < route.nodes.size(); ++i) {
        int idx = route.nodes[i];
        const Node &nd = inst.nodes[idx];
        if (nd.isPlant) continue;
        load += nd.litros;
    }
    return load;
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

bool generateNeighbor2Opt(const Instance &inst, const vector<TruckRoute> &current, vector<TruckRoute> &neighbor)
{
    neighbor = current;

    vector<int> candidateRoutes;
    for (int r = 0; r < (int)neighbor.size(); ++r) {
        if (neighbor[r].nodes.size() >= 4) {
            candidateRoutes.push_back(r);
        }
    }

    if (candidateRoutes.empty()) return false;

    uniform_int_distribution<int> routeDist(0, (int)candidateRoutes.size() - 1);

    int tries = 100;
    while (tries--) {
        int rIdx = candidateRoutes[routeDist(rng)];
        auto &rt = neighbor[rIdx].nodes;
        int m = (int)rt.size();

        if (m < 4) continue;

        uniform_int_distribution<int> iDist(1, m - 3);
        int i = iDist(rng);

        uniform_int_distribution<int> jDist(i + 1, m - 2);
        int j = jDist(rng);

        neighbor = current;
        auto &rt2 = neighbor[rIdx].nodes;
        std::reverse(rt2.begin() + i, rt2.begin() + j + 1);

        return true;
    }

    return false;
}


// ================== GENERATE NEIGHBOR GENERAL ==================

bool generateNeighbor(const Instance &inst, const vector<TruckRoute> &current, vector<TruckRoute> &neighbor)
{
    return generateNeighbor2Opt(inst, current, neighbor);
}

// ================== LECTURA DE INSTANCIA ==================

Instance readInstance(istream &in) {
    Instance inst;

    in >> inst.numTrucks;
    inst.truckCap.resize(inst.numTrucks);
    for (int i = 0; i < inst.numTrucks; ++i) {
        in >> inst.truckCap[i];
    }

    in >> inst.numQualities;
    inst.quotas.resize(inst.numQualities);
    for (int i = 0; i < inst.numQualities; ++i) {
        in >> inst.quotas[i];
    }

    inst.calidad_inst.resize(inst.numQualities);
    for (int i = 0; i < inst.numQualities; ++i) {
        in >> inst.calidad_inst[i];
    }

    int numNodes;
    in >> numNodes;

    inst.nodes.clear();
    inst.nodes.reserve(numNodes);

    inst.plantIndex = -1;

    for (int k = 0; k < numNodes; ++k) {
        Node node;
        char typeChar;
        in >> node.id >> node.x >> node.y >> typeChar >> node.litros;
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

// control + f greedy

vector<TruckRoute> greedyInitialsolucion(const Instance &inst) {

    int N = (int)inst.nodes.size();
    vector<bool> visited(N, false);
    visited[inst.plantIndex] = true;

    vector<double> totalByQ(3, 0.0); // A,B,C (0,1,2)
    for (int i = 0; i < N; ++i) {
        if (inst.nodes[i].isPlant) continue;
        int q = qualityIndex(inst.nodes[i].quality);
        if (q >= 0 && q < 3) {
            totalByQ[q] += inst.nodes[i].litros;
        }
    }

    double totalA = totalByQ[0];
    double totalB = totalByQ[1];
    double totalC = totalByQ[2];

    double quotaA = (inst.numQualities > 0 ? inst.quotas[0] : 0.0);
    double quotaB = (inst.numQualities > 1 ? inst.quotas[1] : 0.0);
    double quotaC = (inst.numQualities > 2 ? inst.quotas[2] : 0.0);

    double collectedA = 0.0;
    double collectedB = 0.0;
    double collectedC = 0.0;

    int T = inst.numTrucks;

    auto chooseTruckForVolume = [&](double volume,
                                    const vector<bool> &used) -> int {
        int best = -1;
        double bestCap = 1e18;
        for (int i = 0; i < T; ++i) {
            if (used[i]) continue;
            double cap = inst.truckCap[i];
            if (cap + 1e-9 >= volume && cap < bestCap) {
                bestCap = cap;
                best = i;
            }
        }
        return best;
    };

    vector<bool> truckUsed(T, false);
    int truckA = -1;
    int truckB = -1;
    int truckC = -1;

    if (totalA > 0.0) {
        truckA = chooseTruckForVolume(totalA, truckUsed);
        if (truckA != -1) {
            truckUsed[truckA] = true;
        }
    }

    if (totalB > 0.0) {
        truckB = chooseTruckForVolume(totalB, truckUsed);
        if (truckB != -1) {
            truckUsed[truckB] = true;
        }
    }

    if (totalC > 0.0) {
        truckC = chooseTruckForVolume(totalC, truckUsed);
        if (truckC != -1) {
            truckUsed[truckC] = true;
        }
    }

    vector<TruckRoute> solution;

    auto isQual = [&](int idx, char c) {
        return inst.nodes[idx].quality == c;
    };

    // FASE 1: Calidad A
    if (truckA != -1 && quotaA > 0.0) {
        TruckRoute route;
        route.camion_id = truckA;
        int current = inst.plantIndex;
        double cap = inst.truckCap[truckA];

        route.nodes.clear();
        route.nodes.push_back(inst.plantIndex);
        bool added = false;

        while (cap > 1e-9 && collectedA < quotaA - 1e-9) {
            int best = -1;
            double bestLitros = -1.0;
            double bestScore = -1e18;

            for (int j = 0; j < N; ++j) {
                if (visited[j]) continue;
                if (!isQual(j,'A')) continue;
                if (inst.nodes[j].litros > cap) continue;

                double litros = inst.nodes[j].litros;
                double d = inst.dist[current][j] + 1e-6;
                double score = litros / d;

                if (litros > bestLitros + 1e-9 ||
                    (fabs(litros - bestLitros) <= 1e-9 && score > bestScore)) {
                    bestLitros = litros;
                    bestScore = score;
                    best = j;
                }
            }

            if (best == -1) break;

            double litrosBest = inst.nodes[best].litros;
            double newCollected = collectedA + litrosBest;

            if (newCollected > quotaA + 1e-9) {
                int chosen = -1;
                double chosenTotal = -1.0;
                bool foundBelow = false;

                for (int j = 0; j < N; ++j) {
                    if (visited[j]) continue;
                    if (!isQual(j,'A')) continue;
                    if (inst.nodes[j].litros > cap) continue;

                    double litros = inst.nodes[j].litros;
                    double candidateTotal = collectedA + litros;

                    if (candidateTotal <= quotaA + 1e-9) {
                        if (!foundBelow || candidateTotal > chosenTotal + 1e-9) {
                            foundBelow = true;
                            chosenTotal = candidateTotal;
                            chosen = j;
                        }
                    } else if (!foundBelow) {
                        double excesoActual = candidateTotal - quotaA;
                        double excesoChosen = (chosen == -1 ? 1e18 : (chosenTotal - quotaA));

                        if (chosen == -1 || excesoActual < excesoChosen - 1e-9) {
                            chosenTotal = candidateTotal;
                            chosen = j;
                        }
                    }
                }

                if (chosen != -1) {
                    best = chosen;
                    litrosBest = inst.nodes[best].litros;
                    newCollected = collectedA + litrosBest;
                }
            }

            route.nodes.push_back(best);
            visited[best] = true;
            added = true;

            cap -= inst.nodes[best].litros;
            collectedA += inst.nodes[best].litros;
            current = best;

            if (collectedA >= quotaA - 1e-9) break;
        }

        route.nodes.push_back(inst.plantIndex);

        if (added) {
            solution.push_back(route);
        }
    }

    // FASE 2: Calidad B (acepta A y B)
    if (truckB != -1 && quotaB > 0.0) {
        TruckRoute route;
        route.camion_id = truckB;
        int current = inst.plantIndex;
        double cap = inst.truckCap[truckB];

        route.nodes.clear();
        route.nodes.push_back(inst.plantIndex);
        bool added = false;

        while (cap > 1e-9 && collectedB < quotaB - 1e-9) {
            int best = -1;
            double bestLitros = -1.0;
            double bestScore = -1e18;

            for (int j = 0; j < N; ++j) {
                if (visited[j]) continue;
                if (inst.nodes[j].isPlant) continue;

                char c = inst.nodes[j].quality;
                if (c != 'A' && c != 'B') continue;
                if (inst.nodes[j].litros > cap) continue;

                double litros = inst.nodes[j].litros;
                double d = inst.dist[current][j] + 1e-6;
                double score = litros / d;

                if (litros > bestLitros + 1e-9 ||
                    (fabs(litros - bestLitros) <= 1e-9 && score > bestScore)) {
                    bestLitros = litros;
                    bestScore = score;
                    best = j;
                }
            }

            if (best == -1) break;

            route.nodes.push_back(best);
            visited[best] = true;
            added = true;

            cap -= inst.nodes[best].litros;
            collectedB += inst.nodes[best].litros;
            current = best;

            if (collectedB >= quotaB - 1e-9) break;
        }

        route.nodes.push_back(inst.plantIndex);

        if (added) {
            solution.push_back(route);
        }
    }

    // FASE 3: Calidad C (acepta A,B,C)
    if (truckC != -1 && quotaC > 0.0) {
        TruckRoute route;
        route.camion_id = truckC;
        int current = inst.plantIndex;
        double cap = inst.truckCap[truckC];

            route.nodes.clear();
        route.nodes.push_back(inst.plantIndex);
        bool added = false;

        while (cap > 1e-9 && collectedC < quotaC - 1e-9) {
            int best = -1;
            double bestLitros = -1.0;
            double bestScore = -1e18;

            for (int j = 0; j < N; ++j) {
                if (visited[j]) continue;
                if (inst.nodes[j].isPlant) continue;
                char qc = inst.nodes[j].quality;
                if (qc != 'A' && qc != 'B' && qc != 'C') continue;
                if (inst.nodes[j].litros > cap) continue;

                double litros = inst.nodes[j].litros;
                double d = inst.dist[current][j] + 1e-6;
                double score = litros / d;

                if (litros > bestLitros + 1e-9 ||
                    (fabs(litros - bestLitros) <= 1e-9 && score > bestScore)) {
                    bestLitros = litros;
                    bestScore = score;
                    best = j;
                }
            }

            if (best == -1) break;

            route.nodes.push_back(best);
            visited[best] = true;
            added = true;

            cap -= inst.nodes[best].litros;
            collectedC += inst.nodes[best].litros;
            current = best;

            if (collectedC >= quotaC - 1e-9) break;
        }

        route.nodes.push_back(inst.plantIndex);

        if (added) {
            solution.push_back(route);
        }
    }

    // fase de completar las granjas

    struct RouteInfo {
        int worstQ;       // recuerda q 0=A, 1=B, 2=C, -1 si no tiene leche
        double volume;    // litros totales de la ruta
    };

    vector<RouteInfo> routeInfo(solution.size());
    for (size_t r = 0; r < solution.size(); ++r) {
        int w = -1;
        double vol = 0.0;
        for (int idx : solution[r].nodes) {
            const Node &nd = inst.nodes[idx];
            if (nd.isPlant) continue;
            int q = qualityIndex(nd.quality);
            if (q >= 0) {
                if (q > w) w = q;
                vol += nd.litros;
            }
        }
        routeInfo[r].worstQ = w;
        routeInfo[r].volume = vol;
    }

    for (const auto &rt : solution) {
        if (rt.camion_id >= 0 && rt.camion_id < inst.numTrucks) {
            truckUsed[rt.camion_id] = true;
        }
    }

    vector<int> unassigned;
    for (int i = 0; i < N; ++i) {
        if (inst.nodes[i].isPlant) continue;
        if (!visited[i]) {
            unassigned.push_back(i);
        }
    }

    sort(unassigned.begin(), unassigned.end(), [&](int a, int b) {
        return inst.nodes[a].litros > inst.nodes[b].litros;
    });

    for (int farmIdx : unassigned) {
        if (visited[farmIdx]) continue;

        const Node &farm = inst.nodes[farmIdx];
        int qFarm = qualityIndex(farm.quality);
        double litrosFarm = farm.litros;

        bool assigned = false;

        for (size_t r = 0; r < solution.size(); ++r) {

            TruckRoute &rt = solution[r];
            RouteInfo &ri = routeInfo[r];

            double cap = inst.truckCap[rt.camion_id];

            if (ri.volume + litrosFarm > cap + 1e-9) continue;

            if (ri.volume > 0.0) {
                int w = ri.worstQ;
                if (qFarm > w) {
                    continue;
                }
            }

            rt.nodes.insert(rt.nodes.end() - 1, farmIdx);

            if (ri.volume <= 0.0) {
                ri.volume = litrosFarm;
                ri.worstQ = qFarm;
            } else {
                ri.volume += litrosFarm;
            }

            visited[farmIdx] = true;
            assigned = true;
            break;
        }

        if (assigned) continue;

        int chosenTruck = -1;
        double bestCap = 1e18;
        for (int t = 0; t < inst.numTrucks; ++t) {
            if (truckUsed[t]) continue;
            if (inst.truckCap[t] + 1e-9 >= litrosFarm && inst.truckCap[t] < bestCap) {
                bestCap = inst.truckCap[t];
                chosenTruck = t;
            }
        }

        if (chosenTruck != -1) {
            TruckRoute newRoute;
            newRoute.camion_id = chosenTruck;
            newRoute.nodes.clear();
            newRoute.nodes.push_back(inst.plantIndex);
            newRoute.nodes.push_back(farmIdx);
            newRoute.nodes.push_back(inst.plantIndex);

            solution.push_back(newRoute);

            RouteInfo newInfo;
            newInfo.volume = litrosFarm;
            newInfo.worstQ = qFarm;
            routeInfo.push_back(newInfo);

            truckUsed[chosenTruck] = true;
            visited[farmIdx] = true;
        }
    }

    return solution;
}

// ================== EVALUACIÓN DE SOLUCIÓN ==================

struct EvalResult {
    double profit;                 // revenue - cost
    double cost;
    double revenue;
    vector<double> totalMilkPerQuality;
    bool quotasSatisfied;
};

EvalResult evaluatesolucion(const Instance &inst, const vector<TruckRoute> &sol) {
    EvalResult res;
    res.cost = 0.0;
    res.revenue = 0.0;
    res.totalMilkPerQuality.assign(inst.numQualities, 0.0);

    for (const auto &route : sol) {
        if (route.nodes.size() < 2) continue;

        for (size_t i = 0; i + 1 < route.nodes.size(); ++i) {
            int u = route.nodes[i];
            int v = route.nodes[i + 1];
            res.cost += inst.dist[u][v];
        }

        int worstQuality = -1; // recuerda q 0=A,1=B,2=C
        double volume = 0.0;

        for (size_t i = 0; i < route.nodes.size(); ++i) {
            int idx = route.nodes[i];
            const Node &nd = inst.nodes[idx];
            if (nd.isPlant) continue;

            int q = qualityIndex(nd.quality);
            if (q == -1) continue;

            if (q > worstQuality) worstQuality = q;
            volume += nd.litros;
        }

        if (volume > 0.0 && worstQuality != -1 &&
            worstQuality < (int)res.totalMilkPerQuality.size()) {
            res.totalMilkPerQuality[worstQuality] += volume;
        }
    }

    for (int t = 0; t < inst.numQualities; ++t) {
        res.revenue += res.totalMilkPerQuality[t] * inst.calidad_inst[t];
    }

    res.profit = res.revenue - res.cost;

    res.quotasSatisfied = true;
    for (int t = 0; t < inst.numQualities; ++t) {
        if (res.totalMilkPerQuality[t] + 1e-9 < inst.quotas[t]) {
            res.quotasSatisfied = false;
            break;
        }
    }

    return res;
}

// ================== IMPRESIÓN ==================

void printsolucion(const Instance &inst, const vector<TruckRoute> &sol, const EvalResult &ev, unsigned long long seed) {
    cout << "Seed: " << seed << "\n";
    cout << "Profit: " << ev.profit << "\n"
         << "Costo Total: " << ev.cost << "\n"
         << "Ingresos por Leche: " << ev.revenue << "\n";
    cout << "-----------------------------------------------------------------------------------------\n";

    for (const auto &route : sol) {
        string path;
        for (size_t i = 0; i < route.nodes.size(); ++i) {
            int idx = route.nodes[i];
            int id = inst.nodes[idx].id;
            if (i > 0) path += "-";
            path += to_string(id);
        }

        double routeCost = 0.0;
        for (size_t i = 0; i + 1 < route.nodes.size(); ++i) {
            int u = route.nodes[i];
            int v = route.nodes[i + 1];
            routeCost += inst.dist[u][v];
        }

        int worstQuality = -1;
        double volume = 0.0;
        for (size_t i = 0; i < route.nodes.size(); ++i) {
            int idx = route.nodes[i];
            const Node &nd = inst.nodes[idx];
            if (nd.isPlant) continue;
            int q = qualityIndex(nd.quality);
            if (q == -1) continue;
            if (q > worstQuality) worstQuality = q;
            volume += nd.litros;
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

    cerr << "Leche final en planta por calidad:\n";
    for (int t = 0; t < inst.numQualities; ++t) {
        cerr << "Calidad " << qualityChar(t)
             << ": " << ev.totalMilkPerQuality[t]
             << " (cuota mínima " << inst.quotas[t] << ")\n";
    }
    cerr << "¿Se cumplen las cuotas?: " << (ev.quotasSatisfied ? "SI" : "NO") << "\n";
}

// control + f simulated annealing

vector<TruckRoute> simulatedAnnealing(const Instance &inst, const vector<TruckRoute> &initial, EvalResult &bestEval, const string &logFilename)
{
    ofstream flog(logFilename);
    flog << "iter,T,current_profit,best_profit,neighbor_profit,accepted\n";

    vector<TruckRoute> current = initial;
    EvalResult currentEval = evaluatesolucion(inst, current);

    if (!currentEval.quotasSatisfied) {
        cerr << "[ERROR] La solución inicial NO es factible. "
             << "No se puede usar SA sin penalización.\n";
        exit(1);
    }

    vector<TruckRoute> best = current;
    EvalResult bestEvalLocal = currentEval;

    double T = 100.0;
    double alpha = 0.98;
    int maxIter = 1000;

    uniform_real_distribution<double> ur(0.0, 1.0);

    for (int iter = 0; iter < maxIter && T > 0.00001; ++iter) {

        vector<TruckRoute> neighbor;
        if (!generateNeighbor(inst, current, neighbor)) {
            T *= alpha;
            continue;
        }

        EvalResult neighEval = evaluatesolucion(inst, neighbor);

        if (!neighEval.quotasSatisfied) {
            flog << iter << "," << T << ","
                 << currentEval.profit << "," << bestEvalLocal.profit << ","
                 << neighEval.profit << "," << 0 << "\n";

            T *= alpha;
            continue;
        }

        double delta = neighEval.profit - currentEval.profit;

        bool accepted = false;
        if (delta >= 0) {
            accepted = true;
        } else {
            double p = exp(delta / T);
            double r = ur(rng);
            if (r < p) accepted = true;
        }

        flog << iter << "," << T << ","
             << currentEval.profit << "," << bestEvalLocal.profit << ","
             << neighEval.profit << "," << (accepted ? 1 : 0) << "\n";

        if (accepted) {
            current = neighbor;
            currentEval = neighEval;
        }

        if (currentEval.profit > bestEvalLocal.profit) {
            best = current;
            bestEvalLocal = currentEval;
        }

        T *= alpha;
    }

    bestEval = bestEvalLocal;
    return best;
}

double routeCost(const Instance &inst, const TruckRoute &route) {
    double c = 0.0;
    for (size_t i = 0; i + 1 < route.nodes.size(); ++i) {
        int u = route.nodes[i];
        int v = route.nodes[i + 1];
        c += inst.dist[u][v];
    }
    return c;
}

// ================== MAIN ==================

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    unsigned long long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    initRNG(seed);

    ifstream in("a80.txt");
    if (!in) {
        cerr << "ERROR: no se pudo abrir la instancia\n";
        return 1;
    }

    Instance inst = readInstance(in);

    vector<TruckRoute> initialSol = greedyInitialsolucion(inst);
    EvalResult greedyEval = evaluatesolucion(inst, initialSol);

    EvalResult saEval;
    vector<TruckRoute> bestSol = simulatedAnnealing(inst, initialSol, saEval, "sa_iterations.csv");


    //aqui crea los archivos para la creacion de las tablas y los graficos
    {
        ofstream fs("summary.csv");
        fs << "solution,profit,cost,revenue\n";
        fs << "GREEDY," << greedyEval.profit << ","
           << greedyEval.cost << ","
           << greedyEval.revenue << "\n";
        fs << "SA," << saEval.profit << ","
           << saEval.cost << ","
           << saEval.revenue << "\n";
    }

    {
        ofstream fg("routes_greedy.csv");
        fg << "route_id,cost\n";
        for (size_t r = 0; r < initialSol.size(); ++r) {
            double c = routeCost(inst, initialSol[r]);
            fg << r << "," << c << "\n";
        }
    }

    {
        ofstream fs("routes_sa.csv");
        fs << "route_id,cost\n";
        for (size_t r = 0; r < bestSol.size(); ++r) {
            double c = routeCost(inst, bestSol[r]);
            fs << r << "," << c << "\n";
        }
    }

    printsolucion(inst, bestSol, saEval, seed);

    return 0;
}
