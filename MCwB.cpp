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
    char quality;   // 'A','B','C' o '-' para planta
    double litros;  // litros de leche
    bool isPlant;
};

struct Instance {
    int numTrucks;
    vector<double> truckCap;

    int numQualities;
    vector<double> quotas;        // cuotas mínimas por calidad
    vector<double> calidad_inst;  // ingresos por litro por calidad

    vector<Node> nodes;
    int plantIndex; 

    vector<vector<double>> dist; 
};

struct TruckRoute {
    int camion_id;         // índice del camión
    vector<int> nodes;     // índices de Instance::nodes
};

struct CustomerPos {
    int routeIndex;
    int pos; // índice dentro de route.nodes
};

// ================== UTILIDADES ==================

double routeLoad(const Instance &inst, const TruckRoute &route) {
    double load = 0.0;
    // clientes están entre 1 y size-2 (0 y size-1 son planta)
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

bool generateNeighbor2Opt(const Instance &inst,
                          const vector<TruckRoute> &current,
                          vector<TruckRoute> &neighbor)
{
    neighbor = current;

    // 1) Recolectar rutas que tengan al menos 2 clientes
    //    (necesitas [planta, c1, c2, planta] => size >= 4)
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
        // 2) Elegir al azar una ruta candidata
        int rIdx = candidateRoutes[routeDist(rng)];
        auto &rt = neighbor[rIdx].nodes;
        int m = (int)rt.size();

        // Clientes están en posiciones [1 .. m-2]
        if (m < 4) continue; // paranoia

        uniform_int_distribution<int> iDist(1, m - 3);
        int i = iDist(rng);

        uniform_int_distribution<int> jDist(i + 1, m - 2);
        int j = jDist(rng);

        // 3) Hacer el 2-opt: invertir el subsegmento [i, j]
        //    Ej: [planta, a, b, c, d, planta] + i=2, j=4 => revierte (b,c,d)
        neighbor = current;  // partir SIEMPRE desde la solución actual
        auto &rt2 = neighbor[rIdx].nodes;
        std::reverse(rt2.begin() + i, rt2.begin() + j + 1);

        // No hace falta revisar capacidad: los mismos clientes, mismo camión
        return true;
    }

    return false;
}


// ================== MOVIMIENTO SWAP (igual, pero lo usamos vía generateNeighbor) ==================

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

    uniform_int_distribution<int> posDist(0, (int)positions.size() - 1);

    int tries = 100;
    while (tries--) {
        Pos p1 = positions[posDist(rng)];
        Pos p2 = positions[posDist(rng)];

        if (p1.r == p2.r && p1.i == p2.i) continue; // misma posición, no sirve

        neighbor = current;

        int &a = neighbor[p1.r].nodes[p1.i];
        int &b = neighbor[p2.r].nodes[p2.i];
        std::swap(a, b);

        auto checkCap = [&](int r) {
            double load = 0.0;
            for (int i = 1; i + 1 < (int)neighbor[r].nodes.size(); ++i) {
                int idx = neighbor[r].nodes[i];
                load += inst.nodes[idx].litros;
            }
            return load <= inst.truckCap[neighbor[r].camion_id] + 1e-9;
        };

        if (checkCap(p1.r) && checkCap(p2.r)) {
            return true; // vecino válido generado con SWAP aleatorio
        }
    }

    return false;
}

// ================== MOVIMIENTO NUEVO: RELOCATE ==================
// *** CAMBIO: movimiento adicional para mejorar exploración

bool generateNeighborRelocate(const Instance &inst,
                              const vector<TruckRoute> &current,
                              vector<TruckRoute> &neighbor) {
    neighbor = current;

    struct Pos { int r; int i; };
    vector<Pos> positions;

    // Posiciones de clientes
    for (int r = 0; r < (int)neighbor.size(); ++r) {
        auto &rt = neighbor[r].nodes;
        if (rt.size() <= 2) continue; // sin clientes
        for (int i = 1; i + 1 < (int)rt.size(); ++i) {
            positions.push_back({r, i});
        }
    }

    if (positions.empty()) return false;

    uniform_int_distribution<int> posDist(0, (int)positions.size() - 1);
    uniform_int_distribution<int> routeDist(0, (int)neighbor.size() - 1);

    int tries = 100;
    while (tries--) {
        neighbor = current;

        // Elegimos una granja origen
        Pos from = positions[posDist(rng)];
        int node = neighbor[from.r].nodes[from.i];

        // Borramos esa posición
        neighbor[from.r].nodes.erase(neighbor[from.r].nodes.begin() + from.i);

        // Elegimos ruta destino (puede ser la misma u otra)
        int rTo = routeDist(rng);
        auto &rtTo = neighbor[rTo].nodes;
        if (rtTo.size() < 2) {
            // ruta vacía, solo planta-planta, insertamos entre medio
            rtTo.insert(rtTo.begin() + 1, node);
        } else {
            // posición de inserción entre planta inicio y planta final
            uniform_int_distribution<int> insDist(1, (int)rtTo.size() - 1);
            int insPos = insDist(rng);
            rtTo.insert(rtTo.begin() + insPos, node);
        }

        auto checkCap = [&](int r) {
            double load = 0.0;
            for (int i = 1; i + 1 < (int)neighbor[r].nodes.size(); ++i) {
                int idx = neighbor[r].nodes[i];
                load += inst.nodes[idx].litros;
            }
            return load <= inst.truckCap[neighbor[r].camion_id] + 1e-9;
        };

        // Chequear capacidad de rutas afectadas
        if (!checkCap(from.r)) continue;
        if (!checkCap(rTo)) continue;

        return true;
    }

    return false;
}

// ================== GENERATE NEIGHBOR GENERAL ==================
// *** CAMBIO: SA ahora puede usar SWAP o RELOCATE aleatoriamente

bool generateNeighbor(const Instance &inst,
                      const vector<TruckRoute> &current,
                      vector<TruckRoute> &neighbor)
{
    uniform_real_distribution<double> ur(0.0, 1.0);
    double x = ur(rng);

    // 50% de las veces SWAP, 50% 2-opt
    if (x < 0.5) {
        return generateNeighborSwap(inst, current, neighbor);
    } else {
        return generateNeighbor2Opt(inst, current, neighbor);
    }
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

// ================== GREEDY INICIAL ==================
vector<TruckRoute> greedyInitialsolucion(const Instance &inst) {

    int N = (int)inst.nodes.size();
    vector<bool> visited(N, false);
    visited[inst.plantIndex] = true;

    // =========================
    // 0) Cantidad de leche por calidad
    // =========================
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



    cout << "=============================\n";
    cout << " CANTIDAD DE LECHE POR CALIDAD\n";
    cout << "=============================\n";
    cout << "Total calidad A: " << totalA << " L (cuota " << quotaA << ")\n";
    cout << "Total calidad B: " << totalB << " L (cuota " << quotaB << ")\n";
    cout << "Total calidad C: " << totalC << " L (cuota " << quotaC << ")\n";

    int T = inst.numTrucks;

    cout << "Camiones disponibles (ID, capacidad):\n";
    for (int i = 0; i < T; ++i) {
        cout << "  Camion " << i << " cap=" << inst.truckCap[i] << "\n";
    }
    cout << "\n";

    // Helper: elegir el camión más chico cuya capacidad soporte todo 'volume'
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
        return best; // -1 si no hay camión que soporte TODO ese volumen
    };

    vector<bool> truckUsed(T, false);
    int truckA = -1;
    int truckB = -1;
    int truckC = -1;

    // =========================
    // 1) Asignar camión a calidad A
    // =========================
    if (totalA > 0.0) {
        truckA = chooseTruckForVolume(totalA, truckUsed);
        if (truckA != -1) {
            truckUsed[truckA] = true;
            cout << "Camion asignado a calidad A: " << truckA
                 << " (cap=" << inst.truckCap[truckA] << ")\n";
        } else {
            cout << "ADVERTENCIA: No hay camión que pueda transportar toda la calidad A (" 
                 << totalA << " L)\n";
        }
    } else {
        cout << "No hay granjas de calidad A.\n";
    }

    // =========================
    // 2) Asignar camión a calidad B
    // =========================
    if (totalB > 0.0) {
        truckB = chooseTruckForVolume(totalB, truckUsed);
        if (truckB != -1) {
            truckUsed[truckB] = true;
            cout << "Camion asignado a calidad B: " << truckB
                 << " (cap=" << inst.truckCap[truckB] << ")\n";
        } else {
            cout << "ADVERTENCIA: No hay camión que pueda transportar toda la calidad B (" 
                 << totalB << " L)\n";
        }
    } else {
        cout << "No hay granjas de calidad B.\n";
    }

    // =========================
    // 3) Asignar camión a calidad C
    // =========================
    if (totalC > 0.0) {
        truckC = chooseTruckForVolume(totalC, truckUsed);
        if (truckC != -1) {
            truckUsed[truckC] = true;
            cout << "Camion asignado a calidad C: " << truckC
                << " (cap=" << inst.truckCap[truckC] << ")\n";
        } else {
            cout << "ADVERTENCIA: No hay camión que pueda transportar toda la calidad C (" 
                << totalC << " L)\n";
        }
    } else {
        cout << "No hay granjas de calidad C.\n";
    }
    cout << "\n";


    cout << "\n";

    vector<TruckRoute> solution;

    auto isQual = [&](int idx, char c) {
        return inst.nodes[idx].quality == c;
    };

    cout << "========= FASE 1: Calidad A =========\n";
    if (truckA != -1 && quotaA > 0.0) {
        TruckRoute route;
        route.camion_id = truckA;
        int current = inst.plantIndex;
        double cap = inst.truckCap[truckA];

        route.nodes.clear();
        route.nodes.push_back(inst.plantIndex);
        bool added = false;

        cout << "Camion " << truckA << " (cap=" << cap << ") recolectando calidad A\n";

        while (cap > 1e-9 && collectedA < quotaA - 1e-9) {
            int best = -1;
            double bestLitros = -1.0;
            double bestScore = -1e18;

            // =========================
            // PRIMERA PASADA: greedy normal (litros/distancia)
            // =========================
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

            // =========================
            // NUEVO BLOQUE:
            // Si la mejor opción se pasa de la cuota,
            // buscamos una alternativa que deje el total
            // lo más cerca posible a la cuota.
            // =========================
            double litrosBest = inst.nodes[best].litros;
            double newCollected = collectedA + litrosBest;

            if (newCollected > quotaA + 1e-9) {
                // Intentar buscar otra granja A que mejore el ajuste a la cuota
                int chosen = -1;
                double chosenTotal = -1.0;     // total recogido con esa granja
                bool foundBelow = false;       // ¿encontramos alguna que NO se pase?

                // Buscamos en TODAS las granjas A disponibles
                for (int j = 0; j < N; ++j) {
                    if (visited[j]) continue;
                    if (!isQual(j,'A')) continue;
                    if (inst.nodes[j].litros > cap) continue;

                    double litros = inst.nodes[j].litros;
                    double candidateTotal = collectedA + litros;

                    if (candidateTotal <= quotaA + 1e-9) {
                        // No se pasa de la cuota → preferimos acercarnos por debajo
                        if (!foundBelow || candidateTotal > chosenTotal + 1e-9) {
                            foundBelow = true;
                            chosenTotal = candidateTotal;
                            chosen = j;
                        }
                    } else if (!foundBelow) {
                        // Se pasa de la cuota, pero aún no encontramos ninguna por debajo
                        // → elegimos la que MENOS se pase (mínimo exceso)
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
                // Si chosen sigue -1, nos quedamos con el best original
            }

            // =========================
            // A partir de aquí, 'best' es la granja elegida final
            // =========================
            cout << "  → Granja A " << inst.nodes[best].id
                << " (" << inst.nodes[best].litros << " L)\n";

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
            cout << "Ruta A cerrada. A recolectado: " 
                << collectedA << " / " << totalA << " L\n";
        } else {
            cout << "  Camion " << truckA << " no pudo recolectar granjas A.\n";
        }
    } else {
        cout << "No se asignó camión específico para A.\n";
    }
    cout << "\n";


    cout << "========= FASE 2: Calidad B (acepta A y B) =========\n";
    if (truckB != -1 && quotaB > 0.0) {
        TruckRoute route;
        route.camion_id = truckB;
        int current = inst.plantIndex;
        double cap = inst.truckCap[truckB];

        route.nodes.clear();
        route.nodes.push_back(inst.plantIndex);
        bool added = false;

        cout << "Camion " << truckB << " (cap=" << cap << ") recolectando calidad B (A/B)\n";

        // 👉 AHORA SOLO MIENTRAS HAYA CAPACIDAD **Y** NO SE HAYA LLEGADO A LA CUOTA B
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

            cout << "  → Granja " << inst.nodes[best].quality << " "
                << inst.nodes[best].id
                << " (" << inst.nodes[best].litros << " L)\n";

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
            cout << "Ruta B cerrada. B recolectado (A+B en este camión): " 
                 << collectedB << " L\n";
        } else {
            cout << "  Camion " << truckB << " no pudo recolectar granjas A/B.\n";
        }
    } else {
        cout << "No se asignó camión específico para B.\n";
    }
    cout << "\n";

    cout << "========= FASE 3: Calidad C =========\n";
    if (truckC != -1 && quotaC > 0.0) {
        TruckRoute route;
        route.camion_id = truckC;
        int current = inst.plantIndex;
        double cap = inst.truckCap[truckC];

        route.nodes.clear();
        route.nodes.push_back(inst.plantIndex);
        bool added = false;

        cout << "Camion " << truckC << " (cap=" << cap << ") recolectando calidad C\n";

        // Mientras haya capacidad y no se llegue a la cuota C
        while (cap > 1e-9 && collectedC < quotaC - 1e-9) {
            int best = -1;
            double bestLitros = -1.0;
            double bestScore = -1e18;

            for (int j = 0; j < N; ++j) {
                if (visited[j]) continue;
                if (inst.nodes[j].isPlant) continue;   // nunca la planta
                char qc = inst.nodes[j].quality;
                if (qc != 'A' && qc != 'B' && qc != 'C') continue;             
                if (inst.nodes[j].litros > cap) continue;

                double litros = inst.nodes[j].litros;
                double d = inst.dist[current][j] + 1e-6;
                double score = litros / d;  // litros / distancia

                if (litros > bestLitros + 1e-9 ||
                    (fabs(litros - bestLitros) <= 1e-9 && score > bestScore)) {
                    bestLitros = litros;
                    bestScore = score;
                    best = j;
                }
            }

            if (best == -1) break;

            cout << "  → Granja C " << inst.nodes[best].id
                << " (" << inst.nodes[best].litros << " L)\n";

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
            cout << "Ruta C cerrada. C recolectado: " 
                << collectedC << " / " << totalC << " L\n";
        } else {
            cout << "  Camion " << truckC << " no pudo recolectar granjas C.\n";
        }
    } else {
        cout << "No se asignó camión específico para C.\n";
    }
    cout << "\n";



    // ====================================================
    // FASE 3 (NUEVA):
    //  - Tomar TODAS las granjas no asignadas
    //  - Intentar meterlas primero en rutas existentes
    //    que tengan capacidad restante y NO empeoren la calidad
    //  - Si no se puede, usar camiones nuevos
    // ====================================================

    cout << "========= FASE 3: COMPLETAR CON GRANJAS RESTANTES =========\n";

    // 1) Construir info por ruta: peor calidad (worstQ) y litros totales
    struct RouteInfo {
        int worstQ;       // 0=A, 1=B, 2=C, -1 si no tiene leche
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
        routeInfo[r].worstQ = w;   // -1 si no tenía leche (caso raro)
        routeInfo[r].volume = vol;
    }

    // 2) Marcar qué camiones ya están siendo usados por alguna ruta
    //vector<bool> truckUsed(inst.numTrucks, false);
    for (const auto &rt : solution) {
        if (rt.camion_id >= 0 && rt.camion_id < inst.numTrucks) {
            truckUsed[rt.camion_id] = true;
        }
    }

    // 3) Construir lista de granjas no asignadas
    vector<int> unassigned;
    for (int i = 0; i < N; ++i) {
        if (inst.nodes[i].isPlant) continue;
        if (!visited[i]) {
            unassigned.push_back(i);
        }
    }

    cout << "Granjas restantes antes de Fase 3: " << unassigned.size() << "\n";

    // Opcional: ordenar granjas restantes por litros (mayor a menor)
    sort(unassigned.begin(), unassigned.end(), [&](int a, int b) {
        return inst.nodes[a].litros > inst.nodes[b].litros;
    });

    // 4) Intentar asignar cada granja primero a rutas existentes
    for (int farmIdx : unassigned) {
        if (visited[farmIdx]) continue; // por si se asignó antes en el loop

        const Node &farm = inst.nodes[farmIdx];
        int qFarm = qualityIndex(farm.quality);
        double litrosFarm = farm.litros;

        bool assigned = false;

        // 4.1) Intentar en rutas existentes (camiones ya usados)
        for (size_t r = 0; r < solution.size(); ++r) {

            TruckRoute &rt = solution[r];
            RouteInfo &ri = routeInfo[r];

            double cap = inst.truckCap[rt.camion_id];

            // Capacidad: no nos pasamos del camión
            if (ri.volume + litrosFarm > cap + 1e-9) continue;

            if (ri.volume > 0.0) {
                // Ruta ya tiene leche ⇒ NO podemos poner granja de
                // calidad peor que la peor calidad actual de la ruta
                // (para no bajar la calidad de la mezcla)
                int w = ri.worstQ; // 0=A,1=B,2=C
                if (qFarm > w) {
                    continue; // esto degradaría la calidad de la ruta → NO
                }
                // Si qFarm <= w, la peor calidad se mantiene igual,
                // así que las cuotas no se rompen.
            } 
            // Caso ri.volume == 0: técnicamente no debería pasar porque
            // todas nuestras rutas tienen al menos una granja, pero si
            // ocurre, puedes permitir cualquier calidad.

            // Insertar la granja justo antes de la planta final
            rt.nodes.insert(rt.nodes.end() - 1, farmIdx);

            // Actualizar info de la ruta
            if (ri.volume <= 0.0) {
                ri.volume = litrosFarm;
                ri.worstQ = qFarm;
            } else {
                ri.volume += litrosFarm;
                // worstQ no cambia porque qFarm <= w (si volumen > 0)
            }

            visited[farmIdx] = true;
            assigned = true;

            cout << "  -> Granja " << farm.id
                 << " (Q=" << farm.quality << ", " << farm.litros
                 << "L) agregada a ruta de camion " << rt.camion_id
                 << " sin degradar calidad.\n";
            break;
        }

        if (assigned) continue;

        // 4.2) Si no se pudo meter en rutas existentes sin estropear calidad,
        //      intentamos crear una nueva ruta con un camión no usado
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

            cout << "  -> Granja " << farm.id
                 << " asignada a NUEVA ruta del camion " << chosenTruck
                 << " (cap=" << inst.truckCap[chosenTruck] << ")\n";

        } else {
            // 4.3) No hay camiones disponibles para esta granja
            cout << "❗ ADVERTENCIA: No hay camión disponible para granja "
                 << farm.id << " (Q=" << farm.quality
                 << ", " << farm.litros << "L)\n";
        }
    }

    // 5) Verificación final de granjas sin asignar
    int unassignedFinal = 0;
    for (int i = 0; i < N; ++i) {
        if (inst.nodes[i].isPlant) continue;
        if (!visited[i]) {
            unassignedFinal++;
            cout << "❗ Granja NO asignada al final: ID="
                 << inst.nodes[i].id
                 << " Calidad=" << inst.nodes[i].quality
                 << " Litros=" << inst.nodes[i].litros << "\n";
        }
    }

    cout << "\n========= RESUMEN GREEDY =========\n";
    cout << "Total de camiones usados: " << solution.size() << "\n";
    cout << "Granjas sin asignar: " << unassignedFinal << "\n";
    cout << "====================================\n";

    return solution;
}









// ================== EVALUACIÓN DE SOLUCIÓN ==================

struct EvalResult {
    double profit;                 // revenue - cost
    double cost;
    double revenue;
    vector<double> totalMilkPerQuality; 
    bool quotasSatisfied;          // true si todas las cuotas se cumplen
};


EvalResult evaluatesolucion(const Instance &inst,
                            const vector<TruckRoute> &sol) {
    EvalResult res;
    res.cost = 0.0;
    res.revenue = 0.0;
    res.totalMilkPerQuality.assign(inst.numQualities, 0.0);

    // 1. Costo + blending
    for (const auto &route : sol) {
        if (route.nodes.size() < 2) continue;

        for (size_t i = 0; i + 1 < route.nodes.size(); ++i) {
            int u = route.nodes[i];
            int v = route.nodes[i + 1];
            res.cost += inst.dist[u][v];
        }

        int worstQuality = -1; // 0=A,1=B,2=C
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

    // 2. Ingreso total
    for (int t = 0; t < inst.numQualities; ++t) {
        res.revenue += res.totalMilkPerQuality[t] * inst.calidad_inst[t];
    }

    // 3. Ganancia pura
    res.profit = res.revenue - res.cost;

    // 4. Chequeo de cuotas (factibilidad)
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

void printsolucion(const Instance &inst,
                   const vector<TruckRoute> &sol,
                   const EvalResult &ev, 
                   unsigned long long seed) {
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
        cout << path << " (Camino de Granjas)\n" 
             << routeCost  << " (Costo de Camino)\n";
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
    cerr << "¿Cuotas satisfechas?: " << (ev.quotasSatisfied ? "SI" : "NO") << "\n";
}

vector<TruckRoute> simulatedAnnealing(const Instance &inst,
                                      const vector<TruckRoute> &initial,
                                      EvalResult &bestEval,
                                      const string &logFilename)
{
    cout << "\n==============================\n";
    cout << " INICIANDO SIMULATED ANNEALING (solo factibles)\n";
    cout << "==============================\n\n";

    // Abrimos archivo de log de iteraciones (para 1.7 y 2.1)
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
    int maxIter = 50000;

    uniform_real_distribution<double> ur(0.0, 1.0);

    for (int iter = 0; iter < maxIter && T > 0.00001; ++iter) {

        bool debug = (iter < 5); // prints sólo primeras 5 iteraciones

        if (debug) {
            cout << "\n---- ITERACIÓN " << iter
                 << " | T=" << T
                 << " | Profit actual=" << currentEval.profit
                 << " ----\n";
        }

        vector<TruckRoute> neighbor;
        if (!generateNeighbor(inst, current, neighbor)) {
            if (debug) cout << "[WARN] No se pudo generar vecino\n";
            T *= alpha;
            continue;
        }

        EvalResult neighEval = evaluatesolucion(inst, neighbor);

        // Si NO es factible (no cumple cuotas), descartamos
        if (!neighEval.quotasSatisfied) {
            if (debug) {
                cout << "Vecino infactible, se DESCARTA (no cumple cuotas)\n";
            }
            flog << iter << "," << T << ","
                << currentEval.profit << "," << bestEvalLocal.profit << ","
                << neighEval.profit << "," << 0 << "\n";

            T *= alpha;
            continue;
        }


        double delta = neighEval.profit - currentEval.profit;

        if (debug) {
            cout << "[NEIGHBOR] Profit=" << neighEval.profit
                 << " | delta=" << delta << "\n";
        }

        bool accepted = false;
        if (delta >= 0) {
            accepted = true;
        } else {
            double p = exp(delta / T);
            double r = ur(rng);
            if (debug) cout << "   p=" << p << "  r=" << r << "\n";
            if (r < p) accepted = true;
        }

        if (debug) {
            if (accepted) cout << " -> Vecino ACEPTADO\n";
            else          cout << " -> Vecino RECHAZADO\n";
        }

        // Escribimos en el log de iteraciones
        flog << iter << "," << T << ","
             << currentEval.profit << "," << bestEvalLocal.profit << ","
             << neighEval.profit << "," << (accepted ? 1 : 0) << "\n";

        if (accepted) {
            current = neighbor;
            currentEval = neighEval;
        }

        // Mejor global por profit
        if (currentEval.profit > bestEvalLocal.profit) {
            best = current;
            bestEvalLocal = currentEval;

            if (debug) {
                cout << " *** NUEVA MEJOR FACTIBLE ***\n";
                cout << " Profit = " << bestEvalLocal.profit << "\n";
            }
        }

        T *= alpha;
    }

    cout << "\n==============================\n";
    cout << " FIN DE SA — SOLO FACTIBLES\n";
    cout << "==============================\n";
    cout << "Mejor profit encontrado: " << bestEvalLocal.profit << "\n";

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

    unsigned long long seed =
        chrono::high_resolution_clock::now().time_since_epoch().count();
    initRNG(seed);

    ifstream in("a33.txt");
    if (!in) {
        cerr << "ERROR: no se pudo abrir la instancia\n";
        return 1;
    }

    Instance inst = readInstance(in);

    // 1) Solución inicial (greedy)
    vector<TruckRoute> initialSol = greedyInitialsolucion(inst);
    EvalResult greedyEval = evaluatesolucion(inst, initialSol);

    // 2) Simulated Annealing (genera sa_iterations.csv)
    EvalResult saEval;
    vector<TruckRoute> bestSol =
        simulatedAnnealing(inst, initialSol, saEval, "sa_iterations.csv");

    // 3) summary.csv (tabla 1.5)
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

    // 4) routes_greedy.csv (para tabla 1.4)
    {
        ofstream fg("routes_greedy.csv");
        fg << "route_id,cost\n";
        for (size_t r = 0; r < initialSol.size(); ++r) {
            double c = routeCost(inst, initialSol[r]);
            fg << r << "," << c << "\n";
        }
    }

    // 5) routes_sa.csv (para tabla 1.4)
    {
        ofstream fs("routes_sa.csv");
        fs << "route_id,cost\n";
        for (size_t r = 0; r < bestSol.size(); ++r) {
            double c = routeCost(inst, bestSol[r]);
            fs << r << "," << c << "\n";
        }
    }

    // (Opcional) imprimir solución final en consola
    printsolucion(inst, bestSol, saEval, seed);

    return 0;
}

