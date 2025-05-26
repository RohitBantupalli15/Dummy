// HEA_density_Seq_vs_Par.cpp
#include "quest.h"
#include <cstdio>
#include <fstream>
#include <vector>
#include <omp.h>
#include <cstdlib>
#include <ctime>

using qreal = qreal;

const qreal FIXED_NOISE = 0.5; // Noise level applied to RX/RY angles

// Define sweep ranges
const std::vector<int> QUBIT_RANGE = {4, 6, 8, 10};     // Safe for density matrices
const std::vector<int> LAYER_RANGE = {10, 25, 50};       // Test different depths

const int NUM_RUNS = 5;      // Average over multiple runs

std::vector<qreal> generateRandomAngles(int numParams) {
    std::vector<qreal> angles(numParams);
    for (int i = 0; i < numParams; ++i)
        angles[i] = ((qreal)rand() / RAND_MAX) * M_PI;
    return angles;
}

void applyNoisyHEALayer(Qureg qureg, const std::vector<qreal>& angles, int layerIndex, qreal noise) {
    int numQubits = qureg.numQubits;
    int offset = layerIndex * numQubits * 2;

    for (int qubit = 0; qubit < numQubits; ++qubit) {
        qreal rx_angle = angles[offset + qubit * 2];
        qreal ry_angle = angles[offset + qubit * 2 + 1];

        if (noise > 0) {
            rx_angle *= (1 + noise * ((qreal)rand() / RAND_MAX - 0.5));
            ry_angle *= (1 + noise * ((qreal)rand() / RAND_MAX - 0.5));
        }

        applyRotateX(qureg, qubit, rx_angle);
        applyRotateY(qureg, qubit, ry_angle);

        mixDephasing(qureg, qubit, noise); // Apply physical dephasing
    }

    for (int qubit = 0; qubit < numQubits - 1; ++qubit) {
        applyControlledMultiQubitNot(qureg, qubit, new int[1]{qubit + 1}, 1);
    }
}

int main() {
    srand(time(0));
    initQuESTEnv();

    // Set a safe number of threads
    omp_set_num_threads(4);

    std::ofstream resultFile("benchmark_optimized.csv");
    if (!resultFile.is_open()) {
        printf("Error: Could not open output file\n");
        return -1;
    }

    resultFile << "Qubits,Layers,Mode,Time(s),AvgError\n";

    for (int NUM_QUBITS : QUBIT_RANGE) {
        for (int NUM_LAYERS : LAYER_RANGE) {

            int totalParams = NUM_QUBITS * 2 * NUM_LAYERS;

            // Sequential Execution
            double seq_start = omp_get_wtime();
            qreal seq_total_error = 0.0;

            for (int run = 0; run < NUM_RUNS; ++run) {
                std::vector<qreal> angles = generateRandomAngles(totalParams);

                Qureg noisyReg = createDensityQureg(NUM_QUBITS);
                Qureg cleanReg = createQureg(NUM_QUBITS);

                initZeroState(noisyReg);
                initZeroState(cleanReg);

                for (int layer = 0; layer < NUM_LAYERS; ++layer) {
                    applyNoisyHEALayer(noisyReg, angles, layer, FIXED_NOISE);
                    applyNoisyHEALayer(cleanReg, angles, layer, 0.0); // No noise
                }

                qreal fidelity = calcFidelity(noisyReg, cleanReg);
                seq_total_error += (1.0 - fidelity);

                destroyQureg(noisyReg);
                destroyQureg(cleanReg);
            }

            double seq_end = omp_get_wtime();
            double seq_avg_time = (seq_end - seq_start) / NUM_RUNS;
            qreal seq_avg_error = seq_total_error / NUM_RUNS;

            // Parallel Execution
            double par_start = omp_get_wtime();
            qreal par_total_error = 0.0;

            #pragma omp parallel reduction(+:par_total_error)
            {
                #pragma omp for schedule(dynamic, 2)
                for (int run = 0; run < NUM_RUNS; ++run) {
                    std::vector<qreal> angles = generateRandomAngles(totalParams);

                    Qureg noisyReg = createDensityQureg(NUM_QUBITS);
                    Qureg cleanReg = createQureg(NUM_QUBITS);

                    initZeroState(noisyReg);
                    initZeroState(cleanReg);

                    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
                        applyNoisyHEALayer(noisyReg, angles, layer, FIXED_NOISE);
                        applyNoisyHEALayer(cleanReg, angles, layer, 0.0); // No noise
                    }

                    qreal fidelity = calcFidelity(noisyReg, cleanReg);
                    par_total_error += (1.0 - fidelity);

                    destroyQureg(noisyReg);
                    destroyQureg(cleanReg);
                }
            }

            double par_end = omp_get_wtime();
            double par_avg_time = (par_end - par_start) / NUM_RUNS;
            qreal par_avg_error = par_total_error / NUM_RUNS;

            // Write results
            resultFile << NUM_QUBITS << "," << NUM_LAYERS << ",Sequential," 
                       << seq_avg_time << "," << seq_avg_error << "\n";
            resultFile << NUM_QUBITS << "," << NUM_LAYERS << ",Parallel," 
                       << par_avg_time << "," << par_avg_error << "\n";

            // Print progress
            printf("Q=%2d L=%3d | Seq=%.4fs Par=%.4fs Speedup=%.2fx Error=%.6f\n",
                   NUM_QUBITS, NUM_LAYERS,
                   seq_avg_time, par_avg_time, seq_avg_time / par_avg_time, par_avg_error);
        }
    }

    resultFile.close();
    printf("Benchmark complete. Optimized results saved to benchmark_optimized.csv\n");

    return 0;
}
