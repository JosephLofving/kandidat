#include "mesh.h"
#include "scattering.h"
#include "potential.h"
#include "computeTMatrix.h"
#include <fstream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuComplex.h>
#include <chrono>

int solveLS();