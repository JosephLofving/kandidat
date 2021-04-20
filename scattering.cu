#include "scattering.h"


/**
	Sets up a complex vector needed to solve the T matrix equation.
	@param k:	Quadrature points
	@param w:	Weights for quadrature points
	@param k0:	On-shell-point
	@return		G0 vector
*/

//const cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);


__device__
cuDoubleComplex operator+(cuDoubleComplex A, cuDoubleComplex B) {
	cuDoubleComplex result = make_cuDoubleComplex(cuCreal(A)+cuCreal(B), cuCimag(A)+cuCimag(B));
	return result;
}

__device__
cuDoubleComplex operator-(cuDoubleComplex A, cuDoubleComplex B) {
	cuDoubleComplex result = make_cuDoubleComplex(cuCreal(A) - cuCreal(B), cuCimag(A) - cuCimag(B));
	return result;
}

__device__
cuDoubleComplex operator-(double a, cuDoubleComplex A) {
	cuDoubleComplex result = cuCsub(make_cuDoubleComplex(a, 0), A);
	return result;
}

__device__
cuDoubleComplex operator-(cuDoubleComplex A, double a) {
	cuDoubleComplex result = cuCsub(A, make_cuDoubleComplex(a, 0));
	return result;
}

__device__
cuDoubleComplex operator*(double scalar, cuDoubleComplex A) {
	cuDoubleComplex result = make_cuDoubleComplex(scalar * cuCreal(A), scalar * cuCimag(A));
	return result;
}

__device__
cuDoubleComplex operator*(cuDoubleComplex A, double scalar) {
	return scalar * A;
}

__device__
cuDoubleComplex operator*(cuDoubleComplex A, cuDoubleComplex B) {
	cuDoubleComplex realProd = cuCreal(A) * B;
	cuDoubleComplex imagProd = cuCimag(A) * B;
	cuDoubleComplex result = cuCadd(realProd, imagProd);
	return result;
}

__device__
cuDoubleComplex operator/(cuDoubleComplex A, cuDoubleComplex B) {
	return cuCdiv(A, B);
}

__device__
cuDoubleComplex operator/(cuDoubleComplex A, double a) {
	return cuCdiv(A, make_cuDoubleComplex(a, 0));
}

__device__
cuDoubleComplex operator/(double a, cuDoubleComplex A) {
	return cuCdiv(make_cuDoubleComplex(a, 0), A);
}

__device__
cuDoubleComplex logCudaComplex(cuDoubleComplex argument) {
	double x = cuCreal(argument);
	double y = cuCimag(argument);
	double real = logf(sqrtf(x * x + y * y));
	double imag = atan2f(y, x);
	cuDoubleComplex result = make_cuDoubleComplex(real, imag);
	return result;
}

__device__
double signCuda(double argument) {
	if (argument > 0) return 1;
	else if (argument == 0) return 0;
	else return -1;
}

__device__
cuDoubleComplex sqrtCudaComplex(cuDoubleComplex argument) {
	double x = cuCreal(argument);
	double y = cuCimag(argument);
	double real = sqrtf((sqrtf(x * x + y * y) + x) / 2);
	double imag = signCuda(y) * sqrtf((sqrtf(x * x + y * y) - x) / 2);
	return make_cuDoubleComplex(real, imag);
}

__device__
cuDoubleComplex atanCudaComplex(cuDoubleComplex argument) {
	cuDoubleComplex numerator = cuCadd(make_cuDoubleComplex(1, 0), cuCmul(make_cuDoubleComplex(0, 1), argument));
	cuDoubleComplex denominator = cuCsub(make_cuDoubleComplex(1, 0), cuCmul(make_cuDoubleComplex(0, 1), argument));
	cuDoubleComplex logOfStuff = logCudaComplex(cuCdiv(numerator, denominator));
	cuDoubleComplex result = cuCmul(make_cuDoubleComplex(0, -0.5), logOfStuff);
	return result;
}

__device__
cuDoubleComplex asinCudaComplex(cuDoubleComplex argument) {
	const cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);
	return I * logCudaComplex(sqrtCudaComplex(1 - argument * argument) - I * argument);
}

__device__
cuDoubleComplex expCudaComplex(cuDoubleComplex argument) {
	double x = cuCreal(argument);
	double y = cuCimag(argument);
	cuDoubleComplex trig = make_cuDoubleComplex(cosf(y), sinf(y));
	cuDoubleComplex result = make_cuDoubleComplex(expf(x), 0) * trig;
	return result;
}

__device__
cuDoubleComplex sinCudaComplex(cuDoubleComplex argument) {
	const cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);
	return (expCudaComplex(I * argument) - expCudaComplex(-1.0 * I * argument)) / 2;
}

__device__
cuDoubleComplex tanCudaComplex(cuDoubleComplex argument) {
	const cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);
	cuDoubleComplex numerator = I * (expCudaComplex(-1.0 * I * argument) - expCudaComplex(I * argument));
	cuDoubleComplex denominator = expCudaComplex(-1.0 * I * argument) + expCudaComplex(I * argument);
	return numerator / denominator;
}




__global__
void setupG0Vector(cuDoubleComplex* G0,
	double* k,
	double* w,
	double* k0,
	int quadratureN,
	int matLength,
	int TLabLength,
	double mu,
	bool coupled) {

	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int width = blockIdx.z * blockDim.z + threadIdx.z;

	double twoMu = (2.0 * mu);
	double twoOverPi = (2.0 / constants::pi);
	double sum = 0;

	if (column < quadratureN && width < TLabLength) {
		G0[column + width * matLength] = make_cuDoubleComplex(twoOverPi * twoMu * k[column] * k[column] * w[column] / (k0[width] * k0[width] - k[column] * k[column]), 0);
		sum += w[column] / (k0[width] * k0[width] - k[column] * k[column]);

		/* If coupled, append G0 to itself to facilitate calculations.
		 * This means the second half of G0 is a copy of the first. */
		if (coupled) {
			G0[quadratureN + 1 + column + width * matLength] = G0[column + width * matLength];
		}

		/* Assign the last element of D */
		G0[quadratureN + width * matLength] = make_cuDoubleComplex(-twoOverPi * twoMu * k0[width] * k0[width] * sum, -twoMu * k0[width]);
		if (coupled) {
			G0[2 * (quadratureN + 1) - 1 + width * matLength] = G0[quadratureN + width * matLength];
		}
	}
}

/**
	Multiplies the potential matrix elements with the G0 vector elements.

	@param channel: Scattering channel
	@param key:		Channel name
	@param V:		Potential matrix
	@param k:		Quadrature points
	@param w:		Weights for quadrature points
	@param k0:		The on-shell-point
	@return			VG kernel
*/
__global__
void setupVGKernel(cuDoubleComplex* VG,
	cuDoubleComplex* V,
	cuDoubleComplex* G0,
	cuDoubleComplex* F,
	double* k,
	double* w,
	double* k0,
	int quadratureN,
	int matLength,
	int TLabLength,
	double mu,
	bool coupled) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int width = blockIdx.z * blockDim.z + threadIdx.z;

	if (row < matLength && column < matLength && width < TLabLength) {
		VG[row + column * matLength + width * matLength * matLength] = cuCmul(V[row + column * matLength + width * matLength * matLength], G0[column + width * matLength]);

		if (row == column) {
			F[row + row * matLength + width * matLength * matLength] = cuCadd(make_cuDoubleComplex(1, 0), cuCmul(make_cuDoubleComplex(-1, 0), VG[row + row * matLength + width * matLength * matLength])); // Diagonal element
		}
		else {
			F[row + column * matLength + width * matLength * matLength] = cuCmul(make_cuDoubleComplex(-1, 0), VG[row + column * matLength + width * matLength * matLength]);
		}

	}

}



	//for (int row = 0; row < matLength; row++) {
	//	for (int column = 0; column < matLength; column++) {
	//		/* Create VG by using VG[i,j] = V[i,j] * G[j] */
	//		VG[row + column * matLength] = cuCmul(V[row + column * matLength], G0[column]);

	//		/* At the same time, create F = delta_ij - VG_ij for computeTMatrix*/
	//		if (row != column) {
	//			F[row + column * matLength] = cuCmul(make_cuDoubleComplex(-1, 0), VG[row + column * matLength]);
	//		}
	//	}
	//	F[row + row * matLength] = cuCadd(make_cuDoubleComplex(1, 0), cuCmul(make_cuDoubleComplex(-1, 0), VG[row + row * matLength])); // Diagonal element
	//}
//}




/**
	Computes the T-matrix from the equation [F][T] = [V]

	@param channel: Scattering channel
	@param key:		Channel name
	@param V:		Potential matrix
	@param k:		Quadrature points
	@param w:		Weights for quadrature points
	@param k0:		On-shell-point
	@return			T matrix
*/


/* TODO: Explain theory for this. */
__device__
void blattToStapp(cuDoubleComplex* phases, 
				  cuDoubleComplex* deltaMinusBB,
				  cuDoubleComplex* deltaPlusBB,
				  cuDoubleComplex* twoEpsilonJBB,
				  int TLabLength) {

	int width = blockIdx.z * blockDim.z + threadIdx.z;
	cuDoubleComplex* twoEpsilonJ = new cuDoubleComplex[TLabLength];
	if (width < TLabLength) {
		twoEpsilonJ[width] = asinCudaComplex(sinCudaComplex(twoEpsilonJBB[width]) * sinCudaComplex(deltaMinusBB[width] - deltaPlusBB[width]));

		phases[0 + width*3] = 0.5 * (deltaPlusBB[width] + deltaMinusBB[width] + asinCudaComplex(tanCudaComplex(twoEpsilonJ[width]) / tanCudaComplex(twoEpsilonJBB[width]))) * constants::rad2deg;
		phases[1 + width*3] = 0.5 * (deltaPlusBB[width] + deltaMinusBB[width] - asinCudaComplex(tanCudaComplex(twoEpsilonJ[width]) / tanCudaComplex(twoEpsilonJBB[width]))) * constants::rad2deg;
		phases[2 + width*3] = 0.5 * twoEpsilonJ[width] * constants::rad2deg;
	}
}


/**
	Computes the phase shift for a given channel and T matrix.

	@param channel: Scattering channel
	@param key:		Channel name
	@param k0:		On-shell-point
	@param T:		T matrix
	@return			Complex phase shifts
*/
__global__
void computePhaseShifts(cuDoubleComplex* phases,
	cuDoubleComplex* T,
	double* k0,
	int quadratureN,
	double mu,
	bool coupled,
	int TLabLength,
	int matLength) {

	int width = blockIdx.z * blockDim.z + threadIdx.z;

	double* rhoT = new double[TLabLength];
	cuDoubleComplex* T11 = new cuDoubleComplex[TLabLength];
	cuDoubleComplex* T12 = new cuDoubleComplex[TLabLength];
	cuDoubleComplex* T22 = new cuDoubleComplex[TLabLength];
	cuDoubleComplex* twoEpsilonJBB = new cuDoubleComplex[TLabLength];
	cuDoubleComplex* deltaPlusBB = new cuDoubleComplex[TLabLength];
	cuDoubleComplex* deltaMinusBB = new cuDoubleComplex[TLabLength];

	if (width < TLabLength) {

		rhoT[width] = 2 * mu * k0[width]; // Equation (2.27) in the theory
		const cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);

		// TODO: Explain theory for the phase shift for the coupled state
		if (coupled) {
			T11[width] = T[(quadratureN)+(quadratureN * quadratureN) + width * matLength * matLength]; //row + column * size
			T12[width] = T[(2 * quadratureN + 1) + (quadratureN * quadratureN) + width * matLength * matLength];
			T22[width] = T[(2 * quadratureN + 1) + (quadratureN * (2 * quadratureN + 1)) + width * matLength * matLength];

			//Blatt - Biedenharn(BB) convention
			twoEpsilonJBB[width] = atanCudaComplex(cuCdiv(cuCmul(make_cuDoubleComplex(2.0, 0), T12[width]), cuCsub(T11[width], T22[width])));
			deltaPlusBB[width] = -0.5 * I * logCudaComplex(1.0 - I * rhoT[width] * (T11[width] + T22[width]) + I * rhoT[width] * (2.0 * T12[width]) / sinCudaComplex(twoEpsilonJBB[width]));
			deltaMinusBB[width] = -0.5 * I * logCudaComplex(1.0 - I * rhoT[width] * (T11[width] + T22[width]) - I * rhoT[width] * (2.0 * T12[width]) / sinCudaComplex(twoEpsilonJBB[width]));

		}
		/* The uncoupled case completely follows equation (2.26). */
		else {
			double T0 = cuCreal(T[(quadratureN)+(quadratureN * quadratureN) + width * matLength ]); //Farligt, detta element kanske inte �r helt reellt. Dock var koden d�lig f�rut is�fall.
			cuDoubleComplex argument = make_cuDoubleComplex(1, -2.0 * rhoT[width] * T0);
			cuDoubleComplex swappedLog = make_cuDoubleComplex(cuCimag(logCudaComplex(argument)), cuCreal(logCudaComplex(argument)));
			cuDoubleComplex delta = cuCmul(make_cuDoubleComplex(-0.5 * constants::rad2deg, 0), swappedLog);
			phases[0] = delta;
		}
	}

	if (coupled) {
		blattToStapp(phases, deltaMinusBB, deltaPlusBB, twoEpsilonJBB, TLabLength);
	}
}