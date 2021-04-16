int main() {
	cudaMalloc((void**)&V_dev, N * N * sizeof(double));
	cudaMemcpy(G0_dev, G0, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(V_host, VG_dev, N * N * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(G0_dev);
}