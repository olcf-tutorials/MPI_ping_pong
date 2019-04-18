CUCOMP  = nvcc
CUFLAGS = -arch=sm_70

INCLUDES  = -I$(OMPI_DIR)/include
LIBRARIES = -L$(OMPI_DIR)/lib -lmpi_ibm

pp_cuda_aware: ping_pong_cuda_aware.o
	$(CUCOMP) $(CUFLAGS) $(LIBRARIES) ping_pong_cuda_aware.o -o pp_cuda_aware

ping_pong_cuda_aware.o: ping_pong_cuda_aware.cu
	$(CUCOMP) $(CUFLAGS) $(INCLUDES) -c ping_pong_cuda_aware.cu

.PHONY: clean

clean:
	rm -f pp_cuda_aware *.o
