CUCOMP  = nvcc
CUFLAGS = -arch=sm_70

INCLUDES  = -I$(OMPI_DIR)/include
LIBRARIES = -L$(OMPI_DIR)/lib -lmpi_ibm

pp_cuda_staged: ping_pong_cuda_staged.o
	$(CUCOMP) $(CUFLAGS) $(LIBRARIES) ping_pong_cuda_staged.o -o pp_cuda_staged

ping_pong_cuda_staged.o: ping_pong_cuda_staged.cu
	$(CUCOMP) $(CUFLAGS) $(INCLUDES) -c ping_pong_cuda_staged.cu

.PHONY: clean

clean:
	rm -f pp_cuda_staged *.o
