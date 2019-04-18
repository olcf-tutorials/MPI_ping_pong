MPICOMP = mpicc

pp: ping_pong.o
	$(MPICOMP) ping_pong.o -o pp

ping_pong.o: ping_pong.c
	$(MPICOMP) -c ping_pong.c

.PHONY: clean

clean:
	rm -f pp *.o
