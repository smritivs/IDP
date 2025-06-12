run:
	nvcc ./src/$(file).cu -I./inc
	./a.out

warn:
	nvcc ./src/$(file).cu -I./inc

prof:
	nvcc ./src/$(file).cu -I./inc
	nvprof ./a.out

clean:
	rm -rf *.o *.out
