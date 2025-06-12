run:
	nvcc ./src/$(file).cu -I./inc
	./a.out

prof:
	nvcc ./src/$(file).cu -I./inc
	nvprof ./a.out

clean:
	rm -rf *.o *.out
