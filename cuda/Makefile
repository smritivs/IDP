LIBTORCH_PATH = /home/bala/libtorch

run:
	rm -rf build
	mkdir build
	cd build; cmake -DCMAKE_PREFIX_PATH=$(LIBTORCH_PATH) ..; make
	mv ./build/ex ./
	nvprof ./ex > ./bench/cudadump.txt
	cd bench; python3 testkanprof.py

clean:
	rm -rf build *.o *.out
