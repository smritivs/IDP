import subprocess
import sys
from consts import libtorch_path

if __name__ == "__main__":
	kernel_name = sys.argv[1]
	subprocess.run(["python3 ./generate_test.py"],shell=True)
	subprocess.run(["mkdir build"],shell=True)
	subprocess.run([f"cd build; cmake -DCMAKE_PREFIX_PATH={libtorch_path} -DFILE={kernel_name} ../..; make;"],shell=True)
	subprocess.run(["mv ./build/ex ./;rm -rf build"],shell=True)
	subprocess.run(["nvprof ./ex > ./cudadump.txt"],shell=True)
	subprocess.run([f"python3 ./test{kernel_name}prof.py"],shell=True)
