import subprocess
from consts import libtorch_path

if __name__ == "__main__":
	subprocess.run(["python3 ./generate_test.py"],shell=True)
	subprocess.run(["mkdir build"],shell=True)
	subprocess.run([f"cd build; cmake -DCMAKE_PREFIX_PATH={libtorch_path} ../..; make;"],shell=True)
	subprocess.run(["mv ./build/ex ./;rm -rf build"],shell=True)
	subprocess.run(["nvprof ./ex > ./cudadump.txt"],shell=True)
	subprocess.run(["python3 ./testkanprof.py"],shell=True)
