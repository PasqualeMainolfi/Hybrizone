FLAG = -Wextra -Wall -O3 -march=native
LIBS = -lhdf5 -lhdf5_cpp -lfftw3 -lportaudio -lsndfile
INCLUDE = -I/opt/homebrew/opt/hdf5/include -I/opt/homebrew/opt/fftw/include -I/opt/homebrew/opt/portaudio/include -I/opt/homebrew/opt/libsndfile/include -I/opt/homebrew/opt/xsimd/include
LIBRARY = -L/opt/homebrew/opt/hdf5/lib -L/opt/homebrew/opt/fftw/lib -L/opt/homebrew/opt/portaudio/lib -L/opt/homebrew/opt/libsndfile/lib

SOURCE = test.cpp src/htools.cpp
BUILD = builded.o

.PHONY = all
all = build run

build:
	clang++ -std=c++2b $(SOURCE) -o $(BUILD) $(FLAG) $(LIBS) $(INCLUDE) $(LIBRARY)

build_debug:
	clang++ -DDEBUG -g -std=c++2b $(SOURCE) -o $(BUILD) $(FLAG) $(LIBS) $(INCLUDE) $(LIBRARY)

run:
	./$(BUILD)
