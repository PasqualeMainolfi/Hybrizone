CC = g++
FLAG = -Wextra -Wall -O3 -march=native
LIBS = -lhdf5 -lhdf5_cpp -lfftw3 -lportaudio -lsndfile

INCLUDE = -I/opt/homebrew/opt/hdf5/include \
          -I/opt/homebrew/opt/fftw/include \
          -I/opt/homebrew/opt/portaudio/include \
          -I/opt/homebrew/opt/libsndfile/include

LIBRARY = -L/opt/homebrew/opt/hdf5/lib \
          -L/opt/homebrew/opt/fftw/lib \
          -L/opt/homebrew/opt/portaudio/lib \
          -L/opt/homebrew/opt/libsndfile/lib

BUILD_FOLDER = build/
SOURCE = test.cpp src/htools.cpp
OBJS = $(BUILD_FOLDER)$(SOURCE:.cpp=.o)
OBJS_DEBUG= $(BUILD_FOLDER)$(SOURCE:.cpp=.o_debug)

BUILD = $(BUILD_FOLDER)builded
BUILD_DEBUG = $(BUILD_FOLDER)builded_debug


.PHONY: all
all: $(BUILD)

$(BUILD): $(OBJS)
	$(CC) -std=c++2b $(FLAG) $(OBJS) $(LIBRARY) $(LIBS) -o $@

$(BUILD_DEBUG): $(OBJS_DEBUG)
$(CC) -std=c++2b $(FLAG) $(OBJS) $(LIBRARY) $(LIBS) -o $(BUILD_DEBUG)

$(BUILD_FOLDER)%.o: %.cpp
	$(CC) -std=c++2b $(FLAG) $(INCLUDE) -c $< -o $@

run: $(BUILD)
	./$(BUILD)

run_debug: $(BUILD_DEBUG)
	./$(BUILD_DEBUG)

clean:
	find $(BUILD_FOLDER) -mindepth 1 -delete
