CC ?= gcc
ARCH_FLAG = -arch $(shell /Users/openclaw/Code/Connect4RL/.venv/bin/python3 -c "import platform; print(platform.machine())")
CFLAGS = -O3 -shared -fPIC -Wall -Wextra -std=c11 $(ARCH_FLAG)
LDFLAGS = -undefined dynamic_lookup
RAYLIB_FLAGS = $(shell pkg-config --cflags --libs raylib 2>/dev/null || echo "-lraylib")
PYTHON_FLAGS = $(shell /Users/openclaw/Code/Connect4RL/.venv/bin/python3 -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")
NUMPY_FLAGS = $(shell /Users/openclaw/Code/Connect4RL/.venv/bin/python3 -c "import numpy; print('-I' + numpy.get_include())")
TARGET = binding$(shell /Users/openclaw/Code/Connect4RL/.venv/bin/python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

.PHONY: all clean test

all: $(TARGET)

$(TARGET): binding.c othello.h negamax.h render.h
	$(CC) $(CFLAGS) $(LDFLAGS) $(PYTHON_FLAGS) $(NUMPY_FLAGS) $(RAYLIB_FLAGS) -o $@ binding.c

clean:
	rm -f $(TARGET) *.o

test:
	/Users/openclaw/Code/Connect4RL/.venv/bin/python3 -m pytest tests/ -v
