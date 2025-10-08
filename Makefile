CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -Werror
TARGET = neuralNetwork
SRC = neuralNetwork.c
LIBS = -lm

all: $(TARGET)

$(TARGET): $(SRC) $(wildcard *.h)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
