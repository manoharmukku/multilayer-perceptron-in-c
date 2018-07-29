# Defines
CC        = gcc
OBJ       = read_csv.o forward_propagation.o back_propagation.o mlp_trainer.o main.o
FLAGS     = -g -Wall
EXE       = mlp

# Compile and Assemble C source files into object files
%.o: %.c
	$(CC) -c *.c

# Generate the executable file and remove the used object files
$(EXE): $(OBJ)
	$(CC) $(FLAGS) $(OBJ) -o $(EXE)
	rm -f $(OBJ)

# Clean the generated executable file
clean:
	rm -f mlp