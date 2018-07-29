# Defines
CC        = gcc
OBJ       = read_csv.o forward_propagation.o back_propagation.o mlp_trainer.o main.o
FLAGS     = -g -Wall
EXE       = mlp

# Compile and Assemble C source files into object files
%.o: %.c
	$(CC) -c *.c

# Generate the executable file
$(EXE): $(OBJ)
	$(CC) $(FLAGS) $(OBJ) -o $(EXE)

# Clean the generated object files
clean:
	rm -f $(OBJ)
clean_exe:
	rm -f mlp