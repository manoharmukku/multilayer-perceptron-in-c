# Defines
CC        = gcc
SRC       = src/*.c
OBJ       = read_csv.o forward_propagation.o back_propagation.o mlp_trainer.o main.o
INCL      = *.h 
EXE       = mlp
FLAGS     = -g -Wall

# Compile and Assemble C source files into object files
# obj/%.o: src/%.c $(INCL)
# 	$(CC) -c $(SRC)
obj/read_csv.o: src/read_csv.c include/read_csv.h
	$(CC) -c src/read_csv.c

# Generate the executable file
$(EXE): $(OBJ)
	$(CC) $(FLAGS) $(OBJ) $(SRCPATH)

# Objects depend on these libraries
$(OBJ): $(INCL)