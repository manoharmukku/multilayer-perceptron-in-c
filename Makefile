# Defines
CC         = gcc
OBJ_DIR    = ./obj
SRC_DIR    = ./src
INCL_DIR   = ./include
OBJECTS    = $(addprefix $(OBJ_DIR)/, read_csv.o forward_propagation.o back_propagation.o mlp_trainer.o mlp_classifier.o main.o)
INCLUDES   = $(addprefix $(INCL_DIR)/, read_csv.h forward_propagation.h back_propagation.h mlp_trainer.h mlp_classifier.h main.h parameters.h)
CFLAGS     = -g -Wall
EXECUTABLE = MLP

# Generate the executable file
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(EXECUTABLE) -I $(INCL_DIR)

# Compile and Assemble C source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(INCLUDES)
	$(CC) $(CFLAGS) -I $(INCL_DIR) -c $< -o $@

# Clean the generated executable file and object files
clean:
	rm -f $(OBJECTS)
	rm -f $(EXECUTABLE)