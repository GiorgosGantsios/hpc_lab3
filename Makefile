NVCC = nvcc
CFLAGS = -g -O4

TARGET = Convolution2D

SRC = Convolution2D.cu

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

LOG_FILE = experiment_12.log

# Clean up
clean:
	rm -f $(TARGET)



experiment_12: $(TARGET)
	@echo "Running $(TARGET) 12 times with filter radius $(FILTER_RADIUS) and image size $(IMAGE_SIZE), writing output to $(LOG_FILE)"
	@rm -f $(LOG_FILE)
	@for i in {1..12}; do \
    	echo "Run $$i:" >> $(LOG_FILE); \
    	./$(TARGET) $(FILTER_RADIUS) $(IMAGE_SIZE) >> $(LOG_FILE); \
    	echo "" >> $(LOG_FILE); \
    done
