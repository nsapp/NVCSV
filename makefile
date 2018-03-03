CC=nvcc
CFLAGS=-O3 -Xptxas -v -use_fast_math -arch=sm_30
SOURCES=nvcsv.cu
BINNAME=nvcsv

$(BINNAME): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $(BINNAME)

clean:
	rm $(BINNAME)
