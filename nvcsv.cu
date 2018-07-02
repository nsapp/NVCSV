/**
	NVCSV : A CUDA-based CSV parser.
	File: nvcsv.cu
	Desc: Entry point for NVCSV.
	Author: Brandon Belna (bbelna)
**/

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <ctime>
#include "nvcsv.h"
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <limits.h>

#define OFFSET 100000	// offset for when the file has near max int lines

int parseCSV(std::string, int, int);	// foreword dec of the parseCSV func

// simple message function
void msg(std::string m) {
	std::cout << m << std::endl;
}

int main(int argc, char** argv) {
	// TODO: replace most std::cout statements with msg()
	std::cout << "NVCSV Version " <<  NVCSV_VERSION << std::endl;
	msg("Initializing CUDA context...");
	cudaFree(0);	// this is a shorthand which initializes the CUDA context
		    	// normally the CUDA context will initialize on the first CUDA command
	if (argc < 4) { // if our arguments are insufficient
		msg("Currently only supports grabbing data from a column of a CSV file.");
		msg("Usage: nvcsv [filename] [index] [field max length]");
		return 1;
	}
	/*
		TODO: allow [filename] to be replaced w/ a directory and have NVCSV parse the whole directory.
		You i.e. would loop parseCSV over the whole directory.

		There's a way (using stat I believe) to be able to tell if a string is a file or a directory.
	*/
	int fieldMaxLength = atoi(*(argv+3));	// field max length
	int index = atoi(*(argv+2));	// the column to grab
	if (index < 0) {
		msg("Error: index must be > 0. Aborting...");
		return -1;
	}
	if (fieldMaxLength <= 0) {
		msg("Error: max length must be > 0. Aborting...");
		return -1;
	}
	std::string fileName(*(argv+1));
	parseCSV(fileName, fieldMaxLength, index);	// parse the csv
	exit(0);	// return
}

int parseCSV(std::string _fileName, int _maxLength, int _parseIndex) {
	// variable init
	int maxLength = _maxLength;
	int parseIndex = _parseIndex;
	std::string fileName = _fileName;
	std::clock_t start1 = std::clock();

	// open file, determine size
	FILE* f = fopen(fileName.c_str(), "r" );
	std::cout << "Determining size of " << fileName << "..." << std::endl;
	fseek(f, 0, SEEK_END);	// seak towards end of file
	struct stat st;
	stat(fileName.c_str(), &st);	// get the file size
	long long fileSize = st.st_size;	// give that size a name
	std::cout << "File size is " << fileSize << "." << std::endl;
	fclose(f); // close the file

	struct stat sb;	// get another statr
	char *p;	// the pointer to our file data
	int fd;		// integer representing the file

	fd = open (fileName.c_str(), O_RDONLY);	// open

	// error handling
	if (fd == -1) {
		perror ("open");
		return 1;
	}

	if (fstat (fd, &sb) == -1) {
		perror ("fstat");
		return 1;
	}

	if (!S_ISREG (sb.st_mode)) {
		fprintf (stderr, "%s is not a file\n", "fileName");
		return 1;
	}	
	
	// create a shared memory map
	p = (char*)mmap (0, fileSize, PROT_READ, MAP_SHARED, fd, 0);

	if (p == MAP_FAILED) {
		perror ("mmap");
		return 1;
	}

	if (close (fd) == -1) {
		perror ("close");
		return 1;
	}
	
	
	/*
		TODO: if fileSize >= INT_MAX-OFFSET, we should set our fileSize variable
		to be that of INT_MAX-OFFSET, and then jump backwards until we reach a newline
		character. This code doesn't work in its current state, I believe.

		After you parse INT_MAX-OFFSET, you need to parse the next INT_MAX-OFFSET (or whatever's
		left of the file). Obviously, there's going to be a problem if you try to get the GPU
		to process the INT_MAX-OFFEST'th byte because the problem lies in the fact that throwing
		such a number into thrust causes the program to crash.

		Thus, you need to create a pointer whose first entry is that of INT_MAX-OFFSET; so that when
		the GPU references the INT_MAX-OFFEST'th byte it's really accessing the first entry of that pointer.
		This removes the problem that thrust can't directly handle files whose size are near INT_MAX.
	*/
	if (fileSize >= INT_MAX-OFFSET) {
		msg("File size >= INT_MAX. Splitting CSV file...");
		// jump to INT MAX. go backwards until new line.
		FILE* fd2 = fopen(fileName.c_str(), "r" );	
		fileSize = INT_MAX-OFFSET;
		fseek(fd2, fileSize, SEEK_SET);
		char t = fgetc(fd2);
		while (t != '\n') {
			fileSize--;
			fseek(fd2, fileSize, SEEK_SET);
			t = fgetc(fd2);
		}	
		fclose(fd2);
	}

	// GPU parsing code
	std::cout << "New file size: " << fileSize << std::endl;
	thrust::device_vector<char> dev(fileSize); // the vector representing the
						   // file's data on the GPU's memory
	msg("Copying file to GPU (this may take a while)...");
	thrust::copy(p, p+fileSize, dev.begin());
	msg("Successful copy to GPU.");
	msg("Counting lines...");
	thrust::device_vector<unsigned long long int> cnt(1);
	cnt[0] = thrust::count(thrust::device, dev.begin(), dev.end(), '\n'); // count the new lines in the file
	std::cout << "There are " << cnt[0] << " total lines in the file." << std::endl;


	/*
		This is ALL GPU processing code. For the most part you shouldn't have
		to touch this.
	*/
	// find all new lines
	thrust::device_vector<int> devPos(cnt[0]+1);
	devPos[0] = -1;
	
	msg("Creating device_vector of newlines...");
	thrust::copy_if(thrust::make_counting_iterator((unsigned int)0), thrust::make_counting_iterator((unsigned int)fileSize),
		dev.begin(), devPos.begin()+1, is_break());
	
	msg("Creating value arrays...");
	thrust::device_vector<char> vals(cnt[0]*25); // where we'll store our values
	thrust::fill(vals.begin(), vals.end(), ' '); // pad whole vector with zeros

	msg("Establishing destination pointer...");
	thrust::device_vector<char*> dest(1);
	dest[0] = thrust::raw_pointer_cast(vals.data()); // destination pointer

	msg("Establishing index vector...");
	thrust::device_vector<unsigned int> index(1); 
	index[0] = parseIndex;

	msg("Establishing max length of field...");
	thrust::device_vector<unsigned int> destLen(1); 
	destLen[0] = maxLength;
	
	thrust::device_vector<unsigned int> indexCount(1);
	indexCount[0] = 1;

	thrust::device_vector<char> seperator(1);
	seperator[0] = ',';

	msg("Parsing column...");
	thrust::counting_iterator<unsigned int> begin(0);
	parse_functor ff((const char*)thrust::raw_pointer_cast(dev.data()),(char**)thrust::raw_pointer_cast(dest.data()), thrust::raw_pointer_cast(index.data()),
		thrust::raw_pointer_cast(indexCount.data()), thrust::raw_pointer_cast(seperator.data()), thrust::raw_pointer_cast(devPos.data()), thrust::raw_pointer_cast(destLen.data()));
	thrust::for_each(begin, begin + cnt[0], ff);
	msg("Successful parse.");
	thrust::device_vector<double> d_float(cnt[0]);
	
	msg("gpu_atof on wanted data...");
	indexCount[0] = maxLength;
	gpu_atof atof_ff((const char*)thrust::raw_pointer_cast(vals.data()),(double*)thrust::raw_pointer_cast(d_float.data()),
			thrust::raw_pointer_cast(indexCount.data()));
	thrust::for_each(begin, begin + cnt[0], atof_ff);
	msg("Successful gpu_atof.");

	msg("Here are the first 10 entries of your desired column:");
	std::cout.precision(10);
	for(int i = 0; i < 10; i++) {
		std::cout << d_float[i] << std::endl;
	}
	msg("Cleaning...");
}
