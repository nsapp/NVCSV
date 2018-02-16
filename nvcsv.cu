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

int main(int argc, char** argv) {
	std::cout << "NVCSV Version " <<  NVCSV_VERSION << std::endl;
	if (argc == 1) {
		std::cout << "Usage: nvcsv [filename] [index count]" << std::endl;
		std::cout << "Currently built for CINF401." << std::endl;
		return 1;
	}
	int iC = atoi(*(argv+2));
	std::string fileName(*(argv+1)); 
	std::clock_t start1 = std::clock();
	FILE* f = fopen(fileName.c_str(), "r" );
	if (f == NULL) {
		std::cout << "failed to open " <<  fileName << ". Does file exist?" << std::endl;
		return 1;
	}
	std::cout << "Determining size of " << fileName << "..." << std::endl;
	fseek(f, 0, SEEK_END);
	struct stat st;
	stat(fileName.c_str(), &st);	
	long long fileSize = st.st_size; 
	thrust::device_vector<char> dev(fileSize); // the vector representing the
						   // file's data on the GPU's memory
	std::cout << "File size is " << fileSize << "." << std::endl;
	fclose(f);
	
	struct stat sb;
	char *p;
	int fd;

	fd = open (fileName.c_str(), O_RDONLY);
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

	p = (char*)mmap (0, fileSize, PROT_READ, MAP_SHARED, fd, 0);

	if (p == MAP_FAILED) {
		perror ("mmap");
		return 1;
	}

	if (close (fd) == -1) {
		perror ("close");
		return 1;
	}


	std::cout << "Copying file to GPU..." << std::endl;
	thrust::copy(p, p+fileSize, dev.begin());

	std::cout << "Counting lines..." << std::endl;
	long long cnt = thrust::count(dev.begin(), dev.end(), '\n'); // count the new lines in the file
	std::cout << "There are " << cnt << " total lines in the file." << std::endl;

	// find all new lines
	thrust::device_vector<int> devPos(cnt+1);
	devPos[0] = -1;
	
	std::cout << "Creating device_vector of newlines..." << std::endl;
	thrust::copy_if(thrust::make_counting_iterator((unsigned int)0), thrust::make_counting_iterator((unsigned int)fileSize),
		dev.begin(), devPos.begin()+1, is_break());
	
	std::cout << "Creating value arrays..." << std::endl;
	thrust::device_vector<char> vals(cnt*25); // where we'll store our values
	thrust::fill(vals.begin(), vals.end(), ' '); // pad whole vector with zeros

	thrust::device_vector<char*> dest(1);
	dest[0] = thrust::raw_pointer_cast(vals.data()); // destination pointer

	thrust::device_vector<unsigned int> index(1); 
	index[0] = 4; // we want the fifth column

	thrust::device_vector<unsigned int> destLen(1); 
	destLen[0] = 25; // max of 25 in length
	
	thrust::device_vector<unsigned int> indexCount(1);
	indexCount[0] = 1;

	thrust::device_vector<char> seperator(1);
	seperator[0] = ',';

	std::cout << "Parsing CSV file on GPU (this may take a while)..." << std::endl;
	thrust::counting_iterator<unsigned int> begin(0);
	parse_functor ff((const char*)thrust::raw_pointer_cast(dev.data()),(char**)thrust::raw_pointer_cast(dest.data()), thrust::raw_pointer_cast(index.data()),
		thrust::raw_pointer_cast(indexCount.data()), thrust::raw_pointer_cast(seperator.data()), thrust::raw_pointer_cast(devPos.data()), thrust::raw_pointer_cast(destLen.data()));
	thrust::for_each(begin, begin+cnt, ff);

	thrust::device_vector<double> d_float(cnt);
	
	std::cout << "gpu_atof on wanted data..." << std::endl;
	indexCount[0] = iC;
	gpu_atof atof_ff((const char*)thrust::raw_pointer_cast(vals.data()),(double*)thrust::raw_pointer_cast(d_float.data()),
			thrust::raw_pointer_cast(indexCount.data()));
	thrust::for_each(begin, begin + cnt, atof_ff);

	std::cout.precision(10);
	for(int i = 0; i < 10; i++) {
		std::cout << d_float[i] << std::endl;
	}
	std::cout << "Success. Exiting..." << std::endl;
}
