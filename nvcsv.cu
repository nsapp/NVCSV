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

int main(int argc, char** argv) {
	std::cout << "NVCSV Version " <<  NVCSV_VERSION << std::endl;
	if (argc == 1) {
		std::cout << "Usage: nvcsv [filename]" << std::endl;
		std::cout << "As of now, currently only runs through file listed to see how long it takes to process." << std::endl;
		return 1;
	}
	std::string fileName(*(argv+1)); 
	std::clock_t start1 = std::clock();
	FILE* f = fopen(fileName.c_str(), "r" );
	if (f == NULL) {
		std::cout << "failed to open " <<  fileName << ". Does file exist?" << std::endl;
		return 1;
	}
	fseek(f, 0, SEEK_END);
	long fileSize = ftell(f);
	thrust::device_vector<char> dev(fileSize);
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

	thrust::copy(p, p+fileSize, dev.begin());

	int cnt = thrust::count(dev.begin(), dev.end(), '\n');
	std::cout << "There are " << cnt << " total lines in a file" << std::endl;

	thrust::device_vector<int> dev_pos(cnt+1);
	dev_pos[0] = -1;

	thrust::copy_if(thrust::make_counting_iterator((unsigned int)0), thrust::make_counting_iterator((unsigned int)fileSize),
					dev.begin(), dev_pos.begin()+1, is_break());

	thrust::device_vector<char> res(cnt*20);
	thrust::fill(res.begin(), res.end(), 0);

	thrust::device_vector<char*> dest(1);
	dest[0] = thrust::raw_pointer_cast(res.data());

	thrust::device_vector<unsigned int> ind(1); //fields positions
	ind[0] = 5;

	thrust::device_vector<unsigned int> dest_len(1); //fields max lengths
	dest_len[0] = 20;

	thrust::device_vector<unsigned int> ind_cnt(1); //fields count
	ind_cnt[0] = 10;

	thrust::device_vector<char> sep(1);
	sep[0] = ',';

	thrust::counting_iterator<unsigned int> begin(0);
	parse_functor ff((const char*)thrust::raw_pointer_cast(dev.data()),(char**)thrust::raw_pointer_cast(dest.data()), thrust::raw_pointer_cast(ind.data()),
					 thrust::raw_pointer_cast(ind_cnt.data()), thrust::raw_pointer_cast(sep.data()), thrust::raw_pointer_cast(dev_pos.data()), thrust::raw_pointer_cast(dest_len.data()));
	thrust::for_each(begin, begin + cnt, ff); // now dev_pos vector contains the indexes of new line characters

	std::cout<< "time0 " <<  ( ( std::clock() - start1 ) / (double)CLOCKS_PER_SEC ) << '\n';
	
	thrust::device_vector<double> d_float(cnt);

	gpu_atof atof_ff((const char*)thrust::raw_pointer_cast(res.data()),(double*)thrust::raw_pointer_cast(d_float.data()),
					 thrust::raw_pointer_cast(ind_cnt.data()));
	thrust::for_each(begin, begin + cnt, atof_ff);

	std::cout.precision(10);
	for(int i = 0; i < 10; i++)
		std::cout << d_float[i] << std::endl;

	return 0;

}
