# NVCSV
**Command-line based CSV parser using CUDA.**

Forked from antonmks' [nvParse.](https://github.com/antonmks/nvParse) Developed to rapidly parse huge CSV files
harnessing the power of Nvidia's CUDA processing. Currently only supports Linux, but potential Windows support in the future.

**Compiling:** Compile with Nvidia's CUDA compiler:
```bash
nvcc -g -G nvcsv.cu -o nvcsv
```