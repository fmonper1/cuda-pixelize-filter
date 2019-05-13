#define _CRT_SECURE_NO_WARNINGS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define FAILURE 0
#define SUCCESS !FAILURE

#define USER_NAME "aca15fm"		//replace with your user name

void print_help();
int process_command_line(int argc, char *argv[]);
int process_output_file(int tile_size);
int process_to_mosaic(int tile_size);
int process_ppm_file(FILE *file, int total_size);
int process_ppm_header(FILE *file);
int do_cpu(FILE *file, int tilse_size);
int do_cuda_processing(int height, int width, int tile_size);
void transform_1D_to_2D(unsigned char* out_array_r, unsigned char* out_array_g, unsigned char* out_array_b, int width, int height);

int IS_BINARY_MODE = 0;

int average_r = 0, average_g = 0, average_b = 0;
__device__ int gpu_average_r, gpu_average_g, gpu_average_b;

double start_timer, timer;

typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;
MODE execution_mode;

unsigned int tile_size = 0;
char *file_name, *output_file;

typedef struct {
	unsigned char red, green, blue;
} PPMPixel;

typedef struct {
	unsigned int *red, *green, *blue;
} device_PPMPixel;

typedef struct {
	int height, width, maxval;
	PPMPixel *data;
	device_PPMPixel *device_data;
} PPMImage;

PPMImage *image;
PPMPixel **image_array; // Stores all the R,G,B values in an array the same size as the original image
PPMPixel **tile_array; // Stores the average RGB values of the Mosaic produced from the image
unsigned char *image_array_r, *image_array_g, *image_array_b;
unsigned char *out_array_r, *out_array_g, *out_array_b;

/* --------------------------------------------------
This functions takes in a ppm file and reads in the
header of the file. It stores the values of the header
in a PPMImage struct and sets the variable IS_BINARY_MODE
to 1 if the magic number is P6
-------------------------------------------------- */
int process_ppm_header(FILE *file) {
	char buff[16];

	//Open the file

	if (file == NULL) {
		printf("File is null");
		return FAILURE;
	}

	image = (PPMImage *)malloc(sizeof(PPMImage));
	if (!image) {
		free(image);
		printf("Error allocating memory for struct");
		return FAILURE;
	}

	// Scan magic number
	if (!fgets(buff, sizeof(buff), file)) {
		//if (fscanf(file, "%s", &image->magic_number) != 1) {
		printf("Magic number error");
		return FAILURE;
	}

	if (buff[0] == 'P' && buff[1] == '6')
		IS_BINARY_MODE = 1;

	// find comments and consume up to newline
	int comment = getc(file);
	while (comment == '#') {
		while (getc(file) != '\n');
		comment = getc(file); // consume the newline
	}

	printf(buff);
	ungetc(comment, file);
	//read size info

	if (fscanf(file, "%d %d", &image->width, &image->height) != 2) {
		return FAILURE;
	}

	if (fscanf(file, "%d", &image->maxval) != 1) {
		printf("Couldnt find colour maxval");
		return FAILURE;
	}

	while (getc(file) != '\n');

	return SUCCESS;
}

/* --------------------------------------------------
This functions takes in a file and the products of the
images width and height. It allocates memory for storing
the image as an array of PPMPixels and analyzes the file
to produce such array.
-------------------------------------------------- */
int process_ppm_file(FILE *file, int total_size) {
	// Check if one of the arguments passed in is null and return FAILURE
	if (file == NULL || total_size == NULL)
		return FAILURE;

	//printf("Processing ppm file \n");

	//int total_r = 0, total_g = 0, total_b = 0;
	int height = *(&image->height);
	int width = *(&image->width);

	// Allocate memory for 2D array of PPMPixel struct
	image_array = (PPMPixel **)malloc(width * height * sizeof(PPMPixel *));
	for (int i = 0; i < *(&image->width); i++) {
		image_array[i] = (PPMPixel *)malloc(width * sizeof(PPMPixel));
	}

	// Scan file and parse R,G,B values into PPMPixel struct
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			// Allocate memory for a certain pixel and parse the R,G,B values
			PPMPixel *pixel = (PPMPixel *)malloc(sizeof(PPMPixel));
			if (pixel == NULL) {
				printf("pixel structure is null");
				return FAILURE;
			}
			// READ PIXEL VALUES, CHECK FOR BINARY MODE OR PLAIN TEXT MODE
			if (IS_BINARY_MODE == 1) {
				pixel->red = fgetc(file);
				pixel->green = fgetc(file);
				pixel->blue = fgetc(file);
			}
			else {
				fscanf(file, "%u %u %u\t", &pixel->red, &pixel->green, &pixel->blue);
			}
			// Store pointer to the pixel in the array
			image_array[i][j] = *(pixel);
		}
	}

	return SUCCESS;
}

int convert_struct_into_arrays() {
	int height = *(&image->height);
	int width = *(&image->width);

	image_array_r = (unsigned char *)malloc(height * width * sizeof(unsigned char));
	image_array_g = (unsigned char *)malloc(height * width * sizeof(unsigned char));
	image_array_b = (unsigned char *)malloc(height * width * sizeof(unsigned char));
	//image_array_1d = (unsigned char *)malloc(height * width * sizeof(PPMPixel));
	// Allocate memory for a temporary 2D array to average colour of pixels

	int count = 0;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			image_array_r[count] = image_array[i][j].red;
			image_array_g[count] = image_array[i][j].green;
			image_array_b[count] = image_array[i][j].blue;
			//printf("r %d, g %d, b %d \n", image_array[i][j].red, image_array[i][j].green, image_array[i][j].blue);
			//printf("r %d, g %d, b %d \n", image_array_r[count], image_array_g[count],image_array_b[count]);
			count++;
		}
	}
	return SUCCESS;
}

int get_average_color_values(FILE *file) {
	int total_r = 0, total_g = 0, total_b = 0;
	int height = *(&image->height);
	int width = *(&image->width);
	int total_size = width * height;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			// Add the R,G,B values to a counter
			total_r += image_array[i][j].red;
			total_g += image_array[i][j].green;
			total_b += image_array[i][j].blue;
		}
	}
	average_r = total_r / total_size;
	average_g = total_g / total_size;
	average_b = total_b / total_size;

	return SUCCESS;
}

/* --------------------------------------------------
This functions takes in the tile size and produces a
copy of the image_array of size (width/tile_size and
height/tile_size) with its pixels being a average for
the pixels in the original array
-------------------------------------------------- */
int process_to_mosaic(int tile_size) {
	const int height = *(&image->height) / tile_size;
	const int width = *(&image->width) / tile_size;
	//printf("Initial width %d height %d, mosaic width %d height %d \n", *(&image->width), *(&image->height), width, height);

	// Allocate memory for a temporary 2D array to average colour of pixels
	tile_array = (PPMPixel **)malloc(width * height * sizeof(PPMPixel *));
	for (int i = 0; i < *(&image->width); i++)
		tile_array[i] = (PPMPixel *)malloc(width * sizeof(PPMPixel));

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			// Multiply i and j by the tile size to skip to the desired pixel
			int new_i = i*tile_size;
			int new_j = j*tile_size;
			int total_r = 0, total_g = 0, total_b = 0;
			for (int i2 = 0; i2 < tile_size; i2++) {
				for (int j2 = 0; j2 < tile_size; j2++) {
					total_r += image_array[new_i + i2][new_j + j2].red;
					total_g += image_array[new_i + i2][new_j + j2].green;
					total_b += image_array[new_i + i2][new_j + j2].blue;
					//printf("%d %d %d \n", image_array[new_i + i2][new_j + j2].red, image_array[new_i + i2][new_j + j2].green, image_array[new_i + i2][new_j + j2].blue);
				}
			}

			// Create a pixel struc and store it in the tile array
			PPMPixel *pixel = (PPMPixel *)malloc(sizeof(PPMPixel));
			pixel->red = total_r / (tile_size * tile_size);
			pixel->green = total_g / (tile_size * tile_size);
			pixel->blue = total_b / (tile_size * tile_size);
			tile_array[i][j] = *pixel;
			//printf("Average for pixels: %d %d %d \n", tile_array[i][j].red, tile_array[i][j].green, tile_array[i][j].blue);

			//printf("------- \n");
		}
	}
	return SUCCESS;
}

/* --------------------------------------------------
This functions takes in the tile size value and
loops over the tile_array to get the colours for
each pixel. Then it creates a ppm file with the dimensions
of the original image.
-------------------------------------------------- */
int process_output_file(int tile_size) {
	FILE *out_file;
	out_file = fopen(output_file, "wb");

	/*-----------------------------
	Output the file header
	-----------------------------*/
	if (IS_BINARY_MODE == 1) {
		fprintf(out_file, "%s\n", "P6");
	}
	else {
		fprintf(out_file, "%s\n", "P3");
	}
	fprintf(out_file, "%d\n", *(&image->width));
	fprintf(out_file, "%d\n", *(&image->height));
	fprintf(out_file, "%d\n", *(&image->maxval));

	// Calculate te size of the mosaic image
	const int height = *(&image->height) / tile_size;
	const int width = *(&image->width) / tile_size;
	// printf("Initial width %d height %d, mosaic width %d height %d \n", *(&image->width), *(&image->height), width, height);

	if (tile_array == NULL) {
		printf("tile_array is NULL");
		return FAILURE;
	}

	for (int i = 0; i < height; i++) {
		for (int i2 = 0; i2 < tile_size; i2++) {
			for (int j = 0; j < width; j++) {
				for (int j2 = 0; j2 < tile_size; j2++) {
					// Multiply i and j by the tile size to skip to the desired pixel
					if (&tile_array[i][j] == NULL)
						return FAILURE;

					if (IS_BINARY_MODE == 1) {
						fwrite(&tile_array[i][j], sizeof(PPMPixel), 1, out_file);
					}
					else {
						fprintf(out_file, "%d %d %d\t", tile_array[i][j].red, tile_array[i][j].green, tile_array[i][j].blue);
					}
					//printf("%d %d %d ", tile_array[i][j].red, tile_array[i][j].green, tile_array[i][j].blue);
				}
			}
			if (IS_BINARY_MODE == 0) {
				fprintf(out_file, "\n");
			}
			//printf("\n");
		}
	}
	fclose(out_file);

	return SUCCESS;
}

int cuda_process_output_file(int tile_size) {
	FILE *out_file;
	out_file = fopen(output_file, "wb");

	/*-----------------------------
	Output the file header
	-----------------------------*/
	if (IS_BINARY_MODE == 1) {
		fprintf(out_file, "%s\n", "P6");
	}
	else {
		fprintf(out_file, "%s\n", "P3");
	}
	fprintf(out_file, "%d\n", *(&image->width));
	fprintf(out_file, "%d\n", *(&image->height));
	fprintf(out_file, "%d\n", *(&image->maxval));

	// Calculate te size of the mosaic image
	const int height = *(&image->height);
	const int width = *(&image->width);
	// printf("Initial width %d height %d, mosaic width %d height %d \n", *(&image->width), *(&image->height), width, height);

	if (tile_array == NULL) {
		printf("tile_array is NULL");
		return FAILURE;
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			// Multiply i and j by the tile size to skip to the desired pixel
			if (&tile_array[i][j] == NULL)
				return FAILURE;

			if (IS_BINARY_MODE == 1) {
				fwrite(&tile_array[i][j], sizeof(PPMPixel), 1, out_file);
			}
			else {
				fprintf(out_file, "%d %d %d\t", tile_array[i][j].red, tile_array[i][j].green, tile_array[i][j].blue);
			}
			//printf("%d,%d:%d %d %d ", i, j, tile_array[i][j].red, tile_array[i][j].green, tile_array[i][j].blue);
				
		}
		if (IS_BINARY_MODE == 0) {
			fprintf(out_file, "\n");
		}
		//printf("\n");
	}
	fclose(out_file);

	return SUCCESS;
}

/* --------------------------------------------------
This functions takes in a file and the products of the
images width and height. It allocates memory for storing
the image as an array of PPMPixels and analyzes the file
to produce such array.
-------------------------------------------------- */
int openmp_process_ppm_file(FILE* file, int total_size) {
	// Check if one of the arguments passed in is null and return FAILURE
	if (file == 0 || total_size == NULL)
		return FAILURE;

	//printf("Processing ppm file \n");

	//int total_r = 0, total_g = 0, total_b = 0;
	int height = *(&image->height);
	int width = *(&image->width);

	// Allocate memory for 2D array of PPMPixel struct
	image_array = (PPMPixel **)malloc(width * height * sizeof(PPMPixel *));
	int i, j;
	for (i = 0; i < *(&image->width); i++)
		image_array[i] = (PPMPixel *)malloc(width * sizeof(PPMPixel));

	// Scan file and parse R,G,B values into PPMPixel struct

	for (i = 0; i < width; i++) {
		for (j = 0; j < height; j++) {
			// Allocate memory for a certain pixel and parse the R,G,B values
			PPMPixel *pixel = (PPMPixel *)malloc(sizeof(PPMPixel));
			if (pixel == NULL) {
				printf("pixel structure is null");
				return FAILURE;
			}
			// READ PIXEL VALUES, CHECK FOR BINARY MODE OR PLAIN TEXT MODE
			if (IS_BINARY_MODE == 1) {
				pixel->red = fgetc(file);
				pixel->green = fgetc(file);
				pixel->blue = fgetc(file);
			}
			else {
				fscanf(file, "%u %u %u\t", &pixel->red, &pixel->green, &pixel->blue);
			}
			// Store pointer to the pixel in the array
			image_array[i][j] = *(pixel);
		}
	}

	return SUCCESS;
}

int openmp_get_average_color_values(FILE *file) {
	int total_r = 0, total_g = 0, total_b = 0;
	int height = *(&image->height);
	int width = *(&image->width);
	int total_size = width * height;
	int i, j;

#pragma omp parallel for private(i, j) shared(image_array, width, height) reduction(+: total_r, total_g, total_b) schedule(static)
	// #pragma omp for nowait
	for (i = 0; i < width; i++) {
		for (j = 0; j < height; j++) {
			// Add the R,G,B values to a counter
			total_r += image_array[i][j].red;
			total_g += image_array[i][j].green;
			total_b += image_array[i][j].blue;
		}
	}
	average_r = total_r / total_size;
	average_g = total_g / total_size;
	average_b = total_b / total_size;
	// printf("Average R,G,B values: %d %d %d \n", total_r, total_g, total_b);

	return SUCCESS;
}

/* --------------------------------------------------
This functions takes in the tile size and produces a
copy of the image_array of size (width/tile_size and
height/tile_size) with its pixels being a average for
the pixels in the original array
-------------------------------------------------- */
int openmp_process_to_mosaic(int tile_size) {
	const int height = (int)(*(&image->height) / tile_size);
	const int width = (int)(*(&image->width) / tile_size);
	//printf("Initial width %d height %d, mosaic width %d height %d \n", *(&image->width), *(&image->height), width, height);

	// Allocate memory for a temporary 2D array to average colour of pixels
	tile_array = (PPMPixel **)malloc(width * height * sizeof(PPMPixel *));
	for (int i = 0; i < *(&image->width); i++)
		tile_array[i] = (PPMPixel *)malloc(width * sizeof(PPMPixel));

	int i, j, new_i, new_j, i2, j2;
	int total_r = 0, total_g = 0, total_b = 0;
#pragma omp parallel private(i, j, new_i, new_j, i2, j2) shared(image_array, tile_size)
	{
		//#pragma omp for reduction(+: total_r, total_g, total_b) schedule(static)
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				// Multiply i and j by the tile size to skip to the desired pixel
				new_i = i*tile_size;
				new_j = j*tile_size;
				total_r = 0, total_g = 0, total_b = 0;
				for (i2 = 0; i2 < tile_size; i2++) {
					for (j2 = 0; j2 < tile_size; j2++) {
						if ((new_i + i2) > image->height || new_j + j2 > image->width) {

						}
						else {
							total_r += image_array[new_i + i2][new_j + j2].red;
							total_g += image_array[new_i + i2][new_j + j2].green;
							total_b += image_array[new_i + i2][new_j + j2].blue;
						}
						//printf("%d %d %d \n", image_array[new_i + i2][new_j + j2].red, image_array[new_i + i2][new_j + j2].green, image_array[new_i + i2][new_j + j2].blue);
					}
				}

				// Create a pixel struc and store it in the tile array
				tile_array[i][j].red = total_r / (tile_size * tile_size);
				tile_array[i][j].green = total_g / (tile_size * tile_size);
				tile_array[i][j].blue = total_b / (tile_size * tile_size);
				//printf("Average for pixels: %d %d %d \n", tile_array[i][j].red, tile_array[i][j].green, tile_array[i][j].blue);

				//printf("------- \n");
			}
		}
	}
	return SUCCESS;
}

/* --------------------------------------------------
//This functions takes in the tile size value and
//loops over the tile_array to get the colours for
//each pixel. Then it creates a ppm file with the dimensions
//of the original image.
-------------------------------------------------- */
int openmp_process_output_file(int tile_size) {
	FILE *out_file;
	out_file = fopen(output_file, "wb");

	/*-----------------------------
	Output the file header
	-----------------------------*/
	if (IS_BINARY_MODE == 1) {
		fprintf(out_file, "%s\n", "P6");
	}
	else {
		fprintf(out_file, "%s\n", "P3");
	}
	fprintf(out_file, "%d\n", *(&image->width));
	fprintf(out_file, "%d\n", *(&image->height));
	fprintf(out_file, "%d\n", *(&image->maxval));

	// Calculate te size of the mosaic image
	const int height = *(&image->height) / tile_size;
	const int width = *(&image->width) / tile_size;
	// printf("Initial width %d height %d, mosaic width %d height %d \n", *(&image->width), *(&image->height), width, height);

	if (tile_array == NULL) {
		printf("tile_array is NULL");
		return FAILURE;
	}

	int i, i2, j, j2;
	for (i = 0; i < height; i++) {
		for (i2 = 0; i2 < tile_size; i2++) {
			for (j = 0; j < width; j++) {
				for (j2 = 0; j2 < tile_size; j2++) {
					// Multiply i and j by the tile size to skip to the desired pixel
					if (&tile_array[i][j] == NULL)
						return FAILURE;

					if (IS_BINARY_MODE == 1) {
						fwrite(&tile_array[i][j], sizeof(PPMPixel), 1, out_file);
					}
					else {
						fprintf(out_file, "%d %d %d\t", tile_array[i][j].red, tile_array[i][j].green, tile_array[i][j].blue);
					}
					//printf("%d %d %d ", tile_array[i][j].red, tile_array[i][j].green, tile_array[i][j].blue);
				}
			}
			if (IS_BINARY_MODE == 0) {
				fprintf(out_file, "\n");
			}
			//printf("\n");
		}
	}
	fclose(out_file);

	return SUCCESS;
}

void print_help() {
	printf("mosaic_%s C M -i input_file -o output_file [options]\n", USER_NAME);

	printf("where:\n");
	printf("\tC              Is the mosaic cell size which should be any positive\n"
		"\t               power of 2 number \n");
	printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
		"\t               ALL. The mode specifies which version of the simulation\n"
		"\t               code should execute. ALL should execute each mode in\n"
		"\t               turn.\n");
	printf("\t-i input_file  Specifies an input image file\n");
	printf("\t-o output_file Specifies an output image file which will be used\n"
		"\t               to write the mosaic image\n");
	printf("[options]:\n");
	printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
		"\t               PPM_PLAIN_TEXT\n ");

	getchar();
}

int process_command_line(int argc, char *argv[]) {
	if (argc < 7) {
		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}

	//first argument is always the executable name

	//read in the non optional command line arguments
	tile_size = (unsigned int)atoi(argv[1]);
	int temp_c = tile_size;
	while ((temp_c % 2) == 0 && temp_c > 1)
		temp_c = temp_c / 2;
	if (temp_c != 1) {
		printf("C has to be equal to 2^n where N is a positive number");
		return(FAILURE);
	}

	//TODO: read in the mode
	if (strcmp(argv[2], "CPU") == 0) {
		execution_mode = CPU;
	}
	if (strcmp(argv[2], "OPENMP") == 0) {
		execution_mode = OPENMP;
	}
	if (strcmp(argv[2], "CUDA") == 0) {
		execution_mode = CUDA;
	}
	if (strcmp(argv[2], "ALL") == 0) {
		execution_mode = ALL;
	}

	//TODO: read in the input image name
	file_name = argv[4];
	//printf(argv[4]);

	//TODO: read in the output image name
	output_file = argv[6];
	//printf(argv[6]);

	return SUCCESS;
}

void transform_1D_to_2D(unsigned char* out_array_r, unsigned char* out_array_g, unsigned char* out_array_b, int width, int height) {
	int theCount = 0;

	tile_array = (PPMPixel **)malloc((*(&image->width)) * (*(&image->height)) * sizeof(PPMPixel *));
	for (int i = 0; i < (*(&image->width)); i++)
		tile_array[i] = (PPMPixel *)malloc((*(&image->width)) * sizeof(PPMPixel));

	// transfer 1D array to 2D
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			//printf("i %d, j %d, r %d, g %d, b %d \n", i, j, out_array_r[theCount], out_array_g[theCount], out_array_b[theCount]);
			tile_array[i][j].red = out_array_r[theCount];
			tile_array[i][j].green = out_array_g[theCount];
			tile_array[i][j].blue = out_array_b[theCount];
			theCount += 1;
		}
	}
}

__device__ unsigned long long gpu_total_r = 0, gpu_total_g = 0, gpu_total_b = 0;

__global__ void get_image_averages(uchar3* gpu_image, int width, int height, int c) {
	// each thread loads one element from global to shared mem
	//unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	atomicAdd(&gpu_total_r, (unsigned long long) gpu_image[i].x);
	atomicAdd(&gpu_total_g, (unsigned long long) gpu_image[i].y);
	atomicAdd(&gpu_total_b, (unsigned long long) gpu_image[i].z);
}



__global__ void cuda_image_pixelize(uchar3* gpu_image, int width, int height, int tilesize) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int output_offset = x + y * blockDim.x * gridDim.x;
	float totalsize = (float)(tilesize*tilesize);
	if ((output_offset == 0 || output_offset % tilesize == 0) && (y == 0 || y % tilesize == 0)) {
		int avg_r = 0, avg_g = 0, avg_b = 0;
		for (int i = 0; i < tilesize; i++) {
			for (int j = 0; j < tilesize; j++) {
				int index = output_offset + i + (j*height);
				uchar3 pixel = gpu_image[index];
				avg_r += pixel.x;
				avg_g += pixel.y;
				avg_b += pixel.z;
			}
		}
		__syncthreads();
		for (int i = 0; i < tilesize; i++) {
			for (int j = 0; j < tilesize; j++) {
				int out_index = output_offset + i + (j * height);
				//printf("out: %d \n",out_index);
				gpu_image[out_index].x = (unsigned char)(avg_r / totalsize);
				gpu_image[out_index].y = (unsigned char)(avg_g / totalsize);
				gpu_image[out_index].z = (unsigned char)(avg_b / totalsize);

			}
		}
	}
}

int do_cuda_processing(int height, int width, int tile_size) {
	cudaEvent_t start, stop;
	float mseconds;
	uchar3 *cpu_pixel;
	uchar3 *gpu_pixel;
	//cuda layout and execution
	dim3 blocksPerGrid2(width / 16, height / 16);
	dim3 threadsPerBlock2(16, 16);
	unsigned long long total_r, total_b, total_g;

	// Allocate memory for the arrays of R, G, B values that are going to be produced
	out_array_r = (unsigned char *)malloc((width)*(height) * sizeof(unsigned char));
	out_array_g = (unsigned char *)malloc((width)*(height) * sizeof(unsigned char));
	out_array_b = (unsigned char *)malloc((width)*(height) * sizeof(unsigned char));

	// Optimization: User uchar3 to keep rgb values coalesced in memory. 
	// We have to copy them from the three arrays into one
	cpu_pixel = (uchar3*)malloc(sizeof(uchar3)*(width)*(height));

	// Copy the r,g,b arrays into the uchar3
	for (int i = 0; i < width*height; i++) {
		cpu_pixel[i].x = image_array_r[i];
		cpu_pixel[i].y = image_array_g[i];
		cpu_pixel[i].z = image_array_b[i];
	}

	printf("width %d, height %d \n", width, height);
	// create timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// starting timing here
	cudaEventRecord(start, 0);

	// Allocate memory in GPU
	cudaMalloc((void**)&gpu_pixel, sizeof(uchar3)*(width)*(height));

	// Copy data into GPU memory
	cudaMemcpy(gpu_pixel, cpu_pixel, sizeof(uchar3)*(width)*(height), cudaMemcpyHostToDevice);

	printf("getting image average...\n");
	get_image_averages <<<blocksPerGrid2, threadsPerBlock2 >>>(gpu_pixel, width, height, tile_size);

	printf("pixelating image...\n");
	cuda_image_pixelize <<<blocksPerGrid2, threadsPerBlock2 >>>(gpu_pixel, width, height, tile_size);
	cudaDeviceSynchronize();

	printf("done pixelating...\n");

	//cudaMemcpyToSymbol(&gpu_total_r, &total_r, sizeof(int));

	cudaMemcpyFromSymbol(&total_r, gpu_total_r, sizeof(unsigned long long));
	cudaMemcpyFromSymbol(&total_g, gpu_total_g, sizeof(unsigned long long));
	cudaMemcpyFromSymbol(&total_b, gpu_total_b, sizeof(unsigned long long));
	printf("done memcopy1\n");

	// Copy data back from gpu
	cudaMemcpy(cpu_pixel, gpu_pixel, sizeof(uchar3)*(width)*(height), cudaMemcpyDeviceToHost);

	// end timing here
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&mseconds, start, stop);

	printf("CUDA mode execution time took %d s and %d ms\n", (int)mseconds / 1000, (int)mseconds % 1000);

	printf("cpu_average_r total is %d \n", total_r / (width*height));
	printf("cpu_average_g total is %d \n", total_g / (width*height));
	printf("cpu_average_b total is %d \n", total_b / (width*height));

	// Improvement
	// Copy data back to the initial format so that the output function still works
	for (int i = 0; i < width*height; i++) {
		out_array_r[i] = cpu_pixel[i].x;
		out_array_g[i] = cpu_pixel[i].y;
		out_array_b[i] = cpu_pixel[i].z;
	}

	// Free GPU memory
	cudaFree(gpu_pixel);
	free(cpu_pixel);

	// cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceReset();
	return SUCCESS;
}

int do_cpu(FILE *file, int tilse_size) {
	//TODO: starting timing here
	clock_t begin, end;
	float mseconds;
	begin = clock();

	if (get_average_color_values(file) == FAILURE) {
		printf("There was a problem averaging the colours");
	}

	if (process_to_mosaic(tile_size) == FAILURE) {
		printf("There was a problem processing the output file");
	}

	// Output the average colour value for the image
	printf("CPU Average image colour red = %d, green = %d, blue = %d \n", average_r, average_g, average_b);

	//TODO: end timing here

	end = clock();
	mseconds = (end - begin) * 1000 / (float)CLOCKS_PER_SEC;
	printf("CPU mode execution time took %d s and %d ms\n", (int)mseconds / 1000, (int)mseconds % 1000);
	// starting timing here

	if (process_output_file(tile_size) == FAILURE) {
		printf("There was a problem processing the output file");
	}

	return SUCCESS;
}

int main(int argc, char *argv[]) {

	if (process_command_line(argc, argv) == FAILURE)
		return 1;

	FILE *file;
	file = fopen(file_name, "rb");

	if (process_ppm_header(file) == FAILURE) {
		printf("There was a problem processing the file header");
		exit(FAILURE);
	}

	int total_size = *(&image->width) * *(&image->height);
	if (process_ppm_file(file, total_size) != 1) {
		printf("There was a problem reading the pixels in the ppm file");
		exit(FAILURE);
	}

	if (tile_size > (unsigned int) *(&image->width) || tile_size >(unsigned int) *(&image->height)) {
		printf("You cant enter a mosaic size bigger than the actual image");
		exit(FAILURE);
	}

	//TODO: execute the mosaic filter based on the mode
	switch (execution_mode) {
		case (CPU): {
			do_cpu(file, tile_size);
			//TODO: starting timing here
			/*start_timer = omp_get_wtime();
			if (get_average_color_values(file) == FAILURE) {
				printf("There was a problem averaging the colours");
			}

			if (process_to_mosaic(tile_size) == FAILURE) {
				printf("There was a problem processing the output file");
			}

			// Output the average colour value for the image
			printf("CPU Average image colour red = %d, green = %d, blue = %d \n", average_r, average_g, average_b);

			//TODO: end timing here
			timer = omp_get_wtime() - start_timer;
			int seconds = (int)timer;
			double milisecs = (timer - seconds) * 1000;
			printf("CPU mode execution time took %d s and %f ms\n", seconds, milisecs);

			if (process_output_file(tile_size) == FAILURE) {
				printf("There was a problem processing the output file");
			}*/
			break;
		}
		case (OPENMP): {
			//TODO: starting timing here
			start_timer = omp_get_wtime();

			if (openmp_get_average_color_values(file) == FAILURE) {

			}
			if (openmp_process_to_mosaic(tile_size) == FAILURE) {
				printf("There was a problem processing the output file");
			}

			// Output the average colour value for the image
			printf("OPENMP Average image colour red = %d, green = %d, blue = %d \n", average_r, average_g, average_b);

			//TODO: end timing here
			timer = omp_get_wtime() - start_timer;
			int seconds = (int)timer;
			double milisecs = (timer - seconds) * 1000;
			printf("OPENMP mode execution time took %d s and %fms\n", seconds, milisecs);
			if (openmp_process_output_file(tile_size) == FAILURE) {
				printf("There was a problem processing the output file");
			}
			break;
		}
		case (CUDA): {
			printf("------------------------------- \n");
			printf("      Launching CUDA Mode \n");
			printf("------------------------------- \n");
			convert_struct_into_arrays();

			do_cuda_processing(*(&image->width), *(&image->height), tile_size);

			printf("transforming 1d to 2d \n");
			transform_1D_to_2D(out_array_r, out_array_g, out_array_b, *(&image->width), *(&image->height));

			printf("processing output file \n");
			if (cuda_process_output_file(tile_size) == FAILURE) {
				printf("There was a problem processing the output file");
			}
			break;
		}
		case (ALL): {
			//TODO: starting timing here
			printf("------------------------------- \n");
			printf("      Launching CPU Mode \n");
			printf("------------------------------- \n");
			start_timer = omp_get_wtime();
			if (get_average_color_values(file) == FAILURE) {
				printf("There was a problem averaging the colours");
			}

			if (process_to_mosaic(tile_size) == FAILURE) {
				printf("There was a problem processing the output file");
			}

			// Output the average colour value for the image
			printf("CPU Average image colour red = %d, green = %d, blue = %d \n", average_r, average_g, average_b);

			//TODO: end timing here
			timer = omp_get_wtime() - start_timer;
			int seconds = (int)timer;
			double milisecs = (timer - seconds) * 1000;
			printf("CPU mode execution time took %d s and %f ms\n", seconds, milisecs);

			if (process_output_file(tile_size) == FAILURE) {
				printf("There was a problem processing the output file");
			}

			printf("------------------------------- \n");
			printf("    Launching OPENMP Mode \n");
			printf("------------------------------- \n");
			//TODO: starting timing here
			start_timer = omp_get_wtime();

			if (openmp_get_average_color_values(file) == FAILURE) {

			}
			if (openmp_process_to_mosaic(tile_size) == FAILURE) {
				printf("There was a problem processing the output file");
			}

			// Output the average colour value for the image
			printf("OPENMP Average image colour red = %d, green = %d, blue = %d \n", average_r, average_g, average_b);

			//TODO: end timing here
			timer = omp_get_wtime() - start_timer;
			printf("OPENMP mode execution time took %d s and %fms\n", seconds, milisecs);
			if (openmp_process_output_file(tile_size) == FAILURE) {
				printf("There was a problem processing the output file");
			}

			printf("------------------------------- \n");
			printf("      Launching CUDA Mode \n");
			printf("------------------------------- \n");
			convert_struct_into_arrays();

			do_cuda_processing(*(&image->width), *(&image->height), tile_size);

			printf("transforming 1d to 2d \n");
			transform_1D_to_2D(out_array_r, out_array_g, out_array_b, *(&image->width), *(&image->height));

			printf("processing output file \n");
			if (cuda_process_output_file(tile_size) == FAILURE) {
				printf("There was a problem processing the output file");
			}
			break;
		}
	}

	free(image_array);

	free(tile_array);
	getchar();

	//save the output image file (from last executed mode)
	return 0;
}