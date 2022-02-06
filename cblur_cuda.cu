#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CUDA_CHECK_RETURN(value)                                                                     \
    {                                                                                                \
        cudaError_t err = value;                                                                     \
        if (err != cudaSuccess)                                                                      \
        {                                                                                            \
            printf("Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
            exit(1);                                                                                 \
        }                                                                                            \
    }

struct BITMAP_header
{
    char name[2];
    unsigned int size;
    int garbage;
    unsigned int image_offset;
};

struct DIB_header
{
    unsigned int header_size;
    unsigned int width;
    unsigned int height;
    unsigned short int colorplanes;
    unsigned short int bitsperpixel;
    unsigned int compression;
    unsigned int image_size;
    unsigned int temp[4];
};

struct RGB
{
    unsigned char blue;
    unsigned char green;
    unsigned char red;
};

struct Image
{
    int height;
    int width;
    struct RGB **rgb;
};

// Función que permite leer los datos de los pixeles de una imagen .bmp.
// Se usan estructuras creadas como el RGB que nos permite guardar los tres valores
// de color por pixel.
struct Image readImage(FILE *fp, int height, int width, int thread_count)
{
    struct Image pic;
    int i;
    pic.rgb = (struct RGB **)malloc(height * sizeof(void *));
    pic.height = height;
    pic.width = width;

    for (i = height - 1; i >= 0; i--)
    {
        pic.rgb[i] = (struct RGB *)malloc(width * sizeof(struct RGB));
        fread(pic.rgb[i], width, sizeof(struct RGB), fp);
    }

    return pic;
};

// Función que libera la memoria usada al usar la imagen.
void freeImage(struct Image pic)
{
    int i;
    for (i = pic.height - 1; i >= 0; i--)
    {
        free(pic.rgb[i]);
    }
    free(pic.rgb);
}

__global__ void processImage(unsigned char *out, unsigned char *in, int *width, int *height, int *FILTER_SIZE)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int sum = 0;
    int count = 0;

    if (x >= *width || y >= *height)
        return;
    for (int i = -(*FILTER_SIZE); i <= (*FILTER_SIZE); i++)
    {
        if (x < *width && y < *height)
        {
            if ((x + i < 0) || (x + i >= *width) ||
                (y + i < 0) ||
                (y + i >= *height))
                return;
            sum += in[(x + y * *width) + i];
            count++;
        }
    }
    if (count != 0)
    {
        sum /= (count);
        out[x + y * *width] = sum;
    }
    else
    {
        out[x + y * *width] = in[x + y * *width];
    }

    __syncthreads();
}

// Función que realiza el filtro de box blur. Obtenemos un kernel de tamaño nxn
// y recorremos nuestra matriz de pixeles modificando cada uno de los RGB de cada pixel
// siempre multiplicando por el valor del kernel.
void RGBImageToBlur(struct Image pic, unsigned int FILTER_SIZE, int thread_count)
{

    int i, j;
    int size = pic.width * pic.height * sizeof(unsigned char);

    int *d_filter_size;
    int *d_width;
    int *d_height;

    int filter = FILTER_SIZE;

    //Alojando memoria en el device de variables a usar: filter_size, width, height

    CUDA_CHECK_RETURN(cudaMalloc(&d_filter_size, sizeof(unsigned int)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_width, sizeof(unsigned int)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_height, sizeof(unsigned int)));

    //Copiando las variables al device

    CUDA_CHECK_RETURN(cudaMemcpy(d_filter_size, &filter, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_width, &pic.width, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_height, &pic.height, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Creando canales RGB del host de entrada y salida

    unsigned char *h_r = (unsigned char *)malloc(size);
    unsigned char *h_g = (unsigned char *)malloc(size);
    unsigned char *h_b = (unsigned char *)malloc(size);

    unsigned char *h_r_n = (unsigned char *)malloc(size);
    unsigned char *h_g_n = (unsigned char *)malloc(size);
    unsigned char *h_b_n = (unsigned char *)malloc(size);

    // Rellenando cada vector de cada color
    int count = 0;
    for (i = 0; i < pic.height; i++)
    {
        for (j = 0; j < pic.width; j++)
        {
            h_r[i * pic.width + j] = pic.rgb[i][j].red;
            h_g[i * pic.width + j] = pic.rgb[i][j].green;
            h_b[i * pic.width + j] = pic.rgb[i][j].blue;
            count++;
        }
    }

    //Crear los canales RGB de salida y de entrada del device y alojarles memoria
    unsigned char *d_r_n;
    unsigned char *d_g_n;
    unsigned char *d_b_n;

    CUDA_CHECK_RETURN(cudaMalloc(&d_r_n, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_g_n, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b_n, size));

    unsigned char *d_r;
    unsigned char *d_g;
    unsigned char *d_b;

    CUDA_CHECK_RETURN(cudaMalloc(&d_r, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_g, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b, size));

    //Copiar vectores RGB desde el host al device

    CUDA_CHECK_RETURN(cudaMemcpy(d_r, h_r, size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_g, h_g, size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    //Calcular los bloques e hilos a usar

    dim3 threadsPerBlock(thread_count, thread_count);
    dim3 blocksPerGrid(ceil((float)pic.width / threadsPerBlock.x), ceil((float)pic.height / threadsPerBlock.y));

    //Invocar el kernel por cada canal

    processImage<<<blocksPerGrid, threadsPerBlock>>>(d_r_n, d_r, d_width, d_height, d_filter_size);
    processImage<<<blocksPerGrid, threadsPerBlock>>>(d_g_n, d_g, d_width, d_height, d_filter_size);
    processImage<<<blocksPerGrid, threadsPerBlock>>>(d_b_n, d_b, d_width, d_height, d_filter_size);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    //Copiar canales RGB desde el device hacia el host

    CUDA_CHECK_RETURN(cudaMemcpy(h_r_n, d_r_n, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_g_n, d_g_n, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_b_n, d_b_n, size, cudaMemcpyDeviceToHost));

    //Pasar los canales RGB hacia la imagen

    for (i = 0; i < pic.height; i++)
    {
        for (j = 0; j < pic.width; j++)
        {
            pic.rgb[i][j].red = h_r_n[i * pic.width + j];
            pic.rgb[i][j].green = h_g_n[i * pic.width + j];
            pic.rgb[i][j].blue = h_b_n[i * pic.width + j];
        }
    }

    //Liberar memoria en device y host

    CUDA_CHECK_RETURN(cudaFree(d_filter_size));
    CUDA_CHECK_RETURN(cudaFree(d_width));
    CUDA_CHECK_RETURN(cudaFree(d_height));

    CUDA_CHECK_RETURN(cudaFree(d_r));
    CUDA_CHECK_RETURN(cudaFree(d_r_n));

    CUDA_CHECK_RETURN(cudaFree(d_g));
    CUDA_CHECK_RETURN(cudaFree(d_g_n));

    CUDA_CHECK_RETURN(cudaFree(d_b));
    CUDA_CHECK_RETURN(cudaFree(d_b_n));

    free(h_r);
    free(h_r_n);

    free(h_g);
    free(h_g_n);

    free(h_b);
    free(h_b_n);
}

//Función que permite crear la imagen de salida y aplicación del filtro.
int createImage(struct BITMAP_header header, struct DIB_header dibheader, struct Image pic, int filter, int kernelSize, char *name, int thread_count)
{
    int i;
    FILE *fpw = fopen(name, "w");
    if (fpw == NULL)
        return 1;

    //Aplicacion del filtro box blur
    if (filter == 2)
    {
        RGBImageToBlur(pic, kernelSize, thread_count);
    }

    fwrite(header.name, 2, 1, fpw);
    fwrite(&header.size, 3 * sizeof(int), 1, fpw);
    fwrite(&dibheader, sizeof(struct DIB_header), 1, fpw); //Puede que sea el problema de archivos corruptos

    for (i = pic.height - 1; i >= 0; i--)
    {
        fwrite(pic.rgb[i], pic.width, sizeof(struct RGB), fpw);
    }
    fclose(fpw);
    return 0;
}

//Función que permite abrir un archivo .bmp y por medio de fread() ir accediendo a los
//bytes del archivo con la información importante para el caso (width, heigth, etc).
void openbmpfile(char *name, char *output, int kernelSize, int thread_count)
{
    FILE *fp = fopen(name, "rb");
    struct BITMAP_header header;
    struct DIB_header dibheader;

    fread(header.name, 2, 1, fp);
    fread(&header.size, 3 * sizeof(int), 1, fp);

    fread(&dibheader, sizeof(struct DIB_header), 1, fp);

    struct Image image = readImage(fp, dibheader.height, dibheader.width, thread_count);

    createImage(header, dibheader, image, 2, kernelSize, output, thread_count);

    fclose(fp);
    freeImage(image);
}

int main(int argc, char *argv[])
{
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);

    int thread_count = atoi(argv[3]);

    if (argc != 5)
    {
        printf("Ingrese los valores correctos:\ninput_name output_name threads kernel_size\n");
        return 1;
    }
    openbmpfile(argv[1], argv[2], atoi(argv[4]), thread_count);

    gettimeofday(&tval_after, NULL);

    timersub(&tval_after, &tval_before, &tval_result);
    printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    printf("done\n");
    return 0;
}