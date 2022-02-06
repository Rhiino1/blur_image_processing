// mpicc test.c -o test_mpi
// mpirun -np 3 test_mpi
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#define root 0

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

struct Image readImage(FILE *fp, int height, int width)
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

void freeImage(struct Image pic)
{
    int i;
    for (i = pic.height - 1; i >= 0; i--)
    {
        free(pic.rgb[i]);
    }
    free(pic.rgb);
}

unsigned char convolution(unsigned char *in, int i, int j, int kernel_size, int width, int height)
{
    float sum = 0.0;
    int counter = 0;
    for (int x = -kernel_size - 1; x <= kernel_size + 1; x++)
    {
        if ((i * width + j) + x > 0 && ((i * width + j) + x) < (width * height * 3))
            sum += in[(i * width + j) + (x)];
        counter++;
    }
    sum = sum / counter;
    return sum;
}

unsigned char *proccesColor(unsigned char *in, int procMax, int procStart, int procEnd, int width, int height, int kernel_size, int color)
{
    unsigned char *output = in;
    for (int i = 0; i < procMax; i++)
    {
        // output[i] = in[i];
        output[i] = convolution(in, i / width, i % width, kernel_size, width, height);
    }
    return output;
}

int createImage(struct BITMAP_header header, struct DIB_header dibheader,
                struct Image pic, int filter, int kernelSize, char *name,
                int numprocs, int processId, int argc, char *argv[])
{
    struct RGB **output;
    int procStart;
    int procEnd;
    int procHeight;
    int procMax;
    int meta[4];

    // if (processId == 0)
    // {
    //     printf("\nLaunching with %i processes\n", numprocs);
    // }

    int size = pic.width * pic.height * sizeof(unsigned char);

    unsigned char *r = (unsigned char *)malloc(size);
    unsigned char *g = (unsigned char *)malloc(size);
    unsigned char *b = (unsigned char *)malloc(size);

    unsigned char *sub_r = malloc(sizeof(unsigned char) * (int)(size / numprocs));
    unsigned char *sub_g = malloc(sizeof(unsigned char) * (int)(size / numprocs));
    unsigned char *sub_b = malloc(sizeof(unsigned char) * (int)(size / numprocs));

    for (int i = 0; i < pic.height; i++)
    {
        for (int j = 0; j < pic.width; j++)
        {
            r[i * pic.width + j] = pic.rgb[i][j].red;
            g[i * pic.width + j] = pic.rgb[i][j].green;
            b[i * pic.width + j] = pic.rgb[i][j].blue;
        }
    }

    if (processId == root)
    {
        procHeight = (int)pic.height / numprocs;
        procStart = root * procHeight;
        procEnd = (root * procHeight) + procHeight;
        procMax = (int)(size / numprocs);
        output = pic.rgb;

        for (int i = 1; i < numprocs; i++)
        {
            meta[0] = procHeight;
            meta[1] = i * procHeight;
            meta[2] = (i * procHeight) + procHeight;
            meta[3] = (int)(size / numprocs);
            MPI_Send((void *)meta, 4, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }
    if (processId != root)
    {
        MPI_Recv(&meta, 4, MPI_INT, root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        procHeight = meta[0];
        procStart = meta[1];
        procEnd = meta[2];
        procMax = meta[3];
        // printf("%d, %d, %d, %d\n", procHeight, procStart, procEnd, procMax);
    }

    MPI_Scatter(r, (int)(size / numprocs), MPI_CHAR, sub_r,
                (int)(size / numprocs), MPI_CHAR, root, MPI_COMM_WORLD);
    MPI_Scatter(g, (int)(size / numprocs), MPI_CHAR, sub_g,
                (int)(size / numprocs), MPI_CHAR, root, MPI_COMM_WORLD);
    MPI_Scatter(b, (int)(size / numprocs), MPI_CHAR, sub_b,
                (int)(size / numprocs), MPI_CHAR, root, MPI_COMM_WORLD);

    unsigned char *sub_r1 = sub_r;
    unsigned char *sub_g1 = sub_g;
    unsigned char *sub_b1 = sub_b;

    sub_r1 = proccesColor(sub_r, procMax, procStart, procEnd, pic.width, pic.height, kernelSize, 0);
    sub_g1 = proccesColor(sub_g, procMax, procStart, procEnd, pic.width, pic.height, kernelSize, 0);
    sub_b1 = proccesColor(sub_b, procMax, procStart, procEnd, pic.width, pic.height, kernelSize, 0);

    unsigned char *sub_rs = NULL;
    unsigned char *sub_gs = NULL;
    unsigned char *sub_bs = NULL;

    if (processId == root)
    {
        sub_rs = malloc(size);
        sub_gs = malloc(size);
        sub_bs = malloc(size);
    }
    MPI_Gather(&sub_r1[0], procMax, MPI_CHAR, sub_rs, procMax, MPI_CHAR, root,
               MPI_COMM_WORLD);
    MPI_Gather(&sub_g1[0], procMax, MPI_CHAR, sub_gs, procMax, MPI_CHAR, root,
               MPI_COMM_WORLD);
    MPI_Gather(&sub_b1[0], procMax, MPI_CHAR, sub_bs, procMax, MPI_CHAR, root,
               MPI_COMM_WORLD);

    if (processId == root)
    {
        int counter = 0;
        for (int i = 0; i < pic.height; i++)
        {
            for (int j = 0; j < pic.width; j++)
            {
                output[i][j].red = sub_rs[i * pic.width + j];
                output[i][j].green = sub_gs[i * pic.width + j];
                output[i][j].blue = sub_bs[i * pic.width + j];
            }
        }

        FILE *fpw = fopen(name, "w");
        if (fpw == NULL)
            return 1;

        fwrite(header.name, 2, 1, fpw);
        fwrite(&header.size, 3 * sizeof(int), 1, fpw);
        fwrite(&dibheader, sizeof(struct DIB_header), 1, fpw);

        for (int i = pic.height - 1; i >= 0; i--)
        {
            fwrite(output[i], pic.width, sizeof(struct RGB), fpw);
        }
        fclose(fpw);
        freeImage(pic);
    }
    return 0;
}

int main(int argc, char *argv[])
{
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);

    if (argc != 4)
    {
        printf("Ingrese los valores correctos:\ninput_name output_name kernel_size\n");
        return 1;
    }

    int processId, numprocs;
    char *name = argv[1];

    FILE *fp = fopen(name, "rb");
    struct BITMAP_header header;
    struct DIB_header dibheader;

    fread(header.name, 2, 1, fp);
    fread(&header.size, 3 * sizeof(int), 1, fp);

    fread(&dibheader, sizeof(struct DIB_header), 1, fp);

    // fseek(fp, header.image_offset, SEEK_SET);
    struct Image image = readImage(fp, dibheader.height, dibheader.width);
    fclose(fp);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    createImage(header, dibheader, image, 2, atoi(argv[3]), argv[2], numprocs, processId, argc, argv);

    if (processId == root)
    {
        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    }
    MPI_Finalize();
    return 0;
}
