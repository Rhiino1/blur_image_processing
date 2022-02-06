#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

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


// Función que realiza el filtro de box blur. Obtenemos un kernel de tamaño nxn
// y recorremos nuestra matriz de pixeles modificando cada uno de los RGB de cada pixel
// siempre multiplicando por el valor del kernel.
void RGBImageToBlur(struct Image pic, int n)
{
    int i, j, x, y;
    struct Image copy = pic;

    float v = 1.0;

    float *kernel[n];
    for (i = 0; i < n; i++)
    {
        float *row = malloc(sizeof(float) * n);
        kernel[i] = row;
    }

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            kernel[i][j] = v;
        }
    }

    for (i = 0; i < pic.height; i++)
    {
        for (j = 0; j < pic.width; j++)
        {
            float sumRed = 0.0;
            float sumGreen = 0.0;
            float sumBlue = 0.0;
            int counter = 0;

            int edges = (int)floor(n / 2);

            // Nuestro algoritmo no toma encuenta los bordes, este if se salta todos.
            if ((i < edges || j < edges) || (i + edges >= pic.height || j + edges >= pic.width))
            {
                continue;
            }
            for (x = -edges; x < edges; x++)
            {
                for (y = -edges; y < edges; y++)
                {

                    sumRed += kernel[x + edges][y + edges] * copy.rgb[i + x][j + y].red;
                    sumGreen += kernel[x + edges][y + edges] * copy.rgb[i + x][j + y].green;
                    sumBlue += kernel[x + edges][y + edges] * copy.rgb[i + x][j + y].blue;
                    counter++;
                }
            }
            //Luego de obtener la suma por cada kernel hay que sumarlo al pixel central.
            pic.rgb[i][j].red = sumRed / counter;
            pic.rgb[i][j].green = sumGreen / counter;
            pic.rgb[i][j].blue = sumBlue / counter;
        }
    }
}

//Función que permite crear la imagen de salida y aplicación del filtro.
int createImage(struct BITMAP_header header, struct DIB_header dibheader, struct Image pic, int filter, int kernelSize, char *name)
{
    int i;
    FILE *fpw = fopen(name, "w");
    if (fpw == NULL)
        return 1;


    //Aplicacion del filtro box blur
    if (filter == 2)
    {
        RGBImageToBlur(pic, kernelSize);
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
void openbmpfile(char *name, char *output, int kernelSize)
{
    FILE *fp = fopen(name, "rb");
    struct BITMAP_header header;
    struct DIB_header dibheader;

    fread(header.name, 2, 1, fp);
    fread(&header.size, 3 * sizeof(int), 1, fp);

    fread(&dibheader, sizeof(struct DIB_header), 1, fp);

    struct Image image = readImage(fp, dibheader.height, dibheader.width);

    createImage(header, dibheader, image, 2, kernelSize, output);

    fclose(fp);
    freeImage(image);
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
    openbmpfile(argv[1], argv[2], atoi(argv[3]));

    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    return 0;
}