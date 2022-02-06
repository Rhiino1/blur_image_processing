#!/bin/bash

gcc cblur.c -o cblur -lm
gcc cblur_omp.c -o cblur_omp -fopenmp -lm
mpicc cblur_mpi.c -o cblur_mpi -lm

echo "Prueba de imagen de 720p" >> tiempos.txt

echo "Paralelizado 4 procesos con Open mpi: " >> tiempos.txt
mpirun -np 4 cblur_mpi ./imagenes/720p.bmp 720p_blur.bmp 10 >> tiempos.txt

echo "Paralelizado a 4 hilos: " >> tiempos.txt
./cblur_omp ./imagenes/720p.bmp 720p_blur.bmp 4 10 >> tiempos.txt

echo "Secuencial: " >> tiempos.txt
./cblur ./imagenes/720p.bmp 720p_blur.bmp 10 >> tiempos.txt



echo "Prueba de imagen de 1080p" >> tiempos.txt

echo "Paralelizado 4 procesos con Open mpi: " >> tiempos.txt
mpirun -np 4 cblur_mpi ./imagenes/1080p.bmp 1080p_blur.bmp 10 >> tiempos.txt

echo "Paralelizado a 4 hilos: " >> tiempos.txt
./cblur_omp ./imagenes/1080p.bmp 1080p_blur.bmp 4 10 >> tiempos.txt

echo "Secuencial: " >> tiempos.txt
./cblur ./imagenes/1080p.bmp 1080p_blur.bmp 10 >> tiempos.txt


echo "Prueba de imagen de 4k" >> tiempos.txt

echo "Paralelizado 4 procesos con Open mpi: " >> tiempos.txt
mpirun -np 4 cblur_mpi ./imagenes/4k.bmp 4k_blur.bmp 10 >> tiempos.txt

echo "Paralelizado a 4 hilos: " >> tiempos.txt
./cblur_omp ./imagenes/4k.bmp 4k_blur.bmp 4 10 >> tiempos.txt

echo "Secuencial: " >> tiempos.txt
./cblur ./imagenes/4k.bmp 4k_blur.bmp 10 >> tiempos.txt