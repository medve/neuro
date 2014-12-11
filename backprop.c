#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

const int MAX_ITERS = 1000000;
const double H = 0.01;
const double ERROR   = 0.00001;

int imagesFromFile(char* filename, double** images, int num_to_read, 
                                                        int data_dim);
void recognize(double*** neurons, int num_layers, int* layer_sizes, 
                         double** images, double** results, int num_images, int data_dim);
void learn(double*** neurons, int num_layers, int* layer_sizes, 
            double** exmpls, double** exmpl_vals, int num_exmpls,  int data_dim);
double activationFunc(double x);
double wsum(double* elems, double* coef, int elems_num);
void out(double** result, int images_num, int results_num);
double quadError(double* values, double* true_values, int num_values);


int main(int argc, char** argv)
{
    int i,rec_images_num,data_dim,num_images,num_exmpls;

    num_images = atoi(argv[6]);
    data_dim = atoi(argv[5]);
    num_exmpls = atoi(argv[4]);
    rec_images_num = num_exmpls;

    /* 
        Memory allocation

        exmpls[num_exmpls][data_dim]
        result[num_images][rec_images_num]
        images[num_images][data_dim]
        neurons[rec_images_num][data_dim]
    */
    double** result = (double**)malloc(num_images*sizeof(double));
    for(i = 0; i < num_images; i++){
        result[i] = (double*)malloc(rec_images_num*sizeof(double));
    }
    double** exmpls = (double**)malloc(num_exmpls*sizeof(double));
    for(i = 0; i < num_exmpls; i++){
        exmpls[i] = (double*)malloc(data_dim*sizeof(double));
    }
    double** exmpl_vals = (double**)malloc(num_exmpls*sizeof(double));
    for(i = 0; i < num_exmpls; i++){
        exmpl_vals[i] = (double*)malloc(num_exmpls*sizeof(double));
    }
    double** images = (double**)malloc(num_images*sizeof(double));
    for(i = 0; i < num_images; i++){
        images[i] = (double*)malloc(data_dim*sizeof(double));
    }
    double*** neurons = (double**)malloc(rec_images_num*sizeof(double));
    for(i = 0; i < rec_images_num; i++){
        neurons[i] = (double*)malloc(data_dim*sizeof(double));
    }
    /*
        Read Data
    */
    if(imagesFromFile(argv[1], exmpls, num_exmpls, data_dim)){
        printf("File can not be openned.\n");
    }//Symbols
    if(imagesFromFile(argv[2], exmpl_vals, num_exmpls, num_exmpls)){
        printf("File can not be openned.\n");
    }//Symbol vals
    if(imagesFromFile(argv[3], images, num_images, data_dim)){
        printf("File can not be openned.\n");
    }//Images
    /*
        Learn and recognize
    */
    learn(neurons, exmpls, exmpl_vals, num_exmpls,
            rec_images_num, data_dim);
    groupRecognize(neurons,images, result, num_images, 
                     rec_images_num, data_dim );
       /*
           Out
       */ 
    out(result, num_images, rec_images_num);

    free(images);
    free(neurons);
    free(exmpls);
    free(result);
    return 0;
}

/*
Нейроны - коэффициенты от первого уровня до последнего
layers - значения на уровнях от нулевого уровня до последнего
нулевой уровень - исходные данные
последний уровень не включает смещение
*/

void recognize(double*** neurons, int num_layers, int* layer_sizes, 
                                     double* image, double** layers)
{
    //размеры уровней включают смещение
    int i,j,k;
    double** layers = (double*)malloc(num_layers*sizeof(double));
    for(i = 0; i < num_layers;i++){
        layers[i] = (double*)malloc(layer_sizes[i]*sizeof(double));
        if(i != num_layers-1){
            //смещение
            layers[i][0] = 1;
        }
    }
    for(i = 0; i < num_layers; i++){
        for(j = 1; j < layer_sizes[i]; j++){
            //для нулевого уровня просто копируем изображение
            //для последующих вычисляем функцию активации
            if(i == 0){
                layers[0][j] = image[i-1];
            } else {
                layers[i][j] = activationFunc(wsum(neurons[i][j],layers[i-1],
                                                           layer_sizes[i-1]));
            }
        }
    }
}

double quadError(double* values, double* true_values, int num_values)
{
    //пропускает смещение
    int i;
    double result = 0.,temp;
    for(i = 1; i < num_values; i++){
        temp = true_values[i] - values[i];
        result += temp*temp;
    }
    return result/2.;
}

void learn(double*** neurons, int num_layers, int* layer_sizes, 
            double** exmpls, double** exmpl_vals, int num_exmpls, int data_dim)
{
    int i,j,k,m;
    double ws,out,e,temp;
    double* sigmas = (double**)malloc(num_layers*sizeof(double*));
    //инициализируем случайными значениями
    for(i = 0; i < num_layers-1; i++){
        for(j = 0; j < layer_sizes[i+1]; j++){
            for(k = 0; k < layer_sizes[i]; k++){
                sigmas[i] = (double*)malloc(layer_sizes[k]*sizeof(double));
                neurons[i][j][k] = 1./(double)rand(1000);
            }
        }
    }

    for(i = 0; i < num_exmpls; i++){
        while(k < NUM_STEPS && e < ERROR){
        //с каждым примером
            //прямое распространение
            recognize(neurons, num_layers, layer_sizes, exmpls[i], sigmas);

            //обратное распространение
            //выходной уровень
            for(i = 0; i < layer_sizes[num_layers-1]; i ++){
                sigmas[0] = sigmas[0][i]*(1. - sigmas[0][i])*(sigmas[0][i] - exmpl_vals[i]);
            }
            //скрытые уровни
            for(i = num_layers - 2; i > 1; i--){
                for (int j = 0; j < layer_sizes[i]; j++){
                    //коэффициенты нейрона проверить
                    //что с чем??
                    for(k = 0; k < layer_sizes[i+1]; k++){
                        temp = neurons[i][j][k]*sigmas[i+1][k];
                    }
                    sigmas[i][j] = sigmas[i][j]*(1-sigmas[i][j])*temp;
                }
            }
        }
    }
    free(sigmas);
}

int imagesFromFile(char* filename, double** images, int num_to_read, int data_dim)
{
    int i,j;
    FILE* file;
    file = fopen(filename,"r");
    if(file == NULL){
        return 1;
    }
    char str[100];
    char* pch = NULL;
    for(i = 0; i < num_to_read && fgets(str,100,file) != NULL; i++){
        pch = strtok (str,",");
        for (j = 0; j < data_dim && pch != NULL; j++){
            images[i][j] = atof(pch);
            pch = strtok (NULL, ",");
        }
    }
    return 0;
}

double wsum(double* elems,double* coef, int elems_num)
{
    int i, s = 0; 
    for(i = 0;i < elems_num; i++){
        s+=elems[i]*coef[i];
    }
    return s;
}

double activationFunc(double x)
{
    return 1./(1. + exp(x));
}

void out(double** result, int images_num, int results_num)
{
    int i,j,m;
    for(i = 0; i < images_num; i++){
        m = result[i][0];
        for(j = 0; j < results_num; j++){
            // printf("%d %d.\n",i, j);
            if(result[i][j] > result[i][m]){
                m = j;
            }
        }
        printf("Image %d is symbol %d.\n",i, m);
    }
}