#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

const int MAX_ITERS = 1000000;
const double DELTA = 0.01;
const double ERROR   = 0.00001;

/*
    Reads images from file
*/
int imagesFromFile(char* filename, double** images, int num_to_read, 
                                                        int data_dim);

/*
    Recognizes one image
*/
void recognize(double** neurons, double* image, double* result, 
                              int rec_images_num, int data_dim);

/*
    Recognizes group of images
*/
void groupRecognize(double** neurons,double** images, double** result,
                     int num_images, int rec_images_num, int data_dim );

/*
    Learns network
*/
void learn(double** neurons, double** exmpls, double** exmpl_vals, 
             int num_exmpls, int recogn_elems_num, int data_dim);

/*
    Learns one neuron
*/
void learnNeuron(double* neuron, double** exmpls, double* exmpl_vals, 
                                        int num_exmpls, int data_dim);

/*
    Updates weights of neuron, using one example
*/
void updateNeuron(double* neuron, double* exmpl, double* result, int data_dim);

/*
    Init neuron coefficients
*/
void initCoefs(double* neuron, int data_dim);

/*
    sigmoid
*/
double activationFunc(double x);

/*

*/
int decisionFunc(double x);

/*
    Weighted sum 
*/
double wsum(double* elems, double* coef, int elems_num);

/*
    Out
*/
void out(double** result, int images_num, int results_num);

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
    double** neurons = (double**)malloc(rec_images_num*sizeof(double));
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

void groupRecognize(double** neurons,double** images, double** result,
                     int num_images, int rec_images_num, int data_dim )
{
    int i;
    for(i = 0; i < num_images; i++){
        recognize(neurons, images[i], result[i], rec_images_num, data_dim);
    }
}

void recognize(double** neurons, double* image, double* result, 
                 int rec_images_num, int data_dim)
{
    int i;
    for(i = 0; i < rec_images_num; i++){
        result[i] = activationFunc(wsum(neurons[i], image, data_dim));
    }
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

int decisionFunc(double x)
{
    return x > 0.5 ? 1 : 0;
}

void initCoefs(double* neuron, int data_dim)
{
    int i;
    for(i = 0; i < data_dim; i++){
        neuron[i] = (double)(rand()%1000)/(double)(rand()+1000);
    }
}

void learnNeuron(double* neuron, double** exmpls, 
                  double* exmpl_vals, int num_exmpls, int data_dim)
{
    int i,j;
    double error, ws, update, ex;
    int k = 0;
    do {
        error = 0.;
        for(i = 0; i < num_exmpls; i++){
        	ws = wsum(neuron,exmpls[i],data_dim);
            ex = exp(ws);
            for(j = 0; j < data_dim; j++ ){
                update = DELTA*(activationFunc(ws) - exmpl_vals[i])
                           *(ex/((1.+ex)*(1.+ex)))*exmpls[i][j];
                error += fabs(update);
                neuron[j] += update;
            }
        }
        k++;
    } while(error > ERROR && k < MAX_ITERS);
    printf("%d\n",k);
}

void learn(double** neurons, double** exmpls, double** exmpl_vals, 
             int num_exmpls, int recogn_elems_num, int data_dim)
{
    int i,j;
    for(i = 0; i < recogn_elems_num; i++){
        initCoefs(neurons[i], data_dim);
        learnNeuron(neurons[i], exmpls, exmpl_vals[i], num_exmpls, data_dim);
    }
}

void out(double** result, int images_num, int results_num)
{
    int i,j;
    for(i = 0; i < images_num; i++){
        printf("Results for image %d:\n",i);
        for(j = 0; j < results_num; j++){
            printf("%f\n", result[i][j]);
            if(decisionFunc(result[i][j]) == 1){
                printf("Image %d is symbol %d.\n",i, j);
            }
        }
    }
}