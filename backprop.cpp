#include <iostream>
#include <math.h>
#include <exception>
#include <stdlib.h>

using namespace std;

class neuronet_not_ready: public exception
{
  virtual const char* what() const throw()
  {
    return "Neuronet is not initialized, yet.\n";
  }
} netnotready;

class neuron_not_exists: public exception
{
  virtual const char* what() const throw()
  {
    return "Neuron that you have called does not exists.\n";
  }
} nne;

class neuron
{
private:
	int num_inputs;
	double*  weights;

	void randInit()
	{
		for(int i = 0; i < num_inputs; i++){
			setWeight(i,1./(double)rand());
		}
	}

	void zeroInit()
	{
		for(int i = 0; i < num_inputs; i++){
			setWeight(i,0);
		}
	}

	double activationFunc(double x)
    {
        return 1./(1. + exp(x));
    }

public:
	neuron()
	{
		this->num_inputs = 0;
		this->weights = NULL;
	}

	neuron(int num)
	{
		setNumInputs(num);
	}

	void setNumInputs(int num)
	{
		this->num_inputs = num;
		this->weights = new double[num];
		randInit();
	}

	void getWeights(double* weights)
	{
		for(int i = 0; i < num_inputs; i++){
			weights[i] = this->weights[i];
		}
	}

	double output(double* inputs)
	{
		double s = 0.; 
        for(int i = 0; i < num_inputs; i++){
            s+=inputs[i]*weights[i];
        }
        return s;
	}

	double actOutput(double* inputs)
	{
		return activationFunc(output(inputs));
	}

	double getWeight(int num)
	{
		if(num < 0 || num > num_inputs)
		{
			throw nne;
		}
		return weights[num];
	}

	void setWeight(int num, int val)
	{
		if(num < 0 || num > num_inputs)
		{
			throw nne;
		}
		weights[num] = val;
	}	

	double updateWeights(double output, double sigma, double alpha, double nu)
	{
		double dw,e;
		for(int j = 0; j < num_inputs; j++){
            dw = alpha*weights[j] + (1 - alpha) * nu * output * sigma;
            e += dw;
            weights[j] += dw;
        }
        return e;
	}

	~neuron()
	{
		delete[] weights;
	}
};

class neuronet
{
private:
    neuron** neurons;
    double** outputs;
    int layers_num;
    int* layer_sizes;
    int data_dim;
    int num_steps;
    double error;
    double alpha;
    double nu;

    void forwardProp(double* image, double** layers)
    {
    	for(int i = 0; i < data_dim; i++){
    		layers[0][i] = image[i];
    	}
        for(int i = 0; i < this->layers_num; i++){
            for(int j = 1; j < layer_sizes[i]; j++){
                    layers[i+1][j] = neurons[i][j].actOutput(image);
            }
        }
    }

    void backProp(double* exmpl_vals, double** output, double** sigmas)
    { 
    	double temp;
        //обратное распространение
        //выходной уровень
        for(int i = 0; i < layer_sizes[layers_num]; i++){
            sigmas[layers_num][i] = output[layers_num][i]*(1. - output[layers_num][i])*(output[layers_num][i] - exmpl_vals[i]);
        }
        //скрытые уровни
        for(int i = layers_num - 1; i > 0; i--){
            for (int j = 0; j < layer_sizes[i]; j++){
                //коэффициенты нейрона проверить
                //что с чем??
                for(int k = 0; k < layer_sizes[i+1]; k++){
                    temp += neurons[i][j].getWeight(k)*sigmas[i+1][k];
                }
                sigmas[i][j] = output[i][j]*(1-output[i][j])*temp;
            }
        } 
    }

    int getFromFile(char* filename, double** images, int num_to_read)
    {
        FILE* file;
        file = fopen(filename,"r");
        if(file == NULL){
            throw netnotready;
        }
        int i,j;
        char str[100];
        char* pch = NULL;
        for(i = 0; i < num_to_read && fgets(str,100,file) != NULL; i++){
            images[i] = new double[data_dim];
            pch = strtok (str,",");
            for (j = 0; j < data_dim && pch != NULL; j++){
                images[i][j] = atof(pch);
                pch = strtok (NULL, ",");
            }
        }
        return i;
    }

public:
    neuronet()
    {
        this->layers_num = 0;
        this->layer_sizes = NULL;
        this->neurons = NULL;
    }

    neuronet(int layers_num, int* layer_sizes, int data_dim)
    {
        setLayers(layers_num,layer_sizes,data_dim);
    }

    void setLayers(int layers_num, int* layer_sizes, int data_dim)
    {
        this->layers_num = layers_num;
        this->data_dim = data_dim;
        this->layer_sizes = new int[layers_num];
        this->neurons = new neuron* [layers_num];
        int cur_layer_size, prev_layer_size;
        for(int i = 0; i < layers_num; i++){
            cur_layer_size = layer_sizes[i]+1;
            this->layer_sizes[i] = cur_layer_size;
            prev_layer_size = (i == 0 ? layer_sizes[i-1] + 1 : data_dim + 1);
            this->neurons[i] = new neuron[cur_layer_size];//number of neurons on layer
            for(int j = 0; j < cur_layer_size; j++){
                this->neurons[i][j].setNumInputs(prev_layer_size);//number of weights for neuron
            }
        }
        this->num_steps = 1000;
        this->error = 0.001;
        this->alpha = 0.1;
        this->nu = 0.1;
        //размеры уровней включают смещение
        this->outputs = new double*[this->layers_num + 1];
        this->outputs[0] = new double [data_dim];
        for(int i = 1; i < this->layers_num+1;i++){
            this->outputs[i] = new double [this->layer_sizes[i]];
            if(i != this->layers_num){
                //смещение
                this->outputs[i][0] = 1;
            }
        }
    }

    void learn(double** exmpls, double** exmpl_vals, int num_examples)
    {
        if(this->layer_sizes == NULL){
            throw netnotready;
        }
        double e,dw;
        int k;
        double** sigmas = new double*[layers_num];
        for(int i = 0; i < layers_num; i++){
        	sigmas[i] = new double[layer_sizes[i]];
        }
        for(int i = 0; i < num_examples; i++){
            k = 0;
            while(k < num_steps && e < error){
                e = 0.;
                forwardProp(exmpls[i], this->outputs);
                backProp(exmpl_vals[i], this->outputs, sigmas);

                for(int m = 0; m < layers_num; m++){
                    for(int n = 0; n < layer_sizes[m]; n++){
                        e+= neurons[n][m].updateWeights(outputs[m][n],
                        	     sigmas[m][n], alpha, nu);
                    }
                }
                k++;
            }    
        }
        delete[] sigmas;
    }

    void learnFromFile(char* exmpls_filename, char* exmpl_vals_filename, int num_to_read)
    {
        if(this->layer_sizes == NULL){
            throw netnotready;
        }
        int i,j,data_dim = this->data_dim;
        double** images = new double*[num_to_read];
        double** exmpl_vals = new double*[num_to_read];
        for(int i = 0; i < num_to_read; i++){
            images[i] = new double[data_dim];
            exmpl_vals[i] = new double[num_to_read];
        }
        int num_exmpls = getFromFile(exmpls_filename, images, num_to_read);
        int num_vals = getFromFile(exmpl_vals_filename, images, num_to_read);
        if(num_exmpls != num_vals){
            throw netnotready;
        }
        learn(images,exmpl_vals,num_exmpls);
        delete[] images;
        delete[] exmpl_vals;
    }

    void recognizeFromFile(char* filename, double** result, int num_to_read)
    {
        double** examples = new double*[num_to_read];
        for(int i = 0; i < num_to_read; i++){
            examples[i] = new double[this->data_dim];
        }
        int num_images = getFromFile(filename,examples,num_to_read);
        for(int i = 0; i < num_images; i++){
            recognize(examples[i], result[i]);
        }
    }

    void recognize(double* example, double* result)
    {
        if(this->layer_sizes == NULL){
            throw netnotready;
        }

        forwardProp(example, this->outputs);
        for(int i = 0;i < this->layer_sizes[this->layers_num]; i++){
            result[i] = this->outputs[this->layers_num][i];
        }
    }

    ~neuronet()
    {
        delete[] this->layer_sizes;
        delete[] this->neurons;
        delete[] outputs;
    }
};

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

int main(int argc, char** argv)
{
	int num_images = atoi(argv[6]);
    int data_dim = atoi(argv[5]);
    int num_exmpls = atoi(argv[4]);
    int rec_images_num = num_exmpls;
	int layer_sizes[] = {64,128,rec_images_num}; 
	cout<<"-1";
    neuronet nn(3,layer_sizes,data_dim);
    cout<<"a";
    nn.learnFromFile(argv[1],argv[2],num_exmpls);
    cout<<"b";
    double** result = new double*[num_images];
    for(int i = 0; i < num_images; i++){
    	result[i] = new double[rec_images_num];
    }
    cout<<"c";
    nn.recognizeFromFile(argv[3],result,num_images);
    cout<<"d";
    out(result,num_images,rec_images_num);

    return 0;
}


