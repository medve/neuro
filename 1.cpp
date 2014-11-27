#include <iostream>
///////////////////////////////////////////////////////////////////////////////////////
//neurons[i][j] - i'th neuron's j'th coefficient                                    //
//exmpls[i][i]  - i'th exmpl's j'th dim                                              //
//exmpls_values[i][j] - says that i'th  example is j'th recognizable element or not  //
//num_neurons' the same as the number of recoznizable elements                       //
//data_dim is dimmension of input data(number of pixels in image)                    //
//the second dimmension of exmpls is the same as data_dim                             //
//num_exmpls - number of examples for learning                                        //
///////////////////////////////////////////////////////////////////////////////////////

const int MAX_ITERS = 1000;

//TODO:interface

void recognize(int** neurons, int* exmpl, int * result, int recogn_elems_num, int data_dim);
int update_function(int prev_val, int wi, int y);
void learn_neuron(int* neuron, int* exmpl, int* exmpl_value, int data_dim);
void learn_exmpl(int** neurons, int* exmpl, int* exmpl_value, int recogn_elems_num, int data_dim);
void init_coef(int** neurons, int recogn_elems_num, int data_dim);
bool validate_result(int** neurons, int** exmpls, int** exmpls_values, int recogn_elems_num, int data_dim, int exmpls_num, int error);
void learn_network(int** neurons, int** exmpls, int** exmpls_values, int recogn_elems_num, int data_dim, int exmpls_num);
int wsum(int* elems,int* coef, int* elems_num);
int decision_func(int val);
void recognize(int** neurons, int* exmpl, int * result, int recogn_elems_num, int data_dim);
void out(int** test_results, int test_set_dim, int recogn_elems_num);

int main(int argc, char** argv)
{
	int recogn_elems_num = 5,
		data_dim = 25,
		exmpls_num = 5,
		test_set_dim = 21;

	int** neurons = new int* [recogn_elems_num];
	for(int i = 0;i < recogn_elems_num; i++)
		neurons[i] = new int [data_dim];

	int** exmpls = new int* [exmpls_num];
	// for(int i = 0;i < exmpls_num;i++)
	// 	exmpls[i] = new int [data_dim];

	int** exmpls_values = new int* [exmpls_num];
	// for(int i = 0;i < exmpls_num;i++)
	// 	exmpls_values[i] = new int [recogn_elems_num];

	int** test_set = new int* [test_set_dim];
	// for(int i = 0;i < test_set_dim;i++)
	// 	test_set[i] = new int [data_dim];

	int** test_results = new int* [test_set_dim];
	for(int i = 0;i < test_set_dim;i++)
		test_results[i] = new int [recogn_elems_num];

	//обучающая выборка
	int exmpls_st[5][25] = {
		{
			 1, 1,-1, 1, 1,
			 1, 1, 1, 1, 1,
			 1,-1, 1,-1, 1,
			 1,-1,-1,-1, 1,
			 1,-1,-1,-1, 1
		},
		/*0'th symbol
			* * - * * 
			* * * * * 
			* - * - * 
			* - - - *
			* - - - *
		*/
		{
			-1,-1, 1,-1,-1,
			-1, 1,-1, 1,-1,
			-1, 1, 1, 1,-1,
			 1,-1,-1,-1, 1,
			 1,-1,-1,-1, 1
		},
		/*1'th symbol
			- - * - - 
			- * - * - 
			- * * * - 
			* - - - *
			* - - - *
		*/
		{
			 1, 1, 1, 1, 1,
			 1,-1,-1,-1,-1,
			 1, 1, 1, 1, 1,
			 1,-1,-1,-1,-1,
			 1, 1, 1, 1, 1
		},
		/*2'th symbol
			* * * * * 
			* - - - - 
			* * * * * 
			* - - - - 
			* * * * *
		*/
		{
			 1, 1, 1, 1,-1,
			 1,-1,-1,-1, 1,
			 1, 1, 1, 1, 1,
			 1,-1,-1,-1, 1,
			 1, 1, 1, 1,-1
		},
		/*3'th symbol
			* * * * - 
			* - - - * 
			* * * * * 
			* - - - * 
			* * * * -
		*/
		{
			-1,-1, 1,-1,-1,
			-1, 1,-1, 1,-1,
			-1, 1, 1, 1,-1,
			 1, 1, 1, 1, 1,
			 1,-1,-1,-1, 1
		},
		/*4'th symbol
			- - * - - 
			- * - * - 
			- * * * - 
			* * * * *
			* - - - *
		*/



	};
	for(int i = 0; i < exmpls_num;i++)
		exmpls[i] = exmpls_st[i]; 
 	
 	//значения для обучающей выборки
 	//1 соответствует правильной букве на iм месте в обучающей выборке
	int exmpls_values_st[5][5] = {
		{
			 1,
			-1,
			-1,
			-1,
			-1
		},
		{
			-1,
			 1,
			-1,
			-1,
			-1
		},
		{
			-1,
			-1,
			 1,
			-1,
			-1
		},
		{
			-1,
			-1,
			-1,
			 1,
			-1
		},
		{
			-1,
			-1,
			-1,
			-1,
			 1
		}
	};

	//костыли
	for(int i = 0; i < exmpls_num;i++)
		exmpls_values[i] = exmpls_values_st[i]; 

	//тестовая выборка
	int test_set_st[21][25] = {
		{
			 1, 1,-1, 1, 1,
			 1, 1, 1, 1, 1,
			 1,-1, 1,-1, 1,
			 1,-1,-1,-1, 1,
			 1,-1,-1,-1, 1
		},
		/*test 0
			* * - * * 
			* * * * * 
			* - * - * 
			* - - - *
			* - - - *
		*/
		{
			-1,-1, 1,-1,-1,
			-1, 1,-1, 1,-1,
			-1, 1, 1, 1,-1,
			 1,-1,-1,-1, 1,
			 1,-1,-1,-1, 1
		},
		/*test 1
			- - * - - 
			- * - * - 
			- * * * - 
			* - - - *
			* - - - *
		*/
		{
			 1, 1,-1, 1, 1,
			 1,-1, 1,-1, 1,
			 1,-1, 1,-1, 1,
			 1,-1,-1,-1, 1,
			 1,-1,-1,-1, 1
		},
		/*test 2
			* * - * * 
			* - * - * 
			* - * - * 
			* - - - *
			* - - - *
		*/
		{
			-1,-1, 1,-1,-1,
			-1, 1,-1, 1,-1,
			 1, 1, 1, 1, 1,
			 1,-1,-1,-1, 1,
			 1,-1,-1,-1, 1
		},
		/*test 3
			- - * - - 
			- * - * - 
			* * * * * 
			* - - - *
			* - - - *
		*/
		{
			 1, 1,-1, 1, 1,
			-1, 1,-1, 1,-1,
			-1,-1, 1,-1, 1,
			 1,-1,-1,-1, 1,
			 1,-1,-1,-1, 1
		},
		/*test 4
			* * - * * 
			- * - * - 
			- - * - * 
			* - - - *
			* - - - *
		*/
		{
			-1,-1, 1,-1,-1,
			 1,-1,-1,-1, 1,
			 1, 1, 1, 1, 1,
			 1,-1,-1,-1, 1,
			 1,-1,-1,-1, 1
		},
		/*test 5
			- - * - - 
			* - - - * 
			* * * * *
			* - - - *
			* - - - *
		*/
		{
			 1,-1,-1,-1, 1,
			 1,-1,-1,-1, 1,
			 1,-1,-1,-1, 1,
			 1,-1,-1,-1, 1,
			 1,-1,-1,-1, 1
		},
		/*test 6
			* - - - * 
			* - - - * 
			* - - - * 
			* - - - *
			* - - - *
		*/
		{
			-1,-1, 1,-1,-1,
			-1,-1,-1,-1,-1,
			-1,-1,-1,-1,-1,
			 1,-1,-1,-1, 1,
			 1,-1,-1,-1, 1
		},
		/*test 7
			- - * - - 
			- - - - - 
			- - - - - 
			- - - - - 
			* - - - *
		*/
		{
			 1, 1, 1, 1, 1,
			-1,-1, 1,-1,-1,
			-1,-1,-1,-1,-1,
			-1,-1,-1,-1,-1,
			-1,-1, 1,-1,-1
		},
		/*test 8
			* * * * * 
			- - * - - 
			- - - - - 
			- - - - - 
			- - * - - 
		*/
		{
			 1,-1,-1,-1, 1,
			 1,-1,-1, 1,-1,
			-1,-1,-1,-1,-1,
			 1,-1,-1, 1,-1,
			 1,-1,-1,-1, 1
		},
		/*test 9
			* - - - * 
			* - - * - 
			- - - - - 
			* - - * - 
			* - - - * 
		*/
		{
			-1,-1,-1,-1, 1,
			-1,-1,-1, 1,-1,
			-1,-1, 1,-1,-1,
			-1,-1,-1, 1,-1,
			 1,-1,-1,-1, 1

		},
		/*test 10
			- - - - * 
			- - - * - 
			- - * - - 
			- - - * - 
			* - - - * 
		*/
		{
			 1,-1,-1,-1,-1,
			-1, 1,-1,-1,-1,
			-1,-1, 1,-1,-1,
			-1, 1,-1,-1,-1,
			 1,-1,-1,-1,-1

		},
		/*test 11
			* - - - - 
			- * - - - 
			- - * - - 
			- * - - - 
			* - - - - 
		*/
		{
			-1,-1,-1,-1, 1,
			 1,-1,-1,-1,-1,
			 1, 1, 1,-1,-1,
			 1,-1,-1,-1,-1,
			-1,-1,-1,-1, 1
		},
		/*test 12
			- - - - * 
			* - - - - 
			* * * - - 
			* - - - - 
			- - - - * 
		*/
		{
			 1,-1,-1,-1, 1,
			-1, 1,-1, 1,-1,
			-1,-1, 1,-1,-1,
			-1, 1,-1, 1,-1,
			 1,-1,-1,-1, 1

		},
		/*test 13
			* - - - * 
			- * - * - 
			- - * - - 
			- * - * - 
			* - - - * 
		*/
		{
			 1,-1,-1,-1, 1,
			 1,-1,-1, 1,-1,
			 1, 1, 1,-1,-1,
			 1,-1,-1, 1,-1,
			 1,-1,-1,-1, 1
		},
		/*test 14
			* - - - * 
			* - - * - 
			* * * - - 
			* - - * - 
			* - - - * 
		*/
		{
			 1, 1, 1, 1, 1,
			-1,-1, 1,-1,-1,
			-1,-1, 1,-1,-1,
			-1,-1, 1,-1,-1,
			-1,-1, 1,-1,-1
		},
		/*test 15
			* * * * * 
			- - * - - 
			- - * - - 
			- - * - - 
			- - * - - 
		*/
		{
			 1, 1, 1, 1, 1,
			-1,-1,-1,-1,-1,
			-1,-1,-1,-1,-1,
			-1,-1,-1,-1,-1,
			-1,-1,-1,-1,-1
		},
		/*test 16
			* * * * * 
			- - - - - 
			- - - - - 
			- - - - - 
			- - - - - 
		*/
		{
			 1,-1,-1,-1, 1,
			-1,-1,-1,-1,-1,
			-1,-1, 1,-1,-1,
			-1,-1,-1,-1,-1,
			 1,-1,-1,-1, 1
		},
		/*test 17
			* - - - * 
			- - - - - 
			- - * - - 
			- - - - - 
			* - - - * 
		*/
		{
			 1, 1, 1, 1, 1,
			 1,-1,-1,-1,-1,
			 1, 1, 1, 1, 1,
			 1,-1,-1,-1,-1,
			 1, 1, 1, 1, 1
		},
		/* test 18
			* * * * * 
			* - - - - 
			* * * * * 
			* - - - - 
			* * * * *
		*/
		{
			 1, 1, 1, 1,-1,
			 1,-1,-1,-1, 1,
			 1, 1, 1, 1, 1,
			 1,-1,-1,-1, 1,
			 1, 1, 1, 1,-1
		},
		/* test 19
			* * * * - 
			* - - - * 
			* * * * * 
			* - - - * 
			* * * * -
		*/
        {
            1, 1, 1, 1, 1,
            1,-1,-1,-1,-1,
            1, 1, 1, 1, 1,
            1,-1,-1,-1, 1,
            1, 1, 1, 1, 1
        }
        /* test 20
         * * * * *
         * - - - -
         * * * * * 
         * - - - * 
         * * * * *
         */

	};

	for(int i = 0; i < test_set_dim;i++)
		test_set[i] = test_set_st[i]; 

    learn_network(neurons, exmpls, exmpls_values, recogn_elems_num, data_dim, exmpls_num);

    for(int i = 0; i < test_set_dim; i++)
    {
    	recognize(neurons, test_set[i], test_results[i], recogn_elems_num, data_dim);
    }

    out(test_results, test_set_dim, recogn_elems_num);

    delete[] test_results;
    delete[] neurons;
    // delete[] test_set;
    // delete[] exmpls_values;
    // delete[] exmpls;
	return 0;
}//end main()

void out(int** test_results, int test_set_dim, int recogn_elems_num)
{
	for(int i = 0; i < test_set_dim; i++)
	{
		for(int j = 0; j < recogn_elems_num; j++)
		{
			if(test_results[i][j] == 1)
			{
				std::cout<<"Test example "<<i<<" is "<<j<<"'th symbol.\n";
			}
		}
	}
}

//функция обновления(Правило Хебба)
int update_function(int prev_val, int x, int y)
{
	return prev_val + x*y;
}

//обучает единственный нейрон на одном примере
void learn_neuron(int* neuron, int* exmpl, int exmpl_value, int data_dim)
{
	//каждый коэффициент нейрона обновляется
	for(int i = 0; i < data_dim; i++)
		neuron[i] = update_function(neuron[i],exmpl[i],exmpl_value);
}

//обучает все нейроны на одном примере
void learn_exmpl(int** neurons, int* exmpl, int* exmpl_value, int recogn_elems_num, int data_dim)
{
	//один пример отправляется каждому из нейронов
	for(int i = 0; i < recogn_elems_num; i++)
	{
		learn_neuron(neurons[i], exmpl, exmpl_value[i], data_dim);
	}
}

//инициализация коэффициентов
void init_coef(int** neurons, int recogn_elems_num, int data_dim)
{
	for (int i = 0; i < recogn_elems_num; ++i)
	{
		for(int j = 0; j < data_dim; j++)
		{
			neurons[i][j] = 0;
		}
	}
}

//условие останова
bool validate_result(int** neurons, int** exmpls, int** exmpls_values,int recogn_elems_num, int data_dim, int exmpls_num,int error)
{
	int e = 0;
	int * result = new int[recogn_elems_num];
	for(int i = 0; i < exmpls_num; i++)
	{
		recognize(neurons, exmpls[i], result, recogn_elems_num, data_dim);
		for(int j = 0; j < recogn_elems_num; j++)
		{
			if(result[j] !=  exmpls_values[i][j])
			{
				e+=1;
			}
		}
	}
	delete[] result;
	return e >= error;
}

//обучает всю сеть
void learn_network(int** neurons, int** exmpls, int** exmpls_values, int recogn_elems_num, int data_dim, int exmpls_num)
{
	int k = 0;
	init_coef(neurons, recogn_elems_num, data_dim);
	while((validate_result(neurons, exmpls, exmpls_values, 
		recogn_elems_num,  data_dim,  exmpls_num, 0) && k < MAX_ITERS) ||k == 0)
	{
		for(int i = 0; i < exmpls_num; i++)
		{
			//каждый пример
			learn_exmpl(neurons, exmpls[i], exmpls_values[i], recogn_elems_num, data_dim);
		}
		k++;
	}
	std::cout<<"iters num "<<k<<std::endl;
}

//взвешенная сумма
int wsum(int* elems,int* coef, int elems_num)
{
	int s = 0; 
	for(int i = 0;i < elems_num; i++)
	{
		s+=elems[i]*coef[i];
	}
	return s;
}

//пороговая функция
int decision_func(int val)
{
	if(val > 0)
		return 1;
	return -1;
}

//распознает
void recognize(int** neurons, int* exmpl, int * result, int recogn_elems_num, int data_dim)
{
	for(int i = 0; i < recogn_elems_num; i++)
	{
		//находит взвешенную сумму
		//пропускает через функция принятия решений
		//result[i] значение вектора, такого как exmpls_values_st
		result[i] = decision_func(wsum(exmpl,neurons[i],data_dim));
	}
}
