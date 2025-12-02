#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <utility>

using namespace std;
using namespace cv;



class MLP {
public:
    ofstream logs;
    MLP(bool t,bool i,int r, int e):training(t),init(i), epochs(e) {
        //logs.open("backprop_log.txt",ios::app);
        registry=r;
        test_correct_guesses=0;
        test_wrong_guesses=0;
        train_correct_guesses=0;
        train_wrong_guesses=0;
        num_total_params;
        train_correct_result.resize(epochs);
        train_wrong_result.resize(epochs);


        if (training) {
            pair<int,int> info=load_mnist_images_labels(training_images_name,training_labels_name,MNIST_train);
            rows=info.first;
            cols=info.second;

        }
        else  {
            pair<int,int> info=load_mnist_images_labels(test_images_name,test_labels_name,MNIST_test);
            rows=info.first;
            cols=info.second;

        }
        input_vector_size=rows*cols;
        neuralNetwork={input_vector_size,256,64,10};
        num_total_layers=neuralNetwork.size()-1;
        num_samples=training?MNIST_train.size():MNIST_test.size();
        calc_total_params();
        //network creation
        layer_chaining();
    }
    ~MLP() {
        //if (logs.is_open()) logs.close();
    }
    void set_training(bool t) {
        training=t;
        if (training) {
            load_mnist_images_labels(training_images_name,training_labels_name,MNIST_train);
        }
        else {
            load_mnist_images_labels(test_images_name,test_labels_name,MNIST_test);
        }
        num_samples=training?MNIST_train.size():MNIST_test.size();
    }
    void set_initializing(bool i) {
        init=i;
    }
    void run() {

        if (training) {
            //loading weights & biases
            if (init) {
                for (auto &layer:main_layers) {
                    layer.initializer();
                }
                save_weights_and_biases(weights_biases_name,main_layers,registry,version,num_total_layers,num_total_neurons,num_total_params,neuralNetwork,activation_function_name);
            }
            else {
                load_weights_and_biases(weights_biases_name,main_layers,registry);
            }
            for (curr_epoch=0;curr_epoch<epochs;curr_epoch++) {
                 cout<<"Current epoch: "<<curr_epoch<<" : ";
                //preparing dataset
                vector<int>keys(MNIST_train.size());
                shuffler(keys,1+curr_epoch*(curr_epoch+1));
                double hyperloss=0.0;
                int progress=0;
                for (int i=0;i<num_samples;i++) {
                    if(i%(num_samples/10)==0){
                        cout<<progress<<"% ";
                        progress+=10;
                    }
                    //forward pass across the network
                    MNIST_image_and_label &sample=MNIST_train[keys[i]];

                    //feeding the network
                    main_layers[0].set_input(sample.image);
                    //forward pass across main layers
                    for (auto &layer:main_layers) {
                        layer.print_input(logs);
                        layer.forward_pass();
                    }
                    //forward pass across output layer and statistics measures
                    output->set_answer(sample.label);
                    tuple<bool,int,double,double> data=output->forward_pass();
                    bool true_prediction=get<0>(data);
                    int prediction=get<1>(data);
                    double prediction_weight=get<2>(data);
                    double loss=get<3>(data);
                    hyperloss+=loss;
                    //output->print_prediction(logs);
                    if (true_prediction) {
                        train_correct_result[curr_epoch].push_back(make_tuple(keys[i],prediction,prediction_weight,loss));
                        train_correct_guesses++;
                    }
                    else {
                        train_wrong_result[curr_epoch].push_back(make_tuple(keys[i],prediction,prediction_weight,loss));
                        train_wrong_guesses++;
                    }
                    //Backpropagation across the network
                    output->backpropagation();
                    int k=0;
                    for (auto it = main_layers.rbegin(); it != main_layers.rend(); ++it) {
                        (*it).backpropagation();
                    }

                if(i==num_samples)save_weights_and_biases(weights_biases_name,main_layers,registry,version,num_total_layers,num_total_neurons,num_total_params,neuralNetwork,activation_function_name);
                }
                // save after every epoch
                save_weights_and_biases(weights_biases_name,main_layers,registry,version,num_total_layers,num_total_neurons,num_total_params,neuralNetwork,activation_function_name);
                train_correct_percentage=(double)train_correct_guesses/(train_wrong_guesses+train_correct_guesses);
                accuracy.push_back(train_correct_percentage);
                train_correct_guesses=0;
                train_wrong_guesses=0;
                if(progress!=100)cout<<"100%";
                cout<<endl;
                cout<<"Epoch: "<<curr_epoch<<" train accuracy = "<<100*train_correct_percentage<<'%'<<" loss = "<<hyperloss/num_samples<<endl;
              }
            }

        else {
            cout<<registry<<endl;
            load_weights_and_biases(weights_biases_name,main_layers,registry);
            for (int i=0;i<num_samples;i++) {
                //forward pass across the network
                MNIST_image_and_label &sample=MNIST_test[i];
                //feeding the network
                main_layers[0].set_input(sample.image);
                //forward pass across main layers
                for (auto &layer:main_layers) {
                    layer.forward_pass();
                }
                //forward pass across output layer and statistics measures
                output->set_answer(sample.label);
                tuple<bool,int,double,double> data=output->forward_pass();
                bool true_prediction=get<0>(data);
                int prediction=get<1>(data);
                double prediction_weight=get<2>(data);
                double loss=get<3>(data);
                if (true_prediction) {
                    test_correct_result.push_back(make_tuple(i,prediction,prediction_weight,loss));
                    test_correct_guesses++;
                }
                else {
                    test_wrong_result.push_back(make_tuple(i,prediction,prediction_weight,loss));
                    test_wrong_guesses++;
                }

            }
        }
    }
    void train_stats(bool correct,int e) {
        cout<<"Epoch : "<<e<<" accuracy= "<<accuracy[e]<<endl;
        output->print_confusion_matrix();
        if (correct) {
            for (const auto &item : train_correct_result[e]) {
                show_mnist_image(MNIST_train,get<0>(item),get<1>(item),get<2>(item),get<3>(item),rows,cols);

            }
        }
        else {
            for (const auto &item : train_wrong_result[e]) {
                show_mnist_image(MNIST_train,get<0>(item),get<1>(item),get<2>(item),get<3>(item),rows,cols);

            }
        }
    }
    void test_stats(bool correct) {
        test_correct_percentage=(double)test_correct_guesses/(test_wrong_guesses+test_correct_guesses);
        cout<<"test_correct_percentage: "<<test_correct_percentage*100<<"%"<<endl;
        output->print_confusion_matrix();
        if (correct) {
            for (const auto &item : test_correct_result) {
                show_mnist_image(MNIST_test,get<0>(item),get<1>(item),get<2>(item),get<3>(item),rows,cols);

            }
        }
        else {
            for (const auto &item : test_wrong_result) {
                show_mnist_image(MNIST_test,get<0>(item),get<1>(item),get<2>(item),get<3>(item),rows,cols);

            }
        }
    }

    private:
    //System
        double version=1.0;
        int registry;
    //Architecture
        vector<int> neuralNetwork;//neurons at layer[i]=neuralNetwork[i]
        int num_total_neurons=330;
        int num_total_layers;
        int num_total_params;
        void calc_total_params() {
            num_total_params=0;
            for(int i=0;i<num_total_layers;i++) {
                num_total_params+=(neuralNetwork[i]+1)*neuralNetwork[i+1];
            }
        }
        string activation_function_name="ReLu";
        double learning_rate=0.0001;
        double clip_threshold=10000000.0;
        int batch=32;
        int epochs;
        int curr_epoch=0;
    //Files I/O
        string weights_biases_name="weights_biases";
        string training_images_name="data/MNIST/train-images.idx3-ubyte";
        string test_images_name="data/MNIST/t10k-images.idx3-ubyte";
        string training_labels_name="data/MNIST/train-labels.idx1-ubyte";
        string test_labels_name="data/MNIST/t10k-labels.idx1-ubyte";
    //MLP state
    bool training;
    bool init;
    //MNIST dataset
        struct MNIST_image_and_label{
            vector<double> image;
            int label;
        };
        vector<MNIST_image_and_label> MNIST_train;
        vector<MNIST_image_and_label> MNIST_test;

        int num_samples;
        int rows;
        int cols;
        int input_vector_size;
    //Statistics
        vector<vector<tuple<int,int,double,double>>> train_correct_result;
        vector<vector<tuple<int,int,double,double>>> train_wrong_result;
        int train_correct_guesses;
        int train_wrong_guesses;
        vector<double> accuracy;
        double train_correct_percentage;
        vector<tuple<int,int,double,double>> test_correct_result;
        vector<tuple<int,int,double,double>> test_wrong_result;
        int test_correct_guesses;
        int test_wrong_guesses;
        double test_correct_percentage;

        class layer {
            public:

                layer(int n_samples,int d,int neurons,int inputs,int batches,double eta,double clip,bool last=false): depth(d),num_neurons(neurons), num_inputs(inputs),num_batches(batches),learning_rate(eta),learning_coef(eta/(double)batches),clip_threshold(clip),last_layer(last) {
                    remaining_training=n_samples%num_batches;
                    quotient=n_samples/num_batches;
                    backprop_index=0;
                    vec_of_weights.resize(num_neurons,vector<double>(num_inputs,0.0));
                    biases.resize(num_neurons,0.0);
                    raw.resize(num_neurons,0.0);
                    output.resize(num_neurons,0.0);
                    grad_loss_to_raw.resize(num_neurons,0.0);
                    grad_loss_to_vec_of_weights.resize(num_neurons,vector<double>(num_inputs,0.0));
                    grad_loss_to_biases.resize(num_neurons,0.0);
                    grad_loss_to_input.resize(num_inputs,0.0);
                    batch_grad_loss_to_vec_of_weights.resize(num_neurons,vector<double>(num_inputs,0.0));
                    batch_grad_loss_to_biases.resize(num_neurons,0.0);
                    weight_normalizer.resize(num_neurons,1.0);
                    bias_normalizer=1.0;


                }
                ~layer() {}
                void initializer () {
                    he_initialize(num_inputs,num_neurons,vec_of_weights,biases);
                }//called by class MLP

                void insert_data(bool weight,bool bias,int neuron,int index,double value) {
                    if (weight && bias) throw logic_error("weight and bias cannot be inserted simultaneously!");
                    if (neuron < 0 || neuron >= num_neurons) throw out_of_range("Trying to insert a out-of-range neuron!");
                    if (index < 0 || index >= num_inputs) throw out_of_range("Trying to insert a out-of-range weight!");

                    if (weight) {
                        vec_of_weights[neuron][index] = value;
                    }
                    else if (bias) {
                        biases[neuron] = value;
                    }//called by function load_weights_and_biases
                }

                void set_input(vector<double> &in) {
                    input=&in;
                }//called by class MLP

                void set_grad_loss_to_output(vector<double> &grd_out) {
                    grad_loss_to_output = &grd_out;
                }//called by class MLP

                const int get_order() const{
                    return depth;
                }//called by function save_weights_and_biases

                const vector<vector<double>> &get_weights_of_all_neurons() const{
                    return vec_of_weights;
                }//called by function save_weights_and_biases

                const vector<double> &get_biases() const{
                    return biases;
                }//called by function save_weights_and_biases

                vector<double> &get_output(){
                    return output;
                }//called by class MLP

                vector<double> &get_grad_loss_to_input(){
                    return grad_loss_to_input;
                }//called by class MLP

                void forward_pass(){
                    for (int n = 0; n < num_neurons; n++) {
                        raw[n]=forward_weighting(*input,vec_of_weights[n],biases[n]);
                    }//raw calculation
                    if(!last_layer)RELU(raw,output);//output calculation
                    else output=raw;
                }//called by class MLP
                void print_input(ofstream &log){
                    log<<"******************************************************"<<endl;
                    log<< "Layer "<<depth<<endl;
                    log<< "Input: "<<endl;
                    for (int n = 0; n < num_inputs; n++) {
                         log<<(*input)[n]<<" ";
                    if(n%10==0)log<<endl;
                  }
                  log<<endl;
                  }
                void print_grad_to_output(ofstream &log)    {
                    log<<"******************************************************"<<endl;
                    log<< "Layer "<<depth<<endl;
                    log<<"grad_loss_to_output: "<<endl;;
                        for (int n = 0; n < num_neurons; n++) {
                        log<<(*grad_loss_to_output)[n]<<" ";
                        if(n%10==0)log<<endl;
                    }
                    log<<endl;
                }
                void print_grad_to_biases(ofstream &log){
                    log<<"******************************************************"<<endl;
                    log<< "Layer "<<depth<<endl;
                    log<<"grad to biases: "<<endl;;
                    for (int n = 0; n < num_neurons; n++) {
                        log<<grad_loss_to_biases[n]<<" ";
                        if(n%10==0)log<<endl;
                    }
                    log<<endl;
                }
                void print_grad_to_b_biases(ofstream &log){
                    log<<"******************************************************"<<endl;
                    log<< "Layer "<<depth<<endl;
                    log<<"grad to batch biases: "<<endl;;
                    for (int n = 0; n < num_neurons; n++) {
                        log<<batch_grad_loss_to_biases[n]<<" ";
                        if(n%10==0)log<<endl;
                    }
                    log<<endl;
                }
                void print_grad_to_raw(ofstream &log){
                        log<<"******************************************************"<<endl;
                        log<< "Layer "<<depth<<endl;
                        log<<"grad to raw: "<<endl;;
                        for (int n = 0; n < num_neurons; n++) {
                            log<<grad_loss_to_raw[n]<<" ";
                            if(n%10==0)log<<endl;
                        }
                        log<<endl;
                    }
            void print_grad_to_input(ofstream &log){
                    log<<"******************************************************"<<endl;
                    log<< "Layer "<<depth<<endl;
                    log<<"grad to input: "<<endl;;
                    for (int n = 0; n < num_inputs; n++) {
                        log<<grad_loss_to_input[n]<<" ";
                        if(n%10==0)log<<endl;
                    }
                    log<<endl;
                }
               void print_grad_to_weights(ofstream &log){
                    log<<"******************************************************"<<endl;
                    log<< "Layer "<<depth<<endl;
                    log<<"grad to weights: "<<endl;;
                    for(auto i = 0; i < num_neurons; i++) {
                    log<<'<'<<i<<'>'<<endl;
                    for (int n = 0; n < num_inputs; n++) {
                        log<<grad_loss_to_vec_of_weights[i][n]<<" ";
                        if(n%10==0)log<<endl;
                    }
                        log<<endl;
                    }

                }
            void print_grad_to_b_weights(ofstream &log){
                    log<<"******************************************************"<<endl;
                    log<< "Layer "<<depth<<endl;
                    log<<"grad to batch weights: "<<endl;;
                    for(auto i = 0; i < num_neurons; i++) {
                        log<<'<'<<i<<'>'<<endl;
                        for (int n = 0; n < num_inputs; n++) {
                            log<<batch_grad_loss_to_vec_of_weights[i][n]<<" ";
                            if(n%10==0)log<<endl;
                        }
                        log<<endl;
                    }

                }
                void backpropagation() {
                    if(!last_layer){
                    for (int n = 0; n < num_neurons; n++) {
                        if (raw[n]>0) {
                            grad_loss_to_raw[n]=(*grad_loss_to_output)[n];
                        }
                        else {
                            grad_loss_to_raw[n]=0.01*(*grad_loss_to_output)[n]; //Leaky RELU
                        }
                      }
                     }
                    else grad_loss_to_raw=(*grad_loss_to_output);
                    //calculated grad_loss_to_raw
                    grad_loss_to_biases=grad_loss_to_raw;//calculated grad_loss_to_biases
                    for (int n=0; n < num_neurons; n++) {
                        scalar_vector_product(grad_loss_to_raw[n],*input,grad_loss_to_vec_of_weights[n]);
                    }//calculated grad_loss_to_vec_of_weights
                    horizontal_vector_dot_matrix(grad_loss_to_raw,vec_of_weights,grad_loss_to_input);//calculated grad_loss_to_input
                    //update of batch_grads
                        A_update_with_B (batch_grad_loss_to_biases,grad_loss_to_biases);
                        for (int n = 0; n < num_neurons; n++) {
                            A_update_with_B (batch_grad_loss_to_vec_of_weights[n],grad_loss_to_vec_of_weights[n]);
                        }
                    backprop_index++;
                    int count=0;

                    if (backprop_index == num_batches) {
                        backprop_index = 0;
                        learning_coef = learning_rate / (double) num_batches;
                        learning();
                        set_batch_grads_to_zero();
                        count++;
                    }
                    else if (quotient==count && remaining_training != 0 && backprop_index == remaining_training) {
                        backprop_index = 0;
                        learning_coef = learning_rate / (double) remaining_training;
                        learning();
                        set_batch_grads_to_zero();
                        learning_coef = learning_rate / (double) num_batches; // RESET
                    }
                }//called by class MLP


            private:
                int depth;
                int num_neurons;
                int num_inputs;
                vector<vector<double>> vec_of_weights;//to be initialised at num_neurons
                vector<double> biases;//to be initialised at num_neurons
                vector<double> *input;
                vector<double> raw;
                vector<double> output;
                vector<double> *grad_loss_to_output;
                vector<double> grad_loss_to_raw;
                vector<vector<double>> grad_loss_to_vec_of_weights;
                vector<double> grad_loss_to_biases;
                vector<double> grad_loss_to_input;
                vector<vector<double>> batch_grad_loss_to_vec_of_weights;
                vector<double> batch_grad_loss_to_biases;
                int num_batches;
                int remaining_training;
                int quotient;
                int backprop_index;
                double learning_rate;
                double learning_coef;
                double clip_threshold;
                double bias_normalizer;
                vector<double> weight_normalizer;
                bool last_layer;


            //functions
                void set_batch_grads_to_zero(){
                    for (int n = 0; n < num_neurons; n++) {
                        fill(batch_grad_loss_to_vec_of_weights[n].begin(), batch_grad_loss_to_vec_of_weights[n].end(), 0.0);
                    }
                    fill(batch_grad_loss_to_biases.begin(), batch_grad_loss_to_biases.end(), 0.0);
                }
                void grad_clipper(){
                     double norm=0.0;
                     bias_normalizer=1.0;
                     fill(weight_normalizer.begin(), weight_normalizer.end(), 1.0);
                    for (auto & item:batch_grad_loss_to_biases){
                            norm+=item*item;
                        }
                    norm=sqrt(norm);
                    if (norm>clip_threshold) {
                    bias_normalizer=clip_threshold/norm;
                    }

                    for (auto it=batch_grad_loss_to_vec_of_weights.begin(); it!=batch_grad_loss_to_vec_of_weights.end(); it++) {
                    norm=0.0;
                    for (auto & item:*it){
                            norm+=item*item;
                    }
                    norm=sqrt(norm);
                    if (norm>clip_threshold) {
                    weight_normalizer[distance(batch_grad_loss_to_vec_of_weights.begin(),it)]=clip_threshold/norm;
                   }
                 }
                }
                void learning () {
                        grad_clipper();
                        scalar_vector_product_in_place(bias_normalizer*learning_coef,batch_grad_loss_to_biases);//normalization
                        A_minus_B_in_place(biases,batch_grad_loss_to_biases);
                        for (int n = 0; n < num_neurons; n++) {
                            scalar_vector_product_in_place(weight_normalizer[n]*learning_coef,batch_grad_loss_to_vec_of_weights[n]);//normalization
                            A_minus_B_in_place(vec_of_weights[n],batch_grad_loss_to_vec_of_weights[n]);
                        }
                    }
                void he_initialize(int num_inputs,int num_outputs,vector<vector<double> > &WEIGHTS,vector<double>&BIASES) {

                    random_device rd;
                    mt19937 gen(rd());
                    double stddev=sqrt(2.0/num_inputs);
                    normal_distribution<double> distribution(0.0,stddev);
                    for(int i=0;i<num_outputs;i++) {
                        for(int j=0;j<num_inputs;j++) {
                            WEIGHTS[i][j]=distribution(gen);
                        }
                    }
                    fill(BIASES.begin(), BIASES.end(), 0.0);
                }
                double forward_weighting(const vector<double> &INPUT,const vector<double> &W,double bias) {
                    double sum = 0.0;
                    for (size_t i = 0; i <INPUT.size(); i++) {
                        sum += INPUT[i]*W[i];
                    }
                    sum += bias;
                    return sum;
                }

                void RELU (const vector<double> &RAW,vector<double> &result) {
                    for (size_t i = 0; i <RAW.size(); i++) {
                        if(RAW[i]>0)result[i]=RAW[i];
                        else result[i]=0.01*RAW[i];
                    }
                }
                inline void scalar_vector_product(double b,const vector<double> &A,vector<double> &result) {
                    for (size_t i = 0; i < A.size(); i++) {
                        result[i]=A[i]*b;
                    }
                }
                inline void scalar_vector_product_in_place(double b,vector<double> &A) {
                    for (size_t i = 0; i < A.size(); i++) {
                        A[i]*=b;
                    }
                }
                void horizontal_vector_dot_matrix(const vector<double> &vec,const vector<vector<double> > &M,vector<double> &result) {
                    int rows = M.size();
                    int cols = M[0].size();
                    int size_vec=vec.size();
                    if (rows != size_vec) throw logic_error("impossible multiplication");
                    //size of result=cols=size of input
                    result.assign(cols, 0.0);
                    for (size_t i = 0; i < cols; i++) {
                        double sum = 0.0;
                        for (size_t j = 0; j < size_vec; j++) {
                            sum+=M[j][i]*vec[j];
                        }
                        result[i]=sum;
                    }
                }
                inline void A_update_with_B(vector<double> &A,const vector<double> &B) {
                    for (size_t i = 0; i < A.size(); i++) {
                        A[i]+=B[i];
                    }
                }
                inline void A_minus_B_in_place(vector<double> &A,const vector<double> &B) {
                    for (size_t i = 0; i < A.size(); i++) {
                        A[i]-=B[i];
                    }
                }
    };

    class output_layer {
        public:
            output_layer(int inputs,const vector<double> &in): num_inputs(inputs),input(in) {
                prediction.resize(num_inputs,0.0);
                answer.resize(num_inputs,0.0);
                confusion_matrix.resize(num_inputs,vector<int>(num_inputs,0));
                grad_loss_to_input.resize(num_inputs,0.0);

            }
            ~output_layer() {}
            void set_answer(int a) {
                ans=a;
                fill(answer.begin(), answer.end(), 0.0);
                answer[ans]=1.0;
            }//called by class MLP

            vector<double> &get_grad_loss_to_input() {
                return grad_loss_to_input;
            }//called by class MLP

            tuple<bool,int,double,double> forward_pass() {
                pair<int,double> result=softmax(input,prediction);
                int max_index=result.first;
                double best=result.second;

                bool true_prediction=false;
                if (ans==max_index) true_prediction=true;
                loss=cross_entropy_loss(prediction,answer);
                confusion_matrix[ans][max_index]++;
                return make_tuple(true_prediction,max_index,best,loss);
            }//called by class MLP

            void backpropagation() {
                A_minus_B(prediction,answer,grad_loss_to_input);
            }//called by class MLP

            void print_confusion_matrix() const{
                    cout << "CONFUSION MATRIX:" << endl;
                    for (int i = 0; i < num_inputs; i++) {
                        for (int j = 0; j < num_inputs; j++) {
                            cout << '['<<i<<']'<<'['<<j<<"] = "<<confusion_matrix[i][j]<<endl;
                        }
                    }
                }//called by class MLP


            void clear_confusion_matrix() {
                confusion_matrix.clear();
            }
            void print_prediction(ofstream &log) const {
                for (int i = 0; i < num_inputs; i++) {
                    log<<i<<" = "<<prediction[i]<<endl;
                }
            }


        private:
            int num_inputs;
            const vector<double> &input;
            vector<double> prediction;
            vector<double> answer;
            int ans;
            vector<vector<int>> confusion_matrix ;
            double loss;
            vector<double> grad_loss_to_input;

            //functions
            pair<int,double> softmax(const vector<double> &OUTPUT,vector<double> &result) {

                double max_val = *max_element(OUTPUT.begin(), OUTPUT.end());
                double sum = 0.0;
                int max_index = 0;
                double greatest_pred=0.0;
                for (size_t i = 0; i < OUTPUT.size(); i++) {
                    result[i]=(exp(OUTPUT[i]-max_val));
                    sum+=result[i];
                }
                for (size_t i = 0; i < OUTPUT.size(); i++) {
                    result[i] /= sum;
                    if (result[i] > greatest_pred) {
                        greatest_pred = result[i];
                        max_index = i;
                    }
                }
                return make_pair(max_index,greatest_pred);
            }

            double cross_entropy_loss(const vector<double> &PREDICTION,const vector<double> &ANSWER) {
                double sum = 0.0;
                const double epsilon = 1e-15;
                for (size_t i = 0; i < PREDICTION.size(); i++) {
                    sum-=ANSWER[i]*log(PREDICTION[i]+epsilon);
                }
                return sum;
            }
            inline void A_minus_B(vector<double> &A,const vector<double> &B,vector<double>&result) {
                for (size_t i = 0; i < A.size(); i++) {
                    result[i]=A[i]-B[i];
            }
        }
    };

    //Layer chaining
    vector<layer> main_layers;
    unique_ptr<output_layer> output;
    //functions
        void layer_chaining() {
            for (int i = 0; i < num_total_layers-1; i++) {
                main_layers.emplace_back(num_samples,i,neuralNetwork[i+1],neuralNetwork[i],batch,learning_rate,clip_threshold);
            }
            main_layers.emplace_back(num_samples,num_total_layers-1,neuralNetwork[num_total_layers],neuralNetwork[num_total_layers-1],batch,learning_rate,clip_threshold,true);
            main_layers[1].set_input(main_layers[0].get_output());
            main_layers[2].set_input(main_layers[1].get_output());
            output=make_unique<output_layer>(neuralNetwork[neuralNetwork.size()-1],main_layers[2].get_output());
            main_layers[2].set_grad_loss_to_output(output->get_grad_loss_to_input());
            main_layers[1].set_grad_loss_to_output(main_layers[2].get_grad_loss_to_input());
            main_layers[0].set_grad_loss_to_output(main_layers[1].get_grad_loss_to_input());

        }
        void load_weights_and_biases(const string &filename, vector<layer> &LAYERS,int &registry){
            ifstream file(filename+"_"+to_string(registry)+".txt");
            string line,bin;
            if (!file.is_open()) throw runtime_error("Error opening file(load mode)");
            while (getline(file, line)) {
                if (line.find("START") != string::npos)
                    break;
            }
            getline(file, line);
            stringstream ss(line);
            ss >>registry; //Example : 1234
            while(getline(file,line)){
                bool weight=false,bias=false;
                int Layer_order,Neuron_order;
                stringstream ss(line);
                bool failed_reading=line.empty();
                string type;
                (ss>>type);
                if(type =="W")weight=true;
                else if(type=="B")bias=true;
                else if(failed_reading)continue;
                else throw runtime_error("Could not parse valid data type (weight or bias)");
                ss>>Layer_order;
                if (weight)ss>>Neuron_order;
                string eq, bracket;
                ss >> eq >> bracket;//Discarding "=",Discarding "["
                if (eq != "=" || bracket != "[")
                       throw runtime_error("Warning: unexpected format after layer/neuron indices");
                layer &temp = LAYERS[Layer_order];
                int index=0;
                string value;
                bool done=false;
                while(!done){
                    while(ss>>value) {
                        if(value=="]") {
                            done=true;
                            break;
                        }
                        temp.insert_data(weight,bias,Neuron_order,index,stod(value));
                        index++;
                    }
                    if (!done && getline(file, line))
                        ss = stringstream(line);
                    else
                        break;
                        }
                    }
                file.close();
            }

        void save_weights_and_biases (const string &filename,const vector<layer> &LAYERS,int &registry,int version,int layers_number,int neurons_number,int weights_number,vector<int>NEURO_MAP,string activation_function) {
            ofstream file(filename+"_"+to_string(registry)+".txt");
            if (!file.is_open()) throw runtime_error("Error opening file (save mode)");

            file  << "AI MNIST MLP "<<version<<" "<<endl;
            file<<endl;
            file << "ECE NTUA"<<endl;
            file << "George Markantonatos"<<endl;
            file<<endl;
            file << "Network's specifications :"<<endl;
            file << "Layers: "<<layers_number<<endl;
            file << "Neurons:"<<neurons_number<<endl;
            file << "Weights:"<<weights_number<<endl;
            file << "Structure of neurons in layers :";
            for (const auto &neuron_num : NEURO_MAP) {
                file <<" "<< neuron_num;
            }
            file<<endl;
            file<<"Activation function:"<<activation_function<<endl;
            file<<endl;
            file << "Weights & Biases record "<<endl;
            file << "================================================"<<endl;
            file<< "START"<<endl;
            file <<++registry<<endl;
            for (const auto &layer : LAYERS) {
                for (auto it= layer.get_weights_of_all_neurons().begin(); it != layer.get_weights_of_all_neurons().end(); it++) {
                file<<"W "<<layer.get_order()<<" "<<distance(layer.get_weights_of_all_neurons().begin(),it)<<" = [ ";
                    int count_for_change_line=0;
                    for (const auto &weight_value:*it) {
                        file<<weight_value<<" ";
                        count_for_change_line++;
                        if (count_for_change_line >= 10) {
                            file<<endl;
                            count_for_change_line=0;
                        }
                }
                    file<<"]"<<endl;
                }//End of saving weights
                const vector<double> &biases=layer.get_biases();
                file<<"B   "<<layer.get_order()<<" = [ ";
                int count_for_change_line=0;
                for (const auto &bias_value:biases) {
                    file<<bias_value<<" ";
                    count_for_change_line++;
                    if (count_for_change_line >= 10) {
                        file<<endl;
                        count_for_change_line=0;
                    }
                }
                file<<"]"<<endl;
                file<<endl;
            }
            file.close();
        }



        pair<int,int> load_mnist_images_labels(const string &filename_images,const string &filename_labels,vector<MNIST_image_and_label> &images_labels) {
            ifstream file_img(filename_images,ios::binary);
            if (!file_img.is_open()) throw runtime_error("Error opening file (load mnist images)");
            uint32_t images_magic_number;
            file_img.read((char*)&images_magic_number,4);//typecasting in order to read the doc as it is
            images_magic_number=__builtin_bswap32(images_magic_number);
            if (images_magic_number != 2051) throw runtime_error("Not a valid MNIST images file");//special configuration for mnist docs

            uint32_t num_images;
            uint32_t num_rows;
            uint32_t num_cols;

            file_img.read((char*)&num_images,4);
            file_img.read((char*)&num_rows,4);
            file_img.read((char*)&num_cols,4);

            num_images=__builtin_bswap32(num_images);
            num_rows=__builtin_bswap32(num_rows);
            num_cols=__builtin_bswap32(num_cols);
            uint32_t image_size=num_rows*num_cols;

            ifstream file_lbl(filename_labels,ios::binary);
            if (!file_lbl.is_open()) throw runtime_error("Error opening file (load mnist labels)");
            uint32_t lbl_magic_number;

            file_lbl.read((char*)&lbl_magic_number,4);
            lbl_magic_number=__builtin_bswap32(lbl_magic_number);
            if (lbl_magic_number != 2049) throw runtime_error("Not a valid MNIST labels file");

            uint32_t num_labels;
            file_lbl.read((char*)&num_labels,4);
            num_labels=__builtin_bswap32(num_labels);

            if (num_images!=num_labels)throw logic_error("The number of images does not match the number of labels");
            int num=num_images;
            images_labels.reserve(num);
            int buffered_set=3000;

            for (uint32_t i = 0; i < num; i+=buffered_set) {
                int current_batch = min(buffered_set, (int)(num - i));
                vector<uint8_t>buffer_img(current_batch*image_size);//read all pixels at one batch
                file_img.read((char*)(buffer_img.data()),current_batch*image_size);
                if (!file_img) throw runtime_error("Error reading image data");

                vector<uint8_t>buffer_lbl(current_batch);
                file_lbl.read((char*)(buffer_lbl.data()),current_batch);
                if (!file_lbl) throw runtime_error("Error reading label data");

                for (int k=0; k<current_batch; k++) {
                    MNIST_image_and_label temp;
                    temp.image.resize(image_size);
                    for (uint32_t j = 0; j < image_size; j++) {
                        temp.image[j]=buffer_img[j+k*image_size]/255.0;// normalize to [0,1]
                    }
                    temp.label=buffer_lbl[k];
                    images_labels.push_back(temp);
                }
            }
            return {num_rows,num_cols};
          }

        void show_mnist_image(const vector<MNIST_image_and_label>& vec,int idx,int pred,double pred_weight,double loss, int rows, int cols) {
            if (vec[idx].image.size() != rows * cols) {
                throw logic_error("Image size does not match rows*cols");
            }

            Mat img(rows, cols, CV_8UC1);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double pixel = vec[idx].image[i * cols + j];//loc function: vector->nxn matrix
                    img.at<uchar>(i, j) = static_cast<uchar>(pixel * 255.0);//pixel by pixel update
                }
            }

            // Zoom
            Mat resized;
            resize(img, resized, Size(), 10, 10, INTER_NEAREST);

            int bar_height = 40;
            Mat bar = Mat::ones(bar_height, resized.cols, CV_8U) * 255;

            Mat final_image;
            vconcat(resized, bar, final_image);

            string text = "MLP: " + to_string(pred)+'_'+to_string(pred_weight)+'_'+to_string(loss);
            putText(final_image, text, Point(10, resized.rows + 25),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
            imshow("MNIST IMAGE", final_image);
            waitKey(0);
        }


        void save_mnist_image(const vector<double>& image, int rows, int cols,int pred,double pred_weight,double loss,const string& filename) {
            if (image.size() != rows * cols)
                throw logic_error("Image size does not match rows*cols");

            Mat img(rows, cols, CV_8UC1);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double pixel = image[i * cols + j];//loc function: vector->nxn matrix
                    img.at<uchar>(i, j) = static_cast<uchar>(pixel * 255.0);//pixel by pixel update
                }
            }

            // Zoom
            Mat resized;
            resize(img, resized, Size(), 10, 10, INTER_NEAREST);

            int bar_height = 40;
            Mat bar = Mat::ones(bar_height, resized.cols, CV_8U) * 255;

            Mat final_image;
            vconcat(resized, bar, final_image);

            string text = "MLP: " + to_string(pred)+'_'+to_string(pred_weight)+'_'+to_string(loss);
            putText(final_image, text, Point(10, resized.rows + 25),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0), 2);
            imwrite(filename, final_image);
        }

        void shuffler(vector<int> &keys,unsigned int seed) {
            iota(keys.begin(), keys.end(), 0);
            mt19937 gen(seed);
            shuffle(keys.begin(), keys.end(), gen);
        }
};


int main() {
    cout<< "Welcome to the MLP program!"<<endl;
    bool train=false, initialize=false, train_stats=false, test_stats=false;
    int registry=0, tested_epoch, num_epochs=30;
    cout<<"Train or test ? (write 'TRAIN' or 'TEST' respectively)"<<endl;
    string f_answer;
    cin>>f_answer;
    while(true){
        if (f_answer=="TRAIN") {
            train=true;
            while (true) {
                cout<<"Initialize weights or load  ? (write 'INITIALIZE' or 'LOAD' respectively)"<<endl;
                string s_answer;
                cin>>s_answer;
                if (s_answer=="INITIALIZE") {
                    initialize=true;
                    break;
                }
                if (s_answer=="LOAD") {
                    cout<< "Please enter a valid weight file registry (the unique number in the filename of weights)"<<endl;
                    cin>>registry;
                    break;

                }
            }
            while (true) {
                cout<<"Please define the number of epochs with a positive integer."<<endl;
                cin>>num_epochs;
                if (num_epochs>0) break;
            }

            while(true){
                cout<<"Would you like training stats for correct or false answers ? (write CORRECT or FALSE respectively)"<<endl;
                string t_answer;
                cin>>t_answer;
                if (t_answer=="CORRECT") {
                    while(true){
                        train_stats=true;
                        cout<<"Type which epoch you would like to assess "<<endl;
                        cin>>tested_epoch;
                        if(0<=tested_epoch && tested_epoch<num_epochs)break;
                    }
                    break;
                }
                if (t_answer=="FALSE") {
                    train_stats=false;
                    break;
                }
            }
            break;
         }
        else if (f_answer=="TEST") {
            cout<< "Please enter a valid weight file registry (the unique number in the filename of weights)"<<endl;
            cin>>registry;
            while(true){
                cout<<"Would you like test stats for correct or false answers ? (write CORRECT or FALSE respectively)"<<endl;
                string t_answer;
                cin>>t_answer;
                if (t_answer=="CORRECT") {
                    test_stats=true;
                    break;
                }
                if (t_answer=="FALSE") {
                    test_stats=false;
                    break;
                }
            }
            break;
        }
    }
    cout<<"Please wait.Process in progress..."<<endl;
    MLP a(train,initialize,registry,num_epochs);
    a.run();
    if(train)a.train_stats(train_stats,tested_epoch);
    else a.test_stats(test_stats);
}

