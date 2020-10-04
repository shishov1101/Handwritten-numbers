import numpy
import scipy.special

class neuralNetwork:
    def __init__(self,inputnodes,hiddennotes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennotes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = (numpy.random.rand(self.hnodes,self.inodes)-0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2) .T
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors=targets-final_outputs
        hidden_errors=numpy.dot(self.who.T,output_errors)
        self.who+=self.lr*(numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs)))
        self.wih +=self.lr *(numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs)))
        pass
    def query(self,inputs_list):
        inputs=numpy.array(inputs_list,ndmin=2) .T
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        return final_outputs
        pass

input_nodes=784
output_nodes=10
hidden_nodes=100

learning_rate=0.1

n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
training_data_file=open("mnist_train_100.csv",'r')
training_data_list=training_data_file.readlines()
training_data_file.close()
for record in training_data_list:
    all_values=record.split(',')
    inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    targets=numpy.zeros(output_nodes)+0.01
    targets[int(all_values[0])]=0.99
    for i in range(100):
        n.train(inputs,targets)

test_data_file=open("mnist_test_10.csv",'r')
test_data_list=test_data_file.readlines()
test_data_file.close()
all_values=test_data_list[0].split(',')

print(all_values[0])
print(n.query((numpy.asfarray(all_values[1:])/255.0*0.99)+0.01))