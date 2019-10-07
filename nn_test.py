import random

from neural_network import NeuralNetwork

inputsize = 2
hiddensize = 4
outputsize = 1

nn = NeuralNetwork(inputsize,hiddensize,outputsize)

print(nn.weights_ih)
print(nn.weights_ho)
print(nn.bias_h)
print(nn.bias_o)

inp = [1]*inputsize
out = nn.predict(inp)
print(out)

inp = [[0,0],[0,1],[1,0],[1,1]]
out = [0, 1, 1, 0]

print('training...')

for _ in range(10000):
	i = random.choice([0,1,2,3])
	nn.train(inp[i],out[i])
	
#~ out = nn.predict([1, 0])
#~ print(out)

while True:
	a = input('Enter 1st input: ')
	b = input('Enter 2nd input: ')
	val = [int(a),int(b)]

	out = nn.predict(val)
	print(out)




