"""
Created on 2017.6.11

@author: tfygg
"""

import lenet
import lenetseq
import utils

# Net
net = lenet.LeNet()
print(net)

for index, param in enumerate(net.parameters()):
    print(list(param.data))
    print(type(param.data), param.size())
    print index, "-->", param


print(net.state_dict())
print(net.state_dict().keys())

for key  in net.state_dict():
    print key, 'corresponds to', list(net.state_dict()[key])


#NetSeq
netSeq = lenetseq.LeNetSeq()
print(netSeq)

utils.initNetParams(netSeq)

for key in netSeq.state_dict():
    print key, 'corresponds to', list(netSeq.state_dict()[key])

