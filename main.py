# ----------------------
#   Code by Radimich   
#   Date: 2024
#   Land:Larnevsk     
# ----------------------
import torch

my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#10lesson_perceptron

def act(x):
    return 0 if x<=0 else 1


w_hidden = torch.FloatTensor([[1,1,-1.5],[1,1,-0.5]])
w_out = torch.FloatTensor([-1,1,-0.5])

data_x = [0.8,0.9]#input data x1,x2
x = torch.FloatTensor(data_x+[1])

z_hidden = torch.matmul(w_hidden,x)
print(z_hidden)
u_hidden = torch.FloatTensor([act(x) for x in z_hidden]+[1])
print(u_hidden)

z_out = torch.dot(w_out,u_hidden)
y = act(z_out)

print(y)


#---------------------------------------------------------------------
# def act(x):
#     return 0 if x<0.5 else 1
#
# def go(house, rock, attr):
#     X = torch.tensor([house, rock, attr], dtype=torch.float32, device=my_device)
#     Wh = torch.tensor([[0.3, 0.3, 0], [0.4, -0.5, 1]], device=my_device)#matrix 2x3
#     Wout = torch.tensor([-1.0, 1.0], device=my_device)#vector 1x2
#
#     Zh = torch.mv(Wh, X)#sum input hidden l
#     print(f'Значение сумм на нейронах скрытого слоя: {Zh}')
#
#     Uh = torch.tensor([act(x) for x in Zh ], dtype=torch.float32, device=my_device)
#     print(f'Значение сумм на выходах скрытого слоя: {Uh}')
#
#     Zout = torch.dot(Wout, Uh)
#     Y = act(Zout)
#
#     print(f'Выходное значение НС: {Y}')
#     return Y
#
# house = 1
# rock = 0
# attr = 1
#
# res = go(house, rock, attr)
# if res==1:
#     print("Ты мне нравишься!")
# else:
#     print('Созвонимся')