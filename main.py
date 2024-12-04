# ----------------------
#   Code by Radimich   
#   Date: 2024
#   Land:Larnevsk     
# ----------------------
import torch

def act(x):
    return 0 if x<0.5 else 1

def go(house, rock, attr):
    X = torch.tensor([house, rock, attr], dtype=torch.float32)
    Wh = torch.tensor([[0.3, 0.3, 0], [0.4, -0.5, 1]])#matrix 2x3
    Wout = torch.tensor([-1.0, 1.0])#vector 1x2

    Zh = torch.mv(Wh, X)#sum input hidden l
    print(f'Значение сумм на нейронах скрытого слоя: {Zh}')

    Uh = torch.tensor([act(x) for x in Zh ], dtype=torch.float32)
    print(f'Значение сумм на выходах скрытого слоя: {Uh}')

    Zout = torch.dot(Wout, Uh)
    Y = act(Zout)

    print(f'Выходное значение НС: {Y}')
    return Y

house = 1
rock = 0
attr = 1

res = go(house, rock, attr)
if res==1:
    print("Ты мне нравишься!")
else:
    print('Созвонимся')