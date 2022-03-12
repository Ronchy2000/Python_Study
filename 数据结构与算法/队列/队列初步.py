# -*- coding: utf-8 -*-
# @Time    : 2022/3/8 19:00
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 队列初步.py
# @Software: PyCharm
'''
FIFO: first in first out
'''
Vqueue ,Nqueue= [] , []
Vhead ,Nhead = 0, 0
Vtail ,Ntail = 0 ,0

def inque(name, type):
  global Vhead, Vtail, Nhead, Ntail, Vqueue , Nqueue

  if (type == 'V'):
      Vqueue.append(name)
      Vtail += 1
  else:
      Nqueue.append(name)
      Ntail += 1
      # print(Vqueue)

def getHead(type):
  global Vhead, Vtail, Nhead, Ntail,Vqueue ,Nqueue

  if (type == 'V'):
      # print(Vhead)
      return Vqueue[Vhead]
  else:
      # print(Nhead)
      return Nqueue[Nhead]

def outque(type):
  global Vhead, Vtail, Nhead, Ntail,Vqueue ,Nqueue
  if (type == 'V'):

      if (Vhead == Vtail):
          return None
      else:
          s = getHead(type)
          Vhead += 1
          return s
  else:
      if (Nhead == Ntail):
          return None
      else:
          s= getHead(type)
          Nhead += 1
          return  s

M = 0
if __name__ == '__main__':
  M = int(input())
  while M > 0:
      M -= 1
      op = input().split() #把输入的字串划分
      print(op[0])
      if op[0] == 'IN':
          inque(op[1], op[2])
          # print('in')
      else:
          outque(op[1])
          # print('out')
      # print("VVVVV",Vqueue)
      # print("NNNN",Nqueue)
      # print(M)

  s = outque('V')
  while s!=None:
      print(s)
      s = outque('V')

  s = outque('N')
  while s != None:
      print(s)
      s = outque('N')
