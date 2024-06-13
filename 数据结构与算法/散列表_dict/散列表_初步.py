# -*- coding: utf-8 -*-
# @Time    : 2022/3/14 18:51
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 散列表_初步.py
# @Software: PyCharm

dict={}
if __name__=='__main__':


  N=int (input())
  ans = 'NO'
  flag = False
  while N>0:
      N-=1
      word=input()
      if(not(flag)) :

          if(dict.get(word)!=None):
              flag=True
              ans=word
          else:
              dict[word]=True
  print(ans)