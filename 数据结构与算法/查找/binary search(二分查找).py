#  列表数值是：
#  小    ------->      大
#  ----------------------
#  left      mid       right
#注意：left mid right 都指下标，第几个。
def binary_search(li,val):
    left = 0
    right = len(li)-1
    while left <= right :#候选区有值
        mid = (left +right) //2
        if li[mid] == val:
            return mid #返回下标
            #return li[mid] #返回值
        elif li[mid] > val:
            right = mid - 1
        else:
            left = mid + 1
    else:
        return None

list = [0,1,5,7,8,10,12,15,16]
#print(list)
print(binary_search(list,12))