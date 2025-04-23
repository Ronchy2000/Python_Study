'''
假设我们有一系列测试用例，
每个测试用例会覆盖测试若干个代码模块。

我们用一个二维数组 cases 来表示这些测试用例的覆盖情况，
其中 cases[i][j]为1表示第i个测试用例覆盖了第j个模块，
为o则表示未覆盖。

求一个最少的测试用例集合，使得该集合能够覆盖所有代码模块。
返回最小集合的大小，
如果不存在能够覆盖所有代码模块的测试用例集合，
则返回-1（n,m≤1000）

输入描述：第一行输出2个整数i,j,
case[i].length = j, case[i][] = 0 or 1, 分别代表用例总数和代码模块总数.
从第二行开始的i航，每一行有j个整数(0 or 1)，每个整数之间用空格分隔，每一行代表一个用例对代码模块的覆盖情况。

输出描述：覆盖所有代码模块使用的最小用例集合大小int，如果不存在能够覆盖所有模块的测试用例集合则返回-1

样例：
输入
3 2
1 0
1 0
1 0
输出：
-1

Np-Hard问题。
'''
def min_cases(i,j, cases):
    uncovered_modules = set(range(j))
    selected_cases_count = 0

    while uncovered_modules:

        max_cover_num = 0






i, j = map(int,input().strip().split(' '))

cases = []

for row in range(i): #每行输入
    input_row= list(map(int,input().strip().split(' ')))
    cases.append(input_row)
# print(cases)

