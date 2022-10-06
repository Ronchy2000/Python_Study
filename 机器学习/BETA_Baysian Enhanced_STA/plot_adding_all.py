import matplotlib.pyplot as plt
import xlrd
import xlwt
from matplotlib.pyplot import MultipleLocator


NUM_CIRCUIT = 14

# 获取baseline值
def get_base():
    excel_model_path = r'Result_add_random.xls'  # 文件路径
    rbook = xlrd.open_workbook(excel_model_path)  # 打开文件
    table = rbook.sheet_by_name('0')  # 索引sheet表

    cols = table.ncols

    baseline = []
    for j in range(1, cols):
        if (j % 2 != 0):
            baseline.append(0)
            continue
        baseline.append(table.cell_value(0, j))

    result = []
    for j in range(cols):
        if(j % 2 == 0):
            continue
        result.append(baseline[j])

    return result

def get_bench(acq,baseline):
    excel_model_path = r'result_add.xlsx'  # 文件路径
    rbook = xlrd.open_workbook(excel_model_path)  # 打开文件
    table = rbook.sheet_by_name(acq)  # 索引sheet表

    cols = table.ncols
    rows = table.nrows

    result = []
    for i in range(1, rows):
        sum = 0
        for j in range(1, cols):
            sum += (baseline[j-1] / table.cell_value(i, j))
        result.append(sum / NUM_CIRCUIT)

    return result


def main():
    baseline = get_base()
    target_MES = get_bench("mes",baseline)
    target_EI = get_bench("ei", baseline)
    target_UCB = get_bench("ucb0.1", baseline)

    # 获取随机数的加速比
    result = []
    excel_random_path = r'Result_add_random.xls'  # 文件路径
    random_rbook = xlrd.open_workbook(excel_random_path)  # 打开文件
    random_table_0 = random_rbook.sheet_by_name("0")  # 索引sheet表
    random_table_1 = random_rbook.sheet_by_name("1")  # 索引sheet表
    random_table_2 = random_rbook.sheet_by_name("2")  # 索引sheet表
    random_table_3 = random_rbook.sheet_by_name("3")  # 索引sheet表
    random_table_4 = random_rbook.sheet_by_name("4")  # 索引sheet表
    # random_table_111 = random_rbook.sheet_by_name("111")  # 索引sheet表
    res_0 = []
    res_1 = []
    res_2 = []
    res_3 = []
    res_4 = []


    random_cols = random_table_0.ncols
    random_rows = random_table_0.nrows
    for i in range(1,random_rows):
        row_sum = 0
        for j in range(1,int(random_cols/2) + 1):
            res_0.append(random_table_0.cell_value(i, j * 2))
            res_1.append(random_table_1.cell_value(i, j * 2))
            res_2.append(random_table_2.cell_value(i, j * 2))
            res_3.append(random_table_3.cell_value(i, j * 2))
            res_4.append(random_table_3.cell_value(i, j * 2))


            ran = (random_table_0.cell_value(i, j * 2) + random_table_1.cell_value(i, j * 2) + random_table_2.cell_value(i, j * 2)
                   + random_table_3.cell_value(i, j * 2) + random_table_4.cell_value(i, j * 2))/5
            # print(ran)
            row_sum = (baseline[j-1] / ran) + row_sum
        result.append(row_sum/NUM_CIRCUIT)
    # print(target)
    # print(result)

    # wbook = xlwt.Workbook()  # 新建工作簿
    # wtable = wbook.add_sheet('0')  # 添加工作页
    #
    #
    # for j in range(20):
    #     wtable.write(0, j, target_MES[j])
    # for j in range(20):
    #     wtable.write(1, j, target_EI[j])
    # for j in range(20):
    #     wtable.write(2, j, target_UCB[j])
    # for j in range(20):
    #     wtable.write(3, j, res_0[j])
    # for j in range(20):
    #     wtable.write(4, j, res_1[j])
    # for j in range(20):
    #     wtable.write(5, j, res_2[j])
    # for j in range(20):
    #     wtable.write(6, j, res_111[j])
    #
    # wbook.save(filename_or_stream='excel_test.xls')  # 一定要保存

    # title
    font1 = {'family': 'Arial',
             'fontweight': 'bold',
             'size': 16
             }
    # label
    font2 = {'family': 'Arial',
             'fontweight': 'bold',
             'size': 14
             }

    marker_size = 70

    # 创建2个列表
    # result[0] = 0.7
    plt.figure(figsize=(7, 4))

    x_values = [i for i in range(1,21)]
    y1_values = target_MES
    y2_values = target_EI
    y3_values = target_UCB
    y4_values = result
    # 使用scatter()绘制散点
    plt.scatter(x_values, y1_values, s=marker_size, c = "dodgerblue", marker='x', label='MES')
    plt.scatter(x_values, y2_values, s=marker_size, c= "orangered", marker='^',label='EI')
    plt.scatter(x_values, y3_values, s=marker_size, c= "limegreen",marker='s', label='UCB')
    plt.scatter(x_values, y4_values, s=marker_size, c= "darkorchid",label='Random')

    plt.legend(fontsize=16, loc="best", frameon = True, ncol = 2, columnspacing = 0.2)  # 添加图例

    plt.scatter(x_values, y1_values, s=marker_size, c="dodgerblue", marker='x', label='MES')

    # 设置图表标title()题并给坐标轴加上标签x/ylabel()
    # plt.title("Bo VS Random(seed=0/1/2/3/4)", fontsize=24)
    plt.xlabel("Epoch", font1)
    plt.ylabel("Average speed-up", font1)

    # 设置刻度标记大小tick_params()
    ax = plt.gca()
    plt.tick_params(axis='both', labelsize=14)
    # plt.legend(fontsize = 14, loc="best")  # 添加图例
    plt.grid()
    x_major_locator = MultipleLocator(5)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()
    # plt.savefig("plot_add_random.eps", bbox_inches='tight')



if __name__ == "__main__":
    main()