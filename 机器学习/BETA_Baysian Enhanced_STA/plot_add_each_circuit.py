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
    name = []
    for j in range(1, cols):
        if (j % 2 != 0):
            baseline.append(0)
            name.append(table.cell_value(0, j))
            continue
        baseline.append(table.cell_value(0, j))

    result = []
    for j in range(cols):
        if(j % 2 == 0):
            continue
        result.append(baseline[j])

    return result, name

def get_bench(acq, baseline, cir_id):
    excel_model_path = r'result_add.xls'  # 文件路径
    rbook = xlrd.open_workbook(excel_model_path)  # 打开文件
    table = rbook.sheet_by_name(acq)  # 索引sheet表

    rows = table.nrows

    result = []
    for i in range(1, rows):
        temp = (baseline[cir_id] / table.cell_value(i, cir_id + 1))
        result.append(temp)

    return result


def get_baseline_opt(excel, baseline, id):
    rbook = xlrd.open_workbook(excel)  # 打开文件
    sheet = rbook.sheet_by_name("data")

    res = []
    nrows = sheet.nrows

    for i in range(1, nrows):
        profit = (baseline[id] / sheet.cell_value(i, id+1))
        if profit < 1:
            profit = 1
        res.append(profit)

    return res


def main():
    for each_id in range(NUM_CIRCUIT):

        baseline, circuit_name = get_base()
        print(circuit_name[each_id])
        target_MES = get_bench("mes", baseline, each_id)
        target_EI = get_bench("ei", baseline, each_id)
        target_UCB = get_bench("ucb0.1", baseline, each_id)

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
            res_0.append(random_table_0.cell_value(i, (each_id + 1) * 2))
            res_1.append(random_table_1.cell_value(i, (each_id + 1) * 2))
            res_2.append(random_table_2.cell_value(i, (each_id + 1) * 2))
            res_3.append(random_table_3.cell_value(i, (each_id + 1) * 2))
            res_4.append(random_table_3.cell_value(i, (each_id + 1) * 2))


            ran = (random_table_0.cell_value(i, (each_id + 1) * 2) + random_table_1.cell_value(i, (each_id + 1) * 2) + random_table_2.cell_value(i, (each_id + 1) * 2)
                   + random_table_3.cell_value(i, (each_id + 1) * 2) + random_table_4.cell_value(i, (each_id + 1) * 2))/5
            # print(ran)
            result.append(baseline[each_id] / ran)
        # print(target)
        # print(result)

        # yichuan
        result_yc = get_baseline_opt(excel="result_yc_add.xls", baseline=baseline, id=each_id)
        result_smac = get_baseline_opt(excel="result_smac_add_v2.xls", baseline=baseline, id=each_id)

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

        y5_values = result_yc
        y6_values = result_smac

        # 使用scatter()绘制散点
        plt.scatter(x_values, y1_values, s=marker_size, c="dodgerblue", marker='x', label='MES')
        plt.scatter(x_values, y2_values, s=marker_size, c="orangered", marker='^', label='EI')
        plt.scatter(x_values, y3_values, s=marker_size, c="limegreen", marker='s', label='UCB')
        plt.scatter(x_values, y4_values, s=marker_size, c="darkorchid", marker='o', label='Random')

        plt.scatter(x_values, y5_values, s=marker_size, c="darkorange", marker='*', label='GA')
        plt.scatter(x_values, y6_values, s=marker_size, c="darkgrey", marker='p', label='SMAC')

        plt.legend(fontsize=16, loc="best", frameon=True, ncol=2, columnspacing=0.2)  # 添加图例

        # plt.scatter(x_values, y1_values, s=marker_size, c="dodgerblue", marker='x', label='MES')

        # 设置图表标title()题并给坐标轴加上标签x/ylabel()
        # plt.title("Bo VS Random(seed=0/1/2/3/4)", fontsize=24)
        plt.xlabel("Epoch", font1)
        plt.ylabel("Speed-up", font1)

        # 设置刻度标记大小tick_params()
        ax = plt.gca()
        plt.tick_params(axis='both', labelsize=14)
        # plt.legend(fontsize = 14, loc="best")  # 添加图例
        plt.grid()
        x_major_locator = MultipleLocator(5)
        ax.xaxis.set_major_locator(x_major_locator)

        # plt.show()
        each_file = r'each_circuit_plot/' + str(each_id) + "_" + circuit_name[each_id] + "_plot.eps"
        each_file_png = r'each_circuit_plot/' + str(each_id) + "_" + circuit_name[each_id] + "_plot.png"

        plt.savefig(each_file_png, bbox_inches='tight')
        # plt.savefig(each_file, bbox_inches='tight')



if __name__ == "__main__":
    main()